"""SEKAI-inspired presentation, independent of the detector's calibrated image.

Perspective mapping adapted from roborhythm/render/renderer.py's _project.
This viewer previews chart targets; actual robot evaluation is rhythmbot.play.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .chart import Chart, demo_chart


class StageRenderer:
    def __init__(self, width=1280, height=720, lanes=6, approach=2.4, contacts_per_arm=2):
        self.contacts_per_arm = contacts_per_arm
        self.width, self.height, self.lanes, self.approach = width, height, lanes, approach

    def project(self, lane, dt):
        d = max(0, min(1, dt / self.approach))
        p = (1 / (1 + 2.5 * d) - 1 / 3.5) / (1 - 1 / 3.5)
        return (self.width * (.5 + ((lane + .5) / self.lanes - .5) * (.17 + .65 * p)),
                self.height * (.19 + .65 * p))

    def render(self, chart, t, touches=None, results=None):
        w, h = self.width, self.height
        im = Image.new('RGB', (w, h))
        d = ImageDraw.Draw(im)
        for y in range(h):
            f = y / h
            d.line((0, y, w, y), fill=(int(12+19*f), int(17+9*f), int(38+32*f)))
        for side in (-1, 1):
            for i in range(7):
                x = w/2 + side * (w*.17 + i*w*.065)
                d.line((w/2, h*.08, x, h), fill=(35, 40, 70), width=2)
        def poly(points, fill, outline=None):
            d.polygon([self.project(a,b) for a,b in points], fill=fill, outline=outline)
        poly([(-.5,0),(-.5,self.approach),(self.lanes-.5,self.approach),(self.lanes-.5,0)], (15,22,43))
        for lane in range(self.lanes+1):
            d.line((self.project(lane-.5,0),self.project(lane-.5,self.approach)), fill=(76,83,114), width=2)
        beat = 60 / chart.bpm
        for i in range(int(t/beat), int((t+self.approach)/beat)+1):
            dt = i*beat-t
            if dt >= 0:
                d.line((self.project(-.5,dt),self.project(self.lanes-.5,dt)), fill=(37,51,73))
        for n in sorted(chart.visible(t,self.approach), key=lambda n:n.time, reverse=True):
            start=max(0,n.time-t)
            color=(105,245,226) if n.kind=='tap' else (115,244,163)
            if n.duration:
                stop=min(n.end_time-t,self.approach)
                steps=max(2,int((stop-start)*50))
                for j in range(steps):
                    a=start+(stop-start)*j/steps
                    b=start+(stop-start)*(j+1)/steps
                    la,lb=n.lane_at(t+a),n.lane_at(t+b)
                    poly([(la-.28,a),(la+.28,a),(lb+.28,b),(lb-.28,b)], (42,133,112))
            lane=n.lane_at(max(t,n.time))
            x,y=self.project(lane,start)
            x0,_=self.project(lane-.43,start); x1,_=self.project(lane+.43,start)
            thick=max(3,10*(1-start/self.approach))
            d.rounded_rectangle((x0,y-thick,x1,y+thick), radius=3, fill=color)
            d.line((x0+3,y-thick+2,x1-3,y-thick+2),fill=(230,255,255),width=2)
        a,b=self.project(-.5,0),self.project(self.lanes-.5,0)
        d.line((a,b),fill=(100,105,177),width=14)
        d.line((a,b),fill=(222,223,255),width=3)
        def text(x,y,s,size=18,fill='white'):
            try: font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',size)
            except OSError:
                try: font=ImageFont.truetype('DejaVuSans.ttf',size)
                except OSError: font=ImageFont.load_default()
            d.text((w*x,h*y),s,font=font,fill=fill)
        text(.04,.045,'RHYTHM / BOT',27)
        text(.04,.10,'VIRTUAL STAGE     /     BIMANUAL',13,(106,236,226))
        text(.68,.05,chart.name.upper()[:32],20)
        text(.68,.10,('SIMULATION / MEASURED CONTACT' if touches is not None else 'CHART PREVIEW  /  TARGET MOTION'),12,(178,184,211))
        for hand,color in ([] if touches is not None else [(0,(99,234,242)),(1,(251,140,207))]):
            targets=[n for n in chart.notes if (n.lane < (self.lanes-1)/2)==(hand==0) and n.end_time>=t]
            target=min(targets,key=lambda n:n.time) if targets else None
            lane=target.lane_at(t) if target else (1 if hand==0 else 4)
            x,y=self.project(lane,0)
            active=target is not None and target.time<=t<=target.end_time+.06
            d.ellipse((x-17,y-17,x+17,y+17),outline=color,width=4)
            if active: d.ellipse((x-8,y-8,x+8,y+8),fill=color)
            text(.08 if hand==0 else .78,.91,'L  /  LEFT HAND' if hand==0 else 'R  /  RIGHT HAND',16,color)
        if touches is not None:
            for k, (down, lane) in enumerate(touches):
                x,y=self.project(lane,0)
                color=(99,234,242) if k<self.contacts_per_arm else (251,140,207)
                d.ellipse((x-12,y-12,x+12,y+12),outline=color,width=3)
                if down: d.ellipse((x-6,y-6,x+6,y+6),fill=color)
            text(.08,.91,'L / MEASURED FINGERTIPS',16,(99,234,242))
            text(.71,.91,'R / MEASURED FINGERTIPS',16,(251,140,207))
        if results is not None:
            completed=sum(r.completed is True for r in results.values())
            misses=sum(r.status=='miss' or r.completed is False for r in results.values())
            text(.73,.28,f'{completed:03d} COMPLETE',23)
            text(.73,.34,f'{misses:03d} MISS',16,(178,184,211))
        text(.39,.94,('PHYSICS REPLAY' if touches is not None else 'SPACE  pause     R  restart'),13,(178,184,211))
        return im


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--chart',type=Path)
    ap.add_argument('--snapshot',type=Path)
    ap.add_argument('--time',type=float,default=5.8)
    args=ap.parse_args()
    chart=Chart.load(args.chart) if args.chart else demo_chart()
    renderer=StageRenderer()
    if args.snapshot:
        renderer.render(chart,args.time).save(args.snapshot)
        return
    import tkinter as tk
    from PIL import ImageTk
    root=tk.Tk(); root.title('RhythmBot · Bimanual stage preview')
    label=tk.Label(root); label.pack()
    state={'t':0.,'last':time.monotonic(),'paused':False}
    def key(event):
        if event.keysym=='space': state['paused']=not state['paused']
        if event.keysym.lower()=='r': state['t']=0.
    root.bind('<Key>',key)
    end=max(n.end_time for n in chart.notes)+2 if chart.notes else 2
    def frame():
        now=time.monotonic()
        if not state['paused']: state['t']=min(end,state['t']+now-state['last'])
        state['last']=now
        label.image=ImageTk.PhotoImage(renderer.render(chart,state['t']))
        label.configure(image=label.image)
        root.after(16,frame)
    frame(); root.mainloop()

if __name__=='__main__': main()
