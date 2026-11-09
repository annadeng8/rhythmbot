"""Original concert-stage artwork for the physical Drake display.

Six control lanes are subdivided into twelve visual strips. Perspective is
identity at the judgment line, preserving the robot's calibrated touch targets.
No changes to the perception/training renderer or judgment windows.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class SekaiScreen:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W, self.H = cfg.perception.frame_size
        self.hit_row = int(self.H * .86)
        self.results = {}
        self.fonts = {}

    def font(self, size):
        size = max(10, int(size * self.W / 768))
        if size not in self.fonts:
            for name in ('DejaVuSans-Bold.ttf','/System/Library/Fonts/Supplemental/Arial Bold.ttf'):
                try:
                    self.fonts[size] = ImageFont.truetype(name,size)
                    break
                except OSError: pass
            else: self.fonts[size] = ImageFont.load_default()
        return self.fonts[size]

    def project(self, lane, dt):
        depth = min(1.,max(0.,dt/self.cfg.layout.approach_time))
        # Contract distant lanes; exactly match physical x/y at contact.
        scale = 1-.76*depth
        x = self.W*(.5+((lane+.5)/self.cfg.layout.n_lanes-.5)*scale)
        y = self.hit_row*(1-.79*depth)
        return x,y

    def render(self, chart, t):
        w,h=self.W,self.H
        img=Image.new('RGB',(w,h));d=ImageDraw.Draw(img)
        for y in range(h):
            f=y/h
            d.line((0,y,w,y),fill=(int(28+22*f),int(19+12*f),int(61+30*f)))
        # Abstract concert set, deliberately original, with moving light beams.
        for side in (-1,1):
            for k in range(4):
                x=w*(.5+side*(.25+k*.09)); swing=math.sin(t*.6+k)*w*.10
                d.polygon([(x,0),(x+side*16,0),(w*.5+swing+side*w*.32,h*.80),
                           (w*.5+swing+side*w*.18,h*.80)],fill=(39+k*3,35+k*3,85+k*5))
        for k in range(10):
            x=w*(k+.5)/10
            d.line((x,h*.26,x,h*.70),fill=(72,61,109),width=2)
            d.ellipse((x-3,h*.30-3,x+3,h*.30+3),fill=(133,112,194))
        # Translucent highway with twelve divisions, and lit side rails.
        layer=Image.new('RGBA',(w,h));ld=ImageDraw.Draw(layer)
        top=self.cfg.layout.approach_time; n=self.cfg.layout.n_lanes
        corners=[self.project(-.5,0),self.project(-.5,top),self.project(n-.5,top),self.project(n-.5,0)]
        ld.polygon(corners,fill=(10,15,36,178))
        for k in range(13):
            lane=k*n/12-.5
            ld.line((self.project(lane,0),self.project(lane,top)),fill=(210,185,255,65 if k%2 else 105),width=1)
        for lane in (-.5,n-.5):
            ld.line((self.project(lane,0),self.project(lane,top)),fill=(229,181,255,240),width=3)
        img=Image.alpha_composite(img.convert('RGBA'),layer)
        notes=Image.new('RGBA',(w,h));nd=ImageDraw.Draw(notes)
        for note in sorted(chart.visible(t,top),key=lambda a:a.time,reverse=True):
            start=max(0,note.time-t); end=min(top,note.end_time-t)
            color=(92,244,255,255) if note.kind=='tap' else (111,255,153,255)
            if note.duration and end>start:
                steps=max(2,int((end-start)*45))
                for j in range(steps):
                    a=start+(end-start)*j/steps;b=start+(end-start)*(j+1)/steps
                    la,lb=note.lane_at(t+a),note.lane_at(t+b)
                    nd.polygon([self.project(la-.37,a),self.project(la+.37,a),
                                self.project(lb+.37,b),self.project(lb-.37,b)],fill=(58,237,132,145))
                    nd.line((self.project(la-.37,a),self.project(lb-.37,b)),fill=(176,255,194,245),width=2)
                    nd.line((self.project(la+.37,a),self.project(lb+.37,b)),fill=(176,255,194,245),width=2)
            lane=note.lane_at(max(t,note.time));x,y=self.project(lane,start)
            a,_=self.project(lane-.45,start);b,_=self.project(lane+.45,start)
            thick=max(3,8*(1-.65*start/top))
            nd.rounded_rectangle((a,y-thick,b,y+thick),radius=3,fill=color,outline=(235,255,255,255),width=2)
            nd.line((a+3,y,b-3,y),fill=(242,255,255,255),width=2)
        # The judge, not the chart, drives feedback and combo.
        judged=sorted((r for r in self.results.values() if r.judgment is not None),
                      key=lambda r:r.time+(r.timing_error or 0))
        combo=0
        for r in judged:
            combo=combo+1 if r.status=='hit' and r.completed is not False else 0
        for r in judged:
            age=t-(r.time+(r.timing_error or 0))
            if r.status=='hit' and 0<=age<.32:
                x,y=self.project(next(n for n in chart.notes if n.id == r.id).lane_at(t),0)
                radius=12+age*100
                nd.ellipse((x-radius,y-radius*.35,x+radius,y+radius*.35),outline=(255,227,167,int(255*(1-age/.32))),width=3)
                for k in range(6):
                    dx=math.cos(k*math.pi/3)*radius;dy=-abs(math.sin(k*math.pi/3)*radius)*1.5
                    nd.ellipse((x+dx-2,y+dy-2,x+dx+2,y+dy+2),fill=(255,246,190,255))
        glow=notes.filter(ImageFilter.GaussianBlur(7))
        img=Image.alpha_composite(img,glow);img=Image.alpha_composite(img,notes)
        d=ImageDraw.Draw(img)
        d.line((0,self.hit_row,w,self.hit_row),fill=(237,201,255),width=5)
        d.line((0,self.hit_row+5,w,self.hit_row+5),fill=(117,86,182),width=4)
        def text(x,y,s,size=18,fill='white',anchor=None):
            d.text((x*w,y*h),s,font=self.font(size),fill=fill,anchor=anchor,stroke_width=1,stroke_fill=(39,28,64))
        d.rounded_rectangle((w*.025,h*.025,w*.63,h*.115),radius=10,fill=(242,242,253))
        score=sum({'perfect':3,'great':2,'good':1}.get(r.judgment,0) for r in judged)
        text(.045,.038,f'SCORE  {score:07d}',20,(57,46,82))
        text(.38,.046,'C    B    A    S',14,(96,83,122))
        d.rounded_rectangle((w*.05,h*.092,w*.59,h*.101),radius=2,fill=(191,179,216))
        d.rounded_rectangle((w*.05,h*.092,w*(.05+.54*score/max(1,3*len(chart.notes))),h*.101),radius=2,fill=(71,211,210))
        text(.72,.04,'RHYTHM / BOT',17,(194,245,255))
        text(.73,.10,'DUAL ARM / LIVE',12,(243,187,255))
        text(.85,.36,f'{combo}',48,(244,206,255),anchor='mt')
        text(.85,.48,'COMBO',16,(244,206,255),anchor='mt')
        if judged:
            r=judged[-1];age=t-(r.time+(r.timing_error or 0))
            if 0<=age<.45:
                text(.5,.73,r.judgment.upper(),28,(230,208,255),anchor='mt')
        text(.5,.935,chart.name.upper()+'  /  LIVE',14,(215,207,244),anchor='mt')
        return np.asarray(img.convert('RGB'))
