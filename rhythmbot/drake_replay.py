"""Render the actual two-arm Drake scene with an occlusion-correct game screen.

HEADLESS=1 python -m rhythmbot.drake_replay --out results/drake-stage
The screen is projected onto its world plane using camera rays and Drake's
body labels; robot pixels remain the real RGB renderer output.
"""
from __future__ import annotations
import argparse
import dataclasses
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import Simulator
from .camera import CameraScreen, CAMERA_NAME
from .chart import Chart, demo_chart
from .config import Camera, Perception, two_arm_config
from .play import build
from .judge import Judge


from .sekai_screen import SekaiScreen


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('results/drake-stage'))
    ap.add_argument('--chart',type=Path)
    ap.add_argument('--duration',type=float)
    ap.add_argument('--fps',type=int,default=15)
    ap.add_argument('--live',action='store_true',help='show the Drake camera while simulating (requires a desktop)')
    args=ap.parse_args()
    if args.fps<=0: ap.error('--fps must be positive')
    chart=Chart.load(args.chart) if args.chart else demo_chart()
    cfg=dataclasses.replace(two_arm_config(),
        camera=Camera(width=960,height=720,fov_y_deg=48,
            eye_offset=(1.05,-1.5,1.5),target_offset=(0,.30,.15),noise_std=0),
        perception=Perception(frame_size=(768,512)))
    diagram,robot,brain,judge=build(chart,cfg,'oracle',None,None,record_camera=True)
    sim=Simulator(diagram); context=sim.get_mutable_context();robot.initialize(context)
    compositor=CameraScreen(chart,cfg,int(robot.plant.GetBodyByName('screen').index()))
    compositor.renderer=SekaiScreen(cfg)
    compositor.renderer.results=judge.results
    station_context=robot.station.GetMyContextFromRoot(context)
    args.out.mkdir(parents=True,exist_ok=True)
    grippers = [robot.plant.GetBodyByName('body', robot.plant.GetModelInstanceByName('wsg'+suffix))
                for suffix in cfg.arms.suffixes]
    min_gripper_distance = float('inf')
    window = None
    if args.live:
        import tkinter as tk
        from PIL import ImageTk
        window=tk.Tk(); window.title('RhythmBot / Drake bimanual')
        panel=tk.Label(window); panel.pack()
        sim.set_target_realtime_rate(1.0)
    end=args.duration if args.duration is not None else chart.end_time+1
    for i,t in enumerate(np.arange(0,end+1e-8,1/args.fps)):
        sim.AdvanceTo(float(t))
        pc = robot.plant.GetMyContextFromRoot(context)
        centers = [robot.plant.EvalBodyPoseInWorld(pc, body).translation() for body in grippers]
        min_gripper_distance = min(min_gripper_distance, float(np.linalg.norm(centers[0]-centers[1])))
        rgb=robot.station.GetOutputPort(f'{CAMERA_NAME}.rgb_image').Eval(station_context).data
        labels=robot.station.GetOutputPort(f'{CAMERA_NAME}.label_image').Eval(station_context).data[...,0]
        frame=Image.fromarray(compositor.compose(rgb,labels,float(t)))
        d=ImageDraw.Draw(frame)
        d.rectangle((0,0,960,62),fill=(15,20,37))
        try: font=ImageFont.truetype('DejaVuSans.ttf',20)
        except OSError: font=ImageFont.load_default()
        complete=sum(r.completed is True for r in judge.results.values())
        d.text((22,12),'DRAKE  /  TWO IIWA + WSG',font=font,fill=(130,246,227))
        d.text((590,12),f'{t:05.2f}s   |   {complete:02d}/{len(chart.notes)} COMPLETE',font=font,fill='white')
        # A readable inset shows the exact same display texture, not a second game.
        canvas=Image.new('RGB',(1440,720),(19,20,38))
        canvas.paste(frame,(0,0))
        screen=Image.fromarray(compositor.renderer.render(chart,float(t))).resize((480,320),Image.Resampling.LANCZOS)
        canvas.paste(screen,(960,180))
        cd=ImageDraw.Draw(canvas)
        cd.text((984,115),'LIVE SCREEN / SAME SIMULATION',font=font,fill=(196,246,239))
        cd.text((984,535),'Two Drake arms + WSG grippers',font=font,fill='white')
        cd.text((984,570),'Contact-aligned judgment line',font=font,fill=(185,184,209))
        frame=canvas
        frame.save(args.out/f'{i:06d}.png')
        if window is not None:
            panel.image=ImageTk.PhotoImage(frame)
            panel.configure(image=panel.image)
            window.update()
    judge.finalize(end)
    report=judge.report(extra={'config':dataclasses.asdict(cfg),'sim_seconds':end,'renderer':'Drake RGB + label-masked screen',
        'min_gripper_center_distance_m': min_gripper_distance})
    Judge.save_report(report,args.out/'report.json');Judge.print_report(report)
    if window is not None: window.mainloop()

if __name__=='__main__': main()
