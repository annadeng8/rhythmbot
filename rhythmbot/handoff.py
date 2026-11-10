"""Partition slide paths into overlapping, physically separated hand segments."""
from dataclasses import replace
import numpy as np
from .chart import SLIDE


def hand_notes(notes, hand, cfg):
    if cfg.arms.n == 1:
        return notes
    out=[]
    middle=(cfg.layout.n_lanes-1)/2
    # Reach with the inner fingertip, leaving the two gripper bodies separated.
    inner_x=0.0 if cfg.gripper.single_tip else cfg.arms.middle_clearance-cfg.layout.lane_width/2
    edge=cfg.layout.x_to_lane((-1 if hand==0 else 1)*inner_x)
    for n in notes:
        if n.kind != SLIDE:
            if cfg.hand_for_lane(n.lane)==hand: out.append(n)
            continue
        wp=n.waypoints or [(0,n.lane),(n.duration,n.lane)]
        times=[0.,n.duration]
        for (a,x),(b,y) in zip(wp,wp[1:]):
            if (x-middle)*(y-middle)<0:
                times.append(a+(b-a)*(middle-x)/(y-x))
            elif x==middle: times.append(a)
            elif y==middle: times.append(b)
        times=sorted(set(times))
        for i,(a,b) in enumerate(zip(times,times[1:])):
            at=lambda t: float(np.interp(t,[p[0] for p in wp],[p[1] for p in wp]))
            if cfg.hand_for_lane(at((a+b)/2))!=hand: continue
            # Both fingertips track through a short overlap at the center.
            start=max(0,a-.04);end=min(n.duration,b+.04)
            bends=[p[0] for p in wp if start<p[0]<end]
            for (ta,xa),(tb,xb) in zip(wp,wp[1:]):
                if (xa-edge)*(xb-edge)<0:
                    clipped_at=ta+(tb-ta)*(edge-xa)/(xb-xa)
                    if start<clipped_at<end: bends.append(clipped_at)
            ts=sorted(set([start,end]+bends))
            clamp=lambda x: min(x,edge) if hand==0 else max(x,edge)
            path=[(t-start,clamp(at(t))) for t in ts]
            out.append(replace(n,key=-(n.key+1)*100-i, lane=path[0][1],
                hit_time=n.hit_time+start,duration=end-start,waypoints=path,
                skipped=False,chosen=None))
    return out
