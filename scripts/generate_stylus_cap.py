"""Obtain a CAP revision with no file/terminal/network tools for the coding agent.

The only supplied context is the public control contract, prior program and
training-only feedback. Never give this command validation/test results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

CONTRACT='''You are the coding-agent condition in a robotics experiment. Produce only a complete Python policy in a single fenced python block, no prose.
The policy must define make_policy(api) returning an object with plan(notes,t_now,X_WT_cmd,width_cmd).
Each instance controls ONE of two Drake iiwa arms holding a single central stylus. You cannot call the classical planner, access files, run tools, or see a chart. Only import numpy, math, dataclasses, typing, collections, itertools, functools, bisect, copy, or rhythmbot.cap.api.
An existing common router assigns tap/hold to the appropriate hand and splits cross-hand slides into overlapping local segments; all experimental policies share it. Your job is local trajectory scheduling, including tap release, long holds and changing slide paths. Notes are perceived and tracked at 30 Hz; replanning uses the previous COMMANDED stylus pose (not perfect physical tracking). No neural model needed.
API:
api.hand is 0 for left, 1 for right. 6 lanes, width 0.12m, lane_x(lane)=(lane+0.5)*0.12-0.36. Left x range [-.36,0], right [0,.36].
api.hit_y is the fixed contact y for this hand; all api.keyframe use it. api.screen_z=0.106 (sphere center for contact with surface); api.contact_z=0.110 (down threshold); api.touch_z=0.104; hover_height=0.035. Closed gripper: width=0, ref=0, rise=0 ALWAYS. Fixed 30deg tool tilt is automatically applied by api.make_plan; do not calculate rotations.
api.current_keyframe(t_now,X_WT_cmd,width_cmd) -> current keyframe; fields t,x,z,width,y,down,ref,rise.
api.keyframe(t,x,z,width=0,down=False,ref=0,rise=0) -> keyframe.
api.make_plan(keyframes,t_now=None) -> Plan through time-sorted positions. It clips the trajectory to this hand's half, legal z [0.104,0.206], and enforces tool geometry. Your plan should start at current_keyframe and contain future commands. Do not set kfs before t_now other than current. Plan linearly interpolates positions, not velocities. Simulator follows pose via differential IK, with small tracking lag.
Every note object has key, kind ('tap','hold','slide'), lane (float), hit_time (absolute seconds), duration, end_time, waypoints [(offset,lane),...], lane_at_offset(off). Mutable note flags skipped/chosen may exist; you need not use them. Notes can be in the future, active or recently finished. Copies may be regenerated at each replan. Plan from these observations only. Local slide segments can start at the midpoint (lane 2.5), including short overlaps at handoff.
Planning frequency 30Hz. Typical reliable descent time ~.22s, lift .12s, lateral speed ~.9m/s, tap dwell .06s. Faster moves may be needed but can cause tracking lag. Support uninterrupted hold/slide even as future notes appear. At most one stylus contact per hand; skip impossible conflicts rather than invent a second fingertip.
Judge: a tap or sustain head needs a FRESH down-contact near the lane inside +/-140ms; perfect +/-45ms; great +/-90ms. Staying down from an earlier note does NOT count as a fresh tap. Holds need >=80% contact. Slides need >=95% on-path contact, <=50ms continuous tracking loss, all final40ms samples close to the end lane. Slide lateral tolerance .042m, hand contact y tolerance .06m. Sustains are scored only at completion. Missing tail -> Miss even with perfect head. Best approach may coordinate hover, descend, contact, follow and lift keyframes with absolute times.
Return code that works entirely through this API. make_policy(api) is called once per hand; internal state may persist. API object contains geometry only, not notes or the classical planner. X_WT_cmd.translation() returns current stylus [x,y,z]. Do not import pydrake or rhythmbot.planner. Use api.keyframe/make_plan.
'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--previous',type=Path)
    ap.add_argument('--feedback',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    prompt=CONTRACT
    if a.previous:
        if not a.feedback:ap.error('revision requires training feedback')
        prompt+='\nPrevious candidate:\n'+a.previous.read_text()+'\nTRAINING-ONLY SIMULATION FEEDBACK:\n'+a.feedback.read_text()+'\nRevise this program using only the supplied feedback. Return the entire new program.'
    (a.out/'prompt.txt').write_text(prompt)
    cwd=tempfile.mkdtemp(prefix='rhythmbot-cap-generation-')
    cmd=['claude','-p','--output-format','json','--tools','','--no-session-persistence',
         '--disable-slash-commands','--setting-sources','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}']
    t=time.time()
    proc=subprocess.run(cmd,input=prompt,text=True,capture_output=True,cwd=cwd,
        env={**os.environ,'CLAUDE_CODE_MAX_OUTPUT_TOKENS':'48000'},timeout=900)
    (a.out/'response.json').write_text(proc.stdout)
    (a.out/'stderr.txt').write_text(proc.stderr)
    if proc.returncode:raise RuntimeError('coding-agent process failed; see saved stderr')
    reply=json.loads(proc.stdout)
    if reply.get('is_error'):raise RuntimeError(reply.get('result'))
    body=reply['result'];match=re.search(r'```(?:python)?\s*\n(.*?)```',body,re.S)
    code=match.group(1) if match else body
    (a.out/'policy.py').write_text(code.strip()+'\n')
    syntax_error=None
    try: compile(code,'policy.py','exec')
    except SyntaxError as exc: syntax_error=str(exc)
    meta={k:reply.get(k) for k in ['modelUsage','total_cost_usd','usage','num_turns','duration_ms']}
    meta.update(wall_seconds=time.time()-t,tools_enabled=False,syntax_error=syntax_error,policy_sha256=hashlib.sha256((a.out/'policy.py').read_bytes()).hexdigest())
    (a.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))
    if syntax_error: raise SystemExit(1)

if __name__=='__main__':main()
