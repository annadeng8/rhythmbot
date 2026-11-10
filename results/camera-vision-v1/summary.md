# RGB camera comparison

Eight identical held-out charts; 78 notes per method/input. Controllers are unchanged. Prior color uses the calibrated flat display; concert-screen and concert-camera use the same concert artwork and the new RGB detector. The direct-concert vs camera-concert pair isolates the camera path (perspective sampling, noise and occlusion).

| Policy | Previous oracle | Previous color | Direct concert RGB | Drake camera RGB |
|---|---:|---:|---:|---:|
| classical | 70/78 (89.7%) | 69/78 (88.5%) | 66/78 (84.6%) | 66/78 (84.6%) |
| cap-v1 | 77/78 (98.7%) | 75/78 (96.2%) | 65/78 (83.3%) | 66/78 (84.6%) |
| bc-mlp | 49/78 (62.8%) | 41/78 (52.6%) | 12/78 (15.4%) | 12/78 (15.4%) |
| bc-diffusion | 44/78 (56.4%) | 36/78 (46.2%) | 28/78 (35.9%) | 28/78 (35.9%) |

| Policy | Input | Tap | Hold | Slide | Equal-chart mean |
|---|---|---:|---:|---:|---:|
| classical | concert-screen | 48/54 | 8/8 | 10/16 | 76.4% |
| classical | concert-camera | 48/54 | 8/8 | 10/16 | 76.4% |
| cap-v1 | concert-screen | 47/54 | 8/8 | 10/16 | 75.9% |
| cap-v1 | concert-camera | 47/54 | 8/8 | 11/16 | 80.1% |
| bc-mlp | concert-screen | 7/54 | 4/8 | 1/16 | 15.7% |
| bc-mlp | concert-camera | 7/54 | 4/8 | 1/16 | 15.7% |
| bc-diffusion | concert-screen | 28/54 | 0/8 | 0/16 | 13.0% |
| bc-diffusion | concert-camera | 28/54 | 0/8 | 0/16 | 13.0% |

The RGB blackout control uses nonempty training charts in full dual-arm camera simulation: zero detections and zero completions. No clean texture, depth, render labels or chart state is available to the detector. The simulator uses render labels only to paint the dynamic display without overwriting occluding robot pixels. This is an occlusion-correct camera composite, not a native animated Drake material or a real-world camera.
