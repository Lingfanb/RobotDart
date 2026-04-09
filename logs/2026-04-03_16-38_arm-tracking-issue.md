# SONIC Sim Filter Arm Tracking Issue Discovery
**Date:** 2026-04-03 16:38
**Session summary:** Discovered that GEAR-SONIC sim filter smooths out arm motions to near-default pose. Decided to use sim filter only for clip selection, training data from original GMR retarget.

## Context
Ran `vis_gmr_filtered.py` to verify filtered motion quality. Rendered MP4 videos showed robot with arms stuck in a raised default-ish pose, not matching original retarget.

## What was done

### Visualization setup
- Fixed headless rendering: `MUJOCO_GL=egl` required for the server (no X11 display)
- Fixed PyOpenGL EGL issue: upgraded `PyOpenGL==3.1.7` (was 3.1.0, broke EGL extension)
- Successfully rendered 5 random clips to `data/verify_g1/filtered_vis/*.mp4`

### Investigation of arm issue
1. **Checked dof_pos values in GMR_filtered npz**: Arms are NOT zero — elbow mean abs ~0.8 rad, shoulder_pitch varies. Data looks non-zero.
2. **Checked MuJoCo model joint order**: Confirmed 29-DOF order matches between npz and model (left_leg → right_leg → waist → left_arm → right_arm).
3. **Checked joint limits**: Elbow range [-1.047, 1.700], only 50/2282 clips exceed limits. Normal.
4. **Checked `batch_sim_record.py`**: `rec_qpos` stores `q_mj_now = mj_data.qpos[7:7+29]` — absolute MuJoCo joint angles, NOT offsets.
5. **Compared sim_recorded vs original GMR retarget**:
   - sim_recorded arms stay close to SONIC default pose (shoulder_pitch=0.2, elbow=0.6)
   - Original GMR retarget has much richer arm motion (shoulder_pitch ranges -2.6 to +1.0, elbow 0 to 1.6)

### Root cause
- GEAR-SONIC's tracking policy **prioritizes balance (legs/torso) over arm tracking**
- Arms are "best effort" — the policy tends to keep them near the default pose `DEFAULT_DOF_POS_MJ`
- Default arm pose: `shoulder_pitch=0.2, shoulder_roll=±0.2, elbow=0.6` (arms slightly raised, elbows bent)
- This is a **fundamental limitation of SONIC WBC**, not a data bug

### Original GMR retarget 43-DOF layout (confirmed)
```
[0:6]   left leg
[6:12]  right leg
[12:15] waist
[15:22] left arm
[22:29] LEFT HAND (all zeros) ← not right arm!
[29:36] right arm
[36:43] RIGHT HAND (all zeros)
```
`convert_gmr_pkl_to_sonic_npz.py` correctly maps: `[0:22] + [29:36]` → 29-DOF

## Key findings
1. **SONIC sim filter destroys arm motion quality** — arms converge to default pose during tracking
2. **sim_recorded data should NOT be used directly for training** if arm motion matters
3. **Better approach**: Use sim filter results only as **clip selection** (which 2282 clips are physically feasible), then use **original GMR retarget PKL** data for those clips
4. Original GMR_retarget/ must be preserved (not deleted) — it's the source of accurate arm data
5. `MUJOCO_GL=egl` and `PyOpenGL>=3.1.7` needed for headless rendering on this server

## Data/files affected

### Created
| File | Description |
|------|-------------|
| `data/verify_g1/filtered_vis/*.mp4` | 5 rendered videos showing arm tracking issue |

### Environment changes
- `PyOpenGL` upgraded from 3.1.0 to 3.1.7 in DART conda env (pyrender wants 3.1.0 but EGL needs 3.1.7)
- `mink`, `qpsolvers`, `daqp` installed in DART conda env (user installed to fix GMR import, but not actually needed)

## Next steps
1. **New data strategy**: Use sim filter as selection only, training data from original GMR retarget
   - Read `sim_recorded/summary.csv` to get list of 2282 successful clip names
   - Match to `GMR_retarget/` PKL files → copy/symlink to `GMR_filtered/` as PKL
   - Update `extract_dataset_g1.py` input path or write new script
2. Regenerate `seq_data_g1/` and `mp_data_g1/` from filtered retarget PKLs
3. Retrain VAE and denoiser
4. Consider: is there a better WBC that preserves arm tracking? (SONIC limitation)
