"""Generate configs/MoGen/data/bones_whitelist.txt + data/G1_Filtered_DATA/BONES_filtered/meta.json.

Whitelist: one clip name per line, only status=='success' clips → for DataLoader filtering.
Meta: dataset-level info (counts, schema, generation params) → for reproducibility.
"""
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

DART_ROOT = Path(__file__).resolve().parents[4]
SONIC_OUT = DART_ROOT / 'data/G1_Filtered_DATA/BONES_filtered'
WHITELIST_PATH = DART_ROOT / 'configs/MoGen/data/bones_whitelist.txt'
META_PATH = SONIC_OUT / 'meta.json'

# ---------------------------------------------------------------------------
# Whitelist
# ---------------------------------------------------------------------------
df = pd.read_csv(SONIC_OUT / 'summary.csv')
white = sorted(df[df.status == 'success']['name'].tolist())
WHITELIST_PATH.parent.mkdir(parents=True, exist_ok=True)
WHITELIST_PATH.write_text('\n'.join(white) + '\n')
print(f'whitelist: {len(white)} clips → {WHITELIST_PATH.relative_to(DART_ROOT)}')

# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
status_counts = df['status'].value_counts().to_dict()

# Per-class success rate (use clip name prefix, drop trailing _001/__A### suffixes)
def get_class(name):
    base = re.sub(r'__A\d+$', '', name)
    base = re.sub(r'_\d{3}$', '', base)
    return base
df['cls'] = df['name'].apply(get_class)
per_class = df.groupby('cls').agg(
    total=('status', 'size'),
    success=('status', lambda s: (s == 'success').sum()),
).reset_index()
per_class['succ_rate'] = (per_class['success'] / per_class['total']).round(4)
# Top-25 most populous classes for the meta (full breakdown stays in summary.csv)
top_classes = per_class.sort_values('total', ascending=False).head(25)
top_classes_dict = top_classes.to_dict(orient='records')

meta = {
    'schema_version': 2,
    'generated_at': datetime.now().isoformat(timespec='seconds'),
    'source': {
        'dataset': 'BONES',
        'subset': 'non-mirror clips',
        'input_dir': str((DART_ROOT / 'data/raw/bones_sonic_input').relative_to(DART_ROOT)),
        'fps_in': 50,
    },
    'sonic_filter': {
        'controller': 'GEAR-SONIC WBC (29-DOF G1)',
        'deploy_dir': '/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy',
        'fps_sim': 50,
        'decimation': 4,                 # mj_step at 200Hz, policy at 50Hz
        'frame0_alignment': 'prepended motion[0] ground truth (zero error)',
        'warmup_steps': 0,
        'fade_steps': 0,
        'elastic_band': 'disabled (target locks to motion[0])',
    },
    'filter_criteria': {
        'fall_pitch_deg': 60.0,
        'fall_height_z': 0.4,
        'knee_below_ground_z_threshold': 0.0,
        'pelvis_drift_abs_floor_m': 0.3,
        'pelvis_drift_ratio_limit': 1.5,
        'foot_contact_force_threshold_N': 5.0,
    },
    'totals': {
        'all_clips': int(len(df)),
        'success': int(status_counts.get('success', 0)),
        'fall': int(status_counts.get('fall', 0)),
        'pelvis_drift': int(status_counts.get('pelvis_drift', 0)),
        'knee_below_ground': int(status_counts.get('knee_below_ground', 0)),
        'error': int(status_counts.get('error', 0)),
        'success_rate': round((df.status == 'success').mean(), 4),
    },
    'top_25_classes_by_clip_count': top_classes_dict,
    'paths': {
        'successful_dir': 'successful/',
        'failed_dir': 'failed/',
        'summary_csv': 'summary.csv',
        'whitelist': str(WHITELIST_PATH.relative_to(DART_ROOT)),
    },
    'npz_schema_per_clip': {
        'orig_dof_pos':       {'shape': '(T, 29)', 'dtype': 'float32', 'units': 'rad', 'note': 'BONES original, MJ joint order, 50fps'},
        'orig_root_pos':      {'shape': '(T, 3)',  'dtype': 'float32', 'units': 'm'},
        'orig_root_quat':     {'shape': '(T, 4)',  'dtype': 'float32', 'note': 'wxyz'},
        'sim_dof_pos':        {'shape': '(T, 29)', 'dtype': 'float32', 'note': 'WBC sim output, frame 0 = motion[0]'},
        'sim_dof_vel':        {'shape': '(T, 29)', 'dtype': 'float32', 'units': 'rad/s', 'note': 'verified via backward-diff'},
        'sim_actions':        {'shape': '(T, 29)', 'dtype': 'float32', 'note': 'IsaacLab order'},
        'sim_torques':        {'shape': '(T, 29)', 'dtype': 'float32', 'units': 'N·m'},
        'sim_root_pos':       {'shape': '(T, 3)',  'dtype': 'float32', 'units': 'm'},
        'sim_root_quat':      {'shape': '(T, 4)',  'dtype': 'float32', 'note': 'wxyz'},
        'pelvis_lin_vel':     {'shape': '(T, 3)',  'dtype': 'float32', 'units': 'm/s', 'note': 'world frame'},
        'pelvis_ang_vel':     {'shape': '(T, 3)',  'dtype': 'float32', 'units': 'rad/s', 'note': 'world frame'},
        'left_foot_contact':  {'shape': '(T,)',    'dtype': 'bool',    'note': 'cfrc_ext > 5N'},
        'right_foot_contact': {'shape': '(T,)',    'dtype': 'bool'},
        'left_foot_force':    {'shape': '(T, 3)',  'dtype': 'float32', 'units': 'N'},
        'right_foot_force':   {'shape': '(T, 3)',  'dtype': 'float32', 'units': 'N'},
        'link_pos_local':     {'shape': '(T, 29, 3)', 'dtype': 'float32', 'note': 'pelvis-local FK, 29 body links'},
        'com_pos':            {'shape': '(T, 3)',  'dtype': 'float32', 'note': 'world frame center of mass'},
        'ref_frame':          {'shape': '(T,)',    'dtype': 'int32',   'note': 'which orig frame each sim frame tracks'},
        'fps':                {'shape': 'scalar',  'dtype': 'float32', 'value': 50.0},
        'segment_boundaries': {'shape': '(K+1,)',  'dtype': 'int32',   'note': 'passed through from input'},
        'segment_labels':     {'shape': '(K,)',    'dtype': 'object',  'note': 'string labels per segment'},
    },
    'summary_csv_columns': list(df.columns),
}

META_PATH.parent.mkdir(parents=True, exist_ok=True)
META_PATH.write_text(json.dumps(meta, indent=2))
print(f'meta.json:  {len(json.dumps(meta))} chars → {META_PATH.relative_to(DART_ROOT)}')
print()
print('--- Quick stats ---')
print(f'  success:           {meta["totals"]["success"]:>6} ({meta["totals"]["success_rate"]*100:.1f}%)')
print(f'  fall:              {meta["totals"]["fall"]:>6}')
print(f'  pelvis_drift:      {meta["totals"]["pelvis_drift"]:>6}')
print(f'  knee_below_ground: {meta["totals"]["knee_below_ground"]:>6}')
print(f'  total:             {meta["totals"]["all_clips"]:>6}')
