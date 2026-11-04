import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]  # repo root
d = str(ROOT / "paper_results" / "yolo" / "s_seed0")
v = json.load(open(f'{d}/val.json', encoding='utf-8'))
t = json.load(open(f'{d}/test.json', encoding='utf-8'))
print('VAL choice', {k: v['checkpoint_choice'][k] for k in ('chosen', 'primary_op_conf')})
print('VAL@op', v['image_level_at_op'])
print('TEST ultra', {k: round(x, 4) for k, x in t['ultralytics_map']['test'].items()})
print('TEST@op', t['image_level_at_op'])
for c in ('0.1', '0.25'):
    print('TEST@' + c, {k: round(x, 4) if isinstance(x, float) else x
                        for k, x in t['at_conf'][c].items()})
print('bootstrap', t.get('bootstrap_at_op'))
print('per_source', t.get('per_source'))
print('keys', list(t.keys()))
sz = json.load(open(f'{d}/size_latency.json', encoding='utf-8'))
print('size_latency', sz)
