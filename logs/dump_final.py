import json, sys
sys.stdout.reconfigure(encoding='utf-8')
base = r'C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\baseline_mnv3'
for row in ('AB_224_seed0', 'AB_224_seed1', 'AB_224_seed2'):
    m = json.load(open(base + '\\' + row + '\\metrics.json', encoding='utf-8'))
    print('==', row, 'thr', round(m['chosen_on_val']['recall_first_threshold'], 3),
          'best_ep', m.get('best_epoch_stage_B') or m.get('stage_B', {}).get('best_epoch'))
    for split in ('val', 'test'):
        d = m[split]['thr_recall_first']
        print('  ', split, {k: round(d[k], 4) for k in
                            ('precision', 'recall', 'balanced_accuracy', 'kappa')})
    sb = m.get('stage_B') or {}
    if sb:
        print('   stageB summary keys:', {k: sb[k] for k in sb if k != 'history' and k != 'history_B'} if isinstance(sb, dict) else sb)
        for split in ('val', 'test'):
            d = (sb.get(split) or {}).get('thr_recall_first') if isinstance(sb.get(split), dict) else None
            if d:
                print('   B', split, {k: round(d[k], 4) for k in
                                      ('precision', 'recall', 'balanced_accuracy', 'kappa')})
print()
y = json.load(open(r'C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\yolo\s_seed0\test_report.json', encoding='utf-8'))
print('YOLO test: ultra', {k: round(v, 4) for k, v in y['ultralytics_map']['test'].items()})
print('choice', {k: y['checkpoint_choice'][k] for k in ('chosen', 'primary_op_conf')})
print('test@op', y['image_level_at_op'])
for c in ('0.1', '0.25'):
    print('test@' + c, y['at_conf'][c])
print('keys', list(y.keys()))
