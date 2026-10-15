import json, sys
sys.stdout.reconfigure(encoding='utf-8')
base = r'C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\baseline_mnv3'
for row in ('A_224_seed0', 'A_448_seed0', 'A_640_seed0', 'AB_224_seed0', 'A_224_b64_seed0'):
    try:
        m = json.load(open(base + '\\' + row + '\\metrics.json', encoding='utf-8'))
    except FileNotFoundError:
        print(row, 'MISSING')
        continue
    ch = m['chosen_on_val']['recall_first_threshold']
    parts = []
    for split in ('val', 'test'):
        d = m[split]['thr_recall_first']
        parts.append(f"{split}: P={d['precision']:.3f} R={d['recall']:.3f} "
                     f"bal={d['balanced_accuracy']:.3f} k={d['kappa']:.3f}")
    st = m.get('stage_B')
    print(row, f'thr={ch:.3f}', ' | '.join(parts), '| stageB' if st else '')
    if st:
        d = (st.get('test_thr_recall_first')
             or st.get('test', {}).get('thr_recall_first')
             or st.get('test_metrics'))
        print('   stageB keys:', list(st.keys())[:8])
