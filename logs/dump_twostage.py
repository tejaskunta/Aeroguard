import json, sys
sys.stdout.reconfigure(encoding='utf-8')
t = json.load(open(r'C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\twostage\s0\test_report.json', encoding='utf-8'))
c = t['chosen_on_val']
print('chosen P', c['P'], 'tau', c['tau'], 'val recall', round(c['recall'], 4), 'val fp', round(c['fp_per_image'], 3))
print('eff', t['chosen_effective'], 'armA', t['arm_a_conf'], 'armB', t['arm_b_conf'])
for k, v in t['arms'].items():
    print(k, 'P', round(v['precision'], 4), 'R', round(v['recall'], 4),
          'fp/img', round(v['fp_per_image'], 3), 'imgR', round(v['img_recall'], 4),
          'Rci', [round(x, 4) for x in v['recall_ci']])
print('paired')
for k, v in t['paired_diff_C_minus_B'].items():
    print(' ', k, 'mean', round(v['mean_diff'], 4), 'ci', [round(x, 4) for x in v['ci95']])
print('terciles')
for arm, d in t['size_terciles'].items():
    row = {}
    for bk, bv in d['recall_by_tercile'].items():
        row[bk] = (round(bv['recall'], 3) if bv['recall'] is not None else None, bv['n'])
    print(' ', arm, row)
print('cost', t['cost'])
