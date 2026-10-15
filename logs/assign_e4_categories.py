import csv, sys
sys.stdout.reconfigure(encoding='utf-8')
p = r'C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\gradcam\e4_failures.csv'

FN_CATS = {
    'Jan_Jan_Morning_0_lwir_lwir_80__FN6.png': 'small',
    'Jan_Jan_Morning_0_lwir_lwir_5__FN4.png': 'camouflaged_or_low_contrast',
    'Jan_Jan_Morning_0_lwir_lwir_48__FN9.png': 'small',
    'Jan_Jan_Morning_0_lwir_lwir_21__FN2.png': 'camouflaged_or_low_contrast',
    'Jan_Jan_Afternoon_0_lwir_59__FN0.png': 'camouflaged_or_low_contrast',
    'Jan_Jan_Afternoon_0_lwir_59__FN2.png': 'camouflaged_or_low_contrast',
    'Jan_Jan_Afternoon_0_lwir_59__FN3.png': 'camouflaged_or_low_contrast',
    'ds0_0801_jpg.rf.7fec8a048a84003b9dcad3f5ee1eb201__FN0.png': 'label_error',
    'ds0_0883_jpg.rf.5a0b82d51317a21637b5f97493e77bcf__FN0.png': 'label_error',
}

def fp_cat(name, image):
    if 'ds0_0840' in image: return 'tree_trunk'
    if 'ds0_0868' in image: return 'fence'
    if 'ds0_0959' in image: return 'structure_wall'
    if 'ds1_0079' in image: return 'structure_edge'
    if 'ds0_0363' in image: return 'sensor_noise'
    if 'ds1_0107' in image: return 'terrain_texture'
    if 'ds1_0112' in image or 'ds1_0197' in image: return 'low_light_ground'
    return 'terrain_texture'   # Jan_Jan grass scenes: bumps/rocks/vegetation texture

rows = list(csv.DictReader(open(p, encoding='utf-8')))
counts = {}
for r in rows:
    if r['kind'] == 'FN':
        r['category'] = FN_CATS.get(r['file'], 'uncategorized')
    else:
        r['category'] = fp_cat(r['file'], r['image'])
    counts[r['category']] = counts.get(r['category'], 0) + 1

with open(p, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

fn = [k for k in counts]
print('counts:', counts)
n_fn = sum(1 for r in rows if r['kind'] == 'FN')
n_fp = sum(1 for r in rows if r['kind'] == 'FP')
print(f'FN {n_fn}, FP {n_fp}, uncategorized {counts.get("uncategorized", 0)}')
