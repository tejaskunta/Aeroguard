import json, sys
sys.stdout.reconfigure(encoding="utf-8")
for s in (0, 1, 2):
    d = json.load(open(rf"C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\yolo\s_seed{s}\test.json", encoding="utf-8"))
    v = json.load(open(rf"C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\yolo\s_seed{s}\val.json", encoding="utf-8"))
    o = d["ops"]["op97_primary"]
    print(
        f"seed {s}: val mAP50 {v['ultralytics_map']['val']['map50']:.4f}, "
        f"test mAP50 {d['ultralytics_map']['test']['map50']:.4f}, "
        f"op conf {o['conf']}, P {o['precision']:.4f} R {o['recall']:.4f} fp/img {o['fp_per_image']:.3f}"
    )
