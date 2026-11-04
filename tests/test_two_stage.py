import random

import numpy as np

import two_stage_eval as ts


def box(x, y, s=10):
    return [x, y, x + s, y + s]


def det(b, conf, p):
    return {"box": b, "conf": conf, "cls_p": p}


def test_iou():
    assert ts.iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert ts.iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(ts.iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_match_basic_and_duplicates():
    gt = [box(0, 0)]
    # perfect hit + a duplicate on the same GT: duplicate must count as FP
    tp, fp, fn = ts.match_image(gt, [det(box(0, 0), 0.9, 1), det(box(1, 0), 0.8, 1)])
    assert (tp, fp, fn) == (1, 1, 0)
    # miss
    assert ts.match_image(gt, [det(box(50, 50), 0.9, 1)]) == (0, 1, 1)
    # no detections
    assert ts.match_image(gt, []) == (0, 0, 1)


def test_perfect_classifier_keeps_recall_removes_fp():
    recs = []
    for i in range(20):
        recs.append({"name": str(i), "gt": [box(0, 0)],
                     "dets": [det(box(0, 0), 0.30, 0.95), det(box(60, 60), 0.20, 0.05)]})
    s1 = ts.evaluate(recs, 0.05)
    s2 = ts.evaluate(recs, 0.05, 0.5)
    assert s1["recall"] == s2["recall"] == 1.0
    assert s1["fp"] == 20 and s2["fp"] == 0
    assert s2["precision"] == 1.0


def test_bad_classifier_costs_recall_and_threshold_respects_constraint():
    rng = random.Random(1)
    recs = []
    for i in range(200):
        p_tp = rng.uniform(0.3, 1.0)           # classifier is imperfect on true mines
        recs.append({"name": str(i), "gt": [box(0, 0)],
                     "dets": [det(box(0, 0), 0.4, p_tp), det(box(60, 60), 0.3, rng.uniform(0.0, 0.6))]})
    base = ts.evaluate(recs, 0.05)["recall"]
    thr = ts.pick_cls_threshold(recs, 0.05, max_recall_drop=0.01)
    after = ts.evaluate(recs, 0.05, thr)
    assert after["recall"] >= base - 0.01 - 1e-12
    assert after["fp"] < ts.evaluate(recs, 0.05)["fp"]          # still removes false positives
    assert ts.evaluate(recs, 0.05, 0.95)["recall"] < base       # an aggressive threshold DOES hurt recall


def test_protect_conf_never_loses_confident_hits():
    recs = [{"name": "a", "gt": [box(0, 0)], "dets": [det(box(0, 0), 0.9, 0.0)]}]  # classifier wrongly says no
    assert ts.evaluate(recs, 0.05, 0.5)["recall"] == 0.0
    assert ts.evaluate(recs, 0.05, 0.5, protect_conf=0.8)["recall"] == 1.0


def test_two_stage_never_beats_stage1_recall_property():
    """Stage 2 only removes detections, so recall can't rise and FP can't rise."""
    rng = random.Random(7)
    for trial in range(300):
        recs = []
        for i in range(8):
            gt = [box(rng.randint(0, 3) * 6, rng.randint(0, 3) * 6) for _ in range(rng.randint(0, 3))]
            dets = [det(box(rng.randint(0, 3) * 6, rng.randint(0, 3) * 6), rng.random(), rng.random())
                    for _ in range(rng.randint(0, 5))]
            recs.append({"name": str(i), "gt": gt, "dets": dets})
        s1 = ts.evaluate(recs, 0.1)
        s2 = ts.evaluate(recs, 0.1, rng.random(), rng.choice([1.01, 0.7]))
        assert s2["tp"] <= s1["tp"], (trial, s1, s2)
        assert s2["fp"] <= s1["fp"], (trial, s1, s2)


def test_matched_recall_baseline_is_fair():
    """YOLO-only baseline must be at the strictest threshold that still reaches the target recall."""
    rng = random.Random(3)
    recs = []
    for i in range(100):
        recs.append({"name": str(i), "gt": [box(0, 0)],
                     "dets": [det(box(0, 0), rng.uniform(0.2, 1.0), 0.9),
                              det(box(60, 60), rng.uniform(0.0, 0.7), 0.1)]})
    target = ts.evaluate(recs, 0.05, 0.5)["recall"]
    b = ts.yolo_only_at_recall(recs, target)
    assert b["recall"] >= target
    higher = ts.evaluate(recs, min(1.0, b["yolo_conf"] + 0.01))
    assert higher["recall"] < target or b["yolo_conf"] >= 0.995


def test_image_level_metrics_and_bootstrap():
    recs = [{"name": "p", "gt": [box(0, 0)], "dets": [det(box(0, 0), 0.9, 1)]},
            {"name": "n", "gt": [], "dets": []},
            {"name": "fp", "gt": [], "dets": [det(box(5, 5), 0.8, 1)]}]
    m = ts.evaluate(recs, 0.05)
    assert (m["img_tp"], m["img_fp"], m["img_fn"], m["img_tn"]) == (1, 1, 0, 1)
    ci = ts.bootstrap(recs, 0.05, n_boot=200, seed=0)
    assert 0.0 <= ci["recall_ci"][0] <= ci["recall_ci"][1] <= 1.0


def test_paired_bootstrap_detects_real_gain_and_null():
    rng = random.Random(11)
    recs = []
    for i in range(150):
        recs.append({"name": str(i), "gt": [box(0, 0)],
                     "dets": [det(box(0, 0), 0.5, 0.95), det(box(60, 60), 0.3, 0.02), det(box(80, 80), 0.2, 0.03)]})
    a = {"yolo_conf": 0.05, "cls_thr": 0.5, "protect_conf": 1.01}
    b = {"yolo_conf": 0.05}
    d = ts.paired_bootstrap_diff(recs, a, b, n_boot=200)
    assert d["fp_per_image"]["ci95"][1] < 0          # two-stage clearly has fewer false alarms
    assert d["recall"]["mean_diff"] == 0.0
    same = ts.paired_bootstrap_diff(recs, b, b, n_boot=200)
    assert same["fp_per_image"]["ci95"] == [0.0, 0.0]  # identical arms: difference is exactly zero


def test_yolo_only_curve_monotone_recall():
    rng = random.Random(5)
    recs = [{"name": str(i), "gt": [box(0, 0)],
             "dets": [det(box(0, 0), rng.random(), 0.5), det(box(60, 60), rng.random(), 0.5)]} for i in range(100)]
    c = ts.yolo_only_curve(recs, n=21)
    rec = [r["recall"] for r in c]
    assert all(rec[i] >= rec[i + 1] - 1e-12 for i in range(len(rec) - 1))   # higher threshold never raises recall
