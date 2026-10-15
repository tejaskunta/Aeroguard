$ErrorActionPreference = 'Stop'
$py  = 'C:\Users\Delta\Downloads\SafeMineAI-main\.venv\Scripts\python.exe'
$wd  = 'C:\Users\Delta\Downloads\SafeMineAI-main'
$dat = 'C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2'
Set-Location $wd

# 1) Phase 3 full eval: single sanctioned test touch (RUN_LOG entry written separately)
& $py phase3_eval.py --run 'runs\detect\runs\detect\landmine_yolov8s_75ep_s0_v2' --data $dat --out "$wd\paper_results\yolo\s_seed0"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2) MNv3 primary row (AB_224) seeds 1 and 2
& $py phase4_mnv3_baseline.py --data $dat --stage AB --imgsz 224 --batch 32 --epochs 20 --steps 25 --seed 1 --workers 2 --out "$wd\paper_results\baseline_mnv3\AB_224_seed1"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py phase4_mnv3_baseline.py --data $dat --stage AB --imgsz 224 --batch 32 --epochs 20 --steps 25 --seed 2 --workers 2 --out "$wd\paper_results\baseline_mnv3\AB_224_seed2"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output 'eval+seeds chain done'
