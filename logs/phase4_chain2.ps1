$ErrorActionPreference = 'Stop'
$py   = 'C:\Users\Delta\Downloads\SafeMineAI-main\.venv\Scripts\python.exe'
$dat  = 'C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2'
$out  = 'C:\Users\Delta\Downloads\SafeMineAI-main\paper_results\baseline_mnv3'
$wd   = 'C:\Users\Delta\Downloads\SafeMineAI-main'
Set-Location $wd

& $py phase4_mnv3_baseline.py --data $dat --stage A  --imgsz 640 --batch 32 --epochs 20 --steps 25 --seed 0 --workers 2 --out "$out\A_640_seed0"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py phase4_mnv3_baseline.py --data $dat --stage AB --imgsz 224 --batch 32 --epochs 20 --steps 25 --seed 0 --workers 2 --out "$out\AB_224_seed0"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py phase4_mnv3_baseline.py --data $dat --stage A  --imgsz 224 --batch 64 --epochs 20 --steps 25 --seed 0 --workers 2 --out "$out\A_224_b64_seed0"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output 'chain done'
