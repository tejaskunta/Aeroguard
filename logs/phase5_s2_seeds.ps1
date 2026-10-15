$ErrorActionPreference = 'Stop'
$py  = 'C:\Users\Delta\Downloads\SafeMineAI-main\.venv\Scripts\python.exe'
$wd  = 'C:\Users\Delta\Downloads\SafeMineAI-main'
Set-Location $wd
$crops = "$wd\paper_results\stage2\crops"
& $py phase5_train_stage2.py --crops $crops --out "$wd\paper_results\stage2\s2_ratio1_s1" --ratio 1 --seed 1 --epochs 30 --patience 6 --batch 64 --workers 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py phase5_train_stage2.py --crops $crops --out "$wd\paper_results\stage2\s2_ratio1_s2" --ratio 1 --seed 2 --epochs 30 --patience 6 --batch 64 --workers 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output 'seeds done'
