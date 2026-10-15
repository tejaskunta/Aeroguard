$ErrorActionPreference = 'Stop'
$py  = 'C:\Users\Delta\Downloads\SafeMineAI-main\.venv\Scripts\python.exe'
$wd  = 'C:\Users\Delta\Downloads\SafeMineAI-main'
Set-Location $wd
$crops = "$wd\paper_results\stage2\crops"
& $py phase5_train_stage2.py --crops $crops --out "$wd\paper_results\stage2\s2_ratio1_s0" --ratio 1 --seed 0 --epochs 30 --patience 6 --batch 64 --workers 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py phase5_train_stage2.py --crops $crops --out "$wd\paper_results\stage2\s2_ratio3_s0" --ratio 3 --seed 0 --epochs 30 --patience 6 --batch 64 --workers 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output 'ratio search done'
