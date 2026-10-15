$ErrorActionPreference = 'Stop'
$py  = 'C:\Users\Delta\Downloads\SafeMineAI-main\.venv\Scripts\python.exe'
$wd  = 'C:\Users\Delta\Downloads\SafeMineAI-main'
Set-Location $wd
foreach ($s in @('1','2')) {
  & $py phase5_pipeline.py --yolo "$wd\runs\detect\runs\detect\landmine_yolov8s_75ep_s0_v2\weights\last.pt" --cls "$wd\paper_results\stage2\s2_ratio1_s$s\best.pt" --data 'C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2' --out "$wd\paper_results\twostage\s$s" --conf 0.05 --arm-a-conf 0.02
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Output 'pipeline seeds done'
