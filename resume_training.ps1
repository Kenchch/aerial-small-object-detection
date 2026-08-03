# Resume an interrupted training run after a reboot.
#
# Ultralytics writes last.pt at the end of every epoch, so a restart costs at
# most the epoch that was in flight. --resume restores optimizer state, EMA and
# the epoch counter, so the LR schedule continues rather than restarting.
#
# Usage:   .\resume_training.ps1                          # resumes n_1024
#          .\resume_training.ps1 -Name other_run
#          .\resume_training.ps1 -Python C:\path\to\python.exe  # non-default env

param(
    [string]$Name = "n_1024",
    [string]$Python = "python"
)

# Resolves the training env's python from PATH by default -- activate the
# conda/venv env before running, or pass -Python explicitly.
$py = (Get-Command $Python -ErrorAction SilentlyContinue).Source
if (-not $py) {
    Write-Host "Could not find '$Python' on PATH. Activate the training env or pass -Python <path-to-python.exe>." -ForegroundColor Yellow
    exit 1
}

# Derived from the script's own location rather than hardcoded, so moving or
# renaming the project directory does not break it.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$last = "$root\runs\$Name\weights\last.pt"

if (-not (Test-Path $last)) {
    Write-Host "No checkpoint at $last -- nothing to resume." -ForegroundColor Yellow
    exit 1
}

$done = 0
$csv = "$root\runs\$Name\results.csv"
if (Test-Path $csv) { $done = (Import-Csv $csv).Count }
Write-Host "Resuming '$Name' from epoch $done" -ForegroundColor Cyan

# Start-Process detaches the run from this shell so closing the terminal
# does not kill it.
$p = Start-Process -FilePath $py `
    -ArgumentList @("$root\src\train.py", "--name", $Name, "--resume") `
    -RedirectStandardOutput "$root\runs\train_$Name.log" `
    -RedirectStandardError  "$root\runs\train_$Name.err" `
    -WindowStyle Hidden -PassThru

$p.Id | Out-File "$root\runs\train_$Name.pid" -Encoding ascii
Write-Host "Started detached, PID $($p.Id)" -ForegroundColor Green
Write-Host "Progress:  Import-Csv '$csv' | Select-Object -Last 3"
