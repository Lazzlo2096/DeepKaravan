param(
    [ValidateSet('up', 'down', 'ps', 'logs', 'setup', 'worker')]
    [string]$Action = 'up'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$root = (Get-Location).Path
$python = Join-Path $root '.venv-amd/Scripts/python.exe'
$state = Join-Path $root '.cache/windows-amd'
$pidFile = Join-Path $state 'worker.pid'
$stdout = Join-Path $state 'worker.log'
$stderr = Join-Path $state 'worker-error.log'
New-Item -ItemType Directory -Force -Path $state | Out-Null

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
function Get-Worker {
    if (Test-Path -LiteralPath $pidFile) {
        $workerId = [int](Get-Content -LiteralPath $pidFile)
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$workerId"
        if ($process -and $process.ExecutablePath -eq $python -and
            $process.CommandLine -match '-m worker') { return $process }
    }
    return $null
}
function Set-WorkerEnvironment {
    $env:PYTHONPATH = "$root;$root/celery"
    $env:HF_HOME = Join-Path $root '.cache/huggingface'
    $env:TOKENIZERS_PARALLELISM = 'false'
    $env:PYTHONUNBUFFERED = '1'
    $env:INFERENCE_DEVICE = 'amd'
    $env:REDIS_BROKER_URL = 'redis://127.0.0.1:6379/0'
    $env:REDIS_RESULTS_URL = 'redis://127.0.0.1:6379/1'
}
$compose = @('compose', '-f', 'compose.yaml', '-f', 'compose.windows-amd.yaml')

switch ($Action) {
    'setup' {
        if (-not (Test-Path -LiteralPath $python)) {
            Invoke-Checked uv -Arguments @('venv', '--python', '3.12', '.venv-amd')
        }
        Invoke-Checked uv -Arguments @('pip', 'install', '--python', $python, '-r', 'requirements/windows-amd.lock')
        Invoke-Checked $python -Arguments @('scripts/check_amd.py')
    }
    'up' {
        & $PSCommandPath setup
        Invoke-Checked docker -Arguments ($compose + @('up', '-d', '--build', '--remove-orphans', 'redis', 'api'))
        if (-not (Get-Worker)) {
            Set-WorkerEnvironment
            $process = Start-Process -FilePath $python -ArgumentList @('-m', 'worker') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
            Set-Content -LiteralPath $pidFile -Value $process.Id
            Start-Sleep -Seconds 3
            if ($process.HasExited) { throw "GPU worker exited. See $stderr" }
            Write-Host "Windows AMD GPU worker started: $($process.Id). Logs: $stderr"
        } else { Write-Host 'Windows AMD GPU worker already running.' }
    }
    'worker' {
        Set-WorkerEnvironment
        Invoke-Checked $python -Arguments @('-m', 'worker')
    }
    'down' {
        $process = Get-Worker
        if ($process) {
            Set-WorkerEnvironment
            # Shutdown is intentionally fire-and-forget: the exiting worker
            # cannot reply to `celery control shutdown` reliably on Windows.
            'from common.celery_app import create_celery_app; from common.config import get_settings; create_celery_app(get_settings()).control.broadcast("shutdown")' | & $python -
            if ($LASTEXITCODE -ne 0) { throw 'Could not request GPU worker shutdown' }
            Wait-Process -Id $process.ProcessId -Timeout 120 -ErrorAction SilentlyContinue
            if (Get-Worker) { throw 'GPU worker is still busy; containers were left running.' }
        }
        Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
        Invoke-Checked docker -Arguments ($compose + @('down'))
    }
    'ps' {
        Invoke-Checked docker -Arguments ($compose + @('ps'))
        $process = Get-Worker
        if ($process) { Write-Host "Windows AMD GPU worker: running (PID $($process.ProcessId))" }
        else { Write-Host 'Windows AMD GPU worker: stopped' }
    }
    'logs' { Get-Content -LiteralPath $stderr -Tail 100 -Wait }
}
