[CmdletBinding()]
param(
    [ValidateRange(1, 32)]
    [int]$Workers = 4,

    [ValidateRange(1, 1000000)]
    [int]$Games = 1000,

    [int]$BaseSeed = 1000,

    [switch]$Resume,

    [string]$RunDirectory = 'runs\parallel'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runPath = Join-Path $repoRoot $RunDirectory

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found at $pythonPath"
}

New-Item -ItemType Directory -Force -Path $runPath | Out-Null

for ($worker = 1; $worker -le $Workers; $worker++) {
    $workerName = "worker-$worker"
    $checkpointPath = Join-Path $runPath "$workerName.pth"
    $logPath = Join-Path $runPath "$workerName.jsonl"
    $seed = $BaseSeed + $worker - 1
    $freshArgument = if ($Resume) { @() } else { @('--fresh') }
    $arguments = @(
        'agent.py',
        '--headless',
        '--max-games', $Games,
        '--seed', $seed
    ) + $freshArgument + @(
        '--checkpoint', "`"$checkpointPath`"",
        '--log-file', "`"$logPath`""
    )

    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -PassThru

    Write-Output "$workerName started (PID $($process.Id), seed $seed)"
    Write-Output "  checkpoint: $checkpointPath"
    Write-Output "  log:        $logPath"
}

Write-Output "Started $Workers independent workers. Their checkpoints are not merged automatically."
