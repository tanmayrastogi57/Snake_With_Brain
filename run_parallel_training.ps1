[CmdletBinding()]
param(
    [ValidateRange(2, 32)]
    [int]$Workers = 4,

    [ValidateRange(1, 1000000)]
    [int]$Games = 1000,

    [int]$Seed = 1000,

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [switch]$Resume,

    [string]$RunDirectory = 'runs\shared'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runPath = Join-Path $repoRoot $RunDirectory
$checkpointPath = Join-Path $runPath 'shared-checkpoint.pth'
$logPath = Join-Path $runPath 'shared-training.jsonl'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python virtual environment not found at $pythonPath"
}

New-Item -ItemType Directory -Force -Path $runPath | Out-Null

$arguments = @(
    'agent.py',
    '--workers', $Workers,
    '--headless',
    '--max-games', $Games,
    '--seed', $Seed,
    '--device', $Device,
    '--checkpoint', "`"$checkpointPath`"",
    '--log-file', "`"$logPath`""
)

if (-not $Resume) {
    $arguments += '--fresh'
}

Write-Output "Starting one shared learner with $Workers environment workers"
Write-Output "Device:     $Device"
Write-Output "Checkpoint: $checkpointPath"
Write-Output "Log:        $logPath"

& $pythonPath @arguments
exit $LASTEXITCODE
