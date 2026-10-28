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

    [switch]$Render,

    [string]$RunDirectory = 'runs\shared',

    [string]$PlotFile = '',

    [ValidateRange(1, 1000000)]
    [int]$PlotEvery = 10,

    [ValidateRange(0, 1000000)]
    [int]$EvalGames = 0,

    [ValidateRange(0, 1000000)]
    [int]$EvalEvery = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runPath = Join-Path $repoRoot $RunDirectory
$checkpointPath = Join-Path $runPath 'shared-checkpoint.pth'
$logPath = Join-Path $runPath 'shared-training.jsonl'
$plotPath = if ($PlotFile) { $PlotFile } else { Join-Path $runPath 'shared-training.png' }

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
    '--checkpoint', $checkpointPath,
    '--log-file', $logPath,
    '--plot-file', $plotPath,
    '--plot-every', $PlotEvery
)

if (-not $Resume) {
    $arguments += '--fresh'
}

if ($Render) {
    $arguments += '--render-workers'
}

if ($EvalGames -gt 0) {
    $arguments += '--eval-games', $EvalGames
    if ($EvalEvery -gt 0) {
        $arguments += '--eval-every', $EvalEvery
    }
}

Write-Output "Starting one shared learner with $Workers environment workers"
Write-Output "Device:     $Device"
Write-Output "Checkpoint: $checkpointPath"
Write-Output "Log:        $logPath"
Write-Output "Graph:      $plotPath"

& $pythonPath @arguments
exit $LASTEXITCODE
