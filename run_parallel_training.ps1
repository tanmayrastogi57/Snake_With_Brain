[CmdletBinding()]
param(
    [ValidateRange(2, 32)]
    [int]$Workers = 4,

    [ValidateRange(0, 1000000)]
    [int]$Games = 1000,

    [int]$Seed = 1000,

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [switch]$Resume,

    [switch]$Render,

    [switch]$ShowGraph,

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
    '--seed', $Seed,
    '--device', $Device,
    '--checkpoint', $checkpointPath,
    '--log-file', $logPath,
    '--plot-file', $plotPath,
    '--plot-every', $PlotEvery
)

if ($Games -gt 0) {
    $arguments += '--max-games', $Games
}

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

if ($Render -or $ShowGraph) {
    $monitorArguments = 'plot_monitor.py --log-file "{0}" --output "{1}"' -f $logPath, $plotPath
    Start-Process -FilePath $pythonPath -WorkingDirectory $repoRoot -ArgumentList $monitorArguments -WindowStyle Normal | Out-Null
    Write-Output "Live graph: launched automatically"
}

Write-Output "Starting one shared learner with $Workers environment workers"
Write-Output "Device:     $Device"
Write-Output "Games:      $(if ($Games -gt 0) { $Games } else { 'unlimited' })"
Write-Output "Checkpoint: $checkpointPath"
Write-Output "Log:        $logPath"
Write-Output "Graph:      $plotPath"

& $pythonPath @arguments
exit $LASTEXITCODE
