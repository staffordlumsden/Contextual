param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { Join-Path $ScriptDir "venv" }
$Python = Join-Path $VenvDir "Scripts\python.exe"
$AppEntry = Join-Path $ScriptDir "contextual2.py"

if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at $VenvDir, or python.exe is missing. Run .\setup_windows.ps1 first."
    exit 1
}

if (-not (Test-Path $AppEntry)) {
    Write-Error "App entrypoint not found at $AppEntry."
    exit 1
}

& $Python $AppEntry @AppArgs
