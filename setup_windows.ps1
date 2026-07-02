param(
    [string]$PythonCmd = $env:PYTHON_CMD,
    [string]$VenvDir = $env:VENV_DIR
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppEntry = Join-Path $ScriptDir "contextual2.py"
$RequirementsFile = Join-Path $ScriptDir "requirements.txt"

if ([string]::IsNullOrWhiteSpace($VenvDir)) {
    $VenvDir = Join-Path $ScriptDir "venv"
}

function Write-Green {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Red {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
}

function Resolve-Python {
    param([string]$Requested)

    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return $Requested
    }

    $Candidates = @("py -3", "python", "python3")
    foreach ($Candidate in $Candidates) {
        $Command = $Candidate.Split(" ")[0]
        if (Get-Command $Command -ErrorAction SilentlyContinue) {
            return $Candidate
        }
    }

    return $null
}

function Invoke-Python {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    $Parts = $Command.Split(" ", 2)
    if ($Parts.Count -eq 2) {
        & $Parts[0] $Parts[1] @Arguments
    } else {
        & $Command @Arguments
    }
}

Write-Green "Starting setup for Contextual on Windows..."

$ResolvedPython = Resolve-Python $PythonCmd
if ([string]::IsNullOrWhiteSpace($ResolvedPython)) {
    Write-Red "Error: Could not locate Python. Install Python 3 and/or set PYTHON_CMD."
    exit 1
}

if (-not (Test-Path $AppEntry)) {
    Write-Red "Missing contextual2.py at: $AppEntry"
    exit 1
}

if (-not (Test-Path $RequirementsFile)) {
    Write-Red "Missing requirements.txt at: $RequirementsFile"
    exit 1
}

if (-not (Test-Path $VenvDir)) {
    Write-Green "Creating virtual environment at: $VenvDir"
    Invoke-Python $ResolvedPython @("-m", "venv", $VenvDir)
} else {
    Write-Host "Virtual environment already exists at $VenvDir."
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Red "Error: venv Python was not found at $VenvPython."
    exit 1
}

Write-Green "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

$TempRequirements = Join-Path $env:TEMP "contextual_windows_requirements.txt"
Get-Content $RequirementsFile |
    Where-Object { $_ -notmatch "^\s*uvloop(\s|#|$)" } |
    Set-Content -Path $TempRequirements -Encoding UTF8

Write-Green "Installing dependencies from requirements.txt, skipping Unix-only uvloop..."
& $VenvPython -m pip install -r $TempRequirements

Write-Green "Setup complete."
Write-Host ""
Write-Host "Run the app with:"
Write-Host "  .\run_chatbot.ps1"
