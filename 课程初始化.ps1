param([switch]$SmokeTest)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $envPath = Join-Path $PSScriptRoot '.venv'
    $python = Join-Path $envPath 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        py -3.14 -m venv $envPath
        if ($LASTEXITCODE -ne 0) {
            throw "Cannot create the course environment. Check py -3.14 -V. Exit code $LASTEXITCODE."
        }
    }

    function Invoke-CoursePython {
        & $python @args
        if ($LASTEXITCODE -ne 0) {
            throw "Course command failed with exit code $LASTEXITCODE."
        }
    }

    Invoke-CoursePython -m pip install --disable-pip-version-check --only-binary=:all: -r '课程工具/课程依赖.txt'
    Invoke-CoursePython -m pip install --disable-pip-version-check torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    Invoke-CoursePython -m ipykernel install --prefix $envPath --name traffic-ai --display-name 'Python (Traffic AI)'
    Invoke-CoursePython -B -X utf8 '课程工具/快速验收.py' env
    if ($SmokeTest) {
        Invoke-CoursePython -B -X utf8 '课程工具/快速验收.py' all
    }
    Write-Host "Course Python: $python"
} finally {
    Pop-Location
}

