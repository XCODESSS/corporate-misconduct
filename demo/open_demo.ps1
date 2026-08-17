<#
.SYNOPSIS
Generates and opens the static XGBoost research demo.

.DESCRIPTION
This command only runs the artifact-driven page generator. It does not train,
tune, score new filings, review candidates, or reopen the frozen test set.
#>

$ErrorActionPreference = "Stop"
$demoDirectory = Split-Path -Parent $PSCommandPath
$projectRoot = Split-Path -Parent $demoDirectory

Push-Location $projectRoot
try {
    python demo/generate_xgboost_demo.py
    Start-Process (Join-Path $demoDirectory "index.html")
}
finally {
    Pop-Location
}
