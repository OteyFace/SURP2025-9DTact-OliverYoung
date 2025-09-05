# PowerShell script to rebuild the CUHK poster PDF
# Usage: pwsh -File 9DTact/force_estimation/build_poster.ps1

$ErrorActionPreference = 'Stop'

# Resolve project root (two levels up from this script directory)
$proj = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$texDir = Join-Path $proj 'CUHK-Poster-Template-main'
$texFile = Join-Path $texDir 'poster.tex'

if (-not (Test-Path $texFile)) {
  Write-Error "Poster TeX not found: $texFile"
}

Push-Location $texDir
try {
  # Run pdflatex twice for references/layout
  pdflatex -interaction=nonstopmode -halt-on-error poster.tex | Out-Null
  pdflatex -interaction=nonstopmode -halt-on-error poster.tex | Out-Null
  Write-Host "Poster built →" (Join-Path $texDir 'poster.pdf')
}
finally {
  Pop-Location
}


