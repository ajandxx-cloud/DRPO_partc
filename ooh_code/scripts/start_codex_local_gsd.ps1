param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CodexArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$localCodexHome = Join-Path $repoRoot ".codex"
$localSkillsDir = Join-Path $localCodexHome "skills"

if (-not (Test-Path $localSkillsDir)) {
    New-Item -ItemType Directory -Path $localSkillsDir -Force | Out-Null
}

$env:CODEX_HOME = $localCodexHome
Write-Host "[local-gsd] CODEX_HOME=$env:CODEX_HOME"

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "codex command not found in PATH."
}

if ($CodexArgs.Count -gt 0) {
    & codex @CodexArgs
    exit $LASTEXITCODE
}

& codex -C $repoRoot
exit $LASTEXITCODE
