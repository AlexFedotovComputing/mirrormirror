$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
$nodePath = if ($nodeCommand) { $nodeCommand.Source } else { Join-Path $PSScriptRoot '.tools/node/node.exe' }
if (-not (Test-Path -LiteralPath $nodePath)) {
    throw 'Install Node.js 22.12+ (LTS), then run npm install. See README.md.'
}
if (-not (Test-Path -LiteralPath 'node_modules/electron/dist/electron.exe')) {
    throw 'Electron is not installed. Run npm install, then npm run setup:electron if needed.'
}
& $nodePath 'desktop/launch.cjs'
exit $LASTEXITCODE
