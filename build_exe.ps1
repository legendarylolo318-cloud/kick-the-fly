# Builds dist\KickTheFly.exe: one file with the game, Python, numpy/scipy/pygame and the brain pack inside.
# Needs data\kick_brain.npz first:  python -m kickthefly.sim.brainpack build   (after the connectome build in README.md)
# data\validation_results.json, if present, is bundled so the real-science popups work.
#   -Python path   the Python 3.11 to build with (default: py -3.11)
#   -GuiSmoke      also open the game window for 3 s as a check (needs a desktop session)
param([string]$Python = "", [switch]$GuiSmoke)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path data\kick_brain.npz)) { throw "data\kick_brain.npz missing: run 'python -m kickthefly.sim.brainpack build' first" }
if (-not (Test-Path .venv-build)) {
    if ($Python) { & $Python -m venv .venv-build } else { py -3.11 -m venv .venv-build }
}
.\.venv-build\Scripts\python.exe -m pip install --upgrade pip
.\.venv-build\Scripts\python.exe -m pip install numpy scipy pygame pyinstaller pillow moderngl pyyaml
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
.\.venv-build\Scripts\python.exe tools\make_icon.py build\icon.png

# Windows file properties (Details tab): version resource generated from version.py
$version = (Select-String -Path kickthefly\core\version.py -Pattern '__version__ = "(.+)"').Matches[0].Groups[1].Value
$parts = ($version.Split(".") + @("0", "0", "0"))[0..3] -join ", "
@"
VSVersionInfo(
  ffi=FixedFileInfo(filevers=($parts), prodvers=($parts), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'legendarylolo318-cloud'),
      StringStruct('FileDescription', 'Kick the Fly'),
      StringStruct('FileVersion', '$version'),
      StringStruct('InternalName', 'KickTheFly'),
      StringStruct('LegalCopyright', 'Connectome data: Janelia FlyEM MaleCNS v1.0, CC BY 4.0'),
      StringStruct('OriginalFilename', 'KickTheFly.exe'),
      StringStruct('ProductName', 'Kick the Fly'),
      StringStruct('ProductVersion', '$version')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"@ | Set-Content -Encoding utf8 build\version_info.txt

$data = @("--add-data", "data\kick_brain.npz;.", "--add-data", "protocols;protocols")
if (Test-Path data\validation_results.json) { $data += @("--add-data", "data\validation_results.json;.") }
else { Write-Warning "data\validation_results.json not found; the build won't show real-science popups" }

.\.venv-build\Scripts\pyinstaller.exe --noconfirm --clean --onefile --windowed --name KickTheFly `
    --icon build\icon.png --version-file build\version_info.txt @data `
    --exclude-module kickthefly.sim.connectome.loader --exclude-module pyarrow `
    --exclude-module tkinter --exclude-module matplotlib --exclude-module pynwb --exclude-module h5py `
    --exclude-module pandas `
    --hidden-import glcontext.wgl --hidden-import glcontext.empty `
    --hidden-import yaml --collect-submodules kickthefly --collect-data kickthefly `
    kick_the_fly.py
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

# build check with no window: load the brain, run the smoke protocol, exit 0 (the exe is a windowed app, so wait on it)
Remove-Item -Recurse -Force build\smoke -ErrorAction SilentlyContinue
$p = Start-Process -FilePath dist\KickTheFly.exe -ArgumentList "--headless --protocol protocols\smoke.yaml --out build\smoke" -Wait -PassThru -NoNewWindow
if ($p.ExitCode -ne 0) { throw "headless smoke run failed with exit code $($p.ExitCode)" }
if (-not (Get-ChildItem build\smoke\*\summary.json)) { throw "headless smoke run wrote no results" }
if ($GuiSmoke) { .\dist\KickTheFly.exe --smoke 3 | Out-Null }
Get-Item dist\KickTheFly.exe | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
