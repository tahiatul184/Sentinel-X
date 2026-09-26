[CmdletBinding()]
param([switch]$FunctionsOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# A native process failure is checked by exit code and can use the next installer.
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Write-Step([string]$Message) {
    Write-Host "[AEROSENTINEL] $Message" -ForegroundColor Cyan
}

function Test-TrustGeoPython([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate) -or -not (Test-Path -LiteralPath $Candidate)) { return $null }
    try {
        $probe = "import sys,struct,tkinter,venv,ensurepip; assert sys.version_info[:2] == (3,12) and struct.calcsize('P')==8; print(sys.executable)"
        $lines = @(& $Candidate -c $probe 2>$null)
        if ($LASTEXITCODE -eq 0 -and $lines.Count -gt 0) {
            $reported = [string]$lines[-1]
            if (Test-Path -LiteralPath $reported.Trim()) { return $reported.Trim() }
        }
    } catch { }
    return $null
}

function Get-PythonManagers {
    $paths = New-Object System.Collections.Generic.List[string]
    $command = Get-Command pymanager.exe -ErrorAction SilentlyContinue
    if ($command) { $paths.Add($command.Source) }
    $windowsApps = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps'
    if (Test-Path -LiteralPath $windowsApps) {
        Get-ChildItem -LiteralPath $windowsApps -Directory -Filter 'PythonSoftwareFoundation.PythonManager_*' -ErrorAction SilentlyContinue | ForEach-Object {
            $candidate = Join-Path $_.FullName 'pymanager.exe'
            if (Test-Path -LiteralPath $candidate) { $paths.Add($candidate) }
        }
    }
    return @($paths | Select-Object -Unique)
}

function Find-TrustGeoPython([string]$ProjectDir) {
    $paths = New-Object System.Collections.Generic.List[string]
    $paths.Add((Join-Path $ProjectDir '.dashboard-venv\Scripts\python.exe'))
    $paths.Add((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'))
    $paths.Add((Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.12-64\python.exe'))
    $paths.Add((Join-Path $env:ProgramFiles 'Python312\python.exe'))
    foreach ($hive in @('HKCU:\Software\Python\PythonCore\3.12\InstallPath', 'HKLM:\Software\Python\PythonCore\3.12\InstallPath')) {
        if (Test-Path $hive) {
            $entry = Get-Item $hive
            $executable = $entry.GetValue('ExecutablePath')
            $directory = $entry.GetValue('')
            if ($executable) { $paths.Add([string]$executable) }
            elseif ($directory) { $paths.Add((Join-Path ([string]$directory) 'python.exe')) }
        }
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notmatch '\\WindowsApps\\python.exe$') { $paths.Add($pythonCommand.Source) }
    foreach ($candidate in @($paths | Select-Object -Unique)) {
        $found = Test-TrustGeoPython $candidate
        if ($found) { return $found }
    }
    foreach ($manager in @(Get-PythonManagers)) {
        try {
            $reported = @(& $manager list '--format=exe' '3.12' 2>$null)
            if ($LASTEXITCODE -eq 0) {
                foreach ($candidate in $reported) {
                    $found = Test-TrustGeoPython (([string]$candidate).Trim())
                    if ($found) { return $found }
                }
            }
        } catch { }
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $reported = @(& $launcher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null)
            if ($LASTEXITCODE -eq 0 -and $reported.Count -gt 0) {
                $found = Test-TrustGeoPython (([string]$reported[-1]).Trim())
                if ($found) { return $found }
            }
        } catch { }
    }
    return $null
}

function Install-TrustGeoPython([string]$ProjectDir) {
    Write-Step 'Installing Python 3.12 for this Windows user. Internet access is required.'
    foreach ($manager in @(Get-PythonManagers)) {
        try {
            & $manager install 3.12 | Out-Host
            if ($LASTEXITCODE -eq 0) {
                $found = Find-TrustGeoPython $ProjectDir
                if ($found) { return $found }
            }
        } catch { Write-Warning "Python manager could not finish: $($_.Exception.Message)" }
    }
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        try {
            & $winget.Source install --id Python.Python.3.12 --exact --scope user --architecture x64 --silent --accept-package-agreements --accept-source-agreements --disable-interactivity | Out-Host
            $found = Find-TrustGeoPython $ProjectDir
            if ($found) { return $found }
        } catch { Write-Warning "Windows Package Manager could not finish: $($_.Exception.Message)" }
    }
    # Official Python installer route when neither install manager is usable.
    Write-Step 'Setting up the official Python Install Manager.'
    try {
        Add-AppxPackage -AppInstallerFile 'https://www.python.org/ftp/python/pymanager/pymanager.appinstaller'
        foreach ($manager in @(Get-PythonManagers)) {
            & $manager install 3.12 | Out-Host
            if ($LASTEXITCODE -eq 0) {
                $found = Find-TrustGeoPython $ProjectDir
                if ($found) { return $found }
            }
        }
    } catch { Write-Warning "Official Python setup could not finish: $($_.Exception.Message)" }
    throw 'Python setup could not complete. Check internet access and Windows installation policy. See installer\setup.log. Python 3.12 (64-bit, with Tcl/Tk and pip) is required.'
}

function New-TrustGeoShortcuts([string]$ProjectDir) {
    $shell = New-Object -ComObject WScript.Shell
    $desktop = [Environment]::GetFolderPath('Desktop')
    $menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'AEROSENTINEL'
    New-Item -ItemType Directory -Path $menu -Force | Out-Null
    foreach ($folder in @($desktop, $menu)) {
        if (-not [string]::IsNullOrWhiteSpace($folder)) {
            $shortcut = $shell.CreateShortcut((Join-Path $folder 'AEROSENTINEL.lnk'))
            $shortcut.TargetPath = Join-Path $ProjectDir 'RUN_ALL.bat'
            $shortcut.WorkingDirectory = $ProjectDir
            $shortcut.Description = 'Open AEROSENTINEL Control Center and dashboard'
            $shortcut.WindowStyle = 7
            $shortcut.Save()
        }
    }
}

function Start-TrustGeoSetup {
    $project = Split-Path -Parent $PSScriptRoot
    foreach ($relative in @('RUN_ALL.bat','start_dashboard.py','desktop.py','app\requirements-dashboard.txt')) {
        if (-not (Test-Path -LiteralPath (Join-Path $project $relative))) {
            throw "Missing $relative. Extract the whole ZIP before opening RUN_ALL.bat."
        }
    }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'This application requires 64-bit Windows.' }
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        throw 'This package uses an Intel/AMD x64 Python runtime. ARM Windows is not supported by this installer.'
    }
    # A project-specific mutex prevents two setup windows changing one environment.
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    $digest = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($project.ToLowerInvariant()))).Replace('-','')
    $hasher.Dispose()
    $mutex = New-Object System.Threading.Mutex($false, ('Local\AEROSENTINEL_SETUP_' + $digest.Substring(0,24)))
    $owned = $false
    $transcript = $false
    try {
        try { $owned = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
        if (-not $owned) {
            Write-Step 'AEROSENTINEL setup is already running. Continue in the first setup window.'
            return
        }
        $log = Join-Path $PSScriptRoot 'setup.log'
        if (Test-Path -LiteralPath $log) {
            if ((Get-Item -LiteralPath $log).Length -gt 5000000) { Move-Item -LiteralPath $log -Destination ($log + '.previous') -Force }
        }
        Start-Transcript -Path $log -Append -Force | Out-Null
        $transcript = $true
        Write-Step '[1/4] Checking Python and desktop support'
        $python = Find-TrustGeoPython $project
        if (-not $python) { $python = Install-TrustGeoPython $project }
        Write-Step '[2/4] Preparing the application packages'
        & $python (Join-Path $project 'start_dashboard.py') --setup-only
        if ($LASTEXITCODE -ne 0) { throw 'Package setup failed. Keep the error above and retry RUN_ALL.bat after fixing the connection or disk-space issue.' }
        $pythonw = Join-Path $project '.dashboard-venv\Scripts\pythonw.exe'
        if (-not (Test-Path -LiteralPath $pythonw)) { throw 'Application runtime is incomplete: pythonw.exe is missing.' }
        New-Item -ItemType Directory -Path (Join-Path $project 'app\highres_inbox') -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $project 'app\data\reports\auto') -Force | Out-Null
        Write-Step '[3/4] Adding Desktop and Start Menu shortcuts' 
        try { New-TrustGeoShortcuts $project } catch { Write-Warning 'Shortcut creation was unavailable; RUN_ALL.bat will still launch AEROSENTINEL.' }
        Write-Step '[4/4] Opening AEROSENTINEL Control Center'
        # Passing a quoted script path preserves spaces and shell punctuation.
        Start-Process -FilePath $pythonw -ArgumentList ('"' + (Join-Path $project 'desktop.py') + '"') -WorkingDirectory $project
        Write-Step 'Control Center is opening. This setup window may close now.'
    } finally {
        if ($transcript) { Stop-Transcript | Out-Null }
        if ($owned) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}

if (-not $FunctionsOnly) {
    try { Start-TrustGeoSetup; exit 0 }
    catch { Write-Host ('[AEROSENTINEL ERROR] ' + $_.Exception.Message) -ForegroundColor Red; exit 1 }
}
