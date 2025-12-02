# scan_updates.ps1
# -----------------------
# Scan missing Windows Updates and post to backend
# -----------------------

# --- Ensure script runs with administrator privileges ---
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Output "Restarting script as Administrator..."
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    $psi.Verb = "runas"
    [System.Diagnostics.Process]::Start($psi) | Out-Null
    exit
}

# Ensure PSWindowsUpdate is installed
if (!(Get-Module -ListAvailable -Name PSWindowsUpdate)) {
    Install-PackageProvider -Name NuGet -Force -Scope CurrentUser -ErrorAction SilentlyContinue
    Install-Module -Name PSWindowsUpdate -Force -Scope CurrentUser
}
Import-Module PSWindowsUpdate

# Get local hostname
$hostname = $env:COMPUTERNAME
Write-Output "Scanning for missing updates on $hostname..."

# Get missing updates from Microsoft Update
$updates = Get-WindowsUpdate -MicrosoftUpdate -IgnoreUserInput -AcceptAll -ErrorAction SilentlyContinue

$patches = @()

foreach ($u in $updates) {
    if ($u.KBArticleIDs -and $u.KBArticleIDs.Count -gt 0) {
        $kbString = ($u.KBArticleIDs | ForEach-Object {
            if ($_ -notmatch "^KB") { "KB$_" } else { $_ }
        }) -join ","
        if ([string]::IsNullOrWhiteSpace($kbString)) { continue }
    } else { continue }

    # Build patch object safely
    $patch = @{
        kb = [string]$kbString
        title = [string]($u.Title -replace "`r|`n", " ") # remove newlines
        severity = if ($u.MsrcSeverity) { [string]$u.MsrcSeverity } else { "Unknown" }
        priority_score = 1.0
    }

    $patches += $patch
}

# Prepare JSON body
$body = @{
    hostname = $hostname
    patches = $patches
} | ConvertTo-Json -Depth 5

Write-Output "JSON to POST:"
Write-Output $body

# Post to backend
$uri = "http://127.0.0.1:8000/api/scan"

try {
    Write-Output "Posting scan results to $uri ..."
    $res = Invoke-RestMethod -Uri $uri -Method Post -ContentType "application/json; charset=utf-8" -Body $body
    Write-Output "Scan posted successfully: $($res.status)"
} catch {
    Write-Error "Failed to post scan: $_"
}
