# ============================
# install_all_updates.ps1
# ============================

param(
    [string]$DBPath = "C:\Users\Adminn\Desktop\fax-app\codepatch\backend\patch_manager.db"
)

# Ensure PSWindowsUpdate module is installed
if (-not (Get-Module -ListAvailable -Name PSWindowsUpdate)) {
    Write-Output "PSWindowsUpdate module not found. Installing..."
    Install-Module -Name PSWindowsUpdate -Force -Confirm:$false
}

Import-Module PSWindowsUpdate

# Check if SQLite CLI exists
if (-not (Get-Command sqlite3.exe -ErrorAction SilentlyContinue)) {
    Write-Error "sqlite3.exe not found. Please install SQLite CLI and add it to PATH."
    exit 1
}

# Retrieve KBs from database
$kbList = & sqlite3.exe "$DBPath" "SELECT DISTINCT kb FROM patches WHERE kb != '';" | ForEach-Object { $_.Trim() }

if ($kbList.Count -eq 0) {
    Write-Output "No KBs found in database."
    exit 0
}

# Loop through each KB and install automatically
foreach ($kb in $kbList) {
    Write-Output "Installing $kb ..."
    try {
        Install-WindowsUpdate -KBArticleID $kb -AcceptAll -IgnoreReboot -Verbose -ErrorAction Stop
        Write-Output "$kb installation completed successfully."
    } catch {
        Write-Warning "Failed to install $kb: $_"
    }
}

Write-Output "All updates processed."
