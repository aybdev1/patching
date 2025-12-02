<#
 install_update.ps1 -KB KBxxxx
 Installs a KB using PSWindowsUpdate module.
 Must be run as Administrator.
#>
Param(
    [Parameter(Mandatory=$true)]
    [string]$KB
)

# Ensure PSWindowsUpdate installed
if (!(Get-Module -ListAvailable -Name PSWindowsUpdate)) {
    Write-Output "PSWindowsUpdate not present. Attempting to install from PSGallery (requires internet & admin)."
    Install-PackageProvider -Name NuGet -Force -Scope CurrentUser -ErrorAction SilentlyContinue
    Install-Module -Name PSWindowsUpdate -Force -Scope CurrentUser
}

Import-Module PSWindowsUpdate

# Install specific KB (works if KB is available from Windows Update)
Write-Output "Attempting to install $KB"
# This might open installer or require reboot. Use -AcceptAll and -IgnoreReboot to suppress prompts as needed.
Get-WindowsUpdate -KBArticleID $KB -Install -AcceptAll -IgnoreReboot -Verbose
