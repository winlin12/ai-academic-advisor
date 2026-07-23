<#
  allow_wsl_llamacpp.ps1 — let WSL reach llama-server running on Windows.

  THE PROBLEM. llama-server is a native Windows process (CUDA 13.3, required for the 5070 Ti's
  Blackwell/sm_120 — the stock cuda-12.4 build will not run on it). The eval harness runs in
  WSL2. In WSL2's default NAT mode, Windows can reach WSL's localhost but NOT the other way
  round: WSL must address the Windows host by its gateway IP, and Windows Firewall drops
  inbound connections from the WSL adapter by default. The symptom is a llama-server that
  starts perfectly, logs "listening", and is nevertheless unreachable — which looks exactly
  like a model that failed to load.

  THE FIX. One inbound allow rule, scoped to the eval port and to private/RFC1918 sources
  only. Run once per machine:

      powershell.exe -ExecutionPolicy Bypass -File setup/allow_wsl_llamacpp.ps1

  It self-elevates (one UAC prompt) because firewall rules need admin.

  SCOPE, deliberately narrow: one TCP port, LocalAddress restricted to the private ranges the
  WSL adapter uses, profile Private. It does not open the port to your LAN or to the internet,
  and it does not touch any other rule. Remove it with:

      Remove-NetFirewallRule -DisplayName "WSL -> llama-server (model_eval)"

  ALTERNATIVE, if you would rather not add a rule: set `networkingMode=mirrored` under [wsl2]
  in C:\Users\<you>\.wslconfig, run `wsl --shutdown`, and set `llamacpp.host: 127.0.0.1` in
  model_eval/config.yaml. Cleaner (WSL and Windows then share localhost, so no firewall
  traversal happens at all) but the shutdown ends every running WSL session.
#>
param(
  [int]$Port = 8099,
  [string]$RuleName = "WSL -> llama-server (model_eval)"
)

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
  Write-Host "Firewall rules require administrator rights — re-launching elevated (accept the UAC prompt)."
  $psi = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Port $Port"
  Start-Process powershell.exe -Verb RunAs -ArgumentList $psi -Wait
  exit
}

$existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($existing) {
  Write-Host "Rule '$RuleName' already exists — updating its port to $Port."
  $existing | Get-NetFirewallPortFilter | Set-NetFirewallPortFilter -LocalPort $Port
} else {
  New-NetFirewallRule `
    -DisplayName $RuleName `
    -Description "Allows the WSL2 virtual adapter to reach llama-server for model_eval runs." `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $Port `
    -Profile Private `
    -RemoteAddress @("172.16.0.0/12", "192.168.0.0/16", "10.0.0.0/8") | Out-Null
  Write-Host "Created firewall rule '$RuleName' for TCP $Port (private sources only)."
}

Write-Host ""
Write-Host "Done. Back in WSL, verify with:"
Write-Host "    cd model_eval && python run.py doctor"
Write-Host ""
Write-Host "If doctor still reports the host unreachable, your WSL adapter may be on a profile"
Write-Host "other than Private. Check with: Get-NetConnectionProfile | Format-Table Name,NetworkCategory"
Read-Host "Press Enter to close"
