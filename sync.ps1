param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("check","upload","download","mkdir")]
  [string]$Mode
)

$ErrorActionPreference = "Stop"

# ===== 設定（必要ならここだけ変える）=====
$HostName = "69.30.85.101"
$Port     = 22117
$User     = "root"

$RemoteProject = "/workspace/atma_22_ca"

$IdentityFile = "~/.ssh/id_ed25519"
# ========================================

function Require-Command($cmd) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    throw "Required command not found: $cmd"
  }
}
Require-Command "wsl"

# プロジェクトルート
$ProjectRoot = (Resolve-Path $PSScriptRoot).Path

# ローカル ./data, ./out
$localData = Join-Path $ProjectRoot "data"
$localOut  = Join-Path $ProjectRoot "out"
if (-not (Test-Path $localData)) { New-Item -ItemType Directory -Path $localData | Out-Null }
if (-not (Test-Path $localOut))  { New-Item -ItemType Directory -Path $localOut  | Out-Null }

function To-WslPath([string]$winPath) {
  $p = $winPath.Replace("\","/")
  if ($p -match "^([A-Za-z]):/(.*)$") {
    $drive = $matches[1].ToLower()
    $rest = $matches[2]
    return "/mnt/$drive/$rest"
  }
  throw "Cannot convert path to WSL path: $winPath"
}

$wslLocalData = To-WslPath $localData
$wslLocalOut  = To-WslPath $localOut

$remote = "$User@$HostName"
$remoteData = "$RemoteProject/data"
$remoteOut  = "$RemoteProject/out"

# rsync（鍵固定・余計な鍵を試さない）
$rsyncBase = "rsync -avzP -e `"ssh -i $IdentityFile -o IdentitiesOnly=yes -p $Port -o StrictHostKeyChecking=accept-new`""

if ($Mode -eq "check") {
  Write-Host "Checking SSH & GPU... ${remote}:$Port"
  wsl bash -lc "ssh -i $IdentityFile -o IdentitiesOnly=yes -p $Port -o StrictHostKeyChecking=accept-new $remote 'echo OK; nvidia-smi -L || true; ls -la $RemoteProject || true'"
  exit 0
}

if ($Mode -eq "mkdir") {
  Write-Host "Creating remote dirs: $remoteData and $remoteOut"
  wsl bash -lc "ssh -i $IdentityFile -o IdentitiesOnly=yes -p $Port -o StrictHostKeyChecking=accept-new $remote 'mkdir -p $remoteData $remoteOut'"
  Write-Host "Done."
  exit 0
}

if ($Mode -eq "upload") {
  Write-Host "Uploading: $localData -> ${remote}:${remoteData}"

  $cmd = @(
    "set -e;",
    "RSYNC_RSH='ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p $Port -o StrictHostKeyChecking=accept-new';",
    "rsync -avzP -e ""$RSYNC_RSH"" '$wslLocalData/' '${remote}:${remoteData}/';"
  ) -join " "

  wsl bash -lc $cmd
  if ($LASTEXITCODE -ne 0) { throw "rsync upload failed (exit=$LASTEXITCODE)" }

  Write-Host "Done."
  exit 0
}


if ($Mode -eq "download") {
  Write-Host "Downloading: ${remote}:${remoteOut} -> $localOut"

  $cmd = @(
    "set -e;",
    "RSYNC_RSH='ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p $Port -o StrictHostKeyChecking=accept-new';",
    "rsync -avzP -e ""$RSYNC_RSH"" '${remote}:${remoteOut}/' '$wslLocalOut/';"
  ) -join " "

  wsl bash -lc $cmd
  if ($LASTEXITCODE -ne 0) { throw "rsync download failed (exit=$LASTEXITCODE)" }

  Write-Host "Done."
  exit 0
}
