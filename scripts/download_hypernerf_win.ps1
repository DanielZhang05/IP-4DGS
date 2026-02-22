param(
    [Parameter(Mandatory=$true)]
    [string]$DestPath
)

# Desc: Download the HyperNeRF models from the official release.
# Args: -DestPath <path> e.g. data/hypernerf

$ErrorActionPreference = "Stop"

# Ensure destination exists
New-Item -ItemType Directory -Force -Path $DestPath | Out-Null

$files = @(
  "interp_chickchicken.zip",
  "interp_torchocolate.zip",
  "misc_americano.zip",
  "misc_espresso.zip",
  "misc_keyboard.zip",
  "misc_split-cookie.zip"
)

$base = "https://github.com/google/hypernerf/releases/download/v0.1"

foreach ($file in $files) {
    $url = "$base/$file"
    $zipPath = Join-Path $DestPath $file

    # Download (skip if exists)
    if (Test-Path $zipPath) {
        Write-Host "File $file already exists. Skipping download."
    } else {
        Write-Host "Downloading $file..."
        try {
            Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
        } catch {
            Write-Warning "Failed to download $file. Skipping... ($($_.Exception.Message))"
            continue
        }
    }

    # Extract
    Write-Host "Extracting $file..."
    try {
        Expand-Archive -Path $zipPath -DestinationPath $DestPath -Force
        Remove-Item $zipPath -Force
        Write-Host "Extraction successful. Deleted $file."
    } catch {
        Write-Warning "Extraction failed for $file. Keeping the zip file for debugging. ($($_.Exception.Message))"
    }
}

Write-Host "Done."