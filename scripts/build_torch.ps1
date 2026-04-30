param(
    [string]$BuildDir = "build-torch",
    [string]$LibtorchRoot = "",
    [string]$Configuration = "Release",
    [string]$Generator = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$sourceDir = Join-Path $root "model"

if (-not $LibtorchRoot) {
    $candidates = @(
        (Join-Path $root "libtorch"),
        (Join-Path $root "libtorch-win-shared-with-deps-2.11.0+cpu (2)\torch")
    )

    $LibtorchRoot = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $LibtorchRoot) {
    throw "Could not find LibTorch. Pass -LibtorchRoot or place it at .\libtorch or .\libtorch-win-shared-with-deps-2.11.0+cpu (2)\torch"
}

$torchConfigDir = Join-Path $LibtorchRoot "share\cmake\Torch"
if (-not (Test-Path $torchConfigDir)) {
    throw "TorchConfig.cmake not found under $torchConfigDir"
}

$cmakeArgs = @(
    "-S", $sourceDir,
    "-B", $BuildDir,
    "-DCMAKE_PREFIX_PATH=$LibtorchRoot"
)

if ($Generator) {
    $cmakeArgs += @("-G", $Generator)
}

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed. On Windows, the downloaded LibTorch package usually requires the MSVC toolchain and a Visual Studio generator."
}

cmake --build $BuildDir --config $Configuration --target inference_torch
if ($LASTEXITCODE -ne 0) {
    throw "Torch build failed."
}

$exe = Join-Path $root "$BuildDir\$Configuration\inference_torch.exe"
if (-not (Test-Path $exe)) {
    $exe = Join-Path $root "$BuildDir\inference_torch.exe"
}

Write-Host "[ok] Built $exe"
