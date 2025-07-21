#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Rubik's Cube Detection System - Universal Build Script for Windows/PowerShell
    
.DESCRIPTION
    This script automatically installs dependencies and builds the project using modern PowerShell.
    Compatible with Windows PowerShell 5.1+ and PowerShell Core 7+.
    
.PARAMETER InstallDeps
    Install system dependencies automatically
    
.PARAMETER DepsOnly
    Only install dependencies, don't build
    
.PARAMETER Clean
    Clean build directory before building
    
.PARAMETER Verbose
    Enable verbose build output
    
.PARAMETER Help
    Show help message
    
.EXAMPLE
    .\build.ps1
    Build project (dependencies must be installed)
    
.EXAMPLE
    .\build.ps1 -InstallDeps
    Install dependencies and build
    
.EXAMPLE
    .\build.ps1 -Clean
    Clean build and rebuild
    
.EXAMPLE
    .\build.ps1 -DepsOnly
    Only install dependencies
#>

[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [switch]$DepsOnly,
    [switch]$Clean,
    [switch]$Verbose,
    [switch]$Help
)

# Script configuration
$script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:BuildDir = Join-Path $ScriptDir "build"
$script:VcpkgRoot = $null

# Color output functions
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Administrator {
    <#
    .SYNOPSIS
    Check if running with administrator privileges
    #>
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        # PowerShell Core cross-platform check
        if ($IsWindows) {
            $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
            return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        } else {
            return (id -u) -eq 0
        }
    } else {
        # Windows PowerShell
        $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
        return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
}

function Install-Chocolatey {
    <#
    .SYNOPSIS
    Install Chocolatey package manager if not present
    #>
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Write-Status "Chocolatey not found. Installing Chocolatey package manager..."
        
        try {
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
            
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            Write-Success "Chocolatey installed successfully"
        }
        catch {
            Write-Error "Failed to install Chocolatey: $($_.Exception.Message)"
            throw
        }
    }
}

function Install-Vcpkg {
    <#
    .SYNOPSIS
    Install and configure vcpkg package manager
    #>
    
    # Check common vcpkg locations
    $possiblePaths = @(
        "C:\vcpkg\vcpkg.exe",
        "C:\tools\vcpkg\vcpkg.exe",
        "$env:VCPKG_ROOT\vcpkg.exe"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $script:VcpkgRoot = Split-Path -Parent $path
            Write-Status "Found vcpkg at: $script:VcpkgRoot"
            return
        }
    }
    
    Write-Status "vcpkg not found. Installing vcpkg package manager..."
    
    try {
        $vcpkgDir = "C:\tools\vcpkg"
        
        if (-not (Test-Path "C:\tools")) {
            New-Item -ItemType Directory -Path "C:\tools" -Force | Out-Null
        }
        
        Push-Location "C:\tools"
        
        if (Test-Path $vcpkgDir) {
            Remove-Item $vcpkgDir -Recurse -Force
        }
        
        Write-Status "Cloning vcpkg repository..."
        & git clone https://github.com/Microsoft/vcpkg.git
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to clone vcpkg repository"
        }
        
        Push-Location $vcpkgDir
        Write-Status "Bootstrapping vcpkg..."
        & .\bootstrap-vcpkg.bat
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to bootstrap vcpkg"
        }
        
        $script:VcpkgRoot = $vcpkgDir
        Write-Success "vcpkg installed successfully at: $script:VcpkgRoot"
        
        Pop-Location
        Pop-Location
    }
    catch {
        Write-Error "Failed to install vcpkg: $($_.Exception.Message)"
        if (Get-Location) { Pop-Location }
        if (Get-Location) { Pop-Location }
        throw
    }
}

function Install-Dependencies {
    <#
    .SYNOPSIS
    Install all required dependencies
    #>
    Write-Status "Installing dependencies for Windows..."
    
    # Check for admin privileges
    if (-not (Test-Administrator)) {
        Write-Error "Administrator privileges required for dependency installation"
        Write-Host "Please run PowerShell as Administrator to install dependencies" -ForegroundColor Yellow
        throw "Admin privileges required"
    }
    
    try {
        # Install Chocolatey if needed
        Install-Chocolatey
        
        # Install basic build tools via Chocolatey
        Write-Status "Installing build tools..."
        $packages = @(
            "cmake",
            "git",
            "visualstudio2022buildtools",
            "visualstudio2022-workload-vctools"
        )
        
        foreach ($package in $packages) {
            Write-Status "Installing $package..."
            & choco install -y $package
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Failed to install $package, continuing..."
            }
        }
        
        # Install and setup vcpkg
        Install-Vcpkg
        
        # Install OpenCV via vcpkg
        Write-Status "Installing OpenCV via vcpkg (this may take a while)..."
        Push-Location $script:VcpkgRoot
        
        & .\vcpkg.exe install opencv4[contrib]:x64-windows
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install OpenCV"
            throw "OpenCV installation failed"
        }
        
        # Integrate vcpkg with Visual Studio
        Write-Status "Integrating vcpkg with Visual Studio..."
        & .\vcpkg.exe integrate install
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "vcpkg integration failed, you may need to specify toolchain manually"
        }
        
        Pop-Location
        Write-Success "Dependencies installed successfully"
    }
    catch {
        if (Get-Location) { Pop-Location }
        throw
    }
}

function Test-Dependencies {
    <#
    .SYNOPSIS
    Check if all required dependencies are available
    #>
    Write-Status "Checking dependencies..."
    
    $missingDeps = @()
    
    # Check for CMake
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        $missingDeps += "CMake"
    }
    
    # Check for Git
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $missingDeps += "Git"
    }
    
    # Check for Visual Studio Build Tools
    $vsLocations = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
    )
    
    $vsFound = $false
    foreach ($location in $vsLocations) {
        if (Test-Path $location) {
            $vsFound = $true
            break
        }
    }
    
    if (-not $vsFound) {
        $missingDeps += "Visual Studio Build Tools"
    }
    
    # Check for vcpkg and OpenCV
    Install-Vcpkg -ErrorAction SilentlyContinue
    
    if (-not $script:VcpkgRoot -or -not (Test-Path "$script:VcpkgRoot\vcpkg.exe")) {
        $missingDeps += "vcpkg"
    } elseif (-not (Test-Path "$script:VcpkgRoot\installed\x64-windows\include\opencv2\opencv.hpp")) {
        $missingDeps += "OpenCV"
    }
    
    if ($missingDeps.Count -gt 0) {
        Write-Error "Missing dependencies: $($missingDeps -join ', ')"
        Write-Status "Run '.\build.ps1 -InstallDeps' to install dependencies automatically"
        throw "Missing dependencies"
    }
    
    Write-Success "All dependencies found"
}

function Test-CMakeVersion {
    <#
    .SYNOPSIS
    Check CMake version
    #>
    try {
        $cmakeVersion = & cmake --version 2>$null | Select-String "cmake version" | ForEach-Object { ($_ -split " ")[2] }
        if (-not $cmakeVersion) {
            throw "Could not determine CMake version"
        }
        Write-Success "CMake version $cmakeVersion found"
    }
    catch {
        Write-Error "Could not determine CMake version"
        throw
    }
}

function Build-Project {
    <#
    .SYNOPSIS
    Build the project using CMake
    #>
    Write-Status "Starting build process..."
    
    try {
        # Create/clean build directory
        if ($Clean -and (Test-Path $script:BuildDir)) {
            Write-Status "Cleaning existing build directory..."
            Remove-Item $script:BuildDir -Recurse -Force
        }
        
        if (-not (Test-Path $script:BuildDir)) {
            New-Item -ItemType Directory -Path $script:BuildDir -Force | Out-Null
        }
        
        Push-Location $script:BuildDir
        
        # Configure with CMake
        Write-Status "Configuring project with CMake..."
        $cmakeArgs = @(
            "-DCMAKE_BUILD_TYPE=Release"
        )
        
        if ($script:VcpkgRoot) {
            $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$script:VcpkgRoot\scripts\buildsystems\vcpkg.cmake"
        }
        
        if ($Verbose) {
            $cmakeArgs += "-DCMAKE_VERBOSE_MAKEFILE=ON"
        }
        
        $cmakeArgs += ".."
        
        & cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        # Build the project
        Write-Status "Building project..."
        $buildArgs = @(
            "--build", ".",
            "--config", "Release",
            "--parallel", $env:NUMBER_OF_PROCESSORS
        )
        
        if ($Verbose) {
            $buildArgs += "--verbose"
        }
        
        & cmake @buildArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        Write-Success "Build completed successfully!"
        
        # Find the executable
        $possibleExes = @(
            "Release\rubiks_cube_cpp_final.exe",
            "rubiks_cube_cpp_final.exe"
        )
        
        foreach ($exe in $possibleExes) {
            $fullPath = Join-Path $script:BuildDir $exe
            if (Test-Path $fullPath) {
                Write-Status "Executable location: $fullPath"
                break
            }
        }
        
        Pop-Location
    }
    catch {
        if (Get-Location) { Pop-Location }
        throw
    }
}

function Test-Build {
    <#
    .SYNOPSIS
    Verify that the build was successful
    #>
    $possibleExes = @(
        "Release\rubiks_cube_cpp_final.exe",
        "rubiks_cube_cpp_final.exe"
    )
    
    foreach ($exe in $possibleExes) {
        $fullPath = Join-Path $script:BuildDir $exe
        if (Test-Path $fullPath) {
            Write-Success "Build verification passed"
            Write-Status "You can now run: $fullPath"
            Write-Status "Make sure your cameras are connected before running"
            return
        }
    }
    
    Write-Error "Build verification failed - executable not found"
    throw "Executable not found"
}

function Show-Usage {
    <#
    .SYNOPSIS
    Display help information
    #>
    Write-Host @"
Rubik's Cube Detection System - PowerShell Build Script

SYNTAX
    .\build.ps1 [[-InstallDeps]] [[-DepsOnly]] [[-Clean]] [[-Verbose]] [[-Help]]

DESCRIPTION
    This script automatically installs dependencies and builds the Rubik's Cube 
    Detection System project using modern PowerShell.

PARAMETERS
    -InstallDeps    Install system dependencies automatically
    -DepsOnly       Only install dependencies, don't build  
    -Clean          Clean build directory before building
    -Verbose        Enable verbose build output
    -Help           Show this help message

EXAMPLES
    .\build.ps1
        Build project (dependencies must be installed)

    .\build.ps1 -InstallDeps
        Install dependencies and build

    .\build.ps1 -Clean
        Clean build and rebuild

    .\build.ps1 -DepsOnly
        Only install dependencies

NOTES
    - Requires PowerShell 5.1+ or PowerShell Core 7+
    - Administrator privileges required for dependency installation
    - Compatible with Windows 10+ systems
    - Uses vcpkg for C++ package management

"@ -ForegroundColor White
}

# Main execution
function Main {
    if ($Help) {
        Show-Usage
        return
    }
    
    Write-Status "Rubik's Cube Detection System - PowerShell Build Script"
    Write-Status "========================================================="
    Write-Status "PowerShell Version: $($PSVersionTable.PSVersion)"
    
    try {
        # Install dependencies if requested
        if ($InstallDeps) {
            Install-Dependencies
            if ($DepsOnly) {
                Write-Success "Dependencies installed successfully!"
                return
            }
        }
        
        # Check dependencies
        Test-Dependencies
        
        # Check CMake version
        Test-CMakeVersion
        
        # Build the project
        Build-Project
        
        # Verify the build
        Test-Build
        
        Write-Success "Build process completed successfully!"
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Connect your two cameras" -ForegroundColor White
        Write-Host "2. Run the executable from the build directory" -ForegroundColor White
        Write-Host "3. Follow the calibration process in the application menu" -ForegroundColor White
        Write-Host ""
        Write-Host "For more information, see README.md" -ForegroundColor Yellow
    }
    catch {
        Write-Error "Build process failed: $($_.Exception.Message)"
        exit 1
    }
}

# Run the main function
Main
