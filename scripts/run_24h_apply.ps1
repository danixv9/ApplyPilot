param(
    [int]$Hours = 24,
    [int]$ApplyWorkers = 4,
    [int]$PrepWorkers = 4,
    [int]$MinScore = 5,
    [int]$PrepIntervalMinutes = 30,
    [string]$Phone = "2048077954",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
# Keep non-zero exit codes from native commands (python) non-terminating.
$PSNativeCommandUseErrorActionPreference = $false

$repoDir = Split-Path -Parent $PSScriptRoot
Set-Location $repoDir

$outDir = Join-Path $repoDir "output\\orchestrator"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$applyOut = Join-Path $outDir "apply_$ts.stdout.log"
$applyErr = Join-Path $outDir "apply_$ts.stderr.log"
$prepLog = Join-Path $outDir "prep_$ts.log"
$ctlLog = Join-Path $outDir "control_$ts.log"

function LogCtl([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    Add-Content -Path $ctlLog -Value $line
    Write-Host $line
}

$env:APPLYPILOT_PHONE = $Phone

$applyArgs = @(
    "-m", "applypilot", "apply",
    "--engine", "playwright",
    "--workers", "$ApplyWorkers",
    "--continuous",
    "--min-score", "$MinScore"
)
if ($Headless) {
    $applyArgs += "--headless"
}

LogCtl "starting continuous apply process"
LogCtl "stdout=$applyOut"
LogCtl "stderr=$applyErr"

function StartApplyProcess {
    param(
        [string[]]$ApplyArgList,
        [string]$StdOut,
        [string]$StdErr
    )
    $proc = Start-Process `
        -FilePath "python" `
        -ArgumentList $ApplyArgList `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $StdOut `
        -RedirectStandardError $StdErr
    LogCtl ("apply pid={0}" -f $proc.Id)
    return $proc
}

$applyProc = StartApplyProcess -ApplyArgList $applyArgs -StdOut $applyOut -StdErr $applyErr

$deadline = (Get-Date).AddHours($Hours)
LogCtl ("deadline={0}" -f $deadline.ToString("s"))

try {
    while ((Get-Date) -lt $deadline) {
        if ($applyProc.HasExited) {
            LogCtl ("apply exited with code {0}; restarting" -f $applyProc.ExitCode)
            $applyProc = StartApplyProcess -ApplyArgList $applyArgs -StdOut $applyOut -StdErr $applyErr
        }

        LogCtl "prep cycle start (discover/enrich/score/tailor/cover/pdf)"
        $prepCmd = "python -m applypilot run discover enrich score tailor cover pdf --min-score $MinScore --workers $PrepWorkers >> `"$prepLog`" 2>&1"
        cmd /c $prepCmd | Out-Null
        $prepExit = $LASTEXITCODE
        if ($prepExit -eq 0) {
            LogCtl "prep cycle complete"
        } else {
            LogCtl ("prep cycle failed (exit={0}), continuing" -f $prepExit)
        }

        if ((Get-Date) -ge $deadline) {
            break
        }
        LogCtl ("sleeping {0} minutes before next prep cycle" -f $PrepIntervalMinutes)
        $remaining = $PrepIntervalMinutes * 60
        while ($remaining -gt 0 -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds ([Math]::Min(30, $remaining))
            $remaining -= 30
            if ($applyProc.HasExited) {
                LogCtl ("apply exited with code {0}; restarting" -f $applyProc.ExitCode)
                $applyProc = StartApplyProcess -ApplyArgList $applyArgs -StdOut $applyOut -StdErr $applyErr
            }
        }
    }
}
catch {
    LogCtl ("orchestrator error: {0}" -f $_.Exception.Message)
    if ($_.ScriptStackTrace) {
        LogCtl ("orchestrator stack: {0}" -f $_.ScriptStackTrace)
    }
}
finally {
    if ($applyProc -and -not $applyProc.HasExited) {
        LogCtl ("stopping apply pid={0}" -f $applyProc.Id)
        Stop-Process -Id $applyProc.Id -Force
    }
    LogCtl "run complete"
}
