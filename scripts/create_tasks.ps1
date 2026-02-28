# NeuralTrader - Windows Task Scheduler Setup
# Single entry point: main_orchestrator_ist.py --mode=<MODE>
# Run this script once to (re)create all 4 tasks.
# All tasks route through the same pipeline and share the same risk laws.

$pythonPath = "C:\Users\Yaniv\AppData\Local\Programs\Python\Python310\python.exe"
$workingDir = "D:\GitHub\NeuralTrader"
$entry      = "main_orchestrator_ist.py"

Write-Host "Creating NeuralTrader automation pipeline (single entry point)..."

# ---------------------------------------------------------------------------
# P1 - Data Sync  |  Daily 16:00  |  --mode=fetch
# Downloads latest OHLCV parquet files for all 2,183 tickers via Tiingo API.
# ---------------------------------------------------------------------------
$action1  = New-ScheduledTaskAction -Execute $pythonPath -Argument "$entry --mode=fetch" -WorkingDirectory $workingDir
$trigger1 = New-ScheduledTaskTrigger -Daily -At 4:00PM
$settings1 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName "NeuralTrader_P1_DataSync" -Action $action1 -Trigger $trigger1 -Settings $settings1 -Description "P1: Fetch daily OHLCV data for all tickers (--mode=fetch)" -Force
Write-Host "  [OK] P1_DataSync        -> $entry --mode=fetch          @ 16:00 daily"

# ---------------------------------------------------------------------------
# P2 - Weekly Model Retrain  |  Saturday 10:00  |  --mode=saturday_retrain
# Brain-Gate protected retraining. Backs up models, validates new ones,
# auto-rollbacks on degradation, sends alert to Architect.
# ---------------------------------------------------------------------------
$action2  = New-ScheduledTaskAction -Execute $pythonPath -Argument "$entry --mode=saturday_retrain" -WorkingDirectory $workingDir
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00AM
$settings2 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask -TaskName "NeuralTrader_P2_WeeklyRetrain" -Action $action2 -Trigger $trigger2 -Settings $settings2 -Description "P2: Brain-Gate model retraining (--mode=saturday_retrain)" -Force
Write-Host "  [OK] P2_WeeklyRetrain   -> $entry --mode=saturday_retrain @ 10:00 Saturday"

# ---------------------------------------------------------------------------
# P3 - Daily Execution  |  Daily 17:15  |  --mode=paper
# AI ensemble scoring -> Risk Machine -> IBKR paper orders (port 7497).
# Requires P1 to have run first (data freshness check enforced).
# ---------------------------------------------------------------------------
$action3  = New-ScheduledTaskAction -Execute $pythonPath -Argument "$entry --mode=paper" -WorkingDirectory $workingDir
$trigger3 = New-ScheduledTaskTrigger -Daily -At 5:15PM
$settings3 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "NeuralTrader_P3_DailyExecution" -Action $action3 -Trigger $trigger3 -Settings $settings3 -Description "P3: AI scoring + IBKR paper execution (--mode=paper)" -Force
Write-Host "  [OK] P3_DailyExecution  -> $entry --mode=paper           @ 17:15 daily"

# ---------------------------------------------------------------------------
# P4 - Daily Report  |  Daily 18:00  |  --mode=report
# Generates HTML dashboard from live broker data and sends email to Architect.
# ---------------------------------------------------------------------------
$action4  = New-ScheduledTaskAction -Execute $pythonPath -Argument "$entry --mode=report" -WorkingDirectory $workingDir
$trigger4 = New-ScheduledTaskTrigger -Daily -At 6:00PM
$settings4 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "NeuralTrader_P4_DailyReport" -Action $action4 -Trigger $trigger4 -Settings $settings4 -Description "P4: HTML dashboard + email report (--mode=report)" -Force
Write-Host "  [OK] P4_DailyReport     -> $entry --mode=report          @ 18:00 daily"

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== VERIFICATION ==="
Get-ScheduledTask | Where-Object { $_.TaskName -like "NeuralTrader_*" } | ForEach-Object {
    $a = $_.Actions[0].Arguments
    $t = $_.Triggers[0].StartBoundary -replace ".*T","" -replace "\+.*",""
    $ok = if ($_.State -eq "Ready") { "[OK]" } else { "[!!]" }
    Write-Host "  $ok $($_.TaskName) | $t | $a"
}
Write-Host ""
Write-Host "Pipeline setup complete. All tasks use single entry point: $entry"
