# Standardize NeuralTrader Automation Pipeline Naming Convention
Write-Host "Standardizing NeuralTrader automation pipeline naming..." -ForegroundColor Yellow

# Step 1: Delete existing NT_ tasks
Write-Host "`nStep 1: Deleting NT_ tasks..." -ForegroundColor Red

# Get all existing NT_ tasks
$existingTasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "NT_*" }

if ($existingTasks) {
    Write-Host "Found NT_ tasks to delete:" -ForegroundColor Red
    $existingTasks | ForEach-Object { Write-Host "  - $($_.TaskName)" }
    
    # Delete all NT_ tasks
    $existingTasks | ForEach-Object {
        try {
            Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false
            Write-Host "Deleted: $($_.TaskName)" -ForegroundColor Green
        } catch {
            Write-Host "Failed to delete: $($_.TaskName) - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "No NT_ tasks found." -ForegroundColor Green
}

# Step 2: Create tasks with NeuralTrader_ prefix
Write-Host "`nStep 2: Creating NeuralTrader_ tasks..." -ForegroundColor Cyan

# Python path and working directory
$pythonPath = "C:\Users\Yaniv\AppData\Local\Programs\Python\Python310\python.exe"
$workingDir = "D:\GitHub\NeuralTrader"

# Task 1: NeuralTrader_P1_DataSync
Write-Host "Creating NeuralTrader_P1_DataSync..." -ForegroundColor Cyan
$action1 = New-ScheduledTaskAction -Execute $pythonPath -Argument "scripts/data_manager.py download-missing" -WorkingDirectory $workingDir
$trigger1 = New-ScheduledTaskTrigger -Daily -At 4:00PM
$settings1 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NeuralTrader_P1_DataSync" -Action $action1 -Trigger $trigger1 -Settings $settings1 -Description "NeuralTrader Data Sync - Download missing tickers" -Force

# Task 2: NeuralTrader_P2_WeeklyRetrain
Write-Host "Creating NeuralTrader_P2_WeeklyRetrain..." -ForegroundColor Cyan
$action2 = New-ScheduledTaskAction -Execute $pythonPath -Argument "src/training/weekly_retrain.py" -WorkingDirectory $workingDir
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00AM
$settings2 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NeuralTrader_P2_WeeklyRetrain" -Action $action2 -Trigger $trigger2 -Settings $settings2 -Description "NeuralTrader Weekly Model Retraining" -Force

# Task 3: NeuralTrader_P3_DailyExecution
Write-Host "Creating NeuralTrader_P3_DailyExecution..." -ForegroundColor Cyan
$action3 = New-ScheduledTaskAction -Execute $pythonPath -Argument "main_orchestrator_ist.py --mode=paper" -WorkingDirectory $workingDir
$trigger3 = New-ScheduledTaskTrigger -Daily -At 5:15PM
$settings3 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NeuralTrader_P3_DailyExecution" -Action $action3 -Trigger $trigger3 -Settings $settings3 -Description "NeuralTrader Daily Execution - Paper Trading Mode" -Force

# Task 4: NeuralTrader_P4_DailyReport
Write-Host "Creating NeuralTrader_P4_DailyReport..." -ForegroundColor Cyan
$action4 = New-ScheduledTaskAction -Execute $pythonPath -Argument "scripts/send_report.py" -WorkingDirectory $workingDir
$trigger4 = New-ScheduledTaskTrigger -Daily -At 6:00PM
$settings4 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NeuralTrader_P4_DailyReport" -Action $action4 -Trigger $trigger4 -Settings $settings4 -Description "NeuralTrader Daily Report Generation" -Force

Write-Host "All NeuralTrader_ tasks created successfully!" -ForegroundColor Green

# Step 3: Verify all tasks
Write-Host "`nStep 3: Verifying all tasks..." -ForegroundColor Yellow

$tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "NeuralTrader_*" }

if ($tasks) {
    Write-Host "Created NeuralTrader_ tasks:" -ForegroundColor Green
    $tasks | ForEach-Object {
        $status = if ($_.State -eq "Ready") { "✅ Ready" } else { "❌ $($_.State)" }
        Write-Host "  $status - $($_.TaskName)" -ForegroundColor $(if($_.State -eq "Ready") {"Green"} else {"Red"})
        Write-Host "    Description: $($_.Description)" -ForegroundColor Gray
        Write-Host "    Trigger: $($_.Triggers.StartBoundary)" -ForegroundColor Gray
    }
} else {
    Write-Host "No NeuralTrader_ tasks found!" -ForegroundColor Red
}

Write-Host "`n🎉 NeuralTrader automation pipeline naming standardization complete!" -ForegroundColor Green
Write-Host "All tasks now use the full 'NeuralTrader_' prefix." -ForegroundColor Cyan
