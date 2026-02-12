# Rebuild NeuralTrader Automation Pipeline - Complete Reset and Rebuild
Write-Host "Rebuilding NeuralTrader automation pipeline..." -ForegroundColor Yellow

# Step 1: WIPE OLD TASKS - Delete all existing NeuralTrader_ and NT_ tasks
Write-Host "`nStep 1: Wiping old tasks..." -ForegroundColor Red

# Get all existing tasks with NeuralTrader_ or NT_ prefix
$existingTasks = Get-ScheduledTask | Where-Object { 
    $_.TaskName -like "NeuralTrader_*" -or 
    $_.TaskName -like "NT_*" 
}

if ($existingTasks) {
    Write-Host "Found existing tasks to delete:" -ForegroundColor Red
    $existingTasks | ForEach-Object { Write-Host "  - $($_.TaskName)" }
    
    # Delete all existing tasks
    $existingTasks | ForEach-Object {
        try {
            Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false
            Write-Host "Deleted: $($_.TaskName)" -ForegroundColor Green
        } catch {
            Write-Host "Failed to delete: $($_.TaskName) - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "No existing NeuralTrader or NT tasks found." -ForegroundColor Green
}

# Step 2: CREATE NEW PRODUCTION TASKS
Write-Host "`nStep 2: Creating new production tasks..." -ForegroundColor Cyan

# Python path and working directory
$pythonPath = "C:\Users\Yaniv\AppData\Local\Programs\Python\Python310\python.exe"
$workingDir = "D:\GitHub\NeuralTrader"

# Task 1: NT_P1_DataSync
Write-Host "Creating NT_P1_DataSync..." -ForegroundColor Cyan
$action1 = New-ScheduledTaskAction -Execute $pythonPath -Argument "scripts/data_manager.py download-missing" -WorkingDirectory $workingDir
$trigger1 = New-ScheduledTaskTrigger -Daily -At 4:00PM
$settings1 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NT_P1_DataSync" -Action $action1 -Trigger $trigger1 -Settings $settings1 -Description "NeuralTrader Data Sync - Download missing tickers" -Force

# Task 2: NT_P2_WeeklyRetrain
Write-Host "Creating NT_P2_WeeklyRetrain..." -ForegroundColor Cyan
$action2 = New-ScheduledTaskAction -Execute $pythonPath -Argument "src/training/weekly_retrain.py" -WorkingDirectory $workingDir
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00AM
$settings2 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NT_P2_WeeklyRetrain" -Action $action2 -Trigger $trigger2 -Settings $settings2 -Description "NeuralTrader Weekly Model Retraining" -Force

# Task 3: NT_P3_DailyExecution
Write-Host "Creating NT_P3_DailyExecution..." -ForegroundColor Cyan
$action3 = New-ScheduledTaskAction -Execute $pythonPath -Argument "main_orchestrator_ist.py --mode=paper" -WorkingDirectory $workingDir
$trigger3 = New-ScheduledTaskTrigger -Daily -At 5:15PM
$settings3 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NT_P3_DailyExecution" -Action $action3 -Trigger $trigger3 -Settings $settings3 -Description "NeuralTrader Daily Execution - Paper Trading Mode" -Force

# Task 4: NT_P4_DailyReport
Write-Host "Creating NT_P4_DailyReport..." -ForegroundColor Cyan
$action4 = New-ScheduledTaskAction -Execute $pythonPath -Argument "main_orchestrator_ist.py --mode=report" -WorkingDirectory $workingDir
$trigger4 = New-ScheduledTaskTrigger -Daily -At 6:00PM
$settings4 = New-ScheduledTaskSettingsSet
Register-ScheduledTask -TaskName "NT_P4_DailyReport" -Action $action4 -Trigger $trigger4 -Settings $settings4 -Description "NeuralTrader Daily Report Generation" -Force

Write-Host "All production tasks created successfully!" -ForegroundColor Green

# Step 3: VERIFY EMAIL CONFIGURATION
Write-Host "`nStep 3: Verifying email configuration..." -ForegroundColor Cyan

# Check if notifier.py exists and has correct email
$notifierPath = "_LEGACY_VAULT\_archive_src\utils\notifier.py"
if (Test-Path $notifierPath) {
    $notifierContent = Get-Content $notifierPath
    if ($notifierContent -match "lugassy\.ai@gmail\.com") {
        Write-Host "✅ Email configuration verified: lugassy.ai@gmail.com" -ForegroundColor Green
    } else {
        Write-Host "❌ Email configuration issue: lugassy.ai@gmail.com not found" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Notifier file not found: $notifierPath" -ForegroundColor Red
}

# Step 4: VERIFY ALL TASKS
Write-Host "`nStep 4: Verifying all tasks..." -ForegroundColor Yellow

$tasks = Get-ScheduledTask | Where-Object { 
    $_.TaskName -like "NT_*" 
}

if ($tasks) {
    Write-Host "Created NT tasks:" -ForegroundColor Green
    $tasks | ForEach-Object {
        $status = if ($_.State -eq "Ready") { "✅ Ready" } else { "❌ $($_.State)" }
        Write-Host "  $status - $($_.TaskName)" -ForegroundColor $(if($_.State -eq "Ready") {"Green"} else {"Red"})
        Write-Host "    Description: $($_.Description)" -ForegroundColor Gray
    }
} else {
    Write-Host "No NT tasks found!" -ForegroundColor Red
}

Write-Host "`n🎉 NeuralTrader automation pipeline rebuild complete!" -ForegroundColor Green
Write-Host "New production tasks are ready with NT_ prefix." -ForegroundColor Cyan
