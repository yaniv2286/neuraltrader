# Reset NeuralTrader Tasks - Delete all existing NeuralTrader_ tasks
Write-Host "Resetting NeuralTrader automation pipeline..." -ForegroundColor Yellow

# Get all existing NeuralTrader tasks
$existingTasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "NeuralTrader_*" }

if ($existingTasks) {
    Write-Host "Found existing NeuralTrader tasks:" -ForegroundColor Red
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
    Write-Host "No existing NeuralTrader tasks found." -ForegroundColor Green
}

Write-Host "Reset complete." -ForegroundColor Cyan
