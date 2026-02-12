# Verify NeuralTrader Tasks
Write-Host "Verifying NeuralTrader automation tasks..." -ForegroundColor Yellow

$tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "NeuralTrader_*" }

if ($tasks) {
    Write-Host "Found NeuralTrader tasks:" -ForegroundColor Green
    $tasks | ForEach-Object {
        $status = if ($_.State -eq "Ready") { "✅ Ready" } else { "❌ $($_.State)" }
        Write-Host "  $status - $($_.TaskName)" -ForegroundColor $(if($_.State -eq "Ready") {"Green"} else {"Red"})
        Write-Host "    Description: $($_.Description)" -ForegroundColor Gray
    }
} else {
    Write-Host "No NeuralTrader tasks found." -ForegroundColor Red
}

Write-Host "`nVerification complete." -ForegroundColor Cyan
