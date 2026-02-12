#!/bin/bash
# NeuralTrader Production Runner
# ==============================
# Executes fetch and trade pipeline in a loop with configurable delay
# All output is captured to timestamped logs

set -e  # Exit on any error

# Configuration
DELAY_MINUTES=${1:-5}  # Default 5 minutes delay
MAX_LOOPS=${2:-0}      # 0 = infinite loops
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run NeuralTrader command
run_neuraltrader() {
    local mode=$1
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_message "=== Starting $mode mode ==="
    log_message "Project Root: $PROJECT_ROOT"
    log_message "Mode: $mode"
    log_message "Loop: $current_loop/$MAX_LOOPS"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Run the batch file
    if ./run_neural_venv.bat "$mode"; then
        local end_time=$(date '+%Y-%m-%d %H:%M:%S')
        log_message "[SUCCESS] $mode completed successfully"
        log_message "Start: $start_time"
        log_message "End: $end_time"
        return 0
    else
        local end_time=$(date '+%Y-%m-%d %H:%M:%S')
        log_message "[ERROR] $mode failed with exit code: $?"
        log_message "Start: $start_time"
        log_message "End: $end_time"
        return 1
    fi
}

# Function to check if market is open
is_market_open() {
    # Simple check: Run on weekdays (Mon-Fri) between 9:30 AM - 4:00 PM EST
    local current_hour=$(TZ='America/New_York' date '+%H')
    local current_day=$(TZ='America/New_York' date '+%u')  # 1=Monday, 7=Sunday
    
    # Check if it's a weekday (1-5)
    if [[ $current_day -gt 5 ]]; then
        log_message "Market closed (weekend)"
        return 1
    fi
    
    # Check if it's market hours (9:30 AM - 4:00 PM EST)
    if [[ $current_hour -ge 10 && $current_hour -lt 16 ]]; then
        return 0  # Market is open
    else
        log_message "Market closed (off hours)"
        return 1
    fi
}

# Main execution loop
main() {
    local current_loop=0
    local delay_seconds=$((DELAY_MINUTES * 60))
    
    log_message "=== NeuralTrader Production Runner Started ==="
    log_message "Delay: $DELAY_MINUTES minutes"
    log_message "Max Loops: $MAX_LOOPS (0 = infinite)"
    log_message "Log Directory: $LOG_DIR"
    log_message "=========================================="
    
    # Create status file
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTED" > "$LOG_DIR/production_status.txt"
    
    while [[ $MAX_LOOPS -eq 0 || $current_loop -lt $MAX_LOOPS ]]; do
        current_loop=$((current_loop + 1))
        
        log_message "--- Loop $current_loop ---"
        
        # Run fetch mode (always)
        if run_neuraltrader "fetch"; then
            log_message "Fetch completed successfully"
        else
            log_message "Fetch failed - continuing to next iteration"
        fi
        
        # Check if market is open for trading
        if is_market_open; then
            log_message "Market is open - running trade mode"
            
            if run_neuraltrader "trade"; then
                log_message "Trade completed successfully"
            else
                log_message "Trade failed - continuing to next iteration"
            fi
        else
            log_message "Market closed - skipping trade mode"
        fi
        
        # Check if we should continue
        if [[ $MAX_LOOPS -ne 0 && $current_loop -ge $MAX_LOOPS ]]; then
            log_message "=== Maximum loops reached ==="
            break
        fi
        
        # Wait before next iteration
        log_message "Waiting $DELAY_MINUTES minutes before next iteration..."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WAITING" > "$LOG_DIR/production_status.txt"
        
        sleep $delay_seconds
    done
    
    log_message "=== NeuralTrader Production Runner Completed ==="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETED" > "$LOG_DIR/production_status.txt"
}

# Handle signals gracefully
trap 'log_message "Received interrupt signal - shutting down..."; echo "[$(date '+%Y-%m-%d %H:%M:%S")] STOPPED" > "$LOG_DIR/production_status.txt"; exit 0' INT TERM

# Start the main execution
main "$@"
