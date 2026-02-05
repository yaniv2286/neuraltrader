# NeuralTrader Architecture Split - Production vs. Simulation
## Date: February 5, 2026

## 🎯 Overview

The NeuralTrader system now has **TWO SEPARATE PIPELINES** to allow on-demand testing without breaking scheduled production logic:

1. **Production Pipeline** (`main_orchestrator_ist.py`) - Strict, scheduled, production-ready
2. **Simulation Pipeline** (`scripts/manual_dry_run.py`) - Flexible, on-demand, testing-focused

## 📊 Architecture Comparison

| Feature | Production Pipeline | Simulation Pipeline |
|---------|-------------------|-------------------|
| **Entry Point** | `main_orchestrator_ist.py` | `scripts/manual_dry_run.py` |
| **Market Hours Check** | ✅ STRICT (NYSE schedule) | ❌ DISABLED (always "open") |
| **Data Source** | YFinance API (scheduled) | Mock data (avoids rate limits) |
| **Logging** | `logs/automation_{timestamp}.log` | `logs/dry_run/simulation_{timestamp}.log` |
| **Trade Log** | `logs/supervision/paper_trades.json` | `logs/supervision/simulation_trades.json` |
| **Email Subject** | `[PROD]` or `[URGENT]` | `[TEST]` |
| **Execution** | Scheduled (Task Scheduler) | On-demand (manual run) |
| **Purpose** | Real production monitoring | Testing & development |

## ✅ Task 1: Global Crash Fix (COMPLETED)

### Unicode Emoji Removal
**File**: `src/utils/daily_logger.py`

**Changes**:
- ✅ Removed all Unicode emojis (✅, 🚀, ⚠️, ❌)
- ✅ Replaced with ASCII tags: `[OK]`, `[INFO]`, `[WARN]`, `[ERROR]`
- ✅ Applied to BOTH pipelines

**Result**: No more `UnicodeEncodeError` crashes in Windows Task Scheduler

### Timestamped Log Files
**Implementation**:
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"automation_{timestamp}.log"  # Production
log_file = f"simulation_{timestamp}.log"  # Simulation
```

**Result**: No more `PermissionError` from file locking

## 🔬 Task 2: Simulation Pipeline (COMPLETED)

### New File: `scripts/manual_dry_run.py`

**Key Features**:
1. **No Market Hours Check** - Assumes market is always OPEN
2. **Mock Data Generation** - Avoids YFinance API rate limits
3. **Separate Logging** - `logs/dry_run/simulation_{timestamp}.log`
4. **Distinct Email** - Subject: `[TEST] Manual Dry Run Results`
5. **Separate Trade Log** - `logs/supervision/simulation_trades.json`

**Usage**:
```bash
python scripts/manual_dry_run.py
```

**What It Does**:
1. Initializes trading components (Risk Manager, Virtual Engine, etc.)
2. Generates 5 days of mock OHLC data for AAPL
3. Generates a trading signal (BUY/SELL/HOLD)
4. Executes virtual trade if signal is not HOLD
5. Logs trade to separate simulation file
6. Sends email with `[TEST]` prefix and log attachment

**Email Example**:
- **Subject**: `[TEST] Manual Dry Run Results - 2026-02-05 10:15 IST`
- **Body**: Simulation report with signal analysis and trade execution
- **Attachment**: Complete simulation log file

## 🏭 Task 3: Production Pipeline (ALREADY CONFIGURED)

### File: `main_orchestrator_ist.py`

**Smart Notification Logic**:

#### Fetch Mode
- **Success**: NO EMAIL (reduces noise)
- **Failure**: Email with subject `[URGENT] NeuralTrader Data Fetch FAILED`
- **Log Attachment**: ✅ Included

#### Report Mode
- **Always sends**: Email with subject `[NEURAL] Daily Executive Brief`
- **Log Attachment**: ✅ Included

#### Saturday Retrain Mode
- **Always sends**: Email with subject `[NEURAL] Weekly Model Update`
- **Log Attachment**: ✅ Included

**Market Hours Check**: ✅ STRICT (NYSE schedule enforced)

## ✅ Task 4: Execution Results

### Manual Dry Run Test
```
[SUCCESS] Manual dry run completed successfully
[INFO] Check your email for [TEST] Manual Dry Run Results

Results:
- Signal: HOLD (50% confidence)
- Trade: No (HOLD signal)
- Email: Sent successfully
- Log: logs/dry_run/simulation_20260205_101552.log
```

**Email Sent**: ✅
- Subject: `[TEST] Manual Dry Run Results - 2026-02-05 10:15 IST`
- Log Attachment: ✅ `simulation_20260205_101552.log`
- Status: Delivered to lugassy.ai@gmail.com

## 📁 File Structure

```
NeuralTrader/
├── main_orchestrator_ist.py          # Production Pipeline
├── scripts/
│   └── manual_dry_run.py              # Simulation Pipeline ✨ NEW
├── logs/
│   ├── automation_{timestamp}.log     # Production logs
│   └── dry_run/
│       └── simulation_{timestamp}.log # Simulation logs ✨ NEW
└── logs/supervision/
    ├── paper_trades.json              # Production trades
    └── simulation_trades.json         # Simulation trades ✨ NEW
```

## 🎯 Use Cases

### Production Pipeline
**When to use**:
- Scheduled daily operations (Task Scheduler)
- Real market hours trading
- Production monitoring
- Automated reporting

**How to run**:
```bash
python main_orchestrator_ist.py --mode=fetch
python main_orchestrator_ist.py --mode=report
```

### Simulation Pipeline
**When to use**:
- Testing new features
- Debugging issues
- On-demand signal generation
- Development and validation

**How to run**:
```bash
python scripts/manual_dry_run.py
```

## 🔒 Safety Features

### Production Pipeline
- ✅ Market hours enforcement
- ✅ Strict scheduling
- ✅ Production-grade error handling
- ✅ Smart notifications (reduce noise)
- ✅ Urgent alerts for failures

### Simulation Pipeline
- ✅ No market hours check (always available)
- ✅ Mock data (no API rate limits)
- ✅ Separate logging (no conflicts)
- ✅ Distinct email prefix `[TEST]`
- ✅ Separate trade tracking

## 📧 Email Prefixes

| Prefix | Pipeline | Meaning |
|--------|----------|---------|
| `[TEST]` | Simulation | Manual dry run results |
| `[URGENT]` | Production | Critical failure alert |
| `[NEURAL]` | Production | Daily/weekly reports |

## ✅ Benefits of Architecture Split

1. **No Production Interference** - Test anytime without breaking production
2. **Faster Development** - No waiting for market hours
3. **Safer Testing** - Separate logs and trade tracking
4. **Clear Separation** - Easy to identify test vs. production emails
5. **No API Limits** - Simulation uses mock data
6. **Better Debugging** - Dedicated simulation logs

## 🚀 Next Steps

1. **Use Simulation for Testing**: Run `python scripts/manual_dry_run.py` anytime
2. **Production Remains Scheduled**: Task Scheduler handles production runs
3. **Monitor Both Pipelines**: Check separate log directories
4. **Review Emails**: `[TEST]` = simulation, `[URGENT]`/`[NEURAL]` = production

---

**Status**: ✅ ARCHITECTURE SPLIT COMPLETE

**Production Pipeline**: Strict, scheduled, production-ready
**Simulation Pipeline**: Flexible, on-demand, testing-focused

Both pipelines are fully operational and independent!
