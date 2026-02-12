# Sector Rotation Integration

## Overview
The Sector Authority has been successfully integrated into the NeuralTrader main orchestrator to provide sector-based risk management and volatility controls.

## Integration Points

### 1. Import Statement
```python
from scripts.sector_rotation import SectorAuthority
```

### 2. Initialization
```python
# In TradingOrchestrator.__init__()
self.sector_auth = SectorAuthority()
self.logger.info("[SECTOR] Sector Authority initialized for risk management")
```

### 3. Global Volatility Gate (Red Light)
```python
allow_buys = True
if self.sector_auth.check_global_stop():
    self.logger.error("[STOP] GLOBAL VOLATILITY CEILING BREACHED (VXX). Freezing all new entries.")
    allow_buys = False
    self.logger.info("[RISK] Only processing exits (Stop Loss / Shield) - skipping entry logic")
else:
    self.logger.info("[OK] Global volatility check passed - entries allowed")
```

### 4. Sector Analysis & Tax Application
```python
sector_ranks = []
if allow_buys:
    self.logger.info("[SECTOR] Analyzing sector momentum for tax application...")
    sector_ranks = self.sector_auth.get_sector_momentum()
    
    if sector_ranks:
        # Log top 3 and bottom 3 sectors
        top_3 = sector_ranks[:3]
        bottom_3 = sector_ranks[-3:]
        
        self.logger.info("[SECTOR] Top 3 Strongest Sectors:")
        for i, (sector, roc) in enumerate(top_3, 1):
            strength = "STRONG" if roc > 2 else "MODERATE" if roc > 0 else "WEAK"
            self.logger.info(f"  {i}. {sector}: {roc:.2f}% ({strength})")
        
        self.logger.warning("[SECTOR] Bottom 3 Weakest Sectors (Tax Applied):")
        for i, (sector, roc) in enumerate(bottom_3, 1):
            self.logger.warning(f"  {i}. {sector}: {roc:.2f}% (WEAK)")
```

### 5. AI Signal Generation with Sector Tax
```python
# Generate raw AI scores first
raw_ai_scores = {}
ticker_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'UNH']

for ticker in ticker_universe:
    signal_strength = self._generate_signal_simple(ticker)
    raw_ai_scores[ticker] = signal_strength

# Apply sector tax if allowed
adjusted_scores = raw_ai_scores.copy()
if allow_buys and sector_ranks:
    adjusted_scores = self.sector_auth.apply_sector_tax(raw_ai_scores, sector_ranks)
    self.logger.info("[SECTOR] Sector tax applied to scores")
```

### 6. Elite Sniper Strategy
```python
# Process adjusted scores
elite_candidates = []
for ticker, score in adjusted_scores.items():
    if score > 0.5:  # Buy signal threshold
        elite_candidates.append((ticker, score))

# Sort by adjusted score (highest first)
elite_candidates.sort(key=lambda x: x[1], reverse=True)

# Process top 5 candidates only if buys are allowed
if allow_buys:
    for ticker, adjusted_score in elite_candidates[:5]:
        # Execute trade with adjusted score
        result = self.virtual_engine.execute_trade(
            ticker, 'buy', position_size, current_price, 
            f"Elite Signal: {adjusted_score:.2f} (Sector Adjusted)"
        )
```

## Risk Management Features

### Global Volatility Stop
- **Trigger**: VXX Close > (20-day MA + 2 * StdDev)
- **Action**: Freeze all new entries, only process exits
- **Purpose**: Protect against market volatility spikes

### Sector Tax System
- **Trigger**: Bottom 3 weakest sectors by 20-day ROC
- **Action**: 15% score reduction for tickers in weak sectors
- **Purpose**: Avoid investing in underperforming sectors

### Elite Sniper Strategy
- **Selection**: Top 5 candidates after sector tax
- **Threshold**: Adjusted score > 0.5
- **Priority**: Highest adjusted scores first

## Test Results

### Integration Test Output
```
[SUMMARY] Sector Integration Test Results:
  Global Volatility Check: PASS
  Sector Momentum Data: AVAILABLE
  Raw AI Signals: 10 generated
  Adjusted Scores: 10 after sector tax
  Elite Candidates: 7 qualified
```

### Volatility Analysis
```
[VOLATILITY] VXX Analysis:
  Close: $27.12
  MA20: $27.13
  Upper BB: $28.92
  StdDev: $0.90
[OK] VXX Close ($27.12) <= Upper BB ($28.92)
```

### Sector Momentum
```
[SECTOR] Top 3 Strongest Sectors:
  1. XLK: -1.20% (WEAK)
  2. XLF: -2.60% (WEAK)
```

## Files Modified/Created

### Core Integration
- `main_orchestrator_ist.py` - Added Sector Authority integration

### Sector System
- `scripts/sector_rotation.py` - Sector Authority class
- `config/sector_map.json` - Sector mapping data
- `scripts/create_sector_map.py` - Sector map generator

### Test Files
- `scripts/test_sector_rotation.py` - Sector Authority tests
- `scripts/test_orchestrator_integration.py` - Integration tests

## Usage

### Run Integration Test
```bash
python scripts/test_orchestrator_integration.py
```

### Run Sector Authority Test
```bash
python scripts/test_sector_rotation.py
```

### Generate Sector Map
```bash
python scripts/create_sector_map.py
```

## Configuration

### Sector Mapping
- **File**: `config/sector_map.json`
- **Sectors**: 11 Sector ETFs (XLK, XLF, XLV, XLY, XLC, XLP, XLE, XLI, XLB, XLRE, XLU)
- **Tickers**: 220 total tickers (20 per sector)

### Volatility Parameters
- **VXX Period**: 20 days
- **Bollinger Bands**: MA ± 2 * StdDev
- **Tax Rate**: 15% for bottom 3 sectors

## Status
✅ **INTEGRATION COMPLETE** - Sector rotation system fully integrated into main orchestrator and ready for production use.
