#!/usr/bin/env python3
"""
Phase 16 - Optimized Backtest: Daily Exits, Weekly Entries, Full Capital Deployment
===================================================================================
Key fixes over Phase 15:
  1. DAILY exit management (SL, TP, trailing stop, timeout) - was monthly
  2. Weekly signal generation for new entries
  3. Proper position sizing: ~7% per position (was 2% = 70% idle capital)
  4. Trailing stop: after +2.5% gain, trail at 50% of peak unrealized
  5. 200 liquid tickers (was 46)
  6. Confidence-weighted sizing: higher confidence = larger position
  7. Stricter short filter: require 0.40+ confidence (was 0.30)
  8. Proper equity tracking: mark-to-market daily (unrealized P&L)
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('Backtest16')

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
REPORTS_DIR = PROJECT_ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# ── Backtest Config ──────────────────────────────────────────────────────────
START_DATE = '2020-01-01'
END_DATE   = '2025-12-31'
INITIAL_CAPITAL = 100_000.0

# Position sizing
MAX_POSITIONS      = 18       # v12: slightly less exposure to reduce DD
BASE_POSITION_PCT  = 0.05     # 5% of capital per position (18 × 5% = 90% max)
CONFIDENCE_BOOST   = 1.5      # Boost high-confidence signals
MIN_POSITION_PCT   = 0.03     # Floor: 3%
MAX_POSITION_PCT   = 0.08     # Cap: 8%

# Exit rules
TP_PCT             = 0.20     # v11: effectively disabled - let trailing stop exit winners
SL_PCT_DEFAULT     = 0.04     # 4% stop loss - cut losers fast
TIMEOUT_DAYS       = 5        # 5d timeout - maximum churn for capital turnover
TRAIL_ACTIVATE_PCT = 0.02     # Trailing stop after +2% gain (early lock-in)
TRAIL_PCT          = 0.012    # Trail 1.2% below peak - tight trail

# Signal thresholds
LONG_THRESHOLD     = 0.38     # v12: slightly higher for better signal quality
SHORT_THRESHOLD    = 1.10     # DISABLED: shorts were -$6k P&L with 49.6% WR
SHORTS_ENABLED     = False    # Master switch for short selling

# Uncle Point
UNCLE_POINT_DD     = 1.00     # DISABLED: Uncle Point was #1 perf killer across all versions
COOLDOWN_DAYS      = 0        # DISABLED

# Rebalancing
REBALANCE_FREQ     = 'W-MON'  # Weekly on Mondays (more entry opportunities)

# ── Universe: 100 most liquid US large-caps (proven runtime ~25 min) ────────────
TICKER_UNIVERSE = [
    # Mega-cap tech (20)
    'AAPL','MSFT','GOOGL','AMZN','TSLA','NVDA','META','AVGO','ORCL','AMD',
    'ADBE','CRM','INTC','CSCO','QCOM','TXN','MU','AMAT','LRCX','KLAC',
    # Software / Cloud (10)
    'NOW','SNOW','PANW','CRWD','ZS','DDOG','TEAM','WDAY','VEEV','FTNT',
    # Finance (12)
    'JPM','BAC','GS','WFC','V','MA','AXP','MS','BLK','SCHW','C','CME',
    # Healthcare (10)
    'UNH','JNJ','LLY','ABBV','MRK','PFE','TMO','ABT','ISRG','REGN',
    # Consumer / Retail (10)
    'HD','WMT','COST','MCD','NKE','SBUX','TGT','LOW','TJX','ROST',
    # Energy (6)
    'XOM','CVX','COP','SLB','EOG','PXD',
    # Industrial (8)
    'CAT','GE','HON','BA','RTX','LMT','DE','UPS',
    # Communication / Media (6)
    'DIS','NFLX','CMCSA','GOOG','T','VZ',
    # Consumer Staples (6)
    'PG','KO','PEP','PM','MO','CL',
    # Materials / Other (4)
    'LIN','APD','ECL','SHW',
    # ETFs (8)
    'SPY','QQQ','IWM','GLD','XLE','XLF','XLK','XLV',
]


class Phase16Backtest:
    def __init__(self):
        from core.ai_models import EnsemblePredictor
        self.ensemble = EnsemblePredictor()
        self.trades = []
        self.equity_curve = []
        self.capital = INITIAL_CAPITAL
        self.peak_capital = INITIAL_CAPITAL
        self.positions = []
        self.cooldown_until = None  # Uncle point cooldown
        self.total_signals_generated = 0

    def load_ticker(self, ticker):
        """Load ticker data with date filtering."""
        path = DATA_DIR / f'{ticker}.parquet'
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            start = pd.to_datetime(START_DATE) - pd.Timedelta(days=400)  # Extra history for features
            end = pd.to_datetime(END_DATE)
            df = df[(df.index >= start) & (df.index <= end)]
            if len(df) < 200:
                return None
            return df
        except Exception:
            return None

    def get_close_col(self, df):
        """Get the adjusted close column name."""
        if 'adjClose' in df.columns:
            return 'adjClose'
        return 'close'

    def get_signal(self, ticker_data, threshold=0.30):
        """Get AI signal for a ticker."""
        try:
            signal, confidence, details = self.ensemble.predict_from_ohlcv(
                ticker_data, threshold=threshold
            )
            return signal, confidence, details
        except Exception:
            return 'HOLD', 0.0, {}

    def calc_position_size(self, confidence, signal):
        """Calculate position size based on confidence and capital."""
        # Base size
        base = self.capital * BASE_POSITION_PCT
        # Confidence boost: scale linearly from 1.0x at threshold to CONFIDENCE_BOOST at 1.0
        threshold = LONG_THRESHOLD if signal == 'BUY' else SHORT_THRESHOLD
        conf_range = 1.0 - threshold
        if conf_range > 0:
            boost = 1.0 + (CONFIDENCE_BOOST - 1.0) * min((confidence - threshold) / conf_range, 1.0)
        else:
            boost = 1.0
        sized = base * boost
        # Clamp
        sized = max(sized, self.capital * MIN_POSITION_PCT)
        sized = min(sized, self.capital * MAX_POSITION_PCT)
        return sized

    def get_current_price(self, ticker, date, price_cache):
        """Get price for a ticker on a date, with nearest-day fallback."""
        if ticker not in price_cache:
            return None
        prices = price_cache[ticker]
        if date in prices.index:
            return float(prices.loc[date])
        # Find nearest prior date
        prior = prices.index[prices.index <= date]
        if len(prior) > 0:
            return float(prices.iloc[prices.index.get_loc(prior[-1])])
        return None

    def mark_to_market(self, date, price_cache):
        """Calculate total portfolio value including unrealized P&L."""
        total = self.capital  # Cash
        for pos in self.positions:
            price = self.get_current_price(pos['ticker'], date, price_cache)
            if price is None:
                continue
            if pos['signal'] == 'BUY':
                unrealized = pos['quantity'] * (price - pos['entry_price'])
            else:
                unrealized = abs(pos['quantity']) * (pos['entry_price'] - price)
            total += unrealized
        return total

    def run(self):
        logger.info(f'=== PHASE 16 BACKTEST: {START_DATE} to {END_DATE} ===')
        logger.info(f'Capital: ${INITIAL_CAPITAL:,.0f} | Max pos: {MAX_POSITIONS} | '
                     f'Size: {BASE_POSITION_PCT:.0%} | TP: {TP_PCT:.0%} | SL: {SL_PCT_DEFAULT:.0%} | '
                     f'Trail: {TRAIL_PCT:.1%} after {TRAIL_ACTIVATE_PCT:.1%} | Timeout: {TIMEOUT_DAYS}d')

        # Filter universe to tickers with data
        tickers = [t for t in TICKER_UNIVERSE if (DATA_DIR / f'{t}.parquet').exists()]
        logger.info(f'Universe: {len(tickers)} tickers (of {len(TICKER_UNIVERSE)} requested)')

        # Pre-load price data
        price_cache = {}
        full_data_cache = {}
        for ticker in tickers:
            df = self.load_ticker(ticker)
            if df is not None:
                col = self.get_close_col(df)
                price_cache[ticker] = df[col]
                full_data_cache[ticker] = df

        logger.info(f'Loaded price data for {len(price_cache)} tickers')

        # Build trading day calendar and rebalancing dates
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        all_bdays = pd.bdate_range(start_dt, end_dt)
        rebal_dates_raw = pd.date_range(start_dt, end_dt, freq=REBALANCE_FREQ)
        # Snap rebal dates to business days
        rebal_set = set()
        for d in rebal_dates_raw:
            candidates = all_bdays[all_bdays >= d]
            if len(candidates) > 0:
                rebal_set.add(candidates[0])

        logger.info(f'Trading days: {len(all_bdays)} | Rebalance dates: {len(rebal_set)} (biweekly)')

        # ── MAIN LOOP: iterate over every business day ──
        for day_idx, today in enumerate(all_bdays):
            # 1. Daily exit management for ALL open positions
            self._daily_exit_check(today, price_cache)

            # 2. Uncle Point check
            mtm = self.mark_to_market(today, price_cache)
            if mtm > self.peak_capital:
                self.peak_capital = mtm
            dd = (self.peak_capital - mtm) / self.peak_capital if self.peak_capital > 0 else 0

            if dd >= UNCLE_POINT_DD and self.cooldown_until is None:
                logger.info(f'[UNCLE POINT] DD={dd:.1%} >= {UNCLE_POINT_DD:.0%} | Liquidating all positions')
                for pos in list(self.positions):
                    self._close_position(pos, today, price_cache, 'UNCLE_POINT')
                self.cooldown_until = today + pd.Timedelta(days=COOLDOWN_DAYS)

            # 3. Record equity (mark-to-market)
            self.equity_curve.append({'date': today, 'equity': mtm})

            # 4. Weekly rebalancing: generate signals and enter new positions
            if today in rebal_set:
                # Skip if in cooldown
                if self.cooldown_until is not None and today < self.cooldown_until:
                    continue
                elif self.cooldown_until is not None and today >= self.cooldown_until:
                    self.cooldown_until = None
                    logger.info(f'[COOLDOWN] Expired, resuming trading')

                open_slots = MAX_POSITIONS - len(self.positions)
                if open_slots <= 0:
                    continue

                # Generate signals
                candidates = []
                held_tickers = {p['ticker'] for p in self.positions}
                for ticker in tickers:
                    if ticker not in price_cache or ticker in held_tickers:
                        continue

                    prices = price_cache[ticker]
                    if today not in prices.index:
                        continue

                    # Need history up to today for feature generation
                    df = full_data_cache.get(ticker)
                    if df is None:
                        continue
                    df_hist = df[df.index <= today]
                    if len(df_hist) < 200:
                        continue

                    signal, confidence, details = self.get_signal(df_hist)
                    self.total_signals_generated += 1

                    # Apply directional thresholds
                    if signal == 'BUY' and confidence >= LONG_THRESHOLD:
                        candidates.append({
                            'ticker': ticker,
                            'signal': signal,
                            'confidence': confidence,
                            'price': float(prices.loc[today])
                        })
                    elif SHORTS_ENABLED and signal == 'SELL_SHORT' and confidence >= SHORT_THRESHOLD:
                        candidates.append({
                            'ticker': ticker,
                            'signal': signal,
                            'confidence': confidence,
                            'price': float(prices.loc[today])
                        })

                # Rank by confidence, take top N
                candidates.sort(key=lambda x: x['confidence'], reverse=True)
                new_entries = candidates[:open_slots]

                for entry in new_entries:
                    pos_size = self.calc_position_size(entry['confidence'], entry['signal'])
                    qty = int(pos_size / entry['price']) if entry['price'] > 0 else 0
                    if qty == 0:
                        continue

                    self.positions.append({
                        'ticker': entry['ticker'],
                        'signal': entry['signal'],
                        'entry_price': entry['price'],
                        'entry_date': today,
                        'quantity': qty if entry['signal'] == 'BUY' else -qty,
                        'confidence': entry['confidence'],
                        'peak_price': entry['price'],  # For trailing stop
                    })

            # Progress logging
            if (day_idx + 1) % 65 == 0:  # Roughly quarterly
                yr = today.year
                logger.info(
                    f'[PROGRESS] {today.strftime("%Y-%m-%d")} | '
                    f'Equity: ${mtm:,.0f} | Pos: {len(self.positions)} | '
                    f'Trades: {len(self.trades)} | DD: {dd:.1%}'
                )

        # Close all remaining positions at end
        for pos in list(self.positions):
            self._close_position(pos, all_bdays[-1], price_cache, 'END_OF_BACKTEST')

        self._report()

    def _daily_exit_check(self, today, price_cache):
        """Check all positions for SL, TP, trailing stop, and timeout DAILY."""
        to_close = []
        for pos in self.positions:
            ticker = pos['ticker']
            price = self.get_current_price(ticker, today, price_cache)
            if price is None:
                continue

            entry_price = pos['entry_price']
            days_held = (today - pos['entry_date']).days

            # P&L calculation
            if pos['signal'] == 'BUY':
                pnl_pct = (price - entry_price) / entry_price
                # Update peak for trailing
                if price > pos['peak_price']:
                    pos['peak_price'] = price
                trail_pnl = (price - pos['peak_price']) / pos['peak_price']
            else:  # SHORT
                pnl_pct = (entry_price - price) / entry_price
                # For shorts, peak is the lowest price
                if price < pos['peak_price']:
                    pos['peak_price'] = price
                trail_pnl = (pos['peak_price'] - price) / pos['peak_price']

            # Peak unrealized gain from entry
            if pos['signal'] == 'BUY':
                peak_gain = (pos['peak_price'] - entry_price) / entry_price
            else:
                peak_gain = (entry_price - pos['peak_price']) / entry_price

            # ── Exit Rules (priority order) ──
            # 1. Stop Loss
            if pnl_pct <= -SL_PCT_DEFAULT:
                to_close.append((pos, 'STOP_LOSS'))
                continue
            # 2. Take Profit
            if pnl_pct >= TP_PCT:
                to_close.append((pos, 'TAKE_PROFIT'))
                continue
            # 3. Trailing Stop (only if position was profitable enough)
            if peak_gain >= TRAIL_ACTIVATE_PCT and trail_pnl <= -TRAIL_PCT:
                to_close.append((pos, 'TRAIL_STOP'))
                continue
            # 4. Timeout
            if days_held >= TIMEOUT_DAYS:
                to_close.append((pos, 'TIMEOUT'))
                continue

        for pos, reason in to_close:
            self._close_position(pos, today, price_cache, reason)

    def _close_position(self, pos, close_date, price_cache, reason):
        """Close a position and record the trade."""
        ticker = pos['ticker']
        exit_price = self.get_current_price(ticker, close_date, price_cache)
        if exit_price is None:
            exit_price = pos['entry_price']

        if pos['signal'] == 'BUY':
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
            pnl_usd = pos['quantity'] * (exit_price - pos['entry_price'])
        else:
            pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
            pnl_usd = abs(pos['quantity']) * (pos['entry_price'] - exit_price)

        self.capital += pnl_usd
        days_held = (close_date - pos['entry_date']).days

        self.trades.append({
            'ticker': ticker,
            'signal': pos['signal'],
            'entry_date': pos['entry_date'],
            'exit_date': close_date,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'days_held': days_held,
            'exit_reason': reason,
            'confidence': pos['confidence'],
        })

        if pos in self.positions:
            self.positions.remove(pos)

    def _report(self):
        """Generate and print backtest results."""
        if not self.trades:
            logger.warning('[BACKTEST] No trades generated!')
            return

        df = pd.DataFrame(self.trades)
        eq = pd.DataFrame(self.equity_curve)

        total = len(df)
        winners = df[df['pnl_pct'] > 0]
        losers = df[df['pnl_pct'] <= 0]
        win_rate = len(winners) / total * 100
        avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0

        longs = df[df['signal'] == 'BUY']
        shorts = df[df['signal'] == 'SELL_SHORT']
        long_wr = len(longs[longs['pnl_pct'] > 0]) / len(longs) * 100 if len(longs) > 0 else 0
        short_wr = len(shorts[shorts['pnl_pct'] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0

        final_eq = eq['equity'].iloc[-1] if len(eq) > 0 else self.capital
        total_return = (final_eq / INITIAL_CAPITAL - 1) * 100
        years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
        cagr = ((final_eq / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if final_eq > 0 else -100

        # Max drawdown from equity curve
        eq_vals = eq['equity'].values
        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / peak * 100
        max_dd = dd.min()

        # Sharpe (daily returns, annualized)
        if len(eq) > 1:
            returns = pd.Series(eq_vals).pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = winners['pnl_usd'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Exit reasons
        exit_reasons = df['exit_reason'].value_counts().to_dict()

        # Long/Short P&L
        long_pnl = longs['pnl_usd'].sum() if len(longs) > 0 else 0
        short_pnl = shorts['pnl_usd'].sum() if len(shorts) > 0 else 0

        results = {
            'period': f'{START_DATE} to {END_DATE}',
            'initial_capital': INITIAL_CAPITAL,
            'final_equity': round(float(final_eq), 2),
            'total_return_pct': round(total_return, 2),
            'cagr_pct': round(cagr, 2),
            'max_drawdown_pct': round(float(max_dd), 2),
            'sharpe_ratio': round(float(sharpe), 2),
            'profit_factor': round(profit_factor, 2),
            'total_trades': total,
            'win_rate_pct': round(win_rate, 1),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_win_rate': round(long_wr, 1),
            'short_win_rate': round(short_wr, 1),
            'long_pnl_usd': round(long_pnl, 2),
            'short_pnl_usd': round(short_pnl, 2),
            'avg_days_held': round(df['days_held'].mean(), 1),
            'exit_reasons': exit_reasons,
            'total_signals': self.total_signals_generated,
            'config': {
                'base_position_pct': BASE_POSITION_PCT,
                'tp_pct': TP_PCT,
                'sl_pct': SL_PCT_DEFAULT,
                'trail_activate': TRAIL_ACTIVATE_PCT,
                'trail_pct': TRAIL_PCT,
                'timeout_days': TIMEOUT_DAYS,
                'long_threshold': LONG_THRESHOLD,
                'short_threshold': SHORT_THRESHOLD,
                'rebalance': REBALANCE_FREQ,
                'max_positions': MAX_POSITIONS,
                'uncle_point': UNCLE_POINT_DD,
            }
        }

        logger.info('=' * 60)
        logger.info('  PHASE 16 BACKTEST RESULTS')
        logger.info('=' * 60)
        logger.info(f'  Period:         {results["period"]}')
        logger.info(f'  Initial:        ${INITIAL_CAPITAL:,.0f}')
        logger.info(f'  Final:          ${final_eq:,.0f}')
        logger.info(f'  Total Return:   {total_return:+.2f}%')
        logger.info(f'  CAGR:           {cagr:+.2f}%')
        logger.info(f'  Max Drawdown:   {max_dd:.2f}%')
        logger.info(f'  Sharpe Ratio:   {sharpe:.2f}')
        logger.info(f'  Profit Factor:  {profit_factor:.2f}')
        logger.info(f'  Total Trades:   {total} (Long: {len(longs)}, Short: {len(shorts)})')
        logger.info(f'  Win Rate:       {win_rate:.1f}% (Long: {long_wr:.1f}%, Short: {short_wr:.1f}%)')
        logger.info(f'  Avg Win:        {avg_win:+.2f}%')
        logger.info(f'  Avg Loss:       {avg_loss:.2f}%')
        logger.info(f'  Avg Hold:       {df["days_held"].mean():.1f} days')
        logger.info(f'  Long P&L:       ${long_pnl:+,.0f}')
        logger.info(f'  Short P&L:      ${short_pnl:+,.0f}')
        logger.info(f'  Exit Reasons:   {exit_reasons}')
        logger.info(f'  Signals Eval:   {self.total_signals_generated}')
        logger.info('=' * 60)

        # Save results
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = REPORTS_DIR / f'backtest_phase16_{ts}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f'[SAVE] Results: {json_path}')

        csv_path = REPORTS_DIR / f'backtest_phase16_trades_{ts}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f'[SAVE] Trades: {csv_path}')

        eq_path = REPORTS_DIR / f'backtest_phase16_equity_{ts}.csv'
        eq.to_csv(eq_path, index=False)
        logger.info(f'[SAVE] Equity: {eq_path}')

        return results


if __name__ == '__main__':
    try:
        bt = Phase16Backtest()
        bt.run()
    except Exception as e:
        logger.error(f'[FATAL] Backtest failed: {e}')
        traceback.print_exc()
        sys.exit(1)
