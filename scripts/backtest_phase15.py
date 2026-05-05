#!/usr/bin/env python3
"""
Phase 15 - Backtest with Realistic TB Models (5% TP, Clean Features)
====================================================================
Fast vectorized backtest: load data once, predict once per ticker, simulate trades.
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('Backtest15')

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
REPORTS_DIR = PROJECT_ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# Backtest config
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.02       # 2% risk per trade
MAX_POSITIONS = 15
TP_PCT = 0.05                  # 5% take profit
SL_PCT_DEFAULT = 0.05          # 5% stop loss default
TIMEOUT_DAYS = 10              # 10 day max hold
REBALANCE_FREQ = 'MS'         # Monthly rebalancing (fast)

# Top 50 liquid tickers for fast backtest
TICKER_UNIVERSE = [
    'SPY','QQQ','IWM','GLD','TLT','VXX',
    'AAPL','MSFT','GOOGL','AMZN','TSLA','NVDA','META',
    'JPM','BAC','GS','WFC','C','MS',
    'XOM','CVX','COP',
    'UNH','JNJ','PFE','ABBV','LLY','MRK',
    'HD','WMT','COST','MCD','NKE',
    'BA','CAT','GE','HON',
    'DIS','NFLX','CRM','ADBE','AMD','INTC',
    'V','MA','PYPL',
    'KO','PEP','PG','ABT'
]


class Phase15Backtest:
    def __init__(self):
        from core.ai_models import EnsemblePredictor
        self.ensemble = EnsemblePredictor()
        self.trades = []
        self.equity_curve = []
        self.capital = INITIAL_CAPITAL
        self.positions = []  # Active positions

    def load_ticker(self, ticker):
        path = DATA_DIR / f'{ticker}.parquet'
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        start = pd.to_datetime(START_DATE)
        end = pd.to_datetime(END_DATE)
        df = df[(df.index >= start) & (df.index <= end)]
        if len(df) < 100:
            return None
        return df

    def get_signal(self, ticker_data, threshold=0.30):
        try:
            signal, confidence, details = self.ensemble.predict_from_ohlcv(ticker_data, threshold=threshold)
            return signal, confidence, details
        except:
            return 'HOLD', 0.0, {}

    def run(self):
        logger.info(f'=== PHASE 15 BACKTEST: {START_DATE} to {END_DATE} ===')
        logger.info(f'Capital: ${INITIAL_CAPITAL:,.0f} | Max positions: {MAX_POSITIONS} | TP: {TP_PCT:.0%} | Timeout: {TIMEOUT_DAYS}d')

        # Use focused ticker universe
        tickers = [t for t in TICKER_UNIVERSE if (DATA_DIR / f'{t}.parquet').exists()]
        logger.info(f'Universe: {len(tickers)} tickers')

        # Get rebalancing dates (monthly for speed)
        rebal_dates = pd.date_range(START_DATE, END_DATE, freq=REBALANCE_FREQ)
        # Snap to nearest business day
        bdays = pd.date_range(START_DATE, END_DATE, freq='B')
        rebal_dates = pd.Index([bdays[bdays >= d][0] if len(bdays[bdays >= d]) > 0 else d for d in rebal_dates])
        logger.info(f'Rebalancing dates: {len(rebal_dates)} (monthly)')

        # Pre-load price data for position management
        price_cache = {}
        for ticker in tickers:
            df = self.load_ticker(ticker)
            if df is not None:
                col = 'adjClose' if 'adjClose' in df.columns else 'close'
                price_cache[ticker] = df[col]

        logger.info(f'Loaded price data for {len(price_cache)} tickers')

        # Main backtest loop
        for i, rebal_date in enumerate(rebal_dates):
            # Check and close expired/stopped positions
            self._manage_positions(rebal_date, price_cache)

            # Skip if at max capacity
            open_slots = MAX_POSITIONS - len(self.positions)
            if open_slots <= 0:
                self._record_equity(rebal_date)
                continue

            # Generate signals for all tickers
            candidates = []
            for ticker in tickers:
                if ticker not in price_cache:
                    continue
                # Skip if already holding
                if any(p['ticker'] == ticker for p in self.positions):
                    continue

                prices = price_cache[ticker]
                if rebal_date not in prices.index:
                    continue

                # Need enough history for features
                hist_end_idx = prices.index.get_loc(rebal_date)
                if hist_end_idx < 200:
                    continue

                # Load full data up to rebal_date for feature generation
                df = self.load_ticker(ticker)
                if df is None:
                    continue
                df_hist = df[df.index <= rebal_date]
                if len(df_hist) < 200:
                    continue

                signal, confidence, details = self.get_signal(df_hist)

                if signal in ('BUY', 'SELL_SHORT') and confidence >= 0.30:
                    candidates.append({
                        'ticker': ticker,
                        'signal': signal,
                        'confidence': confidence,
                        'price': float(prices.loc[rebal_date])
                    })

            # Rank by confidence, take top N
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            new_entries = candidates[:open_slots]

            for entry in new_entries:
                pos_size = self.capital * POSITION_SIZE_PCT
                qty = int(pos_size / entry['price']) if entry['price'] > 0 else 0
                if qty == 0:
                    continue

                self.positions.append({
                    'ticker': entry['ticker'],
                    'signal': entry['signal'],
                    'entry_price': entry['price'],
                    'entry_date': rebal_date,
                    'quantity': qty if entry['signal'] == 'BUY' else -qty,
                    'confidence': entry['confidence'],
                })

            self._record_equity(rebal_date)

            if (i + 1) % 26 == 0:
                logger.info(f'[PROGRESS] Week {i+1}/{len(rebal_dates)} | Equity: ${self.capital:,.0f} | Positions: {len(self.positions)} | Trades: {len(self.trades)}')

        # Close all remaining positions at end
        end_dt = pd.to_datetime(END_DATE)
        for pos in list(self.positions):
            self._close_position(pos, end_dt, price_cache, 'END_OF_BACKTEST')

        self._report()

    def _manage_positions(self, current_date, price_cache):
        to_close = []
        for pos in self.positions:
            ticker = pos['ticker']
            if ticker not in price_cache:
                to_close.append((pos, 'NO_DATA'))
                continue

            prices = price_cache[ticker]
            if current_date not in prices.index:
                continue

            current_price = float(prices.loc[current_date])
            entry_price = pos['entry_price']
            days_held = (current_date - pos['entry_date']).days

            # Calculate PnL based on direction
            if pos['signal'] == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SELL_SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Exit rules
            if pnl_pct >= TP_PCT:
                to_close.append((pos, 'TAKE_PROFIT'))
            elif pnl_pct <= -SL_PCT_DEFAULT:
                to_close.append((pos, 'STOP_LOSS'))
            elif days_held >= TIMEOUT_DAYS:
                to_close.append((pos, 'TIMEOUT'))

        for pos, reason in to_close:
            self._close_position(pos, current_date, price_cache, reason)

    def _close_position(self, pos, close_date, price_cache, reason):
        ticker = pos['ticker']
        if ticker in price_cache and close_date in price_cache[ticker].index:
            exit_price = float(price_cache[ticker].loc[close_date])
        else:
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

    def _record_equity(self, date):
        self.equity_curve.append({'date': date, 'equity': self.capital})

    def _report(self):
        if not self.trades:
            logger.warning('[BACKTEST] No trades generated')
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

        final_eq = self.capital
        total_return = (final_eq / INITIAL_CAPITAL - 1) * 100
        years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
        cagr = ((final_eq / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if final_eq > 0 else -100

        # Max drawdown
        eq_vals = eq['equity'].values
        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / peak * 100
        max_dd = dd.min()

        # Sharpe ratio
        if len(eq) > 1:
            returns = pd.Series(eq_vals).pct_change().dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(52) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = winners['pnl_usd'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Exit reason breakdown
        exit_reasons = df['exit_reason'].value_counts().to_dict()

        results = {
            'period': f'{START_DATE} to {END_DATE}',
            'initial_capital': INITIAL_CAPITAL,
            'final_equity': round(final_eq, 2),
            'total_return_pct': round(total_return, 2),
            'cagr_pct': round(cagr, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 2),
            'profit_factor': round(profit_factor, 2),
            'total_trades': total,
            'win_rate_pct': round(win_rate, 1),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_win_rate': round(long_wr, 1),
            'short_win_rate': round(short_wr, 1),
            'avg_days_held': round(df['days_held'].mean(), 1),
            'exit_reasons': exit_reasons,
            'tb_params': f'TP={TP_PCT:.0%} SL={SL_PCT_DEFAULT:.0%} Timeout={TIMEOUT_DAYS}d',
            'features': '64 clean (no raw OHLCV)',
            'models': 'XGBoost+LightGBM+HGB (40/40/20)',
        }

        logger.info('=' * 60)
        logger.info('  PHASE 15 BACKTEST RESULTS')
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
        logger.info(f'  Exit Reasons:   {exit_reasons}')
        logger.info('=' * 60)

        # Save
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = REPORTS_DIR / f'backtest_phase15_{ts}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f'[SAVE] Results: {json_path}')

        csv_path = REPORTS_DIR / f'backtest_phase15_trades_{ts}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f'[SAVE] Trades: {csv_path}')


if __name__ == '__main__':
    bt = Phase15Backtest()
    bt.run()
