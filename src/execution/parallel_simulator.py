"""
NeuralTrader Parallel Simulator - High-Performance Backtesting
================================================================

Parallel Daily Scoring for accelerated 26-year simulations.
Uses multiprocessing to process 4-8 trading days simultaneously.

Features:
- Multiprocessing pool for parallel day processing
- Thread-safe trade aggregation
- Memory-efficient chunked processing
- Progress tracking and performance monitoring
- Maintains TradeHistorian compatibility

Usage:
    from src.execution.parallel_simulator import ParallelSimulator
    
    simulator = ParallelSimulator(num_processes=6)
    performance_metrics = simulator.run_parallel_simulation(date_range, mode)
"""

import os
import sys
import json
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.execution.simulation_utils import (
    UnifiedDataManager,
    evaluate_exit_conditions, 
    check_risk_shields, 
    generate_ai_scores, 
    apply_risk_machine,
    calculate_portfolio_value
)
from src.execution.historian import TradeHistorian

logger = logging.getLogger(__name__)

class ParallelSimulator:
    """
    High-performance parallel simulator for accelerated backtesting
    """
    
    def __init__(self, num_processes: int = 6, chunk_size: int = 50):
        """
        Initialize parallel simulator
        
        Args:
            num_processes: Number of parallel processes (4-8 recommended)
            chunk_size: Number of trading days per chunk
        """
        self.num_processes = num_processes
        self.chunk_size = chunk_size
        self.data_manager = UnifiedDataManager()
        
        logger.info(f"[PARALLEL] Initialized with {num_processes} processes, chunk size {chunk_size}")
    
    def run_parallel_simulation(self, date_range: str, mode: str = 'simulation') -> Dict:
        """
        Run parallel simulation for date range
        
        Args:
            date_range: Date range in format 'YYYY-MM-DD,YYYY-MM-DD'
            mode: Simulation mode
            
        Returns:
            Performance metrics from aggregated results
        """
        try:
            logger.info("=" * 80)
            logger.info(f"[PARALLEL] Starting Parallel Simulation - {date_range}")
            logger.info(f"[PARALLEL] Processes: {self.num_processes}, Chunk Size: {self.chunk_size}")
            logger.info("=" * 80)
            
            # Initialize historian for trade aggregation
            historian = TradeHistorian(buffer_size=100)
            
            # Get trading dates
            trading_dates = self.data_manager.get_trading_dates(date_range)
            total_days = len(trading_dates)
            
            logger.info(f"[PARALLEL] Processing {total_days} trading days")
            
            # Split into chunks for parallel processing
            date_chunks = self._create_date_chunks(trading_dates)
            
            # Initialize portfolio state
            initial_portfolio = {
                'cash': 100000.0,
                'positions': {},
                'history': [],
                'peak_portfolio_value': 100000.0,
                'circuit_breaker_cooldown_until': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Process chunks in parallel
            start_time = time.time()
            all_results = []
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all chunks
                future_to_chunk = {}
                for i, chunk in enumerate(date_chunks):
                    future = executor.submit(
                        self._process_date_chunk,
                        chunk,
                        initial_portfolio.copy(),
                        i
                    )
                    future_to_chunk[future] = (i, chunk)
                
                # Collect results as they complete
                completed_chunks = 0
                for future in as_completed(future_to_chunk):
                    chunk_id, chunk_dates = future_to_chunk[future]
                    
                    try:
                        chunk_result = future.result()
                        all_results.append(chunk_result)
                        completed_chunks += 1
                        
                        # Log progress
                        progress = (completed_chunks / len(date_chunks)) * 100
                        elapsed = time.time() - start_time
                        rate = completed_chunks / elapsed if elapsed > 0 else 0
                        eta = (len(date_chunks) - completed_chunks) / rate if rate > 0 else 0
                        
                        logger.info(f"[PARALLEL] Chunk {chunk_id+1}/{len(date_chunks)} completed "
                                  f"({progress:.1f}%) - ETA: {eta/60:.1f}min")
                        
                    except Exception as e:
                        logger.error(f"[PARALLEL] Chunk {chunk_id} failed: {e}")
                        continue
            
            # Aggregate results
            logger.info("[PARALLEL] Aggregating results...")
            final_portfolio = self._aggregate_results(all_results, historian)
            
            # Calculate final metrics
            performance_metrics = historian.finalize()
            
            # Add parallel processing stats
            total_time = time.time() - start_time
            performance_metrics['parallel_stats'] = {
                'total_days': total_days,
                'processes_used': self.num_processes,
                'chunk_size': self.chunk_size,
                'total_time_seconds': total_time,
                'days_per_second': total_days / total_time,
                'chunks_processed': len(date_chunks)
            }
            
            logger.info("=" * 80)
            logger.info("[PARALLEL] SIMULATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Total Days: {performance_metrics['parallel_stats']['total_days']}")
            logger.info(f"Processes: {performance_metrics['parallel_stats']['processes_used']}")
            logger.info(f"Total Time: {total_time/60:.1f} minutes")
            logger.info(f"Processing Rate: {performance_metrics['parallel_stats']['days_per_second']:.2f} days/sec")
            logger.info(f"Total Trades: {performance_metrics['total_trades']}")
            logger.info(f"Final Capital: ${performance_metrics['final_capital']:,.2f}")
            logger.info(f"CAGR: {performance_metrics['cagr']:.2%}")
            logger.info(f"Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"[PARALLEL] Simulation failed: {e}", exc_info=True)
            raise
    
    def _create_date_chunks(self, trading_dates: List[pd.Timestamp]) -> List[List[pd.Timestamp]]:
        """Split trading dates into chunks for parallel processing"""
        chunks = []
        for i in range(0, len(trading_dates), self.chunk_size):
            chunk = trading_dates[i:i + self.chunk_size]
            chunks.append(chunk)
        
        logger.info(f"[PARALLEL] Created {len(chunks)} chunks from {len(trading_dates)} trading days")
        return chunks
    
    def _process_date_chunk(self, date_chunk: List[pd.Timestamp], 
                          initial_portfolio: Dict, chunk_id: int) -> Dict:
        """
        Process a chunk of trading dates in a separate process
        
        Args:
            date_chunk: List of trading dates to process
            initial_portfolio: Initial portfolio state
            chunk_id: Chunk identifier for logging
            
        Returns:
            Chunk processing results
        """
        try:
            # Setup process-specific logging
            process_logger = logging.getLogger(f"Process_{chunk_id}")
            
            chunk_results = {
                'chunk_id': chunk_id,
                'dates_processed': len(date_chunk),
                'trades': [],
                'final_portfolio': initial_portfolio.copy(),
                'equity_curve': []
            }
            
            # Process each date in the chunk
            current_portfolio = initial_portfolio.copy()
            
            for i, current_date in enumerate(date_chunk):
                date_str = current_date.strftime('%Y-%m-%d')
                
                if i % 10 == 0:  # Progress update within chunk
                    process_logger.debug(f"[Chunk {chunk_id}] Processing day {i+1}/{len(date_chunk)}: {date_str}")
                
                # Get available tickers for this date
                available_tickers = self.data_manager.get_available_tickers_for_date(current_date)
                
                if not available_tickers:
                    continue
                
                # Load data for available tickers
                market_data = self.data_manager.load_ticker_data_for_date(
                    available_tickers, current_date, lookback_days=252
                )
                
                if not market_data:
                    continue
                
                # Process this trading day
                day_result = self._process_trading_day(
                    current_date, current_portfolio, market_data, process_logger
                )
                
                # Update portfolio
                current_portfolio = day_result['portfolio']
                
                # Record trades
                chunk_results['trades'].extend(day_result['trades'])
                
                # Record equity
                chunk_results['equity_curve'].append({
                    'date': date_str,
                    'portfolio_value': day_result['portfolio_value']
                })
            
            chunk_results['final_portfolio'] = current_portfolio
            return chunk_results
            
        except Exception as e:
            logger.error(f"[PARALLEL] Chunk {chunk_id} processing failed: {e}")
            raise
    
    def _process_trading_day(self, current_date: pd.Timestamp, 
                          portfolio: Dict, market_data: Dict[str, pd.DataFrame],
                          logger: logging.Logger) -> Dict:
        """
        Process a single trading day
        
        Returns:
            Dictionary with portfolio, trades, and portfolio value
        """
        try:
            # 1. Evaluate exits
            exit_trades = evaluate_exit_conditions(portfolio, market_data, current_date)
            
            # Execute exits
            for trade in exit_trades:
                ticker = trade['ticker']
                position = portfolio['positions'].get(ticker, {})
                
                if position:
                    entry_price = position.get('cost_basis', 0)
                    exit_price = trade['price']
                    shares = position.get('shares', 0)
                    pnl_realized = (exit_price - entry_price) * shares
                    
                    # Update portfolio
                    portfolio['cash'] += exit_price * shares
                    del portfolio['positions'][ticker]
                    
                    # Add trade record
                    trade['pnl_realized'] = pnl_realized
                    trade['date'] = current_date.strftime('%Y-%m-%d')
            
            # 2. Check risk shields
            shields_passed = check_risk_shields(market_data, current_date)
            
            if not shields_passed:
                return {
                    'portfolio': portfolio,
                    'trades': exit_trades,
                    'portfolio_value': calculate_portfolio_value(portfolio, market_data)
                }
            
            # 3. Generate AI scores
            ai_scores = generate_ai_scores(market_data, min_confidence=0.60)
            
            if not ai_scores:
                return {
                    'portfolio': portfolio,
                    'trades': exit_trades,
                    'portfolio_value': calculate_portfolio_value(portfolio, market_data)
                }
            
            # 4. Apply risk machine
            entry_trades, circuit_active = apply_risk_machine(
                ai_scores, portfolio, market_data, current_date
            )
            
            if circuit_active:
                return {
                    'portfolio': portfolio,
                    'trades': exit_trades,
                    'portfolio_value': calculate_portfolio_value(portfolio, market_data)
                }
            
            # 5. Execute entries
            all_trades = exit_trades.copy()
            
            for trade in entry_trades:
                ticker = trade['ticker']
                price = trade['price']
                quantity = trade['quantity']
                ai_score = trade.get('ai_score', 0.0)
                
                # Check if we have enough cash
                cost = price * quantity
                if portfolio['cash'] >= cost:
                    # Update portfolio
                    portfolio['cash'] -= cost
                    portfolio['positions'][ticker] = {
                        'shares': quantity,
                        'cost_basis': price,
                        'entry_date': current_date.strftime('%Y-%m-%d'),
                        'ai_score': ai_score
                    }
                    
                    # Add trade record
                    trade['date'] = current_date.strftime('%Y-%m-%d')
                    all_trades.append(trade)
            
            # Update portfolio peak value
            current_value = calculate_portfolio_value(portfolio, market_data)
            if current_value > portfolio['peak_portfolio_value']:
                portfolio['peak_portfolio_value'] = current_value
            
            return {
                'portfolio': portfolio,
                'trades': all_trades,
                'portfolio_value': current_value
            }
            
        except Exception as e:
            logger.error(f"[PARALLEL] Failed to process trading day {current_date}: {e}")
            return {
                'portfolio': portfolio,
                'trades': [],
                'portfolio_value': calculate_portfolio_value(portfolio, market_data)
            }
    
    def _aggregate_results(self, all_results: List[Dict], historian: TradeHistorian) -> Dict:
        """Aggregate results from all parallel chunks"""
        try:
            logger.info(f"[PARALLEL] Aggregating {len(all_results)} chunk results")
            
            # Sort results by chunk_id to maintain chronological order
            all_results.sort(key=lambda x: x['chunk_id'])
            
            # Log all trades to historian
            total_trades = 0
            for chunk_result in all_results:
                for trade in chunk_result['trades']:
                    historian.log_trade(
                        date=trade.get('date', ''),
                        ticker=trade.get('ticker', ''),
                        action=trade.get('action', ''),
                        price=trade.get('price', 0.0),
                        quantity=trade.get('quantity', 0),
                        ai_score=trade.get('ai_score', ''),
                        exit_reason=trade.get('reason', ''),
                        pnl_realized=trade.get('pnl_realized', 0.0)
                    )
                    total_trades += 1
            
            logger.info(f"[PARALLEL] Aggregated {total_trades} trades from all chunks")
            
            # Return final portfolio state (from last chunk)
            if all_results:
                return all_results[-1]['final_portfolio']
            else:
                return {}
                
        except Exception as e:
            logger.error(f"[PARALLEL] Result aggregation failed: {e}")
            return {}

def main():
    """Test function for parallel simulator"""
    try:
        simulator = ParallelSimulator(num_processes=6, chunk_size=50)
        
        # Test with a small date range
        date_range = "2023-01-01,2023-01-31"
        
        logger.info(f"[TEST] Running parallel simulation test: {date_range}")
        
        performance_metrics = simulator.run_parallel_simulation(date_range)
        
        logger.info("[TEST] Parallel simulation test completed successfully")
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"[TEST] Parallel simulation test failed: {e}")
        return {}

if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
