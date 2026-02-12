"""
Signal Cache System
===================

Cache generated signals to avoid redundant computation.
Signals are deterministic - same tickers + date range = same signals.

Cache key: MD5(tickers + start_date + end_date + strategy_type)
"""

import pandas as pd
import hashlib
import pickle
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class SignalCache:
    """Cache for trading signals."""
    
    def __init__(self, cache_dir: str = "signals/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        strategy_type: str
    ) -> str:
        """Generate unique cache key from parameters."""
        # Sort tickers for consistency
        sorted_tickers = sorted(tickers)
        
        # Create hash input
        hash_input = f"{','.join(sorted_tickers)}|{start_date}|{end_date}|{strategy_type}"
        
        # Generate MD5 hash
        cache_key = hashlib.md5(hash_input.encode()).hexdigest()
        
        return cache_key
    
    def save_signals(
        self,
        signals: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str,
        strategy_type: str,
        metadata: dict = None
    ):
        """
        Save signals to cache.
        
        Args:
            signals: DataFrame with signals
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            strategy_type: Strategy identifier (e.g., 'sweetspot', 'sweetspot_v2')
            metadata: Optional metadata dictionary
        """
        cache_key = self._generate_cache_key(tickers, start_date, end_date, strategy_type)
        
        # Save signals
        signals_path = self.cache_dir / f"{cache_key}_signals.pkl"
        with open(signals_path, 'wb') as f:
            pickle.dump(signals, f)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'cache_key': cache_key,
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'strategy_type': strategy_type,
            'signal_count': len(signals),
            'cached_at': datetime.now().isoformat()
        })
        
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Signals cached: {cache_key}")
        print(f"   📁 Cache path: {signals_path}")
    
    def load_signals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        strategy_type: str
    ) -> Optional[pd.DataFrame]:
        """
        Load signals from cache.
        
        Returns:
            DataFrame with signals if cached, None otherwise
        """
        cache_key = self._generate_cache_key(tickers, start_date, end_date, strategy_type)
        
        signals_path = self.cache_dir / f"{cache_key}_signals.pkl"
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
        
        if not signals_path.exists():
            return None
        
        # Load signals
        with open(signals_path, 'rb') as f:
            signals = pickle.load(f)
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"   ✅ Loaded cached signals: {cache_key}")
            print(f"   📊 Cached at: {metadata.get('cached_at', 'unknown')}")
            print(f"   📊 Signal count: {metadata.get('signal_count', len(signals))}")
        
        return signals
    
    def clear_cache(self, strategy_type: Optional[str] = None):
        """
        Clear signal cache.
        
        Args:
            strategy_type: If specified, only clear cache for this strategy type
        """
        if strategy_type:
            # Clear specific strategy cache
            count = 0
            for file in self.cache_dir.glob(f"*_metadata.json"):
                with open(file, 'r') as f:
                    metadata = json.load(f)
                
                if metadata.get('strategy_type') == strategy_type:
                    cache_key = metadata['cache_key']
                    
                    # Remove signals and metadata
                    signals_file = self.cache_dir / f"{cache_key}_signals.pkl"
                    if signals_file.exists():
                        signals_file.unlink()
                    file.unlink()
                    count += 1
            
            print(f"   🗑️ Cleared {count} cached signal files for strategy: {strategy_type}")
        else:
            # Clear all cache
            count = 0
            for file in self.cache_dir.glob("*"):
                file.unlink()
                count += 1
            
            print(f"   🗑️ Cleared {count} cached signal files")
    
    def list_cache(self) -> List[dict]:
        """List all cached signals."""
        cached_signals = []
        
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            cached_signals.append(metadata)
        
        return cached_signals
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cached_signals = self.list_cache()
        
        total_signals = sum(s.get('signal_count', 0) for s in cached_signals)
        
        strategies = {}
        for signal in cached_signals:
            strategy = signal.get('strategy_type', 'unknown')
            if strategy not in strategies:
                strategies[strategy] = 0
            strategies[strategy] += 1
        
        return {
            'total_cached': len(cached_signals),
            'total_signals': total_signals,
            'strategies': strategies,
            'cache_dir': str(self.cache_dir)
        }


# Global signal cache instance
_signal_cache = None

def get_signal_cache() -> SignalCache:
    """Get global signal cache instance."""
    global _signal_cache
    if _signal_cache is None:
        _signal_cache = SignalCache()
    return _signal_cache


if __name__ == "__main__":
    # Test signal cache
    cache = get_signal_cache()
    
    print("\n" + "=" * 70)
    print("📊 SIGNAL CACHE TEST")
    print("=" * 70)
    
    # Create test signals
    test_signals = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'ticker': ['AAPL'] * 10,
        'signal': [1] * 10,
        'confidence': [0.8] * 10
    })
    
    # Save signals
    print("\n1. Saving test signals...")
    cache.save_signals(
        signals=test_signals,
        tickers=['AAPL', 'GOOGL'],
        start_date='2020-01-01',
        end_date='2020-12-31',
        strategy_type='test_strategy'
    )
    
    # Load signals
    print("\n2. Loading test signals...")
    loaded_signals = cache.load_signals(
        tickers=['AAPL', 'GOOGL'],
        start_date='2020-01-01',
        end_date='2020-12-31',
        strategy_type='test_strategy'
    )
    
    if loaded_signals is not None:
        print(f"   ✅ Loaded {len(loaded_signals)} signals")
    
    # Cache stats
    print("\n3. Cache statistics:")
    stats = cache.get_cache_stats()
    print(f"   Total cached: {stats['total_cached']}")
    print(f"   Total signals: {stats['total_signals']}")
    print(f"   Strategies: {stats['strategies']}")
    
    # Clear cache
    print("\n4. Clearing test cache...")
    cache.clear_cache('test_strategy')
    
    print("\n✅ Signal cache test complete!")
