"""
Strategy Registry - Config-Driven Parallel Execution
=====================================================

Implements the strategy registry contract:
1. Config-driven strategy registration
2. Inheritance via deep-merge overrides
3. No code duplication
4. No branching logic inside baseline
5. All strategies run through SAME backtest engine

Baseline: VT_SweetSpot_v1 (IMMUTABLE - do not modify)
Variants: Inherit from baseline, declare only differences
"""

import json
import copy
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod


class StrategyRegistryError(Exception):
    """Raised when strategy registry contract is violated."""
    pass


@dataclass
class StrategyConfig:
    """
    Configuration for a trading strategy.
    
    All strategies share this config structure.
    Variants override specific fields via deep-merge.
    """
    # Identity
    strategy_id: str
    strategy_name: str
    parent_id: Optional[str] = None  # For inheritance
    is_baseline: bool = False
    
    # Timeframe
    timeframe: str = '1D'
    
    # Signal timing (FROZEN for all strategies)
    signal_time: str = 'close'  # Signal generated at close of Day T
    execution_time: str = 'open'  # Execute at open of Day T+1
    one_bar_delay: bool = True  # MANDATORY
    
    # Data policy (FROZEN for all strategies)
    data_provider: str = 'tiingo'
    price_policy: str = 'adjusted'
    
    # Indicators - SMA periods
    sma_periods: List[int] = field(default_factory=lambda: [25, 50, 100, 200])
    
    # Indicators - Stochastic
    daily_stoch_k: int = 10
    daily_stoch_d: int = 3
    daily_stoch_smooth: int = 3
    weekly_stoch_k: int = 19
    weekly_stoch_d: int = 4
    weekly_stoch_smooth: int = 4
    
    # Entry thresholds
    stoch_entry_threshold: int = 80  # K >= 80 AND D >= 80
    volume_multiplier: float = 1.0  # Volume >= multiplier * 30-day avg
    
    # Exit rules
    stoch_exit_threshold: int = 80  # Exit when K < 80 OR D < 80
    trailing_stop_pct: float = 0.03  # 3% default, allowed range 1-5%
    
    # Position sizing
    max_position_pct: float = 0.10  # 10% per position
    max_positions: int = 10
    
    # Constraints
    allow_shorts: bool = False  # v1 = no shorts
    
    # Overrides tracking (for audit)
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self):
        """Validate configuration constraints."""
        errors = []
        
        # Trailing stop must be 1-5%
        if not (0.01 <= self.trailing_stop_pct <= 0.05):
            errors.append(
                f"trailing_stop_pct must be 1-5%, got {self.trailing_stop_pct*100:.1f}%"
            )
        
        # One bar delay is mandatory
        if not self.one_bar_delay:
            errors.append("one_bar_delay must be True (mandatory)")
        
        # Price policy must be adjusted
        if self.price_policy != 'adjusted':
            errors.append(f"price_policy must be 'adjusted', got '{self.price_policy}'")
        
        # Stoch thresholds must be valid
        if not (0 <= self.stoch_entry_threshold <= 100):
            errors.append(f"stoch_entry_threshold must be 0-100")
        
        if not (0 <= self.stoch_exit_threshold <= 100):
            errors.append(f"stoch_exit_threshold must be 0-100")
        
        if errors:
            raise StrategyRegistryError(
                f"Strategy '{self.strategy_id}' config validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )


# ============================================================
# BASELINE CONFIGURATION - IMMUTABLE
# ============================================================

VT_SWEETSPOT_V1_CONFIG = StrategyConfig(
    strategy_id='VT_SweetSpot_v1',
    strategy_name='VT Sweet Spot v1 (Baseline)',
    parent_id=None,
    is_baseline=True,
    
    # Timeframe
    timeframe='1D',
    
    # Signal timing (FROZEN)
    signal_time='close',
    execution_time='open',
    one_bar_delay=True,
    
    # Data policy (FROZEN)
    data_provider='tiingo',
    price_policy='adjusted',
    
    # Indicators - SMA
    sma_periods=[25, 50, 100, 200],
    
    # Indicators - Stochastic (FROZEN)
    daily_stoch_k=10,
    daily_stoch_d=3,
    daily_stoch_smooth=3,
    weekly_stoch_k=19,
    weekly_stoch_d=4,
    weekly_stoch_smooth=4,
    
    # Entry thresholds (FROZEN)
    stoch_entry_threshold=80,
    volume_multiplier=1.0,
    
    # Exit rules
    stoch_exit_threshold=80,
    trailing_stop_pct=0.03,  # 3% default
    
    # Position sizing
    max_position_pct=0.10,
    max_positions=10,
    
    # Constraints
    allow_shorts=False,
    
    # No overrides for baseline
    overrides={}
)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge override into base config.
    Override values take precedence.
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


class StrategyRegistry:
    """
    Registry for all trading strategies.
    
    - Baseline is registered automatically and IMMUTABLE
    - Variants inherit from baseline via deep-merge
    - All strategies validated before registration
    """
    
    def __init__(self):
        self._strategies: Dict[str, StrategyConfig] = {}
        self._baseline_id: str = 'VT_SweetSpot_v1'
        
        # Register baseline (IMMUTABLE)
        self._register_baseline()
    
    def _register_baseline(self):
        """Register the immutable baseline strategy."""
        baseline = copy.deepcopy(VT_SWEETSPOT_V1_CONFIG)
        baseline.validate()
        self._strategies[baseline.strategy_id] = baseline
    
    def register_variant(
        self,
        strategy_id: str,
        strategy_name: str,
        overrides: Dict[str, Any]
    ) -> StrategyConfig:
        """
        Register a variant strategy that inherits from baseline.
        
        Args:
            strategy_id: Unique identifier for the variant
            strategy_name: Human-readable name
            overrides: Dictionary of config overrides
            
        Returns:
            The registered StrategyConfig
            
        Raises:
            StrategyRegistryError: If registration fails
        """
        if strategy_id == self._baseline_id:
            raise StrategyRegistryError(
                f"Cannot override baseline strategy '{self._baseline_id}'. "
                f"Baseline is IMMUTABLE."
            )
        
        if strategy_id in self._strategies:
            raise StrategyRegistryError(
                f"Strategy '{strategy_id}' already registered."
            )
        
        # Get baseline config as dict
        baseline_dict = self._strategies[self._baseline_id].to_dict()
        
        # Deep merge overrides
        merged = deep_merge(baseline_dict, overrides)
        
        # Update identity fields
        merged['strategy_id'] = strategy_id
        merged['strategy_name'] = strategy_name
        merged['parent_id'] = self._baseline_id
        merged['is_baseline'] = False
        merged['overrides'] = overrides
        
        # Create and validate config
        config = StrategyConfig.from_dict(merged)
        config.validate()
        
        # Register
        self._strategies[strategy_id] = config
        
        return config
    
    def get_strategy(self, strategy_id: str) -> StrategyConfig:
        """Get a registered strategy by ID."""
        if strategy_id not in self._strategies:
            raise StrategyRegistryError(
                f"Strategy '{strategy_id}' not found in registry."
            )
        return copy.deepcopy(self._strategies[strategy_id])
    
    def get_baseline(self) -> StrategyConfig:
        """Get the baseline strategy."""
        return self.get_strategy(self._baseline_id)
    
    def get_all_strategies(self) -> List[StrategyConfig]:
        """Get all registered strategies."""
        return [copy.deepcopy(s) for s in self._strategies.values()]
    
    def get_strategy_ids(self) -> List[str]:
        """Get all registered strategy IDs."""
        return list(self._strategies.keys())
    
    def validate_baseline_unchanged(self) -> bool:
        """
        Validate that baseline has not been modified.
        
        This is called before generating Excel output.
        FAIL RUN if baseline shows any modification.
        """
        current = self._strategies.get(self._baseline_id)
        if current is None:
            raise StrategyRegistryError("Baseline strategy missing from registry!")
        
        # Compare to frozen baseline
        frozen = VT_SWEETSPOT_V1_CONFIG
        
        # Check critical fields
        critical_fields = [
            'strategy_id', 'is_baseline', 'one_bar_delay', 'price_policy',
            'sma_periods', 'daily_stoch_k', 'daily_stoch_d', 'daily_stoch_smooth',
            'weekly_stoch_k', 'weekly_stoch_d', 'weekly_stoch_smooth',
            'stoch_entry_threshold', 'stoch_exit_threshold', 'allow_shorts'
        ]
        
        for field in critical_fields:
            current_val = getattr(current, field)
            frozen_val = getattr(frozen, field)
            if current_val != frozen_val:
                raise StrategyRegistryError(
                    f"FAIL RUN: Baseline '{field}' was modified! "
                    f"Expected {frozen_val}, got {current_val}. "
                    f"Baseline must remain IMMUTABLE."
                )
        
        return True
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies for comparison."""
        baseline = self.get_baseline()
        
        summary = {
            'baseline_id': baseline.strategy_id,
            'baseline_config': baseline.to_dict(),
            'variants': []
        }
        
        for strategy_id, config in self._strategies.items():
            if strategy_id == self._baseline_id:
                continue
            
            summary['variants'].append({
                'strategy_id': config.strategy_id,
                'strategy_name': config.strategy_name,
                'overrides': config.overrides,
                'full_config': config.to_dict()
            })
        
        return summary


# Global registry instance
_registry: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """Get global strategy registry instance."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry


def reset_strategy_registry():
    """Reset global registry (for testing)."""
    global _registry
    _registry = None


# ============================================================
# EXAMPLE VARIANT CONFIGURATIONS
# ============================================================

def register_experimental_variants(registry: StrategyRegistry):
    """Register experimental variant strategies."""
    
    # Variant 1: Tighter trailing stop
    registry.register_variant(
        strategy_id='VT_SweetSpot_v1_tight_stop',
        strategy_name='VT Sweet Spot v1 (Tight Stop 2%)',
        overrides={
            'trailing_stop_pct': 0.02  # 2% instead of 3%
        }
    )
    
    # Variant 2: Wider trailing stop
    registry.register_variant(
        strategy_id='VT_SweetSpot_v1_wide_stop',
        strategy_name='VT Sweet Spot v1 (Wide Stop 5%)',
        overrides={
            'trailing_stop_pct': 0.05  # 5% instead of 3%
        }
    )
    
    # Variant 3: More positions
    registry.register_variant(
        strategy_id='VT_SweetSpot_v1_more_positions',
        strategy_name='VT Sweet Spot v1 (15 Positions)',
        overrides={
            'max_positions': 15,
            'max_position_pct': 0.07  # Smaller per position
        }
    )
    
    # Variant 4: Experimental (combined changes)
    registry.register_variant(
        strategy_id='VT_SweetSpot_v1_experimental',
        strategy_name='VT Sweet Spot v1 (Experimental)',
        overrides={
            'trailing_stop_pct': 0.04,  # 4%
            'max_positions': 12,
            'max_position_pct': 0.08
        }
    )
