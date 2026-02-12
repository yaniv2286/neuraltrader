"""
Cost Model - Mandatory for all backtests.
All backtests and evaluations must include commissions and bid-ask spread.
If costs are not applied → results are invalid.
"""
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeCosts:
    """Breakdown of costs for a single trade."""
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    total_cost: float = 0.0
    
    def __post_init__(self):
        self.total_cost = self.commission + self.spread_cost + self.slippage_cost


class CostModel:
    """
    Implements cost model from Trading Constitution.
    All costs are mandatory and must be applied to every trade.
    """
    
    def __init__(self, config_path: str = "config/trading_constitution.json"):
        self.config = self._load_config(config_path)
        cost_config = self.config.get('cost_model', {})
        
        self.enabled = cost_config.get('enabled', True)
        self.commission_per_trade = cost_config.get('commission_per_trade', 1.00)
        self.spread_estimate_pct = cost_config.get('spread_estimate_pct', 0.001)
        self.slippage_estimate_pct = cost_config.get('slippage_estimate_pct', 0.0005)
    
    def _load_config(self, path: str) -> dict:
        """Load trading constitution config."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'cost_model': {
                    'enabled': True,
                    'commission_per_trade': 1.00,
                    'spread_estimate_pct': 0.001,
                    'slippage_estimate_pct': 0.0005
                }
            }
    
    def calculate_entry_costs(
        self,
        trade_value: float,
        actual_spread_pct: Optional[float] = None
    ) -> TradeCosts:
        """
        Calculate costs for entering a trade.
        
        Args:
            trade_value: Dollar value of the trade
            actual_spread_pct: Actual spread if known, otherwise use estimate
        
        Returns:
            TradeCosts with breakdown
        """
        if not self.enabled:
            return TradeCosts()
        
        spread_pct = actual_spread_pct if actual_spread_pct else self.spread_estimate_pct
        
        return TradeCosts(
            commission=self.commission_per_trade,
            spread_cost=trade_value * spread_pct / 2,  # Half spread on entry
            slippage_cost=trade_value * self.slippage_estimate_pct
        )
    
    def calculate_exit_costs(
        self,
        trade_value: float,
        actual_spread_pct: Optional[float] = None
    ) -> TradeCosts:
        """
        Calculate costs for exiting a trade.
        
        Args:
            trade_value: Dollar value of the trade at exit
            actual_spread_pct: Actual spread if known, otherwise use estimate
        
        Returns:
            TradeCosts with breakdown
        """
        if not self.enabled:
            return TradeCosts()
        
        spread_pct = actual_spread_pct if actual_spread_pct else self.spread_estimate_pct
        
        return TradeCosts(
            commission=self.commission_per_trade,
            spread_cost=trade_value * spread_pct / 2,  # Half spread on exit
            slippage_cost=trade_value * self.slippage_estimate_pct
        )
    
    def calculate_round_trip_costs(
        self,
        entry_value: float,
        exit_value: float,
        actual_spread_pct: Optional[float] = None
    ) -> TradeCosts:
        """
        Calculate total costs for a round-trip trade (entry + exit).
        
        Args:
            entry_value: Dollar value at entry
            exit_value: Dollar value at exit
            actual_spread_pct: Actual spread if known
        
        Returns:
            TradeCosts with total breakdown
        """
        entry_costs = self.calculate_entry_costs(entry_value, actual_spread_pct)
        exit_costs = self.calculate_exit_costs(exit_value, actual_spread_pct)
        
        return TradeCosts(
            commission=entry_costs.commission + exit_costs.commission,
            spread_cost=entry_costs.spread_cost + exit_costs.spread_cost,
            slippage_cost=entry_costs.slippage_cost + exit_costs.slippage_cost
        )
    
    def apply_costs_to_pnl(
        self,
        gross_pnl: float,
        entry_value: float,
        exit_value: float,
        actual_spread_pct: Optional[float] = None
    ) -> tuple:
        """
        Apply costs to gross P&L to get net P&L.
        
        Args:
            gross_pnl: Gross profit/loss before costs
            entry_value: Dollar value at entry
            exit_value: Dollar value at exit
            actual_spread_pct: Actual spread if known
        
        Returns:
            (net_pnl, costs) tuple
        """
        costs = self.calculate_round_trip_costs(entry_value, exit_value, actual_spread_pct)
        net_pnl = gross_pnl - costs.total_cost
        return net_pnl, costs
    
    def get_config_dict(self) -> dict:
        """Get cost model configuration as dictionary."""
        return {
            'enabled': self.enabled,
            'commission_per_trade': self.commission_per_trade,
            'spread_estimate_pct': self.spread_estimate_pct,
            'slippage_estimate_pct': self.slippage_estimate_pct
        }
    
    def estimate_breakeven_move(self, trade_value: float) -> float:
        """
        Estimate the minimum price move needed to break even after costs.
        
        Args:
            trade_value: Dollar value of the trade
        
        Returns:
            Breakeven move as percentage
        """
        costs = self.calculate_round_trip_costs(trade_value, trade_value)
        return costs.total_cost / trade_value
    
    def validate_trade_profitability(
        self,
        expected_return_pct: float,
        trade_value: float
    ) -> tuple:
        """
        Check if expected return covers costs.
        
        Args:
            expected_return_pct: Expected return percentage
            trade_value: Dollar value of the trade
        
        Returns:
            (is_profitable, net_expected_return_pct)
        """
        breakeven = self.estimate_breakeven_move(trade_value)
        net_return = expected_return_pct - breakeven
        return net_return > 0, net_return
