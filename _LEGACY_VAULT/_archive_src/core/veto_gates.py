"""
Veto Gates - Hard rules that can reject trades.
A trade is INVALID if ANY veto rule applies.
No overrides. Ever. "No trade" is a valid and preferred outcome.
"""
import json
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum

class VetoReason(Enum):
    """Standardized veto reason codes."""
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    POOR_RISK_REWARD = "POOR_RISK_REWARD"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    HIGH_SPREAD = "HIGH_SPREAD"
    BLACKLISTED = "BLACKLISTED"
    SECTOR_LIMIT = "SECTOR_LIMIT"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    MAX_POSITIONS = "MAX_POSITIONS"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    REGIME_UNFAVORABLE = "REGIME_UNFAVORABLE"

@dataclass
class VetoResult:
    """Result of veto gate check."""
    passed: bool
    reason: Optional[VetoReason] = None
    details: str = ""
    
    def __str__(self):
        if self.passed:
            return "PASSED"
        return f"VETOED: {self.reason.value} - {self.details}"


class VetoGates:
    """
    Implements all veto rules from Trading Constitution.
    All gates must pass for a trade to be valid.
    """
    
    def __init__(self, config_path: str = "config/trading_constitution.json"):
        self.config = self._load_config(config_path)
        self.veto_rules = self.config.get('veto_rules', {})
        self.risk_priorities = self.config.get('risk_priorities', {})
        self.portfolio_rules = self.config.get('portfolio_rules', {})
        self.blacklist = set(self.config.get('blacklist', []))
    
    def _load_config(self, path: str) -> dict:
        """Load trading constitution config."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default config if file not found."""
        return {
            'veto_rules': {
                'low_confidence_threshold': 0.45,
                'min_risk_reward_ratio': 1.5,
                'min_liquidity_volume': 100000,
                'max_spread_pct': 0.005
            },
            'risk_priorities': {
                'max_portfolio_drawdown': 0.20,
                'max_trade_risk': 0.02,
                'daily_loss_limit': 0.05
            },
            'portfolio_rules': {
                'max_sector_exposure': 0.30,
                'max_correlation_exposure': 0.50,
                'max_positions': 20
            },
            'blacklist': []
        }
    
    def check_all(
        self,
        ticker: str,
        confidence: float,
        risk_reward_ratio: float,
        volume: float,
        spread_pct: float,
        current_positions: int = 0,
        sector_exposure: float = 0.0,
        daily_pnl_pct: float = 0.0,
        portfolio_drawdown_pct: float = 0.0
    ) -> Tuple[bool, List[VetoResult]]:
        """
        Run all veto gates. Returns (passed, list of results).
        Trade is valid ONLY if ALL gates pass.
        """
        results = []
        
        # Gate 1: Blacklist check
        results.append(self._check_blacklist(ticker))
        
        # Gate 2: Confidence check
        results.append(self._check_confidence(confidence))
        
        # Gate 3: Risk/Reward check
        results.append(self._check_risk_reward(risk_reward_ratio))
        
        # Gate 4: Liquidity check
        results.append(self._check_liquidity(volume))
        
        # Gate 5: Spread check
        results.append(self._check_spread(spread_pct))
        
        # Gate 6: Position limit check
        results.append(self._check_position_limit(current_positions))
        
        # Gate 7: Sector exposure check
        results.append(self._check_sector_exposure(sector_exposure))
        
        # Gate 8: Daily loss limit check
        results.append(self._check_daily_loss(daily_pnl_pct))
        
        # Gate 9: Portfolio drawdown check
        results.append(self._check_portfolio_drawdown(portfolio_drawdown_pct))
        
        # ALL must pass
        all_passed = all(r.passed for r in results)
        
        return all_passed, results
    
    def _check_blacklist(self, ticker: str) -> VetoResult:
        """Check if ticker is blacklisted."""
        if ticker.upper() in self.blacklist:
            return VetoResult(
                passed=False,
                reason=VetoReason.BLACKLISTED,
                details=f"{ticker} is in blacklist"
            )
        return VetoResult(passed=True)
    
    def _check_confidence(self, confidence: float) -> VetoResult:
        """Check if confidence meets threshold."""
        threshold = self.veto_rules.get('low_confidence_threshold', 0.45)
        if confidence < threshold:
            return VetoResult(
                passed=False,
                reason=VetoReason.LOW_CONFIDENCE,
                details=f"Confidence {confidence:.2f} < threshold {threshold:.2f}"
            )
        return VetoResult(passed=True)
    
    def _check_risk_reward(self, ratio: float) -> VetoResult:
        """Check if risk/reward ratio is acceptable."""
        min_ratio = self.veto_rules.get('min_risk_reward_ratio', 1.5)
        if ratio < min_ratio:
            return VetoResult(
                passed=False,
                reason=VetoReason.POOR_RISK_REWARD,
                details=f"R/R ratio {ratio:.2f} < min {min_ratio:.2f}"
            )
        return VetoResult(passed=True)
    
    def _check_liquidity(self, volume: float) -> VetoResult:
        """Check if volume meets liquidity requirements."""
        min_volume = self.veto_rules.get('min_liquidity_volume', 100000)
        if volume < min_volume:
            return VetoResult(
                passed=False,
                reason=VetoReason.LOW_LIQUIDITY,
                details=f"Volume {volume:,.0f} < min {min_volume:,.0f}"
            )
        return VetoResult(passed=True)
    
    def _check_spread(self, spread_pct: float) -> VetoResult:
        """Check if spread is acceptable."""
        max_spread = self.veto_rules.get('max_spread_pct', 0.005)
        if spread_pct > max_spread:
            return VetoResult(
                passed=False,
                reason=VetoReason.HIGH_SPREAD,
                details=f"Spread {spread_pct:.3%} > max {max_spread:.3%}"
            )
        return VetoResult(passed=True)
    
    def _check_position_limit(self, current_positions: int) -> VetoResult:
        """Check if position limit is reached."""
        max_positions = self.portfolio_rules.get('max_positions', 20)
        if current_positions >= max_positions:
            return VetoResult(
                passed=False,
                reason=VetoReason.MAX_POSITIONS,
                details=f"Positions {current_positions} >= max {max_positions}"
            )
        return VetoResult(passed=True)
    
    def _check_sector_exposure(self, sector_exposure: float) -> VetoResult:
        """Check if sector exposure limit is reached."""
        max_exposure = self.portfolio_rules.get('max_sector_exposure', 0.30)
        if sector_exposure >= max_exposure:
            return VetoResult(
                passed=False,
                reason=VetoReason.SECTOR_LIMIT,
                details=f"Sector exposure {sector_exposure:.1%} >= max {max_exposure:.1%}"
            )
        return VetoResult(passed=True)
    
    def _check_daily_loss(self, daily_pnl_pct: float) -> VetoResult:
        """Check if daily loss limit is reached."""
        limit = self.risk_priorities.get('daily_loss_limit', 0.05)
        if daily_pnl_pct <= -limit:
            return VetoResult(
                passed=False,
                reason=VetoReason.DAILY_LOSS_LIMIT,
                details=f"Daily P&L {daily_pnl_pct:.1%} <= limit -{limit:.1%}"
            )
        return VetoResult(passed=True)
    
    def _check_portfolio_drawdown(self, drawdown_pct: float) -> VetoResult:
        """Check if portfolio drawdown limit is reached."""
        max_dd = self.risk_priorities.get('max_portfolio_drawdown', 0.20)
        if abs(drawdown_pct) >= max_dd:
            return VetoResult(
                passed=False,
                reason=VetoReason.MAX_DRAWDOWN,
                details=f"Drawdown {drawdown_pct:.1%} >= max {max_dd:.1%}"
            )
        return VetoResult(passed=True)
    
    def get_passed_gates_string(self, results: List[VetoResult]) -> str:
        """Get string of passed gates for trade record."""
        passed = [f"Gate{i+1}" for i, r in enumerate(results) if r.passed]
        return ",".join(passed)
    
    def get_veto_summary(self, results: List[VetoResult]) -> str:
        """Get summary of veto results."""
        vetoed = [r for r in results if not r.passed]
        if not vetoed:
            return "ALL_PASSED"
        return "; ".join(str(r) for r in vetoed)
