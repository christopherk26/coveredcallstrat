from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class Position:
    """Represents a single 100-share position and its associated call option"""
    position_id: int
    entry_price: float
    current_shares: int = 100
    cash_balance: float = 0.0
    
    def __post_init__(self):
        self.trades: List[Trade] = []
        self.current_call: Optional[Call] = None
        self.current_volatility: float = None  # Track current IV for the position

@dataclass
class Call:
    """Represents a single call option contract"""
    write_week: int
    expiry_week: int
    strike: float
    premium: float
    stock_price_at_write: float
    volatility_at_write: float  # Add IV at time of writing

@dataclass
class Trade:
    """Records the result of each option trade"""
    position_id: int
    write_week: int
    expiry_week: int
    stock_price_at_write: float
    stock_price_at_expiry: float
    strike: float
    premium: float
    assigned: bool
    volatility_at_write: float  # Track IV for the trade
    rebuy_cost: Optional[float] = None

class CoveredCallStrategy:
    def __init__(
        self,
        num_positions: int,
        target_delta: float,
        weeks_to_expiry: int,
        risk_free_rate: float,
        base_volatility: float,
        vega_scale: float = 0.1402  # Added vega scaling factor from original implementation
    ):
        self.num_positions = num_positions
        self.target_delta = target_delta
        self.weeks_to_expiry = weeks_to_expiry
        self.risk_free_rate = risk_free_rate
        self.base_volatility = base_volatility
        self.vega_scale = vega_scale
        self.positions: List[Position] = []
        
    def initialize_positions(self, initial_price: float):
        """Initialize all positions with starting price and base volatility"""
        self.positions = [
            Position(position_id=i, entry_price=initial_price)
            for i in range(self.num_positions)
        ]
        for position in self.positions:
            position.current_volatility = self.base_volatility

    def update_volatility(self, position: Position, current_price: float) -> float:
        """Update implied volatility based on price movement"""
        if position.current_call:
            price_change = (current_price - position.current_call.stock_price_at_write) / position.current_call.stock_price_at_write
            vol_adjustment = self.vega_scale * price_change
            new_vol = max(0.1, position.current_volatility + vol_adjustment)  # Floor at 10% IV
            position.current_volatility = new_vol
            return new_vol
        return self.base_volatility

    def calculate_strike_price(self, current_price: float, volatility: float) -> float:
        """Calculate strike price for desired delta using Newton's method with current IV"""
        T = self.weeks_to_expiry / 52
        initial_guess = current_price * (1 + 0.5 * volatility * np.sqrt(T))
        
        def objective(K):
            d1 = (np.log(current_price/K) + (self.risk_free_rate + 0.5*volatility**2)*T) / (volatility*np.sqrt(T))
            calculated_delta = norm.cdf(d1)
            return calculated_delta - self.target_delta
        
        def derivative(K):
            d1 = (np.log(current_price/K) + (self.risk_free_rate + 0.5*volatility**2)*T) / (volatility*np.sqrt(T))
            return -norm.pdf(d1)/(K*volatility*np.sqrt(T))
        
        K = initial_guess
        for _ in range(50):
            diff = objective(K)
            if abs(diff) < 1e-5:
                break
            K = K - diff/derivative(K)
        
        return K

    def calculate_premium(self, stock_price: float, strike_price: float, volatility: float) -> float:
        """Calculate option premium using Black-Scholes with current IV"""
        T = self.weeks_to_expiry / 52
        
        d1 = (np.log(stock_price/strike_price) + (self.risk_free_rate + 0.5*volatility**2)*T) / (volatility*np.sqrt(T))
        d2 = d1 - volatility*np.sqrt(T)
        
        call_price = stock_price*norm.cdf(d1) - strike_price*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)
        return call_price

    def write_call(self, position: Position, current_week: int, stock_price: float) -> Call:
        """Write a new call for the given position using current IV"""
        current_vol = self.update_volatility(position, stock_price)
        strike = self.calculate_strike_price(stock_price, current_vol)
        premium = self.calculate_premium(stock_price, strike, current_vol)
        
        call = Call(
            write_week=current_week,
            expiry_week=current_week + self.weeks_to_expiry,
            strike=strike,
            premium=premium,
            stock_price_at_write=stock_price,
            volatility_at_write=current_vol
        )
        
        position.current_call = call
        position.cash_balance += premium * 100
        
        return call

    def check_expiration(self, position: Position, current_week: int, stock_price: float) -> Optional[Trade]:
        """Check if current call is expiring and handle assignment if necessary"""
        if not position.current_call or position.current_call.expiry_week != current_week:
            return None
            
        call = position.current_call
        assigned = stock_price > call.strike
        
        if assigned:
            position.cash_balance += call.strike * 100
            rebuy_cost = stock_price * 100
            position.cash_balance -= rebuy_cost
        else:
            rebuy_cost = None
            
        trade = Trade(
            position_id=position.position_id,
            write_week=call.write_week,
            expiry_week=current_week,
            stock_price_at_write=call.stock_price_at_write,
            stock_price_at_expiry=stock_price,
            strike=call.strike,
            premium=call.premium,
            assigned=assigned,
            volatility_at_write=call.volatility_at_write,
            rebuy_cost=rebuy_cost
        )
        
        position.trades.append(trade)
        position.current_call = None
        
        return trade

    def process_path(self, price_path: np.ndarray) -> List[Trade]:
        """Process an entire price path for all positions"""
        self.initialize_positions(price_path[0])
        all_trades = []
        
        for week, stock_price in enumerate(price_path):
            for position in self.positions:
                # Check for expiring calls
                if position.current_call and position.current_call.expiry_week == week:
                    trade = self.check_expiration(position, week, stock_price)
                    if trade:
                        all_trades.append(trade)
                
                # Write new calls if needed
                if not position.current_call:
                    self.write_call(position, week, stock_price)
        
        return all_trades

    def get_strategy_metrics(self) -> Dict:
        """Calculate and return strategy metrics including volatility info"""
        all_trades = []
        for position in self.positions:
            all_trades.extend(position.trades)
            
        if not all_trades:
            return {}
            
        df = pd.DataFrame([vars(t) for t in all_trades])
        
        return {
            'total_premium_collected': df['premium'].sum() * 100,
            'number_of_assignments': df['assigned'].sum(),
            'total_rebuy_cost': df['rebuy_cost'].sum(),
            'average_premium_per_trade': df['premium'].mean() * 100,
            'assignment_rate': (df['assigned'].sum() / len(df)) * 100,
            'average_volatility': df['volatility_at_write'].mean(),
            'max_volatility': df['volatility_at_write'].max(),
            'min_volatility': df['volatility_at_write'].min()
        }

    def get_position_metrics(self) -> pd.DataFrame:
        """Get metrics broken down by position"""
        position_data = []
        
        for position in self.positions:
            trades_df = pd.DataFrame([vars(t) for t in position.trades])
            
            if not trades_df.empty:
                position_data.append({
                    'position_id': position.position_id,
                    'total_premium': trades_df['premium'].sum() * 100,
                    'num_assignments': trades_df['assigned'].sum(),
                    'current_cash': position.cash_balance,
                    'num_trades': len(trades_df),
                    'avg_volatility': trades_df['volatility_at_write'].mean()
                })
        
        return pd.DataFrame(position_data)