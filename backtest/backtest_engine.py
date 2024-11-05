import pandas as pd
import numpy as np
from typing import Dict, Optional
from .portfolio import Portfolio

class BacktestEngine:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        
    def run(self) -> Dict:
        """백테스트 실행"""
        if not self.portfolio.validate():
            raise ValueError("포트폴리오 데이터가 유효하지 않습니다")
            
        # 포트폴리오 가치 계산
        portfolio_values = self._calculate_portfolio_values()
        
        # 벤치마크 가치 계산
        benchmark_values = self._calculate_benchmark_values()
        
        # 성과 지표 계산
        portfolio_return, benchmark_return = self.portfolio.get_period_returns()
        
        return {
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return
        }
        
    def _calculate_portfolio_values(self) -> pd.Series:
        """일별 포트폴리오 가치 계산"""
        portfolio_value = self.portfolio.initial_capital
        portfolio_values = []
        
        returns = self.portfolio.stock_prices.pct_change()
        for date in returns.index:
            daily_return = np.sum(returns.loc[date] * self.portfolio.weights)
            portfolio_value *= (1 + daily_return)
            portfolio_values.append(portfolio_value)
            
        return pd.Series(portfolio_values, index=returns.index)
        
    def _calculate_benchmark_values(self) -> pd.Series:
        """벤치마크 가치 계산"""
        return self.portfolio.initial_capital * (
            self.portfolio.benchmark / self.portfolio.benchmark.iloc[0]
        )
    
    def _calculate_statistics(self) -> Dict:
        """백테스트 통계치 계산"""
        # TODO: 추가적인 통계치 계산 (MaxDrawdown, 승률 등) 구현
        pass