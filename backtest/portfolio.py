from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


@dataclass
class Portfolio:
    """포트폴리오 데이터를 저장하는 데이터 클래스"""

    initial_capital: float
    start_date: str
    end_date: str
    stock_prices: Optional[pd.DataFrame] = None
    benchmark: Optional[pd.Series] = None
    weights: Optional[np.ndarray] = None
    stock_info: Optional[pd.DataFrame] = None

    @property
    def n_assets(self) -> int:
        """포트폴리오 내 자산 수 반환"""
        if self.stock_prices is None:
            return 0
        return len(self.stock_prices.columns)

    def validate(self) -> bool:
        """포트폴리오 데이터 유효성 검증"""
        if self.stock_prices is None or self.stock_prices.empty:
            return False
        if self.benchmark is None or len(self.benchmark) == 0:
            return False
        if self.weights is not None and len(self.weights) != self.n_assets:
            return False
        return True

    def get_period_returns(self) -> Tuple[float, float]:
        """기간 수익률 계산"""
        if not self.validate():
            raise ValueError("포트폴리오 데이터가 유효하지 않습니다")

        portfolio_return = (self.stock_prices * self.weights).sum(axis=1)
        portfolio_return = (
            portfolio_return.iloc[-1] / portfolio_return.iloc[0] - 1
        ) * 100
        benchmark_return = (self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1) * 100

        return portfolio_return, benchmark_return
