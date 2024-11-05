from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional, List
from stock_crawler import StockData


@dataclass
class AnalysisResult:
    """주식 분석 결과를 저장하는 데이터 클래스"""

    code: str
    name: str
    annual_return: float
    volatility: float
    sharpe_ratio: float
    liquidity: float
    dividend_yield: float


class StockAnalyzer:
    """주식 데이터 분석을 수행하는 클래스"""

    def __init__(self, config):
        """
        Parameters:
            config (Config): 설정 객체
        """
        self.config = config

    def calculate_metrics(self, stock_data: StockData) -> Optional[AnalysisResult]:
        """
        개별 종목의 지표들을 계산합니다.

        Parameters:
            stock_data (StockData): 원본 주가 데이터

        Returns:
            Optional[AnalysisResult]: 계산된 지표들
        """
        try:
            returns = self._calculate_returns(stock_data.prices)

            annual_return = self._calculate_annual_return(returns)
            volatility = self._calculate_volatility(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(annual_return, volatility)
            liquidity = self._calculate_liquidity(stock_data.trading_values)

            return AnalysisResult(
                code=stock_data.code,
                name=stock_data.name,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                liquidity=liquidity,
                dividend_yield=stock_data.dividend_yield,
            )

        except Exception as e:
            print(f"분석 실패 - {stock_data.code}: {str(e)}")
            return None

    def _calculate_returns(self, prices: List[float]) -> pd.Series:
        """
        일간 수익률을 계산합니다.

        Parameters:
            prices (List[float]): 주가 리스트

        Returns:
            pd.Series: 일간 수익률
        """
        return pd.Series(prices).pct_change().dropna()

    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """
        연간 수익률을 계산합니다.

        Parameters:
            returns (pd.Series): 일간 수익률

        Returns:
            float: 연간 수익률
        """
        # 전체 기간 수익률 계산
        total_return = (1 + returns).prod() - 1

        # 연간화
        n_years = len(returns) / self.config.analysis.TRADING_DAYS
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        return annual_return

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """
        변동성을 계산합니다.

        Parameters:
            returns (pd.Series): 일간 수익률

        Returns:
            float: 연간화된 변동성
        """
        return returns.std() * np.sqrt(self.config.analysis.TRADING_DAYS)

    def _calculate_sharpe_ratio(self, annual_return: float, volatility: float) -> float:
        """
        샤프 비율을 계산합니다.

        Parameters:
            annual_return (float): 연간 수익률
            volatility (float): 변동성

        Returns:
            float: 샤프 비율
        """
        excess_return = annual_return - self.config.analysis.RISK_FREE_RATE
        return excess_return / volatility if volatility != 0 else 0

    def _calculate_liquidity(self, trading_values: List[float]) -> float:
        """
        유동성(일평균거래대금)을 계산합니다.

        Parameters:
            trading_values (List[float]): 일별 거래대금

        Returns:
            float: 일평균거래대금
        """
        return np.mean(trading_values)
