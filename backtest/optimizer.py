import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple
from .portfolio import Portfolio


class PortfolioOptimizer:
    def __init__(
        self,
        min_weight: float = 0.05,
        max_weight: float = 0.40,
        target_return: float = 0.05,
        risk_free_rate: float = 0.02,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_return = target_return
        self.risk_free_rate = risk_free_rate

    def optimize(self, portfolio: Portfolio) -> np.ndarray:
        """포트폴리오 최적화 수행"""
        if portfolio.stock_prices is None:
            raise ValueError("주가 데이터가 필요합니다")

        returns = portfolio.stock_prices.pct_change().dropna()
        n_assets = len(returns.columns)

        # 제약조건 설정
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # 비중 합 = 1
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # 최적화
        best_weights = None
        lowest_risk = np.inf

        for _ in range(20):
            try:
                # 초기 비중 설정
                initial_weights = np.random.random(n_assets)
                initial_weights /= initial_weights.sum()

                result = minimize(
                    lambda w: self._calculate_objective(w, returns),
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )

                if result.success:
                    risk = self._calculate_risk(result.x, returns)
                    if risk < lowest_risk:
                        lowest_risk = risk
                        best_weights = result.x

            except Exception as e:
                print(f"최적화 시도 실패: {str(e)}")
                continue

        if best_weights is None:
            print("최적화 실패. 동일 비중으로 설정합니다.")
            best_weights = np.array([1 / n_assets] * n_assets)

        portfolio.weights = best_weights
        self._print_optimization_result(portfolio)

        return best_weights

    def _calculate_objective(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """최적화 목적함수 계산"""
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_risk = self._calculate_risk(weights, returns)

        # 수익률 제약조건 위반 시 패널티
        return_penalty = max(0, self.target_return - portfolio_return) * 100

        return portfolio_risk + return_penalty

    def _calculate_risk(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """포트폴리오 위험(표준편차) 계산"""
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))) * 100

    def _calculate_portfolio_stats(
        self, weights: np.ndarray, returns: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """포트폴리오 통계치 계산"""
        port_return = np.sum(returns.mean() * weights) * 252
        port_risk = self._calculate_risk(weights, returns)
        sharpe = (port_return - self.risk_free_rate) / (port_risk / 100)

        return port_return, port_risk, sharpe

    def _print_optimization_result(self, portfolio: Portfolio) -> None:
        """최적화 결과 출력"""
        returns = portfolio.stock_prices.pct_change().dropna()
        port_return, port_risk, sharpe = self._calculate_portfolio_stats(
            portfolio.weights, returns
        )

        print("\n=== 포트폴리오 최적화 결과 ===")
        for code, weight in zip(portfolio.stock_prices.columns, portfolio.weights):
            stock_info = portfolio.stock_info[
                portfolio.stock_info["code"] == code
            ].iloc[0]
            print(f"{code} ({stock_info['name']}): {weight*100:.1f}%")

        print(f"\n예상 연간 수익률: {port_return*100:.1f}%")
        print(f"예상 연간 표준편차: {port_risk:.1f}%")
        print(f"샤프 비율: {sharpe:.2f}")
