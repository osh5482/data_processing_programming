import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from io import BytesIO
import base64
from .portfolio import Portfolio


class BacktestVisualizer:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self._setup_style()

    def _setup_style(self):
        """시각화 스타일 설정"""
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_theme()
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

    def generate_results(self, backtest_results: Dict) -> Dict:
        """백테스트 결과를 시각화하고 성과지표를 계산하여 반환"""
        # 성과지표 계산
        performance_metrics = self._calculate_performance_metrics(backtest_results)

        # 시각화 생성 및 인코딩
        visualization_data = self._generate_visualizations(backtest_results)

        return {"metrics": performance_metrics, "visualizations": visualization_data}

    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """모든 성과지표 계산"""
        # 개별 종목 통계
        stock_stats = self._calculate_stock_stats()

        # KOSPI 통계
        kospi_stats = self._calculate_kospi_statistics()

        # 포트폴리오 통계
        portfolio_stats = self._calculate_portfolio_statistics(results)

        # 포트폴리오 구성 정보
        portfolio_composition = [
            {
                "code": code,
                "name": self.portfolio.stock_info[
                    self.portfolio.stock_info["code"] == code
                ].iloc[0]["name"],
                "weight": weight * 100,
                "dividend_yield": float(
                    self.portfolio.stock_info[
                        self.portfolio.stock_info["code"] == code
                    ].iloc[0]["dividend_yield"]
                ),
            }
            for code, weight in zip(
                self.portfolio.stock_info["code"], self.portfolio.weights
            )
        ]

        # 추가 성과지표 계산
        returns = pd.Series(results["portfolio_values"]).pct_change().dropna()
        max_drawdown = self._calculate_max_drawdown(results["portfolio_values"])

        return {
            "period": {
                "start_date": self.portfolio.start_date,
                "end_date": self.portfolio.end_date,
            },
            "portfolio": {
                "composition": portfolio_composition,
                "final_value": float(results["portfolio_values"].iloc[-1]),
                "total_return": float(portfolio_stats["return"]),
                "annual_volatility": float(portfolio_stats["volatility"]),
                "sharpe_ratio": float(self._calculate_sharpe_ratio(returns)),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(len(returns[returns > 0]) / len(returns) * 100),
            },
            "benchmark": {
                "final_value": float(results["benchmark_values"].iloc[-1]),
                "total_return": float(kospi_stats["return"]),
                "annual_volatility": float(kospi_stats["volatility"]),
            },
            "individual_stocks": stock_stats.to_dict("records"),
        }

    def _generate_visualizations(self, results: Dict) -> Dict:
        """모든 시각화 생성 및 인코딩"""
        visualizations = {}

        # 1. 포트폴리오 가치 변화 그래프
        fig = plt.figure(figsize=(15, 8))
        self._plot_value_changes(fig, results)
        visualizations["value_changes"] = self._fig_to_base64(fig)
        plt.close(fig)

        # 2. 포트폴리오 구성 파이 차트
        fig = plt.figure(figsize=(10, 10))
        self._plot_portfolio_composition(fig)
        visualizations["composition"] = self._fig_to_base64(fig)
        plt.close(fig)

        # 3. 위험-수익 산점도
        fig = plt.figure(figsize=(12, 8))
        self._plot_risk_return(fig, results)
        visualizations["risk_return"] = self._fig_to_base64(fig)
        plt.close(fig)

        return visualizations

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """matplotlib 그림을 base64 문자열로 변환"""
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.03
    ) -> float:
        """샤프 비율 계산"""
        excess_returns = returns - risk_free_rate / 252  # 일별 무위험수익률
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """최대 낙폭 계산"""
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        return float(drawdown.min() * 100)

    def _calculate_stock_stats(self) -> pd.DataFrame:
        """개별 종목들의 수익률과 변동성 계산"""
        stats = []
        for code in self.portfolio.stock_prices.columns:
            returns = self.portfolio.stock_prices[code].pct_change().dropna()
            stats.append(
                {
                    "code": code,
                    "name": self.portfolio.stock_info[
                        self.portfolio.stock_info["code"] == code
                    ].iloc[0]["name"],
                    "return": returns.mean() * 252 * 100,
                    "volatility": returns.std() * 100,
                }
            )
        return pd.DataFrame(stats)

    def _calculate_kospi_statistics(self) -> Dict:
        """KOSPI 수익률과 변동성 계산"""
        kospi_returns = self.portfolio.benchmark.pct_change().dropna()
        return {
            "return": kospi_returns.mean() * 252 * 100,
            "volatility": kospi_returns.std() * 100,
        }

    def _calculate_portfolio_statistics(self, results: Dict) -> Dict:
        """포트폴리오 수익률과 변동성 계산"""
        portfolio_returns = pd.Series(results["portfolio_values"]).pct_change().dropna()
        return {
            "return": portfolio_returns.mean() * 252 * 100,
            "volatility": portfolio_returns.std() * 100,
        }

    def _plot_value_changes(self, fig, results: Dict):
        """가치 변화 그래프"""
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

        # 개별 종목들의 가치 변화
        for code in self.portfolio.stock_prices.columns:
            stock_returns = (
                self.portfolio.stock_prices[code]
                / self.portfolio.stock_prices[code].iloc[0]
            )
            stock_values = self.portfolio.initial_capital * stock_returns
            stock_info = self.portfolio.stock_info[
                self.portfolio.stock_info["code"] == code
            ].iloc[0]
            ax1.plot(
                stock_values, label=f"{stock_info['name']}", linestyle="--", alpha=0.5
            )

        # 포트폴리오와 KOSPI
        ax1.plot(
            results["portfolio_values"], label="Portfolio", linewidth=2.5, color="red"
        )
        ax1.plot(
            results["benchmark_values"],
            label="KOSPI",
            linewidth=2.5,
            color="black",
            alpha=0.7,
        )

        ax1.set_title("포트폴리오 가치 변화", fontsize=12, pad=15)
        ax1.set_ylabel("포트폴리오 가치 (원)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True)

    def _plot_portfolio_composition(self, fig):
        """포트폴리오 구성 파이 차트"""
        ax2 = plt.subplot2grid((2, 2), (1, 1))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.portfolio.weights)))

        wedges, texts, autotexts = ax2.pie(
            self.portfolio.weights,
            labels=self.portfolio.stock_info["name"],
            autopct="%1.1f%%",
            colors=colors,
        )
        ax2.set_title("포트폴리오 구성", fontsize=12, pad=15)

    def _plot_risk_return(self, fig, results: Dict):
        """위험-수익 산점도"""
        ax3 = plt.subplot2grid((2, 2), (1, 0))

        # 개별 종목 통계
        stock_stats = self._calculate_stock_stats()

        # KOSPI 통계
        kospi_stats = self._calculate_kospi_statistics()

        # 포트폴리오 통계
        portfolio_stats = self._calculate_portfolio_statistics(results)

        # 산점도 그리기
        ax3.scatter(
            stock_stats["volatility"],
            stock_stats["return"],
            label="개별 종목",
            alpha=0.6,
        )

        # 종목명 표시
        for idx, row in stock_stats.iterrows():
            ax3.annotate(
                row["name"],
                (row["volatility"], row["return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # KOSPI와 포트폴리오 점 추가
        ax3.scatter(
            kospi_stats["volatility"],
            kospi_stats["return"],
            label="KOSPI",
            marker="s",
            s=100,
            color="red",
        )

        ax3.scatter(
            portfolio_stats["volatility"],
            portfolio_stats["return"],
            label="포트폴리오",
            marker="^",
            s=100,
            color="green",
        )

        ax3.set_title("위험-수익 비교", fontsize=12, pad=15)
        ax3.set_xlabel("표준편차")
        ax3.set_ylabel("연간 수익률 (%)")
        ax3.legend()
        ax3.grid(True)
