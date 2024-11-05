import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
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

    def plot_results(self, results: Dict):
        """백테스트 결과 시각화"""
        fig = plt.figure(figsize=(15, 10))

        # 1. 포트폴리오 가치 변화 (좌상단)
        self._plot_value_changes(fig, results)

        # 2. 포트폴리오 구성 (우하단)
        self._plot_portfolio_composition(fig)

        # 3. 위험-수익 산점도 (좌하단)
        self._plot_risk_return(fig, results)

        plt.tight_layout()
        plt.show()

        # 텍스트 결과 출력
        self._print_results(results)

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

        ax1.set_title("포트폴리오 가치 변화", fontsize=12, pad=15)  # 한글로 변경
        ax1.set_ylabel("포트폴리오 가치 (원)")  # 한글로 변경
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
        ax2.set_title("포트폴리오 구성", fontsize=12, pad=15)  # 한글로 변경

    def _plot_risk_return(self, fig, results: Dict):
        """위험-수익 산점도"""
        ax3 = plt.subplot2grid((2, 2), (1, 0))

        # 개별 종목 통계
        stock_stats = self._calculate_individual_statistics()

        # KOSPI 통계
        kospi_stats = self._calculate_kospi_statistics()

        # 포트폴리오 통계
        portfolio_stats = self._calculate_portfolio_statistics(results)

        # 산점도 그리기
        ax3.scatter(
            stock_stats["volatility"],
            stock_stats["return"],
            label="개별 종목",  # 한글로 변경
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
            label="포트폴리오",  # 한글로 변경
            marker="^",
            s=100,
            color="green",
        )

        ax3.set_title("위험-수익 비교", fontsize=12, pad=15)  # 한글로 변경
        ax3.set_xlabel("표준편차")  # 한글로 변경
        ax3.set_ylabel("연간 수익률 (%)")  # 한글로 변경
        ax3.legend()
        ax3.grid(True)

    def _calculate_individual_statistics(self) -> pd.DataFrame:
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
        portfolio_returns = results["portfolio_values"].pct_change().dropna()
        return {
            "return": portfolio_returns.mean() * 252 * 100,
            "volatility": portfolio_returns.std() * 100,
        }

    def _print_results(self, results: Dict):
        """백테스트 결과 출력"""
        print("\n=== 백테스트 결과 ===")
        print(f"기간: {self.portfolio.start_date} ~ {self.portfolio.end_date}")

        print("\n1. 포트폴리오 구성")
        for code, name, weight in zip(
            self.portfolio.stock_info["code"],
            self.portfolio.stock_info["name"],
            self.portfolio.weights,
        ):
            print(f"{code} ({name}): {weight*100:.1f}%")

        # 통계 계산
        portfolio_stats = self._calculate_portfolio_statistics(results)
        kospi_stats = self._calculate_kospi_statistics()

        print("\n2. 수익률 비교")
        print(f"포트폴리오 수익률: {portfolio_stats['return']:.1f}%")
        print(f"포트폴리오 표준편차: {portfolio_stats['volatility']:.1f}%")
        print(f"KOSPI 수익률: {kospi_stats['return']:.1f}%")
        print(f"KOSPI 표준편차: {kospi_stats['volatility']:.1f}%")
