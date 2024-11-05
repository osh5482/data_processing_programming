import pandas as pd
import numpy as np
from datetime import datetime
import FinanceDataReader as fdr
from pykrx import stock
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import sqlite3
from typing import List, Tuple, Dict
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from config import Config


class DividendPortfolioBacktest:
    def __init__(
        self,
        initial_capital: float = 100_000_000,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
    ):
        self.config = Config()
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.benchmark = None
        self.stock_prices = None  # 주가 데이터를 저장할 변수 추가
        self._initialize_db_connection()

    def _initialize_db_connection(self):
        """데이터베이스 연결 초기화 및 기본 데이터 로드"""
        with sqlite3.connect(self.config.file.DB_PATH) as conn:
            query = """
            SELECT 
                code,
                name,
                annual_return,
                volatility,
                dividend_yield,
                liquidity
            FROM stock_analysis
            ORDER BY annual_return DESC
            """
            self.stock_data = pd.read_sql(query, conn)

    def get_valid_stock_data(self, n_stocks: int = 5) -> pd.DataFrame:
        """필터링된 유효 주식 데이터와 KOSPI 데이터 반환"""
        if self.stock_data is None:
            raise ValueError("데이터베이스 연결이 필요합니다")

        print("\n주가 데이터 수집 중...")
        valid_stocks = {}

        # 상위 n_stocks개 종목만 처리
        for _, stock in self.stock_data.head(n_stocks).iterrows():
            try:
                df = fdr.DataReader(stock["code"], self.start_date, self.end_date)

                if not df["Close"].isna().any():
                    valid_stocks[stock["code"]] = df["Close"]
                    print(
                        f"[성공] {stock['code']} ({stock['name']}): "
                        f"연간수익률 {stock['annual_return']*100:.1f}%"
                    )
                else:
                    print(f"[제외] {stock['code']} ({stock['name']}): NaN 값 존재")

            except Exception as e:
                print(
                    f"[제외] {stock['code']} ({stock['name']}): "
                    f"데이터 수집 실패 - {str(e)}"
                )

        # 최종 선택된 종목들의 정보 저장
        self.top_dividend_stocks = self.stock_data[
            self.stock_data["code"].isin(valid_stocks.keys())
        ].copy()

        # KOSPI 데이터 수집
        print("\nKOSPI 데이터 수집 중...")
        self.benchmark = self._fetch_kospi_data()
        if self.benchmark is None:
            raise ValueError("KOSPI 데이터 수집 실패")
        print("KOSPI 데이터 수집 완료")

        # 주가 데이터프레임 저장
        self.stock_prices = pd.DataFrame(valid_stocks)

        return self.stock_prices

    def calculate_individual_statistics(self) -> pd.DataFrame:
        """개별 종목들의 수익률과 변동성 계산"""
        stats = []
        for code in self.stock_prices.columns:
            returns = self.stock_prices[code].pct_change().dropna()
            annual_return = returns.mean() * 252 * 100  # 연간 수익률 (%)
            volatility = returns.std() * 100  # 수익률의 표준편차 (%)
            stock_info = self.top_dividend_stocks[
                self.top_dividend_stocks["code"] == code
            ].iloc[0]

            stats.append(
                {
                    "code": code,
                    "name": stock_info["name"],
                    "return": annual_return,
                    "volatility": volatility,
                }
            )

        return pd.DataFrame(stats)

    def _portfolio_statistics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """포트폴리오 수익률과 변동성, 샤프비율 계산"""
        # 일간 수익률 계산
        returns = self.stock_prices.pct_change().dropna()

        # 연간 수익률과 변동성 계산
        port_return = np.sum(returns.mean() * weights) * 252
        port_vol = (
            np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * 100
        )  # 수익률의 표준편차

        # 샤프비율 계산 (무위험수익률 2% 가정)
        rf_rate = 0.02
        sharpe_ratio = (port_return - rf_rate) / (port_vol / 100) if port_vol > 0 else 0

        return port_return, port_vol, sharpe_ratio

    def _minimize_volatility(self, weights: np.ndarray) -> float:
        """최적화 목적함수: 변동성 최소화 (제약조건: 일정 수준 이상의 수익률)"""
        port_return, port_vol, _ = self._portfolio_statistics(weights)

        # 페널티: 수익률이 목표치(5%)보다 낮으면 패널티 부과
        target_return = 0.05
        penalty = max(0, target_return - port_return) * 100

        return port_vol + penalty

    def optimize_portfolio(self) -> np.ndarray:
        """최소 변동성 포트폴리오 최적화"""
        if self.stock_prices is None:
            raise ValueError("먼저 get_stock_data를 실행하세요.")

        # 결측치 처리된 수익률 계산
        returns = self.stock_prices.pct_change().fillna(0)

        n_assets = len(self.stock_prices.columns)
        if n_assets < 1:
            raise ValueError("최소 1개 이상의 종목이 필요합니다.")

        # 제약조건 설정
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # 비중 합 = 1
        ]

        # 비중 제한 설정 (각 종목 최소 5%, 최대 40%)
        bounds = tuple((0.05, 0.40) for _ in range(n_assets))

        # 여러 초기값으로 최적화 시도
        best_result = None
        lowest_volatility = np.inf

        for _ in range(20):  # 20번 시도
            try:
                # 랜덤 초기 비중 (합이 1이 되도록)
                initial_weights = np.random.random(n_assets)
                initial_weights /= initial_weights.sum()

                # 최적화 실행
                result = minimize(
                    self._minimize_volatility,
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )

                if result.success:
                    port_vol = self._portfolio_statistics(result.x)[1]
                    if port_vol < lowest_volatility:
                        lowest_volatility = port_vol
                        best_result = result

            except Exception as e:
                print(f"최적화 시도 실패: {str(e)}")
                continue

        if best_result is None:
            print("최적화 실패. 동일 비중으로 설정합니다.")
            self.optimal_weights = np.array([1 / n_assets] * n_assets)
        else:
            self.optimal_weights = best_result.x

        # 최적화 결과 출력
        port_return, port_vol, sharpe = self._portfolio_statistics(self.optimal_weights)
        print("\n=== 포트폴리오 최적화 결과 ===")

        # 각 종목별 비중 출력
        for code, weight in zip(self.stock_prices.columns, self.optimal_weights):
            stock_info = self.top_dividend_stocks[
                self.top_dividend_stocks["code"] == code
            ].iloc[0]
            print(f"{code} ({stock_info['name']}): {weight*100:.1f}%")

        print(f"\n예상 연간 수익률: {port_return*100:.1f}%")
        print(f"예상 연간 변동성: {port_vol*100:.1f}%")
        print(f"샤프 비율: {sharpe:.2f}")

        return self.optimal_weights

    def run_backtest(self) -> Dict:
        """백테스트 실행"""
        if self.optimal_weights is None:
            raise ValueError("먼저 optimize_portfolio를 실행하세요.")

        # 포트폴리오 가치 계산
        portfolio_value = self.initial_capital
        portfolio_values = []

        returns = self.stock_prices.pct_change()
        for date in returns.index:
            daily_return = np.sum(returns.loc[date] * self.optimal_weights)
            portfolio_value *= 1 + daily_return
            portfolio_values.append(portfolio_value)

        portfolio_values = pd.Series(portfolio_values, index=returns.index)

        # 벤치마크(KOSPI) 수익률 계산
        benchmark_values = self.initial_capital * (
            self.benchmark / self.benchmark.iloc[0]
        )

        # 성과 지표 계산
        portfolio_return = (
            (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        )
        benchmark_return = (
            (benchmark_values[-1] - self.initial_capital) / self.initial_capital * 100
        )

        return {
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
        }

    def plot_results(self, backtest_results: Dict):
        """백테스트 결과 시각화"""
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_theme()

        # 한글 폰트 설정
        plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 기본 한글 폰트
        plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

        fig = plt.figure(figsize=(15, 10))

        # 1. 포트폴리오 가치 변화 (좌상단)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

        # 개별 종목들의 가치 변화 추가
        for code in self.stock_prices.columns:
            stock_returns = self.stock_prices[code] / self.stock_prices[code].iloc[0]
            stock_values = self.initial_capital * stock_returns
            stock_info = self.top_dividend_stocks[
                self.top_dividend_stocks["code"] == code
            ].iloc[0]
            ax1.plot(
                stock_values, label=f"{stock_info['name']}", linestyle="--", alpha=0.5
            )

        # 포트폴리오와 KOSPI 그리기
        ax1.plot(
            backtest_results["portfolio_values"],
            label="Portfolio",
            linewidth=2.5,
            color="red",
        )
        ax1.plot(
            backtest_results["benchmark_values"],
            label="KOSPI",
            linewidth=2.5,
            color="black",
            alpha=0.7,
        )

        ax1.set_title("Portfolio Value Over Time", fontsize=12, pad=15)
        ax1.set_ylabel("Portfolio Value (KRW)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True)

        # 2. 포트폴리오 구성 (우하단)
        ax2 = plt.subplot2grid((2, 2), (1, 1))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.optimal_weights)))
        wedges, texts, autotexts = ax2.pie(
            self.optimal_weights,
            labels=self.top_dividend_stocks["name"],
            autopct="%1.1f%%",
            colors=colors,
        )
        ax2.set_title("Portfolio Composition", fontsize=12, pad=15)

        # 3. 위험-수익 산점도 (좌하단)
        ax3 = plt.subplot2grid((2, 2), (1, 0))

        # 개별 종목들의 통계치 계산
        individual_stats = self.calculate_individual_statistics()

        # KOSPI 수익률과 변동성 계산
        kospi_returns = self.benchmark.pct_change().dropna()
        kospi_return = kospi_returns.mean() * 252 * 100
        kospi_vol = kospi_returns.std() * 100  # 수익률의 표준편차

        # 포트폴리오 수익률과 변동성
        portfolio_returns = backtest_results["portfolio_values"].pct_change().dropna()
        portfolio_return = portfolio_returns.mean() * 252 * 100
        portfolio_vol = portfolio_returns.std() * 100  # 수익률의 표준편차

        # 산점도 그리기
        # 개별 종목
        ax3.scatter(
            individual_stats["volatility"],
            individual_stats["return"],
            label="Individual Stocks",
            alpha=0.6,
        )

        # 종목명 표시
        for idx, row in individual_stats.iterrows():
            ax3.annotate(
                row["name"],
                (row["volatility"], row["return"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # KOSPI
        ax3.scatter(
            kospi_vol, kospi_return, label="KOSPI", marker="s", s=100, color="red"
        )

        # 포트폴리오
        ax3.scatter(
            portfolio_vol,
            portfolio_return,
            label="Portfolio",
            marker="^",
            s=100,
            color="green",
        )

        ax3.set_title("Risk-Return Comparison", fontsize=12, pad=15)
        ax3.set_xlabel("Standard Deviation (%)")  # 변동성을 표준편차로 변경
        ax3.set_ylabel("Annual Return (%)")
        ax3.legend()
        ax3.grid(True)

        # 레이아웃 조정
        plt.tight_layout()
        plt.show()

        # 텍스트 결과 출력
        print("\n=== 백테스트 결과 ===")
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print("\n1. 포트폴리오 구성")
        for code, name, weight in zip(
            self.top_dividend_stocks["code"],
            self.top_dividend_stocks["name"],
            self.optimal_weights,
        ):
            print(f"{code} ({name}): {weight*100:.1f}%")

        print("\n2. 수익률 비교")
        print(f"포트폴리오 수익률: {portfolio_return:.1f}%")
        print(f"포트폴리오 표준편차: {portfolio_vol:.1f}%")  # 변동성을 표준편차로 변경
        print(f"KOSPI 수익률: {kospi_return:.1f}%")
        print(f"KOSPI 표준편차: {kospi_vol:.1f}%")  # 변동성을 표준편차로 변경


def run_backtest(
    initial_capital: float = 100_000_000,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    n_stocks: int = 5,
):
    """백테스트 실행 함수"""

    # 백테스트 인스턴스 생성
    backtest = DividendPortfolioBacktest(
        initial_capital=initial_capital, start_date=start_date, end_date=end_date
    )

    # NaN이 없는 주가 데이터 수집 (상위 n_stocks개 종목)
    backtest.get_valid_stock_data(n_stocks=n_stocks)

    # 포트폴리오 최적화
    backtest.optimize_portfolio()

    # 백테스트 실행
    results = backtest.run_backtest()

    # 결과 시각화
    backtest.plot_results(results)


if __name__ == "__main__":
    run_backtest()
