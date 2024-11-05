from typing import Tuple, Optional
from backtest.portfolio import Portfolio
from backtest.data_loader import DataLoader
from backtest.optimizer import PortfolioOptimizer
from backtest.backtest_engine import BacktestEngine
from backtest.visualizer import BacktestVisualizer


def run_backtest(
    # 기본 설정
    initial_capital: float = 100_000_000,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    # 종목 선정 조건
    n_stocks: int = 5,
    min_dividend: float = 2.0,  # 최소 배당수익률
    min_liquidity: float = 100,  # 최소 일평균 거래대금 (백만원)
    max_volatility: float = float("inf"),  # 최대 변동성
    # 포트폴리오 최적화 조건
    min_weight: float = 0.05,  # 최소 투자 비중
    max_weight: float = 0.90,  # 최대 투자 비중
    target_return: float = 0.05,  # 목표 수익률
    risk_free_rate: float = 0.03,  # 무위험 수익률
    # 기타 설정
    db_path: str = "data/stock_data.db",
) -> Tuple[Portfolio, dict]:
    """백테스트 실행 함수

    Args:
        initial_capital: 초기 투자금액 (원)
        start_date: 백테스트 시작일 (YYYY-MM-DD)
        end_date: 백테스트 종료일 (YYYY-MM-DD)

        n_stocks: 포트폴리오에 포함할 종목 수
        min_dividend: 최소 배당수익률 (%)
        min_liquidity: 최소 일평균 거래대금 (백만원)
        max_volatility: 최대 허용 변동성 (%)

        min_weight: 종목당 최소 투자 비중 (0.05 = 5%)
        max_weight: 종목당 최대 투자 비중 (0.40 = 40%)
        target_return: 목표 수익률 (0.05 = 5%)
        risk_free_rate: 무위험 수익률 (0.02 = 2%)

        db_path: 데이터베이스 파일 경로
    """
    try:
        # 1. 포트폴리오 초기화
        portfolio = Portfolio(
            initial_capital=initial_capital, start_date=start_date, end_date=end_date
        )

        # 2. 데이터 로드
        print("\n1. 데이터 로드 중...")
        data_loader = DataLoader(db_path)
        data_loader.load_stock_data(
            portfolio=portfolio,
            n_stocks=n_stocks,
            min_dividend=min_dividend,
            min_liquidity=min_liquidity,
            max_volatility=max_volatility,
        )

        # 3. 포트폴리오 최적화
        print("\n2. 포트폴리오 최적화 중...")
        optimizer = PortfolioOptimizer(
            min_weight=min_weight,
            max_weight=max_weight,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
        )
        optimizer.optimize(portfolio)

        # 4. 백테스트 실행
        print("\n3. 백테스트 실행 중...")
        engine = BacktestEngine(portfolio)
        results = engine.run()

        # 5. 결과 시각화
        print("\n4. 결과 시각화...")
        visualizer = BacktestVisualizer(portfolio)
        visualizer.plot_results(results)

        return portfolio, results

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return None, None


if __name__ == "__main__":
    run_backtest(
        n_stocks=5,
        min_dividend=2.0,
    )
