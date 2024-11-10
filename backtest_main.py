from typing import Tuple, Optional, Dict
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
) -> Dict:
    """백테스트 실행 함수

    Returns:
        Dict: 백테스트 결과를 담은 딕셔너리
        {
            "metrics": {
                "period": {
                    "start_date": str,
                    "end_date": str
                },
                "portfolio": {
                    "composition": List[Dict],  # 포트폴리오 구성 종목 정보
                    "final_value": float,       # 최종 포트폴리오 가치
                    "total_return": float,      # 총 수익률
                    "annual_volatility": float, # 연간 변동성
                    "sharpe_ratio": float,      # 샤프 비율
                    "max_drawdown": float,      # 최대 낙폭
                    "win_rate": float          # 승률
                },
                "benchmark": {
                    "final_value": float,
                    "total_return": float,
                    "annual_volatility": float
                },
                "individual_stocks": List[Dict]  # 개별 종목 성과
            },
            "visualizations": {
                "value_changes": str,     # Base64 인코딩된 포트폴리오 가치 변화 그래프
                "composition": str,       # Base64 인코딩된 포트폴리오 구성 파이 차트
                "risk_return": str        # Base64 인코딩된 위험-수익 산점도
            }
        }
    """
    try:
        # 1. 포트폴리오 초기화
        portfolio = Portfolio(
            initial_capital=initial_capital, start_date=start_date, end_date=end_date
        )

        # 2. 데이터 로드
        data_loader = DataLoader(db_path)
        data_loader.load_stock_data(
            portfolio=portfolio,
            n_stocks=n_stocks,
            min_dividend=min_dividend,
            min_liquidity=min_liquidity,
            max_volatility=max_volatility,
        )

        # 3. 포트폴리오 최적화
        optimizer = PortfolioOptimizer(
            min_weight=min_weight,
            max_weight=max_weight,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
        )
        optimizer.optimize(portfolio)

        # 4. 백테스트 실행
        engine = BacktestEngine(portfolio)
        backtest_results = engine.run()

        # 5. 결과 생성
        visualizer = BacktestVisualizer(portfolio)
        results = visualizer.generate_results(backtest_results)

        return results

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return {"error": str(e), "metrics": None, "visualizations": None}


if __name__ == "__main__":
    import json
    import base64
    from PIL import Image
    import io
    import os
    from datetime import datetime
    import webbrowser

    # 테스트를 위한 실행
    results = run_backtest(
        n_stocks=5,
        min_dividend=3.0,
    )

    # 결과 확인
    if results.get("metrics"):
        print("\n=== 백테스트 결과 ===")
        print(
            f"포트폴리오 총 수익률: {results['metrics']['portfolio']['total_return']:.2f}%"
        )
        print(
            f"벤치마크 총 수익률: {results['metrics']['benchmark']['total_return']:.2f}%"
        )
        print(f"샤프 비율: {results['metrics']['portfolio']['sharpe_ratio']:.2f}")
        print(f"최대 낙폭: {results['metrics']['portfolio']['max_drawdown']:.2f}%")

        print("\n포트폴리오 구성:")
        for stock in results["metrics"]["portfolio"]["composition"]:
            print(
                f"{stock['name']}: {stock['weight']:.1f}% (배당수익률: {stock['dividend_yield']:.1f}%)"
            )

        print("\n시각화 데이터 포함 여부:")
        for key in results["visualizations"].keys():
            print(f"- {key}: {'성공' if results['visualizations'][key] else '실패'}")

        # print("\n=== 전체 메트릭스 상세 정보 ===")
        # print(json.dumps(results, indent=2, ensure_ascii=False))

        # # HTML 파일 생성
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # img_dir = "backtest_results"
        # if not os.path.exists(img_dir):
        #     os.makedirs(img_dir)

        # html_content = f"""
        # <!DOCTYPE html>
        # <html>
        # <head>
        #     <title>백테스트 결과 - {timestamp}</title>
        #     <style>
        #         body {{
        #             font-family: Arial, sans-serif;
        #             max-width: 1200px;
        #             margin: 0 auto;
        #             padding: 20px;
        #         }}
        #         .metrics {{
        #             background-color: #f5f5f5;
        #             padding: 20px;
        #             border-radius: 5px;
        #             margin-bottom: 20px;
        #         }}
        #         .visualization {{
        #             margin-bottom: 30px;
        #             text-align: center;
        #         }}
        #         .visualization img {{
        #             max-width: 100%;
        #             height: auto;
        #             border: 1px solid #ddd;
        #             border-radius: 5px;
        #         }}
        #         h2 {{
        #             color: #333;
        #             border-bottom: 2px solid #ddd;
        #             padding-bottom: 10px;
        #         }}
        #         table {{
        #             width: 100%;
        #             border-collapse: collapse;
        #             margin: 10px 0;
        #         }}
        #         th, td {{
        #             padding: 8px;
        #             text-align: left;
        #             border-bottom: 1px solid #ddd;
        #         }}
        #         th {{
        #             background-color: #f0f0f0;
        #         }}
        #     </style>
        # </head>
        # <body>
        #     <h1>백테스트 결과 ({timestamp})</h1>

        #     <div class="metrics">
        #         <h2>포트폴리오 성과</h2>
        #         <table>
        #             <tr>
        #                 <th>지표</th>
        #                 <th>값</th>
        #             </tr>
        #             <tr>
        #                 <td>포트폴리오 총 수익률</td>
        #                 <td>{results['metrics']['portfolio']['total_return']:.2f}%</td>
        #             </tr>
        #             <tr>
        #                 <td>벤치마크 총 수익률</td>
        #                 <td>{results['metrics']['benchmark']['total_return']:.2f}%</td>
        #             </tr>
        #             <tr>
        #                 <td>샤프 비율</td>
        #                 <td>{results['metrics']['portfolio']['sharpe_ratio']:.2f}</td>
        #             </tr>
        #             <tr>
        #                 <td>최대 낙폭</td>
        #                 <td>{results['metrics']['portfolio']['max_drawdown']:.2f}%</td>
        #             </tr>
        #         </table>

        #         <h2>포트폴리오 구성</h2>
        #         <table>
        #             <tr>
        #                 <th>종목명</th>
        #                 <th>비중</th>
        #                 <th>배당수익률</th>
        #             </tr>
        #             {"".join([f'''
        #             <tr>
        #                 <td>{stock['name']}</td>
        #                 <td>{stock['weight']:.1f}%</td>
        #                 <td>{stock['dividend_yield']:.1f}%</td>
        #             </tr>
        #             ''' for stock in results['metrics']['portfolio']['composition']])}
        #         </table>
        #     </div>

        #     <h2>포트폴리오 가치 변화</h2>
        #     <div class="visualization">
        #         <img src="data:image/png;base64,{results['visualizations']['value_changes']}" alt="Portfolio Value Changes">
        #     </div>

        #     <h2>포트폴리오 구성 비율</h2>
        #     <div class="visualization">
        #         <img src="data:image/png;base64,{results['visualizations']['composition']}" alt="Portfolio Composition">
        #     </div>

        #     <h2>위험-수익 분석</h2>
        #     <div class="visualization">
        #         <img src="data:image/png;base64,{results['visualizations']['risk_return']}" alt="Risk-Return Analysis">
        #     </div>
        # </body>
        # </html>
        # """

        # html_path = os.path.join(img_dir, f"backtest_report_{timestamp}.html")
        # with open(html_path, "w", encoding="utf-8") as f:
        #     f.write(html_content)

        # print(f"\n리포트가 생성되었습니다: {html_path}")
        # # 웹 브라우저로 HTML 파일 열기
        # webbrowser.open(f"file://{os.path.abspath(html_path)}")
