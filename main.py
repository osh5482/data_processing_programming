import asyncio
import sys
from datetime import datetime
from typing import Optional, List

from config import Config
from data_loader import DataLoader, LoadedData
from stock_crawler import StockCrawler, StockData
from analyzer import StockAnalyzer, AnalysisResult
from database import Database


class StockAnalysisApp:
    """주식 분석 애플리케이션의 메인 클래스"""

    def __init__(self):
        """애플리케이션 초기화"""
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.crawler = StockCrawler(self.config)
        self.analyzer = StockAnalyzer(self.config)
        self.database = Database(self.config)

    async def run(self):
        """
        메인 실행 로직

        전체 프로세스:
        1. CSV 파일에서 배당주 데이터 로드
        2. 각 종목의 주가 데이터 수집
        3. 수집된 데이터 분석
        4. 분석 결과 데이터베이스 저장
        """
        try:
            self._print_header()

            # 1. 데이터 로드
            loaded_data = self._load_data()
            if loaded_data is None:
                return

            # 2. 주가 데이터 수집
            stock_data = await self._collect_stock_data(loaded_data)
            if not stock_data:
                return

            # 3. 데이터 분석
            analysis_results = self._analyze_data(stock_data)
            if not analysis_results:
                return

            # 4. 데이터베이스 저장
            self._save_results(analysis_results)

            self._print_footer()

        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
        finally:
            self.cleanup()

    def _print_header(self):
        """시작 헤더를 출력합니다."""
        print(f"\n{'='*50}")
        print(f"배당주 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")

    def _print_footer(self):
        """종료 푸터를 출력합니다."""
        print(f"\n{'='*50}")
        print(f"배당주 분석 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")

    def _load_data(self) -> Optional[LoadedData]:
        """
        CSV 파일에서 배당주 데이터를 로드합니다.

        Returns:
            Optional[LoadedData]: 로드된 데이터 또는 None
        """
        try:
            return self.data_loader.load_dividend_stocks()
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {str(e)}")
            return None

    async def _collect_stock_data(self, loaded_data: LoadedData) -> List[StockData]:
        """
        주가 데이터를 수집합니다.

        Parameters:
            loaded_data (LoadedData): 로드된 배당주 데이터

        Returns:
            List[StockData]: 수집된 주가 데이터
        """
        print("\n[1/3] 주가 데이터 수집 중...")
        stock_data = await self.crawler.fetch_all_stocks(loaded_data.dataframe)
        if not stock_data:
            print("주가 데이터 수집 실패")
            return []
        print(f"수집 완료: {len(stock_data)}개 종목\n")
        return stock_data

    def _analyze_data(self, stock_data: List[StockData]) -> List[AnalysisResult]:
        """
        수집된 주가 데이터를 분석합니다.

        Parameters:
            stock_data (List[StockData]): 수집된 주가 데이터

        Returns:
            List[AnalysisResult]: 분석 결과 리스트
        """
        print("[2/3] 데이터 분석 중...")
        analysis_results = []
        total = len(stock_data)

        for i, data in enumerate(stock_data, 1):
            sys.stdout.write(f"\r진행률: {i}/{total} ({(i/total)*100:.1f}%)")
            sys.stdout.flush()

            result = self.analyzer.calculate_metrics(data)
            if result:
                analysis_results.append(result)

        print(f"\n분석 완료: {len(analysis_results)}개 종목\n")
        return analysis_results

    def _save_results(self, analysis_results: List[AnalysisResult]):
        """
        분석 결과를 데이터베이스에 저장합니다.

        Parameters:
            analysis_results (List[AnalysisResult]): 저장할 분석 결과
        """
        print("[3/3] 분석 결과 저장 중...")
        try:
            # AnalysisResult 객체를 딕셔너리로 변환하고 값들을 문자열에서 float으로 변환
            data_to_save = []
            for result in analysis_results:
                data_dict = {
                    "code": result.code,
                    "name": result.name,
                    "annual_return": float(result.annual_return),
                    "volatility": float(result.volatility),
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "liquidity": float(result.liquidity),
                    "dividend_yield": float(result.dividend_yield),
                }
                data_to_save.append(data_dict)

            # 데이터베이스에 저장
            self.database.save_analysis(data_to_save)

        except Exception as e:
            print(f"결과 저장 중 오류 발생: {str(e)}")

    def cleanup(self):
        """리소스 정리"""
        try:
            self.database.close()
        except Exception as e:
            print(f"정리 중 오류 발생: {str(e)}")


async def main():
    """
    애플리케이션 시작점
    """
    app = StockAnalysisApp()
    await app.run()


if __name__ == "__main__":
    # Windows에서 실행할 때 필요한 설정
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 메인 함수 실행
    asyncio.run(main())
