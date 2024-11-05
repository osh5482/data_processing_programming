import pandas as pd
import numpy as np
from datetime import datetime
import FinanceDataReader as fdr
from pykrx import stock
import sqlite3
from typing import Tuple, Optional
from .portfolio import Portfolio


class DataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def load_stock_data(
        self,
        portfolio: Portfolio,
        n_stocks: int = 5,
        min_dividend: float = 2.0,
        min_liquidity: float = 500,
        max_volatility: float = 40.0,
    ) -> None:
        """주식 데이터 로드"""
        # DB에서 기본 데이터 로드
        stock_data = self._load_db_data(
            limit=n_stocks,
            min_dividend=min_dividend,
            min_liquidity=min_liquidity,
            max_volatility=max_volatility,
        )

        # 주가 데이터 수집
        valid_stocks = {}
        for _, row in stock_data.iterrows():
            try:
                df = self._fetch_stock_price(row["code"], portfolio)

                if df is not None and not df.empty and not df["Close"].isna().any():
                    valid_stocks[row["code"]] = df["Close"]
                    print(f"[성공] {row['code']} ({row['name']})")
                    print(f"       배당수익률: {row['dividend_yield']:.1f}%")
                    print(
                        f"       일평균거래대금: {row['liquidity']/1_000_000:.0f}백만원"
                    )
                    print(f"       변동성: {row['volatility']:.1f}%")
                else:
                    print(f"[제외] {row['code']} ({row['name']}): 유효하지 않은 데이터")

            except Exception as e:
                print(f"[제외] {row['code']} ({row['name']}): {str(e)}")

        if not valid_stocks:
            raise ValueError("유효한 주가 데이터가 없습니다")

        # 데이터 저장
        portfolio.stock_prices = pd.DataFrame(valid_stocks)
        portfolio.stock_info = stock_data[stock_data["code"].isin(valid_stocks.keys())]
        portfolio.benchmark = self._fetch_benchmark_data(portfolio)

    def _load_db_data(
        self,
        limit: int,
        min_dividend: float,
        min_liquidity: float,
        max_volatility: float,
    ) -> pd.DataFrame:
        """DB에서 조건을 만족하는 종목 로드"""
        query = """
        SELECT 
            code, 
            name, 
            annual_return, 
            volatility, 
            dividend_yield, 
            liquidity
        FROM stock_analysis
        WHERE dividend_yield >= ?
            AND liquidity >= ?
            AND volatility <= ?
        ORDER BY annual_return DESC
        LIMIT ?
        """

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                query,
                conn,
                params=(
                    min_dividend,
                    min_liquidity * 1_000_000,  # 백만원 → 원
                    max_volatility,
                    limit,
                ),
            )

        if df.empty:
            raise ValueError(
                f"조건을 만족하는 종목이 없습니다.\n"
                f"- 최소 배당수익률: {min_dividend}%\n"
                f"- 최소 거래대금: {min_liquidity}백만원\n"
                f"- 최대 변동성: {max_volatility}%"
            )

        print(f"\n=== 조회된 종목 ({len(df)}개) ===")
        for _, row in df.iterrows():
            print(f"{row['code']} ({row['name']}):")
            print(f"  배당수익률: {row['dividend_yield']:.1f}%")
            print(f"  일평균거래대금: {row['liquidity']/1_000_000:.0f}백만원")
            print(f"  변동성: {row['volatility']:.1f}%")

        return df

    def _fetch_stock_price(
        self, code: str, portfolio: Portfolio
    ) -> Optional[pd.DataFrame]:
        """개별 종목 주가 데이터 수집"""
        try:
            formatted_code = str(code).zfill(6)
            df = fdr.DataReader(
                formatted_code, portfolio.start_date, portfolio.end_date
            )
            return df
        except Exception as e:
            print(f"주가 데이터 수집 실패 ({code}): {str(e)}")
            return None

    def _fetch_benchmark_data(self, portfolio: Portfolio) -> pd.Series:
        """벤치마크(KOSPI) 데이터 수집"""
        start = datetime.strptime(portfolio.start_date, "%Y-%m-%d")
        end = datetime.strptime(portfolio.end_date, "%Y-%m-%d")

        kospi = stock.get_index_ohlcv_by_date(
            start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), "1001"
        )

        if kospi.empty:
            raise ValueError("벤치마크 데이터를 찾을 수 없습니다")

        return kospi["종가"]
