from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
try:
    import FinanceDataReader as fdr
except ImportError:
    print("FinanceDataReader 모듈이 설치되어 있지 않습니다.")
    print("pip install finance-datareader 명령어로 설치해주세요.")
import asyncio
from typing import List, Optional
from datetime import datetime
from tqdm import tqdm
from config import Config

@dataclass
class StockData:
    """주식 데이터를 저장하는 데이터 클래스"""
    code: str
    name: str
    prices: List[float]
    trading_values: List[float]
    dividend_yield: float = 0.0

class StockCrawler:
    """주식 데이터 수집을 위한 크롤러 클래스"""
    
    def __init__(self, config: Config):
        """
        Parameters:
            config (Config): 설정 객체
        """
        self.config = config
        
    async def fetch_stock_data(self, code: str, name: str) -> Optional[StockData]:
        """
        단일 종목의 주가 데이터를 수집합니다.
        
        Parameters:
            code (str): 종목 코드
            name (str): 종목명
            
        Returns:
            Optional[StockData]: 수집된 주가 데이터
        """
        try:
            # 비동기 실행을 위해 이벤트 루프에서 실행
            df = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._fetch_stock_history,
                code
            )
            
            if df.empty:
                print(f"데이터 없음 - {code}: {name}")
                return None
                
            return StockData(
                code=code,
                name=name,
                prices=df['Close'].tolist(),
                trading_values=(df['Close'] * df['Volume']).tolist(),
                dividend_yield=0.0  # CSV에서 나중에 설정
            )
                    
        except Exception as e:
            print(f"예외 발생 - {code}: {str(e)}")
            return None
            
    def _fetch_stock_history(self, code: str) -> pd.DataFrame:
        """
        종목의 과거 데이터를 조회합니다.
        
        Parameters:
            code (str): 종목 코드
            
        Returns:
            pd.DataFrame: 주가 데이터
        """
        return fdr.DataReader(
            code,
            self.config.analysis.START_DATE.strftime(self.config.analysis.DATE_FORMAT),
            self.config.analysis.END_DATE.strftime(self.config.analysis.DATE_FORMAT)
        )
    
    @staticmethod
    def _format_stock_code(code: str) -> str:
        """
        종목코드를 표준 형식으로 변환합니다.
        
        Parameters:
            code (str): 원본 종목코드
            
        Returns:
            str: 변환된 종목코드
        """
        if str(code).isdigit():
            return str(code).zfill(6)
        return str(code).strip()[-6:]  # 우선주의 경우 마지막 6자리만
    
    async def fetch_all_stocks(self, stocks_df: pd.DataFrame) -> List[StockData]:
        """
        모든 종목의 주가 데이터를 수집합니다.
        
        Parameters:
            stocks_df (pd.DataFrame): 수집할 종목 목록
            
        Returns:
            List[StockData]: 수집된 모든 주가 데이터
        """
        # 종목코드 형식 처리
        stocks_df['종목코드'] = stocks_df['종목코드'].apply(self._format_stock_code)
        
        # 데이터 수집 태스크 생성
        tasks = [
            self.fetch_stock_data(row['종목코드'], row['종목명'])
            for _, row in stocks_df.iterrows()
        ]
        
        # 프로그레스바로 진행상황 표시
        results = []
        for task in tqdm(
            asyncio.as_completed(tasks), 
            total=len(tasks), 
            desc="주가 데이터 수집",
            ncols=100
        ):
            stock_data = await task
            if stock_data:
                # CSV에서 읽은 배당수익률 추가
                stock_info = stocks_df[stocks_df['종목코드'] == stock_data.code].iloc[0]
                stock_data.dividend_yield = float(stock_info['배당수익률'])
                results.append(stock_data)
        
        print(f"\n성공적으로 수집한 종목 수: {len(results)}/{len(tasks)}")
        return results