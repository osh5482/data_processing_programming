from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
import pandas as pd
from typing import Optional, List
from datetime import datetime

from config import Config
from database import Base, StockAnalysis

class DBViewer:
    """SQLAlchemy를 사용한 데이터베이스 조회 클래스"""
    
    def __init__(self, config: Config):
        """
        Parameters:
            config (Config): 설정 객체
        """
        self.config = config
        self.engine = None
        self.Session = None
    
    def connect(self):
        """데이터베이스 연결을 생성합니다."""
        try:
            self.engine = create_engine(f'sqlite:///{self.config.file.DB_PATH}')
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            print("데이터베이스 연결 성공")
        except Exception as e:
            print(f"데이터베이스 연결 실패: {str(e)}")
    
    def close(self):
        """데이터베이스 연결을 종료합니다."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None

    def get_all_stocks(self) -> Optional[pd.DataFrame]:
        """
        모든 종목의 분석 데이터를 조회합니다.
        
        Returns:
            Optional[pd.DataFrame]: 전체 종목 데이터 또는 None
        """
        if not self.Session:
            self.connect()
            
        session = None
        try:
            session = self.Session()
            stocks = session.query(StockAnalysis).order_by(desc(StockAnalysis.dividend_yield)).all()
            
            data = []
            for stock in stocks:
                data.append({
                    '종목코드': stock.code,
                    '종목명': stock.name,
                    '연간수익률(%)': round(stock.annual_return * 100, 2),
                    '변동성(%)': round(stock.volatility * 100, 2),
                    '샤프비율': round(stock.sharpe_ratio, 2),
                    '일평균거래대금(백만원)': round(stock.liquidity / 1_000_000, 0),
                    '배당수익률(%)': round(stock.dividend_yield, 2)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"데이터 조회 실패: {str(e)}")
            return None
        finally:
            if session:
                session.close()

    def get_top_dividend_stocks(self, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        배당수익률 상위 종목을 조회합니다.
        
        Parameters:
            limit (int): 조회할 종목 수
            
        Returns:
            Optional[pd.DataFrame]: 상위 종목 데이터 또는 None
        """
        if not self.Session:
            self.connect()
            
        try:
            session = self.Session()
            stocks = session.query(StockAnalysis).order_by(
                desc(StockAnalysis.dividend_yield)
            ).limit(limit).all()
            
            data = []
            for stock in stocks:
                data.append({
                    '종목코드': stock.code,
                    '종목명': stock.name,
                    '배당수익률(%)': round(stock.dividend_yield, 2),
                    '연간수익률(%)': round(stock.annual_return * 100, 2),
                    '변동성(%)': round(stock.volatility * 100, 2),
                    '샤프비율': round(stock.sharpe_ratio, 2)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"데이터 조회 실패: {str(e)}")
            return None
        finally:
            session.close()

    def get_stock_by_conditions(self, 
                              min_dividend: float = 0, 
                              max_volatility: float = 100,
                              min_liquidity: float = 0) -> Optional[pd.DataFrame]:
        """
        조건에 맞는 종목을 조회합니다.
        
        Parameters:
            min_dividend (float): 최소 배당수익률
            max_volatility (float): 최대 변동성
            min_liquidity (float): 최소 일평균거래대금(백만원)
            
        Returns:
            Optional[pd.DataFrame]: 조건에 맞는 종목 데이터 또는 None
        """
        if not self.Session:
            self.connect()
            
        try:
            session = self.Session()
            stocks = session.query(StockAnalysis).filter(
                StockAnalysis.dividend_yield >= min_dividend,
                StockAnalysis.volatility * 100 <= max_volatility,
                StockAnalysis.liquidity >= min_liquidity * 1_000_000
            ).order_by(desc(StockAnalysis.dividend_yield)).all()
            
            data = []
            for stock in stocks:
                data.append({
                    '종목코드': stock.code,
                    '종목명': stock.name,
                    '배당수익률(%)': round(stock.dividend_yield, 2),
                    '연간수익률(%)': round(stock.annual_return * 100, 2),
                    '변동성(%)': round(stock.volatility * 100, 2),
                    '샤프비율': round(stock.sharpe_ratio, 2),
                    '일평균거래대금(백만원)': round(stock.liquidity / 1_000_000, 0)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"데이터 조회 실패: {str(e)}")
            return None
        finally:
            session.close()

def main():
    """
    DB 조회 예시
    """
    config = Config()
    viewer = DBViewer(config)
    
    try:
        # 전체 종목 조회
        print("\n=== 전체 종목 데이터 ===")
        all_stocks = viewer.get_all_stocks()
        if all_stocks is not None:
            print(f"총 {len(all_stocks)}개 종목")
            print(all_stocks.head())
        
        # 배당수익률 상위 종목 조회
        print("\n=== 배당수익률 상위 10종목 ===")
        top_dividend = viewer.get_top_dividend_stocks(10)
        if top_dividend is not None:
            print(top_dividend)
        
        # 조건별 종목 조회
        print("\n=== 조건별 종목 조회 ===")
        print("조건: 배당수익률 3% 이상, 변동성 20% 이하, 일평균거래대금 10억원 이상")
        condition_stocks = viewer.get_stock_by_conditions(
            min_dividend=3.0,
            max_volatility=20.0,
            min_liquidity=1000
        )
        if condition_stocks is not None:
            print(f"조건에 맞는 종목 수: {len(condition_stocks)}")
            print(condition_stocks)
            
    finally:
        viewer.close()

if __name__ == "__main__":
    main()