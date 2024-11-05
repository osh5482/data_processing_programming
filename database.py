from __future__ import annotations
from sqlalchemy import create_engine, Column, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from config import Config

Base = declarative_base()

class StockAnalysis(Base):
    """주식 분석 결과를 저장하는 테이블 모델"""
    __tablename__ = 'stock_analysis'

    # 기본 정보
    code = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    
    # 수익성 지표
    annual_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    liquidity = Column(Float)
    dividend_yield = Column(Float)

class Database:
    """SQLAlchemy를 사용한 데이터베이스 처리 클래스"""
    
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
            # config.file.DB_PATH 사용
            self.engine = create_engine(f'sqlite:///{self.config.file.DB_PATH}')
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            print("데이터베이스 연결 성공")
        except Exception as e:
            print(f"데이터베이스 연결 실패: {str(e)}")
    
    def save_analysis(self, analyzed_data: List[Dict[str, Any]]):
        """
        분석된 데이터를 데이터베이스에 저장합니다.
        
        Parameters:
            analyzed_data (List[Dict[str, Any]]): 저장할 분석 데이터
        """
        if not self.Session:
            self.connect()
            
        try:
            session = self.Session()
            
            # 기존 데이터 삭제
            session.query(StockAnalysis).delete()
            
            # 새로운 데이터 추가
            stocks = []
            for data in analyzed_data:
                stock = StockAnalysis(
                    code=data['code'],
                    name=data['name'],
                    annual_return=float(data['annual_return']),
                    volatility=float(data['volatility']),
                    sharpe_ratio=float(data['sharpe_ratio']),
                    liquidity=float(data['liquidity']),
                    dividend_yield=float(data['dividend_yield'])
                )
                stocks.append(stock)
            
            # 벌크 insert 수행
            session.bulk_save_objects(stocks)
            session.commit()
            
            print(f"데이터 저장 완료: {len(analyzed_data)}개 종목")
            
        except Exception as e:
            if session:
                session.rollback()
            print(f"데이터 저장 실패: {str(e)}")
            raise  # 에러를 상위로 전파하여 디버깅 용이하게 함
        finally:
            if session:
                session.close()
    
    def close(self):
        """데이터베이스 연결을 종료합니다."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None

    def get_total_count(self) -> int:
        """
        저장된 총 종목 수를 반환합니다.
        
        Returns:
            int: 총 종목 수
        """
        if not self.Session:
            self.connect()
            
        session = self.Session()
        try:
            return session.query(StockAnalysis).count()
        finally:
            session.close()