from dataclasses import dataclass
from datetime import datetime, timedelta

import os

@dataclass
class FileConfig:
    """파일 관련 설정"""
    DATA_DIR: str = "data"
    INPUT_FILE: str = os.path.join("data", "dividend_stocks.csv")
    DB_PATH: str = os.path.join("data", "stock_data.db")

@dataclass
class AnalysisConfig:
    """분석 관련 설정"""
    TRADING_DAYS: int = 252  # 1년 거래일 수
    RISK_FREE_RATE: float = 0.035  # 무위험 수익률 (3.5% 가정)
    
    # 기간 설정
    END_DATE: datetime = datetime.now()
    START_DATE: datetime = END_DATE - timedelta(days=365)
    
    # 데이터 형식
    DATE_FORMAT: str = "%Y-%m-%d"

class Config:
    """전체 설정을 관리하는 클래스"""
    
    def __init__(self):
        self.file = FileConfig()
        self.analysis = AnalysisConfig()