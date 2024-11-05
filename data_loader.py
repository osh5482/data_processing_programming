from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Optional
import requests as rq
import re
from io import BytesIO
from bs4 import BeautifulSoup
from config import Config

@dataclass
class LoadedData:
    """로드된 데이터를 저장하는 데이터 클래스"""
    dataframe: pd.DataFrame
    count: int
    columns: list[str]

class DataLoader:
    """주식 데이터 로딩을 처리하는 클래스"""
    
    def __init__(self, config: Config):
        """
        Parameters:
            config (Config): 설정 객체
        """
        self.config = config
        
    def download_krx_data(self) -> bool:
        """
        KRX에서 전체 주식 데이터를 다운로드합니다.
        
        Returns:
            bool: 다운로드 성공 여부
        """
        try:
            # 영업일 구하기
            url = "https://finance.naver.com/sise/sise_deposit.naver"
            data = rq.get(url)
            data_html = BeautifulSoup(data.content, 'html.parser')
            
            parse_day = data_html.select_one(
                "div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah"
            ).text
            
            biz_day = re.findall("[0-9]+", parse_day)
            biz_day = "".join(biz_day)
            
            # KRX 데이터 다운로드
            gen_otp_url = "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
            gen_otp_ksq = {
                "searchType": "1",
                "mktId": "ALL",
                "trdDd": biz_day,
                "csvxls_isNo": "false",
                "name": "fileDown",
                "url": "dbms/MDC/STAT/standard/MDCSTAT03501",
            }
            
            headers = {
                "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
            }
            
            otp = rq.post(gen_otp_url, gen_otp_ksq, headers=headers).text
            down_url = "http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
            krx_ind = rq.post(down_url, {"code": otp}, headers=headers)
            
            krx_ind = pd.read_csv(BytesIO(krx_ind.content), encoding="EUC-KR")
            krx_ind["종목명"] = krx_ind["종목명"].str.strip()
            krx_ind["기준일"] = biz_day
            
            krx_ind.to_csv("krx_data.csv", encoding="utf-8-sig")
            print("KRX 데이터 다운로드 완료")
            return True
            
        except Exception as e:
            print(f"KRX 데이터 다운로드 실패: {str(e)}")
            return False
    
    def filter_dividend_stocks(self) -> bool:
        """
        다운로드된 KRX 데이터에서 배당주만 필터링합니다.
        
        Returns:
            bool: 필터링 성공 여부
        """
        try:
            stocks_df = pd.read_csv("krx_data.csv")
            stocks_df = stocks_df[["종목코드", "종목명", "배당수익률", "주당배당금", "기준일"]]
            
            dividend_stocks_df = stocks_df[stocks_df["배당수익률"] > 0]
            dividend_stocks_df = dividend_stocks_df.sort_values(by=["배당수익률"], ascending=False)
            
            dividend_stocks_df.to_csv(
                self.config.file.INPUT_FILE, 
                index=False, 
                encoding="utf-8-sig"
            )
            print("배당주 필터링 완료")
            return True
            
        except Exception as e:
            print(f"배당주 필터링 실패: {str(e)}")
            return False
    
    def update_dividend_stocks(self) -> bool:
        """
        KRX 데이터를 다운로드하고 배당주를 필터링하여 데이터를 갱신합니다.
        
        Returns:
            bool: 갱신 성공 여부
        """
        if self.download_krx_data():
            return self.filter_dividend_stocks()
        return False
    
    def load_dividend_stocks(self) -> Optional[LoadedData]:
        """
        배당주 목록이 있는 CSV 파일을 로드합니다.
        
        Returns:
            Optional[LoadedData]: 로드된 데이터 또는 에러 시 None
        """
        try:
            df = pd.read_csv(self.config.file.INPUT_FILE, encoding='utf-8')
            
            loaded_data = LoadedData(
                dataframe=df,
                count=len(df),
                columns=df.columns.tolist()
            )
            
            print(f"데이터 로드 완료: {loaded_data.count}개 종목")
            return loaded_data
            
        except Exception as e:
            print(f"데이터 로드 실패: {str(e)}")
            return None