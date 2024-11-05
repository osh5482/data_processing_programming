import os
import requests as rq
import re
from io import BytesIO
from bs4 import BeautifulSoup
import pandas as pd

def get_business_day():
    """네이버 금융에서 최근 영업일을 가져옵니다."""
    try:
        url = "https://finance.naver.com/sise/sise_deposit.naver"
        data = rq.get(url)
        data_html = BeautifulSoup(data.content, 'html.parser')
        
        parse_day = data_html.select_one(
            "div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah"
        ).text
        
        biz_day = re.findall("[0-9]+", parse_day)
        return "".join(biz_day)
    except Exception as e:
        print(f"영업일 조회 실패: {str(e)}")
        return None

def download_krx_data():
    """KRX에서 전체 주식 데이터를 다운로드합니다."""
    try:
        biz_day = get_business_day()
        if not biz_day:
            return None
            
        # KRX OTP 생성
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
        
        # 데이터 다운로드
        otp = rq.post(gen_otp_url, gen_otp_ksq, headers=headers).text
        down_url = "http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
        krx_data = rq.post(down_url, {"code": otp}, headers=headers)
        
        # DataFrame으로 변환
        df = pd.read_csv(BytesIO(krx_data.content), encoding="EUC-KR")
        df["종목명"] = df["종목명"].str.strip()
        df["기준일"] = biz_day
        
        return df
        
    except Exception as e:
        print(f"KRX 데이터 다운로드 실패: {str(e)}")
        return None

def filter_dividend_stocks(df):
    """배당주만 필터링하여 정렬합니다."""
    try:
        # 필요한 컬럼만 선택
        stocks_df = df[["종목코드", "종목명", "배당수익률", "주당배당금", "기준일"]]
        
        # 배당주만 필터링하고 정렬
        dividend_stocks = stocks_df[stocks_df["배당수익률"] > 0]
        dividend_stocks = dividend_stocks.sort_values(by=["배당수익률"], ascending=False)
        
        return dividend_stocks
        
    except Exception as e:
        print(f"배당주 필터링 실패: {str(e)}")
        return None

def update_dividend_stocks():
    """배당주 데이터를 갱신합니다."""
    print("\n=== 배당주 데이터 갱신 시작 ===")
    
    # data 디렉토리 확인 및 생성
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"'{data_dir}' 디렉토리 생성됨")
    
    print("1. KRX 데이터 다운로드 중...")
    krx_data = download_krx_data()
    if krx_data is None:
        print("데이터 갱신 실패")
        return False
        
    print("2. 배당주 필터링 중...")
    dividend_stocks = filter_dividend_stocks(krx_data)
    if dividend_stocks is None:
        print("데이터 갱신 실패")
        return False
    
    print("3. 파일 저장 중...")
    try:
        # KRX 원본 데이터 저장
        krx_file_path = os.path.join(data_dir, "krx_data.csv")
        krx_data.to_csv(krx_file_path, index=False, encoding="utf-8-sig")
        
        # 필터링된 배당주 데이터 저장
        dividend_file_path = os.path.join(data_dir, "dividend_stocks.csv")
        dividend_stocks.to_csv(
            dividend_file_path,
            index=False,
            encoding="utf-8-sig"
        )
        print(f"데이터 갱신 완료: {len(dividend_stocks)}개 종목\n")
        return True
        
    except Exception as e:
        print(f"파일 저장 실패: {str(e)}")
        return False

if __name__ == "__main__":
    update_dividend_stocks()