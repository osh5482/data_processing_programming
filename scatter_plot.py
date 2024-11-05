import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import matplotlib.font_manager as fm
import numpy as np


def plot_risk_return_scatter():
    """배당주들의 위험-수익 산점도를 그립니다."""

    # 데이터 로드
    db_path = os.path.join("data", "stock_data.db")
    engine = create_engine(f"sqlite:///{db_path}")

    query = """
    SELECT code, name, annual_return, volatility, dividend_yield
    FROM stock_analysis
    WHERE dividend_yield > 0
    """

    stocks_df = pd.read_sql(query, engine)

    # 수익률과 변동성을 퍼센트로 변환
    stocks_df["annual_return"] = stocks_df["annual_return"] * 100
    stocks_df["volatility"] = stocks_df["volatility"] * 100

    # 한글 폰트 설정
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # FHD 해상도에 맞는 그래프 생성 (16:9 비율)
    plt.figure(figsize=(16, 9), dpi=120)

    # 산점도 그리기 - 극명한 대비를 위해 'RdYlBu_r' 컬러맵 사용
    scatter = plt.scatter(
        stocks_df["volatility"],
        stocks_df["annual_return"],
        c=stocks_df["dividend_yield"],
        cmap="RdYlBu_r",  # Red(높음) - Yellow - Blue(낮음) 컬러맵
        s=stocks_df["dividend_yield"] ** 2,  # 점 크기 증가
        alpha=0.7,
        vmin=stocks_df["dividend_yield"].min(),
        vmax=stocks_df["dividend_yield"].max(),
    )

    # 컬러바 추가 - 세로로 긴 형태로 변경
    cbar = plt.colorbar(scatter, label="배당수익률 (%)", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # 축 레이블과 제목
    plt.xlabel("변동성 (%)", fontsize=12)
    plt.ylabel("연간 수익률 (%)", fontsize=12)
    plt.title("배당주 위험-수익 분포", fontsize=14, pad=20)

    # 격자 추가
    plt.grid(True, linestyle="--", alpha=0.3)

    # 그래프에 종목명 표시 (높은 배당수익률 순으로 상위 10개만)
    top_10 = stocks_df.nlargest(10, "dividend_yield")
    for idx, row in top_10.iterrows():
        plt.annotate(
            f"{row['name']}\n({row['dividend_yield']:.1f}%)",  # 종목명과 배당수익률을 두 줄로 표시
            (row["volatility"], row["annual_return"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.7
            ),  # 텍스트 배경 추가
            ha="left",
            va="bottom",
        )

    # 축 범위 설정 - 여유 공간 추가
    x_margin = (stocks_df["volatility"].max() - stocks_df["volatility"].min()) * 0.1
    y_margin = (
        stocks_df["annual_return"].max() - stocks_df["annual_return"].min()
    ) * 0.1

    plt.xlim(
        stocks_df["volatility"].min() - x_margin,
        stocks_df["volatility"].max() + x_margin,
    )
    plt.ylim(
        stocks_df["annual_return"].min() - y_margin,
        stocks_df["annual_return"].max() + y_margin,
    )

    # 여백 조정
    plt.tight_layout()

    # 그래프 저장 (FHD 해상도)
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(
        os.path.join(data_dir, "risk_return_scatter.png"),
        dpi=120,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close()

    # 통계 정보 출력
    print("\n=== 배당주 위험-수익 분석 ===")
    print(f"총 종목 수: {len(stocks_df)}")
    print("\n상위 10개 배당주:")

    # 보기 좋게 포맷팅하여 출력
    summary_df = top_10[
        ["name", "dividend_yield", "annual_return", "volatility"]
    ].copy()
    summary_df.columns = ["종목명", "배당수익률(%)", "연간수익률(%)", "변동성(%)"]
    print(summary_df.to_string(float_format=lambda x: "{:.2f}".format(x)))


if __name__ == "__main__":
    plot_risk_return_scatter()
