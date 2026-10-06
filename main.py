import pandas as pd
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 데이터를 활용하여 경과 연수에 따른 선형 회귀 모델을 생성하고 기온을 예측합니다.")

# 1. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_and_process_data():
    # CSV 파일 읽기
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜에서 연도 추출 및 기온 숫자형 변환
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 관측일 수가 300일 이상이고, 2025년 이하 데이터만 필터링
    valid_years = []
    for year, group in df.groupby("연도"):
        if year <= 2025 and group["평균기온"].dropna().count() >= 300:
            valid_years.append(year)

    df_filtered = df[df["연도"].isin(valid_years)]

    # 연도별 평균기온 집계
    annual_df = (
        df_filtered.groupby("연도")["평균기온"].mean().reset_index()
    )

    # 1908년부터 지난 연수 (독립변수 X)
    annual_df["경과연수"] = annual_df["연도"] - 1908

    return annual_df


try:
    df_annual = load_and_process_data()

    start_year = int(df_annual["연도"].min())
    end_year = int(df_annual["연도"].max())
    total_count = len(df_annual)

    # 2. 선형 회귀 파라미터 직접 계산 (최소제곱법, sklearn 미사용)
    X = df_annual["경과연수"]
    Y = df_annual["평균기온"]

    n = len(X)
    sum_x = X.sum()
    sum_y = Y.sum()
    sum_x2 = (X**2).sum()
    sum_xy = (X * Y).sum()

    # 기울기(slope) 및 절편(intercept)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - (sum_x**2))
    intercept = (sum_y - slope * sum_x) / n

    # 회귀선 예측값 추가
    df_annual["회귀선"] = slope * df_annual["경과연수"] + intercept

    # 피어슨 상관계수 계산
    corr = df_annual["경과연수"].corr(df_annual["평균기온"])

    # 3. 화면 정보 출력
    col1, col2, col3 = st.columns(3)
    col1.metric("학습 데이터 연도 수", f"{total_count}개 해")
    col2.metric("시작 연도 ~ 끝 연도", f"{start_year}년 ~ {end_year}년")
    col3.metric("연수-기온 상관계수", f"{corr:.4f}")

    st.markdown("---")

    # 4. 연도 선택 슬라이더 및 기온 예측
    st.subheader("🔮 예상 기온 예측하기")
    target_year = st.slider(
        "예측하고 싶은 연도를 선택하세요",
        min_value=1900,
        max_value=2100,
        value=2025,
        step=1,
    )

    passed_years = target_year - 1908
    predicted_temp = slope * passed_years + intercept

    st.markdown(
        f"""
    <div style="background-color:#f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="margin:0; color:#333;">{target_year}년 서울 예상 평균기온</h3>
        <h1 style="color:#ff4b4b; font-size: 50px; margin: 10px 0;">{predicted_temp:.2f} °C</h1>
        <p style="color:#666; margin:0;">(1908년 기준 경과 연수: {passed_years}년
