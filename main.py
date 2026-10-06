import pandas as pd
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write(
    "서울 기온 데이터를 활용하여 경과 연수에 따른 선형 회귀 모델을 생성하고 기온을 예측합니다."
)

# 1. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_and_process_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜 처리 및 평균기온 숫자 변환
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 관측일 수가 300일 이상이고 2025년 이하 데이터만 필터링
    valid_years = []
    for year, group in df.groupby("연도"):
        if year <= 2025 and group["평균기온"].dropna().count() >= 300:
            valid_years.append(year)

    df_filtered = df[df["연도"].isin(valid_years)]

    # 연도별 평균기온 계산
    annual_df = df_filtered.groupby("연도")["평균기온"].mean().reset_index()

    # 1908년부터 지난 연수 (독립변수 X)
    annual_df["경과연수"] = annual_df["연도"] - 1908

    return annual_df


try:
    df_annual = load_and_process_data()

    start_year = int(df_annual["연도"].min())
    end_year = int(df_annual["연도"].max())
    total_count = len(df_annual)

    # 2. 선형 회귀 파라미터 계산 (최소제곱법)
    X = df_annual["경과연수"]
    Y = df_annual["평균기온"]

    n = len(X)
    sum_x = X.sum()
    sum_y = Y.sum()
    sum_x2 = (X**2).sum()
    sum_xy = (X * Y).sum()

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - (sum_x**2))
    intercept = (sum_y - slope * sum_x) / n

    df_annual["회귀선"] = slope * df_annual["경과연수"] + intercept

    # 상관계수 계산
    corr = df_annual["경과연수"].corr(df_annual["평균기온"])

    # 3. 데이터 요약 출력
    col1, col2, col3 = st.columns(3)
    col1.metric("학습 데이터 연도 수", f"{total_count}개 해")
    col2.metric("시작 연도 ~ 끝 연도", f"{start_year}년 ~ {end_year}년")
    col3.metric("연수-기온 상관계수", f"{corr:.4f}")

    st.divider()

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

    # 예측 결과 크게 표시 (Streamlit 순수 metric 카드 사용)
    st.metric(
        label=f"{target_year}년 서울 예상 평균기온 (1908년 기준 경과연수: {passed_years}년)",
        value=f"{predicted_temp:.2f} °C",
    )

    st.divider()

    # 5. 차트 시각화
    st.subheader("📈 서울 연평균기온 및 회귀 직선")

    chart_data = df_annual.set_index("연도")[["평균기온", "회귀선"]]
    chart_data.columns = ["실제 관측 기온(°C)", "회귀 직선(°C)"]

    st.line_chart(chart_data)

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
