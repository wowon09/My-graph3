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

    # 연도별 관측일수(count) 및 평균기온(mean) 집계
    all_years = (
        df.groupby("연도")["평균기온"]
        .agg(count="count", mean="mean")
        .reset_index()
    )
    all_years = all_years[all_years["연도"] <= 2025]

    return df, all_years


try:
    df_raw, all_years = load_and_process_data()

    # 관측일 수가 300일 이상인 해만 추출하여 연평균 데이터프레임 생성
    df_annual = all_years[all_years["count"] >= 300].rename(
        columns={"mean": "평균기온"}
    )
    df_annual["경과연수"] = df_annual["연도"] - 1908

    start_year = int(df_annual["연도"].min())
    end_year = int(df_annual["연도"].max())
    total_count = len(df_annual)

    # 2. 선형 회귀 파라미터 계산 (최소제곱법 - 순수 파라미터 계산)
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

    st.divider()

    # =========================================================
    # 도전 — 직선 대신 곡선을 쓰면 (numpy 완전 제거 버전)
    # =========================================================
    st.subheader("도전 — 직선 대신 곡선을 쓰면")

    연평균 = all_years[all_years["count"] >= 300].rename(
        columns={"mean": "기온"}
    )  # 앞에서 만든 연평균 표
    학습 = 연평균[연평균["연도"] < 2005]  # 2005년 이전은 훈련용
    평가 = 연평균[연평균["연도"] >= 2005]  # 그 뒤는 한 번도 안 본 테스트용
    x = (
        lambda y: (y - 1950) / 100
    )  # 연도를 작은 수로 바꿔야 고차 곡선이 안정된다


    # numpy 대신 순수 파이썬 다항식 피팅(최소제곱법 및 가우스 소거법) 함수
    def fit_poly(x_list, y_list, degree):
        n_dim = degree + 1
        # 정규 방정식 (Normal Equation) 행렬 생성
        M = [
            [
                sum(xv ** (2 * degree - i - j) for xv in x_list)
                for j in range(n_dim)
            ]
            for i in range(n_dim)
        ]
        V = [
            sum((xv ** (degree - i)) * yv for xv, yv in zip(x_list, y_list))
            for i in range(n_dim)
        ]

        # 가우스 소거법 (Gaussian Elimination)
        for i in range(n_dim):
            max_row = max(range(i, n_dim), key=lambda r: abs(M[r][i]))
            M[i], M[max_row] = M[max_row], M[i]
            V[i], V[max_row] = V[max_row], V[i]

            pivot = M[i][i]
            for j in range(i, n_dim):
                M[i][j] /= pivot
            V[i] /= pivot

            for k in range(n_dim):
                if k != i:
                    factor = M[k][i]
                    for j in range(i, n_dim):
                        M[k][j] -= factor * M[i][j]
                    V[k] -= factor * V[i]
        return V


    # 다항식 계산 함수
    def eval_poly(coeffs, x_val):
        res = 0
        for coeff in coeffs:
            res = res * x_val + coeff
        return res


    x_train = [x(y) for y in 학습["연도"]]
    y_train = list(학습["기온"])
    x_test = [x(y) for y in 평가["연도"]]
    y_test = list(평가["기온"])

    rows = []
    for 차수 in [1, 3, 9]:
        계수 = fit_poly(x_train, y_train, 차수)

        # 평가 데이터 예측 오차(MAE) 계산
        preds = [eval_poly(계수, xv) for xv in x_test]
        평가오차 = sum(abs(p - yt) for p, yt in zip(preds, y_test)) / len(
            y_test
        )

        # 2050년 예측값 계산
        pred_2050 = eval_poly(계수, x(2050))

        rows.append(
            {
                "곡선": f"{차수}차",
                "테스트 오차(℃)": round(평가오차, 2),
                "2050년 예측(
