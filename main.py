import streamlit as st
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")

# 데이터 로드 및 전처리
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 열 이름 공백 제거 및 날짜 파싱
    df.columns = df.columns.str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 2025년까지의 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    # 연도별 관측일 수 및 평균기온 계산
    yearly = df.groupby('연도')['평균기온'].agg(
        count='count',
        mean_temp='mean'
    ).reset_index()
    
    # 관측일 수가 300일 이상인 해만 필터링
    yearly_valid = yearly[yearly['count'] >= 300].copy()
    yearly_valid = yearly_valid.sort_values('연도').reset_index(drop=True)
    
    return yearly_valid

# 데이터 읽기
df_yearly = load_and_process_data()

# 메타데이터 계산
num_years = len(df_yearly)
start_year = int(df_yearly['연도'].min())
end_year = int(df_yearly['연도'].max())

# 회귀 분석 (독립 변수: 1908년부터 지난 연수)
df_yearly['X'] = df_yearly['연도'] - 1908
X = df_yearly['X'].tolist()
y = df_yearly['mean_temp'].tolist()

# 순수 파손(파이썬 기본 연산) 최소제곱법 선형 회귀 계산
n = len(X)
sum_x = sum(X)
sum_y = sum(y)
sum_x2 = sum(x ** 2 for x in X)
sum_xy = sum(x * y_i for x, y_i in zip(X, y))

# 기울기(slope) 및 절편(intercept)
slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
intercept = (sum_y - slope * sum_x) / n

# 상관계수(r) 계산
mean_x = sum_x / n
mean_y = sum_y / n
var_x = sum((x - mean_x) ** 2 for x in X)
var_y = sum((y_i - mean_y) ** 2 for y_i in y)
cov_xy = sum((x - mean_x) * (y_i - mean_y) for x, y_i in zip(X, y))
corr_coef = cov_xy / ((var_x * var_y) ** 0.5)

# 1. 요약 정보 출력
st.subheader("📌 데이터 및 분석 개요")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{corr_coef:.4f}")

st.markdown("---")

# 2. 연도 선택 슬라이더 및 예측값 강조 표시
st.subheader("🔮 연도별 기온 예측")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 1908년부터 지난 연수를 기준으로 기온 예측
elapsed_years = selected_year - 1908
pred_temp = slope * elapsed_years + intercept

st.markdown(
    f"""
    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 12px; text-align: center; margin: 20px 0;">
        <span style="font-size: 1.2rem; color: #555; font-weight: bold;">{selected_year}년 예상 서울 연평균 기온</span>
        <h1 style="color: #ff4b4b; font-size: 3.2rem; margin: 10px 0 0 0;">{pred_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 3. Streamlit 내장 차트로 데이터 및 회귀선 시각화
st.subheader("📈 연도별 평균기온 및 회귀 예측선")

# 1900~2100년 범위의 회귀선 데이터 생성
years_range = list(range(1900, 2101))
chart_df = pd.DataFrame({
    '연도': years_range,
    '회귀 예측선': [slope * (yr - 1908) + intercept for yr in years_range]
})

# 실제 관측 데이터 결합
chart_df = chart_df.merge(df_yearly[['연도', 'mean_temp']], on='연도', how='left')
chart_df.rename(columns={'mean_temp': '실제 연평균기온'}, inplace=True)

# 시각화를 위한 인덱스 설정
chart_df.set_index('연도', inplace=True)

# 라인 차트 표시 (회귀선 및 실제 기온 표출)
st.line_chart(chart_df)
