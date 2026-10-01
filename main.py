import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️️", layout="wide")

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
X = df_yearly['X'].values
y = df_yearly['mean_temp'].values

# 최소제곱법을 이용한 선형 회귀 계수 구하기 (기울기, 절편)
slope, intercept = np.polyfit(X, y, 1)

# 상관계수 계산
corr_coef = np.corrcoef(df_yearly['연도'], y)[0, 1]

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
    unsafe_allow_dict_style=True
)

# 3. Plotly 산점도 및 회귀 직선 시각화
years_range = np.arange(1900, 2101)
trend_y = slope * (years_range - 1908) + intercept

fig = go.Figure()

# 관측값 산점도
fig.add_trace(go.Scatter(
    x=df_yearly['연도'],
    y=df_yearly['mean_temp'],
    mode='markers',
    name='실제 연평균기온',
    marker=dict(color='#1f77b4', size=7, opacity=0.8)
))

# 회귀 직선
fig.add_trace(go.Scatter(
    x=years_range,
    y=trend_y,
    mode='lines',
    name='회귀 직선 (예측선)',
    line=dict(color='#ff7f0e', width=2.5, dash='dash')
))

# 선택한 연도 강조 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[pred_temp],
    mode='markers+text',
    name='선택 연도 예측치',
    text=[f"{pred_temp:.2f}°C"],
    textposition="top center",
    marker=dict(color='#d62728', size=13, symbol='star')
))

fig.update_layout(
    title="서울 연도별 평균기온 분포 및 회귀 예측선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)
