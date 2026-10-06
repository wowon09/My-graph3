import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.markdown("서울의 과거 기온 데이터를 바탕으로 선형 회귀 모델을 생성하고 미래 기온을 예측합니다.")

# 데이터 불러오기 함수 (캐싱 처리)
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 데이터 파싱 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'].str.strip())
    df['연도'] = df['날짜'].dt.year
    
    # 평균기온 결측치 제거
    df_clean = df.dropna(subset=['평균기온'])
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = df_clean.groupby('연도').agg(
        count=('평균기온', 'count'),
        mean_temp=('평균기온', 'mean')
    ).reset_index()
    
    # 조건 필터링: 2025년 이하 & 관측일수 300일 이상
    valid_years = yearly_stats[(yearly_stats['연도'] <= 2025) & (yearly_stats['count'] >= 300)].copy()
    
    # 독립변수 X: 1908년부터 경과한 연수 (연도 - 1908)
    valid_years['years_since_1908'] = valid_years['연도'] - 1908
    
    return valid_years

# 데이터 로드
data = load_and_process_data()

# 회귀 모델 학습 (scikit-learn)
X = data[['years_since_1908']]
y = data['mean_temp']

model = LinearRegression()
model.fit(X, y)

# 상관계수 계산 (Pandas 사용)
corr = data['연도'].corr(data['mean_temp'])

# 메트릭 정보 계산
num_years = len(data)
start_year = int(data['연도'].min())
end_year = int(data['연도'].max())

# 요약 정보 출력
st.subheader("📊 학습 데이터 및 모델 정보")
col1, col2, col3, col4 = st.columns(4)
col1.metric("학습 데이터 수", f"{num_years}개 연도")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("피어슨 상관계수", f"{corr:.3f}")

st.divider()

# 연도 선택 슬라이더 및 예측값 표시
st.subheader("🔮 연도별 예상 기온 예측")
selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요 (1900년 ~ 2100년)",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택 연도의 예측값 산출
selected_x = pd.DataFrame({'years_since_1908': [selected_year - 1908]})
predicted_temp = model.predict(selected_x)[0]

# 크게 강조하여 표시
st.markdown(
    f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
        <h3 style="margin: 0; color: #333;">{selected_year}년 예상 서울 연평균 기온</h3>
        <h1 style="margin: 10px 0 0 0; color: #ff4b4b; font-size: 3rem;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("📈 연도별 평균기온 추이 및 회귀 직선")

# 1900년부터 2100년까지의 전체 데이터프레임 생성
all_years = list(range(1900, 2101))
df_chart = pd.DataFrame({'연도': all_years})
df_chart['years_since_1908'] = df_chart['연도'] - 1908

# 전체 기간 회귀 예측값 계산
df_chart['회귀 직선'] = model.predict(df_chart[['years_since_1908']])

# 실제 관측 데이터 병합
df_chart = df_chart.merge(data[['연도', 'mean_temp']], on='연도', how='left')
df_chart.rename(columns={'mean_temp': '실제 연평균 기온'}, inplace=True)

# 차트용 데이터 정리 및 연도 인덱스 설정
chart_data = df_chart.set_index('연도')[['실제 연평균 기온', '회귀 직선']]

# Streamlit 기본 차트 출력
st.line_chart(chart_data)
