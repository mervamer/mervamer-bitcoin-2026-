import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# 페이지 설정
st.set_page_config(page_title="비트코인 가격 분석 대시보드", layout="wide")

@st.cache_data
def load_data():
    # 데이터 로드 (업로드된 파일의 세미콜론 구분자 반영)
    try:
        df = pd.read_csv('coin.csv', sep=';')
        
        # 시간 관련 컬럼 데이터 타입 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 분석을 위해 날짜순 정렬
        df = df.sort_values('timestamp')
        return df
    except FileNotFoundError:
        st.error("파일을 찾을 수 없습니다. 'coin.csv' 파일이 동일한 경로에 있는지 확인해주세요.")
        return None

try:
    df = load_data()

    if df is not None:
        # 사이드바 설정
        st.sidebar.header("📊 대시보드 설정")
        
        # 날짜 범위 선택기
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "조회 기간 선택",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # 데이터 필터링
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            filtered_df = df.loc[mask].copy()
        else:
            filtered_df = df.copy()
            start_date, end_date = min_date, max_date

        # 메인 타이틀
        st.title("📈 비트코인(BTC) 데이터 분석 및 예측 대시보드")
        st.markdown(f"**{start_date}** 부터 **{end_date}** 까지의 데이터를 기반으로 합니다.")

        # 상단 주요 지표
        if not filtered_df.empty:
            latest_data = filtered_df.iloc[-1]
            prev_data = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            price_diff = latest_data['close'] - prev_data['close']
            price_diff_pct = (price_diff / prev_data['close']) * 100

            col1.metric("현재 종가 (Close)", f"{latest_data['close']:,.0f} KRW", f"{price_diff_pct:.2f}%")
            col2.metric("기간 내 최고가", f"{filtered_df['high'].max():,.0f} KRW")
            col3.metric("평균 거래량", f"{filtered_df['volume'].mean():,.0f}")
            col4.metric("시가총액 (Market Cap)", f"{latest_data['marketCap']:,.0e}")

            # 차트 섹션
            st.divider()
            
            st.subheader("BTC 가격 상세 추이 (캔들스틱)")
            fig_candle = go.Figure(data=[go.Candlestick(x=filtered_df['timestamp'],
                            open=filtered_df['open'],
                            high=filtered_df['high'],
                            low=filtered_df['low'],
                            close=filtered_df['close'],
                            name='BTC')])
            fig_candle.update_layout(xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig_candle, use_container_width=True)

            # --- 내일 가격 예측 섹션 (선형 회귀 모델) ---
            st.divider()
            st.subheader("🔮 내일 가격 예측 (Linear Regression)")
            
            # 머신러닝 데이터 준비 (날짜를 숫자로 변환)
            df_ml = filtered_df.copy()
            df_ml['date_ordinal'] = df_ml['timestamp'].apply(lambda x: x.toordinal())
            
            X = df_ml[['date_ordinal']].values
            y = df_ml['close'].values
            
            # 모델 훈련
            model = LinearRegression()
            model.fit(X, y)
            
            # 내일 날짜 예측
            tomorrow_date = df_ml['timestamp'].max() + timedelta(days=1)
            tomorrow_ordinal = np.array([[tomorrow_date.toordinal()]])
            predicted_price = model.predict(tomorrow_ordinal)[0]
            
            # 현재가와 비교
            current_price = latest_data['close']
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            
            p_col1, p_col2 = st.columns([1, 2])
            
            with p_col1:
                st.write(f"**예측 기준일:** {tomorrow_date.date()}")
                if change > 0:
                    st.success(f"🚀 **상승 예측**: 약 {change_pct:.2f}% 상승할 것으로 보입니다.")
                else:
                    st.error(f"📉 **하락 주의**: 약 {change_pct:.2f}% 하락할 것으로 보입니다.")
                
                st.metric("내일 예상 종가", f"{predicted_price:,.0f} KRW", f"{change:,.0f} KRW")
                st.caption("※ 선형 회귀 모델은 과거의 추세가 지속된다는 가정하에 계산되므로 투자 참고용으로만 사용하세요.")

            with p_col2:
                # 예측 시각화
                future_dates = pd.date_range(start=df_ml['timestamp'].min(), end=tomorrow_date)
                future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                trend_line = model.predict(future_ordinals)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df_ml['timestamp'], y=df_ml['close'], mode='lines', name='실제 가격'))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=trend_line, mode='lines', name='추세선 (예측)', line=dict(dash='dash', color='red')))
                fig_pred.update_layout(title="가격 추세선 및 예측 결과", height=400)
                st.plotly_chart(fig_pred, use_container_width=True)

            # 나머지 차트
            st.divider()
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("날짜별 거래량")
                fig_volume = px.bar(filtered_df, x='timestamp', y='volume',
                                    labels={'volume': '거래량', 'timestamp': '날짜'},
                                    color='volume', color_continuous_scale='Viridis')
                st.plotly_chart(fig_volume, use_container_width=True)

            with col_right:
                st.subheader("시가총액 추이")
                fig_mc = px.area(filtered_df, x='timestamp', y='marketCap',
                                 labels={'marketCap': '시가총액', 'timestamp': '날짜'},
                                 color_discrete_sequence=['gold'])
                st.plotly_chart(fig_mc, use_container_width=True)

            # 데이터 요약 정보
            st.divider()
            with st.expander("원본 데이터 확인 및 통계 요약"):
                tab1, tab2 = st.tabs(["데이터 통계", "Raw Data"])
                with tab1:
                    st.write(filtered_df[['open', 'high', 'low', 'close', 'volume', 'marketCap']].describe())
                with tab2:
                    st.dataframe(filtered_df.sort_values('timestamp', ascending=False), use_container_width=True)
        else:
            st.warning("선택한 기간에 해당하는 데이터가 없습니다.")

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
