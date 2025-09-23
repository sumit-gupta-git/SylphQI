
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh

# Page configuration
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
FLASK_API_URL = "http://localhost:5000"  # Your Flask API URL
CITIES = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
WEATHER_PARAMS = ["Max_Temperature_C", "Min_Temperature_C", "Avg_Temperature_C", 
                  "Humidity_Percent", "Rainfall_mm", "Wind_Speed_kmh", 
                  "Atmospheric_Pressure_hPa", "Visibility_km"]

# AQI Categories
AQI_CATEGORIES = {
    (0, 50): {"category": "Good", "color": "#00E400", "description": "Air quality is satisfactory"},
    (51, 100): {"category": "Moderate", "color": "#FFFF00", "description": "Air quality is acceptable"},
    (101, 150): {"category": "Unhealthy for Sensitive Groups", "color": "#FF7E00", "description": "Sensitive people may experience health effects"},
    (151, 200): {"category": "Unhealthy", "color": "#FF0000", "description": "Everyone may experience health effects"},
    (201, 300): {"category": "Very Unhealthy", "color": "#8F3F97", "description": "Health alert for everyone"},
    (301, 500): {"category": "Hazardous", "color": "#7E0023", "description": "Emergency conditions for everyone"}
}

@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_api_data(endpoint):
    """Fetch data from Flask API with caching"""
    try:
        response = requests.get(f"{FLASK_API_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_aqi_category(aqi_value):
    """Get AQI category information"""
    if aqi_value is None:
        return {"category": "Unknown", "color": "#808080", "description": "No data available"}
    
    for (min_val, max_val), info in AQI_CATEGORIES.items():
        if min_val <= aqi_value <= max_val:
            return info
    return {"category": "Hazardous", "color": "#7E0023", "description": "Extremely dangerous"}

def create_aqi_gauge(city_data):
    """Create AQI gauge chart"""
    if not city_data:
        return go.Figure()
    
    aqi = city_data.get("AQI", 0)
    city_name = city_data.get("city", "Unknown")
    category_info = get_aqi_category(aqi)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = aqi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{city_name} AQI"},
        delta = {'reference': 100, 'relative': True},
        gauge = {
            'axis': {'range': [None, 300], 'tickwidth': 1},
            'bar': {'color': category_info["color"]},
            'steps': [
                {'range': [0, 50], 'color': "#00E400"},
                {'range': [50, 100], 'color': "#FFFF00"}, 
                {'range': [100, 150], 'color': "#FF7E00"},
                {'range': [150, 200], 'color': "#FF0000"},
                {'range': [200, 300], 'color': "#8F3F97"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 200
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_pollutant_comparison(all_cities_data):
    """Create pollutant comparison chart"""
    if not all_cities_data:
        return go.Figure()
    
    cities = list(all_cities_data.keys())
    pollutant_data = {pollutant: [] for pollutant in POLLUTANTS}
    
    for city in cities:
        city_data = all_cities_data.get(city, {})
        for pollutant in POLLUTANTS:
            pollutant_data[pollutant].append(city_data.get(pollutant, 0))
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3[:len(POLLUTANTS)]
    
    for i, pollutant in enumerate(POLLUTANTS):
        fig.add_trace(go.Bar(
            name=pollutant,
            x=cities,
            y=pollutant_data[pollutant],
            marker_color=colors[i],
            text=[f"{val:.1f}" for val in pollutant_data[pollutant]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Pollutant Levels Across Cities',
        xaxis_title='Cities',
        yaxis_title='Concentration (μg/m³)',
        barmode='group',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_weather_comparison(all_cities_data):
    """Create weather comparison chart"""
    if not all_cities_data:
        return go.Figure()
    
    cities = list(all_cities_data.keys())
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Humidity & Rainfall', 'Wind Speed', 'Pressure & Visibility'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Temperature data
    max_temps = [all_cities_data[city].get('Max_Temperature_C', 0) for city in cities]
    min_temps = [all_cities_data[city].get('Min_Temperature_C', 0) for city in cities]
    avg_temps = [all_cities_data[city].get('Avg_Temperature_C', 0) for city in cities]
    
    fig.add_trace(go.Bar(name='Max Temp', x=cities, y=max_temps, marker_color='red'), row=1, col=1)
    fig.add_trace(go.Bar(name='Min Temp', x=cities, y=min_temps, marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Bar(name='Avg Temp', x=cities, y=avg_temps, marker_color='orange'), row=1, col=1)
    
    # Humidity and Rainfall
    humidity = [all_cities_data[city].get('Humidity_Percent', 0) for city in cities]
    rainfall = [all_cities_data[city].get('Rainfall_mm', 0) for city in cities]
    
    fig.add_trace(go.Bar(name='Humidity %', x=cities, y=humidity, marker_color='lightblue'), row=1, col=2)
    fig.add_trace(go.Bar(name='Rainfall mm', x=cities, y=rainfall, marker_color='darkblue', yaxis='y2'), row=1, col=2, secondary_y=True)
    
    # Wind Speed
    wind_speed = [all_cities_data[city].get('Wind_Speed_kmh', 0) for city in cities]
    fig.add_trace(go.Bar(name='Wind Speed', x=cities, y=wind_speed, marker_color='green'), row=2, col=1)
    
    # Pressure and Visibility  
    pressure = [all_cities_data[city].get('Atmospheric_Pressure_hPa', 0) for city in cities]
    visibility = [all_cities_data[city].get('Visibility_km', 0) for city in cities]
    
    fig.add_trace(go.Bar(name='Pressure hPa', x=cities, y=pressure, marker_color='purple'), row=2, col=2)
    fig.add_trace(go.Bar(name='Visibility km', x=cities, y=visibility, marker_color='brown', yaxis='y2'), row=2, col=2, secondary_y=True)
    
    fig.update_layout(height=600, showlegend=True, title_text="Weather Parameters Comparison")
    
    return fig

def create_trend_chart(trend_data, parameter, city):
    """Create trend analysis chart using Prophet results"""
    if not trend_data or 'trends' not in trend_data:
        return go.Figure()
    
    trends = trend_data['trends']
    
    fig = go.Figure()
    
    # Historical data
    if 'historical' in trends:
        historical = trends['historical']
        hist_dates = [item['ds'] for item in historical]
        hist_values = [item['y'] for item in historical]
        
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
    
    # Forecast data
    if 'forecast' in trends:
        forecast = trends['forecast']
        forecast_dates = [item['ds'] for item in forecast]
        forecast_values = [item['yhat'] for item in forecast]
        upper_bound = [item['yhat_upper'] for item in forecast]
        lower_bound = [item['yhat_lower'] for item in forecast]
        
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Confidence Interval'
        ))
    
    fig.update_layout(
        title=f'{parameter} Trend Analysis - {city}',
        xaxis_title='Date',
        yaxis_title=f'{parameter} Value',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .stAlert > div {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🌫️ Real-Time AQI Prediction Dashboard")
    st.markdown("*Monitoring air quality and weather patterns for major Indian cities*")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto-refresh Dashboard", value=True)
    if auto_refresh:
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval (seconds)",
            [30, 60, 120, 300], 
            index=2
        )
        count = st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data Now"):
        st.cache_data.clear()
        st.rerun()
    
    # API Status Check
    with st.sidebar.expander("📊 API Status"):
        health_data = fetch_api_data("/api/health")
        if health_data and health_data.get("status") == "healthy":
            st.success("✅ Flask API Connected")
            st.write(f"Cities: {len(health_data.get('cities', []))}")
            st.write(f"Background Thread: {'Active' if health_data.get('background_thread_active') else 'Inactive'}")
        else:
            st.error("❌ Flask API Disconnected")
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col2:
        st.markdown("### 📊 Last Updated")
        st.info(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch real-time data
    with st.spinner("Fetching real-time data..."):
        api_response = fetch_api_data("/api/realtime/all")
    
    if not api_response or api_response.get("status") != "success":
        st.error("Failed to fetch data from API. Please check your Flask backend.")
        st.stop()
    
    all_cities_data = api_response.get("data", {})
    
    if not all_cities_data:
        st.warning("No data available from the API.")
        st.stop()
    
    # Display summary metrics
    st.markdown("### 📈 Current AQI Summary")
    
    # Create columns for summary metrics
    metric_cols = st.columns(5)
    
    for i, (city, data) in enumerate(all_cities_data.items()):
        with metric_cols[i]:
            aqi_value = data.get("AQI", 0)
            category_info = get_aqi_category(aqi_value)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color:{category_info['color']}">{city}</h4>
                <h2 style="margin:0; color:{category_info['color']}">{aqi_value:.1f}</h2>
                <p style="margin:0; font-size:0.8em; color:{category_info['color']}">{category_info['category']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 AQI Gauges", "🔬 Pollutant Analysis", "🌤️ Weather Analysis", "📊 Trend Analysis"])
    
    with tab1:
        st.markdown("### 🎯 Real-Time AQI Levels")
        
        # Create gauge charts
        gauge_cols = st.columns(3)
        
        city_list = list(all_cities_data.keys())
        for i in range(0, len(city_list), 3):
            batch = city_list[i:i+3]
            cols = st.columns(len(batch))
            
            for j, city in enumerate(batch):
                with cols[j]:
                    city_data = all_cities_data[city]
                    fig_gauge = create_aqi_gauge(city_data)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Additional city info
                    aqi = city_data.get("AQI", 0)
                    category_info = get_aqi_category(aqi)
                    st.markdown(f"**Status:** {category_info['category']}")
                    st.markdown(f"**Description:** {category_info['description']}")
    
    with tab2:
        st.markdown("### 🔬 Pollutant Concentration Analysis")
        
        # Pollutant comparison chart
        fig_pollutants = create_pollutant_comparison(all_cities_data)
        st.plotly_chart(fig_pollutants, use_container_width=True)
        
        # Detailed pollutant table
        st.markdown("### 📋 Detailed Pollutant Data")
        
        pollutant_df_data = []
        for city, data in all_cities_data.items():
            row = {"City": city}
            row.update({pollutant: data.get(pollutant, 0) for pollutant in POLLUTANTS})
            pollutant_df_data.append(row)
        
        pollutant_df = pd.DataFrame(pollutant_df_data)
        st.dataframe(pollutant_df, use_container_width=True)
        
        # Download button
        csv_data = pollutant_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Pollutant Data (CSV)",
            data=csv_data,
            file_name=f"pollutant_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.markdown("### 🌤️ Weather Parameter Analysis")
        
        # Weather comparison chart
        fig_weather = create_weather_comparison(all_cities_data)
        st.plotly_chart(fig_weather, use_container_width=True)
        
        # Weather data table
        st.markdown("### 📋 Current Weather Data")
        
        weather_df_data = []
        for city, data in all_cities_data.items():
            row = {"City": city}
            row.update({param: data.get(param, 0) for param in WEATHER_PARAMS})
            weather_df_data.append(row)
        
        weather_df = pd.DataFrame(weather_df_data)
        st.dataframe(weather_df, use_container_width=True)
    
    with tab4:
        st.markdown("### 📊 Trend Analysis (Prophet Forecasting)")
        
        # Trend analysis controls
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            selected_city = st.selectbox("Select City for Trend Analysis", CITIES)
        
        with trend_col2:
            selected_parameter = st.selectbox(
                "Select Parameter", 
                POLLUTANTS + ["AQI"] + WEATHER_PARAMS
            )
        
        # Forecast period
        forecast_days = st.slider("Forecast Days", 7, 60, 30)
        
        if st.button("🔮 Generate Trend Analysis"):
            with st.spinner(f"Generating {selected_parameter} trends for {selected_city}..."):
                trend_data = fetch_api_data(f"/api/trends/{selected_city}/{selected_parameter}?days={forecast_days}")
                
                if trend_data and trend_data.get("status") == "success":
                    # Display trend chart
                    fig_trend = create_trend_chart(trend_data, selected_parameter, selected_city)
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Trend insights
                    trends = trend_data.get("trends", {})
                    if "forecast" in trends and trends["forecast"]:
                        avg_forecast = np.mean([item["yhat"] for item in trends["forecast"]])
                        st.info(f"📈 **Average {selected_parameter} forecast for next {forecast_days} days:** {avg_forecast:.2f}")
                    
                    # Components analysis
                    if "components" in trends:
                        components = trends["components"]
                        st.markdown("#### 📊 Trend Components")
                        
                        comp_col1, comp_col2, comp_col3 = st.columns(3)
                        
                        with comp_col1:
                            if components.get("trend"):
                                avg_trend = np.mean(components["trend"])
                                st.metric("Overall Trend", f"{avg_trend:.2f}")
                        
                        with comp_col2:
                            if components.get("weekly"):
                                avg_weekly = np.mean(components["weekly"])
                                st.metric("Weekly Seasonality", f"{avg_weekly:.2f}")
                        
                        with comp_col3:
                            if components.get("yearly"):
                                avg_yearly = np.mean(components["yearly"])
                                st.metric("Yearly Seasonality", f"{avg_yearly:.2f}")
                
                else:
                    st.error("Failed to generate trend analysis. Insufficient historical data.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    💡 **Data Sources**: OpenAQ Global Platform, Visual Crossing Weather API, Real-time Government Monitoring Stations
    
    🔄 **Update Frequency**: Real-time data refreshes every 5 minutes | Trend analysis uses Prophet forecasting
    
    ⚠️ **Note**: Air quality predictions are based on machine learning models and should be used as estimates only
    """)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📋 Dashboard Info")
        st.write("**Total Cities Monitored:**", len(CITIES))
        st.write("**Parameters Tracked:**", len(POLLUTANTS + WEATHER_PARAMS))
        st.write("**ML Models Used:**", "RandomForest + Prophet")
        
        if auto_refresh:
            st.success(f"🔄 Auto-refreshing every {refresh_interval}s")
        
        # API endpoint info
        with st.expander("🔧 API Endpoints"):
            st.code(f"""
Real-time Data: {FLASK_API_URL}/api/realtime/all
City Data: {FLASK_API_URL}/api/realtime/<city>
Trends: {FLASK_API_URL}/api/trends/<city>/<param>
Health: {FLASK_API_URL}/api/health
            """)

if __name__ == "__main__":
    main()
