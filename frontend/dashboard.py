#File: frontend/dashboard.py

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
import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Configure page
st.set_page_config(
    page_title="Health First AI Supply Chain Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .status-active {
        color: #4caf50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

class DashboardAPI:
    """Handle API calls to the backend"""

    @staticmethod
    def get_system_status():
        """Get system status"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"overall_status": "unknown", "agents": {}}
        except Exception as e:
            st.error(f"Failed to connect to backend API: {str(e)}")
            return {"overall_status": "disconnected", "agents": {}}

    @staticmethod
    def get_dashboard_data():
        """Get comprehensive dashboard data"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/analytics/dashboard-data", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            st.error(f"Failed to fetch dashboard data: {str(e)}")
            return {}

    @staticmethod
    def get_kpis():
        """Get KPI metrics"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/analytics/kpis", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            return {}

    @staticmethod
    def get_market_intelligence():
        """Get market intelligence data"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/market-intelligence/summary", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            return {}

    @staticmethod
    def get_demand_forecast():
        """Get demand forecast data"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/demand/forecast", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            return {}

    @staticmethod
    def get_inventory_levels():
        """Get inventory levels"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/inventory/levels", timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            return []

    @staticmethod
    def get_supplier_performance():
        """Get supplier performance"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/suppliers/performance", timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            return []

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🏥 Health First AI Supply Chain Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

def render_system_status():
    """Render system status in sidebar"""
    st.sidebar.markdown("## 🔧 System Status")

    status_data = DashboardAPI.get_system_status()
    overall_status = status_data.get('overall_status', 'unknown')

    # Overall status
    if overall_status == 'healthy':
        st.sidebar.markdown('<p class="status-active">🟢 System Healthy</p>', unsafe_allow_html=True)
    elif overall_status == 'degraded':
        st.sidebar.markdown('<p class="status-error">🟡 System Degraded</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-error">🔴 System Offline</p>', unsafe_allow_html=True)

    # Agent statuses
    agents = status_data.get('agents', {})
    for agent_name, agent_info in agents.items():
        status = agent_info.get('status', 'unknown')
        if status == 'active':
            st.sidebar.markdown(f"✅ {agent_name.replace('_', ' ').title()}")
        else:
            st.sidebar.markdown(f"❌ {agent_name.replace('_', ' ').title()}")

def render_kpi_metrics():
    """Render KPI metrics"""
    st.markdown("## 📊 Key Performance Indicators")

    kpis = DashboardAPI.get_kpis()

    if kpis:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            forecast_accuracy = kpis.get('forecast_accuracy', 0.0)
            st.metric(
                label="Forecast Accuracy",
                value=f"{forecast_accuracy:.1%}",
                delta=f"{(forecast_accuracy - 0.85)*100:.1f}%" if forecast_accuracy > 0 else None
            )

        with col2:
            inventory_turnover = kpis.get('inventory_turnover', 0.0)
            st.metric(
                label="Inventory Turnover",
                value=f"{inventory_turnover:.1f}x",
                delta=f"{inventory_turnover - 5.0:.1f}" if inventory_turnover > 0 else None
            )

        with col3:
            stockout_rate = kpis.get('stockout_rate', 0.0)
            st.metric(
                label="Stockout Rate",
                value=f"{stockout_rate:.1%}",
                delta=f"{-(stockout_rate - 0.02)*100:.1f}%" if stockout_rate > 0 else None,
                delta_color="inverse"
            )

        with col4:
            fill_rate = kpis.get('order_fill_rate', 0.0)
            st.metric(
                label="Order Fill Rate",
                value=f"{fill_rate:.1%}",
                delta=f"{(fill_rate - 0.95)*100:.1f}%" if fill_rate > 0 else None
            )

        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            supplier_perf = kpis.get('supplier_performance_avg', 0.0)
            st.metric(
                label="Avg Supplier Performance",
                value=f"{supplier_perf:.1%}",
                delta=f"{(supplier_perf - 0.90)*100:.1f}%" if supplier_perf > 0 else None
            )

        with col6:
            cost_savings = kpis.get('cost_savings_mtd', 0.0)
            st.metric(
                label="Cost Savings (MTD)",
                value=f"${cost_savings:,.0f}",
                delta=f"${cost_savings - 25000:,.0f}" if cost_savings > 0 else None
            )

        with col7:
            emergency_orders = kpis.get('emergency_orders_count', 0)
            st.metric(
                label="Emergency Orders",
                value=f"{emergency_orders}",
                delta=f"{emergency_orders - 5}" if emergency_orders > 0 else None,
                delta_color="inverse"
            )

        with col8:
            inventory_value = kpis.get('total_inventory_value', 0.0)
            st.metric(
                label="Total Inventory Value",
                value=f"${inventory_value:,.0f}",
                delta=None
            )
    else:
        st.warning("KPI data not available. Please ensure the backend API is running.")

def render_market_intelligence():
    """Render market intelligence section"""
    st.markdown("## 🌐 Market Intelligence")

    market_data = DashboardAPI.get_market_intelligence()

    if market_data:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Social Media Monitoring")
            social_media = market_data.get('social_media', {})

            if social_media:
                # Sentiment pie chart
                sentiment_data = {
                    'Positive': social_media.get('sentiment_positive', 0),
                    'Negative': social_media.get('sentiment_negative', 0),
                    'Neutral': social_media.get('sentiment_neutral', 0)
                }

                fig_sentiment = px.pie(
                    values=list(sentiment_data.values()),
                    names=list(sentiment_data.keys()),
                    title="Social Media Sentiment",
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Negative': '#F44336',
                        'Neutral': '#FFC107'
                    }
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

                # Mentions metric
                mentions = social_media.get('mentions_7d', 0)
                change_pct = social_media.get('mentions_change_pct', 0)
                st.metric(
                    label="Mentions (7 days)",
                    value=f"{mentions:,}",
                    delta=f"{change_pct:+.1f}%"
                )

                # Trending topics
                trending_topics = social_media.get('trending_topics', [])
                if trending_topics:
                    st.markdown("**Trending Topics:**")
                    for topic in trending_topics[:5]:
                        st.markdown(f"• {topic}")

        with col2:
            st.markdown("### Competitor Analysis")
            competitors = market_data.get('competitors', [])

            if competitors:
                # Market share chart
                market_share_data = {
                    comp['competitor_name']: comp['market_share'] 
                    for comp in competitors
                }

                fig_market_share = px.bar(
                    x=list(market_share_data.keys()),
                    y=list(market_share_data.values()),
                    title="Market Share by Competitor",
                    labels={'x': 'Competitor', 'y': 'Market Share'}
                )
                fig_market_share.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_market_share, use_container_width=True)

                # Supply chain status
                st.markdown("**Supply Chain Status:**")
                for comp in competitors[:5]:
                    status = comp.get('supply_chain_status', 'normal')
                    status_emoji = {
                        'normal': '🟢',
                        'disrupted': '🔴',
                        'optimized': '🟡'
                    }.get(status, '⚪')
                    st.markdown(f"{status_emoji} {comp['competitor_name']}: {status.title()}")

        # Active alerts
        active_triggers = market_data.get('active_triggers', [])
        if active_triggers:
            st.markdown("### 🚨 Active Market Alerts")
            for trigger in active_triggers[:5]:
                severity = trigger.get('severity', 'low')
                alert_class = f"alert-{severity}"
                st.markdown(
                    f'<div class="metric-card {alert_class}">'
                    f'<strong>{trigger.get("type", "").replace("_", " ").title()}</strong><br>'
                    f'{trigger.get("description", "")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Market intelligence data not available. Starting with simulated data...")

def render_demand_forecasting():
    """Render demand forecasting section"""
    st.markdown("## 📈 Demand Forecasting")

    forecast_data = DashboardAPI.get_demand_forecast()

    if forecast_data and 'forecasts' in forecast_data:
        forecasts = forecast_data['forecasts']

        if forecasts:
            # Convert to DataFrame for easier handling
            df_forecasts = pd.DataFrame([
                {
                    'Date': forecast['forecast_date'],
                    'Product Category': forecast['product_category'],
                    'Region': forecast['region'],
                    'Predicted Demand': forecast['predicted_demand'],
                    'Lower Bound': forecast['confidence_interval_lower'],
                    'Upper Bound': forecast['confidence_interval_upper']
                }
                for forecast in forecasts[:100]  # Limit for performance
            ])

            df_forecasts['Date'] = pd.to_datetime(df_forecasts['Date'])

            col1, col2 = st.columns(2)

            with col1:
                # Product category filter
                categories = df_forecasts['Product Category'].unique()
                selected_category = st.selectbox("Select Product Category", categories)

                # Filter data
                filtered_df = df_forecasts[df_forecasts['Product Category'] == selected_category]

                # Aggregate by date
                daily_forecast = filtered_df.groupby('Date').agg({
                    'Predicted Demand': 'sum',
                    'Lower Bound': 'sum',
                    'Upper Bound': 'sum'
                }).reset_index()

                # Forecast chart
                fig_forecast = go.Figure()

                # Add prediction line
                fig_forecast.add_trace(go.Scatter(
                    x=daily_forecast['Date'],
                    y=daily_forecast['Predicted Demand'],
                    mode='lines',
                    name='Predicted Demand',
                    line=dict(color='blue')
                ))

                # Add confidence interval
                fig_forecast.add_trace(go.Scatter(
                    x=daily_forecast['Date'],
                    y=daily_forecast['Upper Bound'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))

                fig_forecast.add_trace(go.Scatter(
                    x=daily_forecast['Date'],
                    y=daily_forecast['Lower Bound'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='Confidence Interval',
                    fillcolor='rgba(0,100,80,0.2)'
                ))

                fig_forecast.update_layout(
                    title=f"Demand Forecast - {selected_category}",
                    xaxis_title="Date",
                    yaxis_title="Predicted Demand"
                )

                st.plotly_chart(fig_forecast, use_container_width=True)

            with col2:
                # Regional breakdown
                regional_demand = df_forecasts.groupby('Region')['Predicted Demand'].sum().reset_index()

                fig_regional = px.bar(
                    regional_demand,
                    x='Region',
                    y='Predicted Demand',
                    title="Demand Forecast by Region"
                )
                fig_regional.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_regional, use_container_width=True)

                # Model performance
                performance = forecast_data.get('model_performance', {})
                if performance:
                    st.markdown("**Model Performance:**")
                    r2_score = performance.get('r2_score', 0)
                    mae = performance.get('mean_absolute_error', 0)

                    st.metric(label="R² Score", value=f"{r2_score:.3f}")
                    st.metric(label="Mean Absolute Error", value=f"{mae:.2f}")
        else:
            st.info("No forecast data available")
    else:
        st.info("Demand forecasting data not available")

def render_inventory_management():
    """Render inventory management section"""
    st.markdown("## 📦 Inventory Management")

    inventory_data = DashboardAPI.get_inventory_levels()

    if inventory_data:
        # Convert to DataFrame
        df_inventory = pd.DataFrame([
            {
                'Product ID': item['product_id'],
                'Product Name': item['product_name'],
                'Category': item['product_category'],
                'Location': item['location'],
                'Current Stock': item['current_stock'],
                'Safety Stock': item['safety_stock_level'],
                'Reorder Point': item['reorder_point'],
                'Max Stock': item['max_stock_level'],
                'Unit Cost': item['unit_cost'],
                'Total Value': item['total_value'],
                'Stock Status': 'Low' if item['current_stock'] <= item['safety_stock_level'] else 'Normal'
            }
            for item in inventory_data['items']
        ])

        col1, col2 = st.columns(2)

        with col1:
            # Inventory value by category
            category_value = df_inventory.groupby('Category')['Total Value'].sum().reset_index()

            fig_category_value = px.pie(
                category_value,
                values='Total Value',
                names='Category',
                title="Inventory Value by Category"
            )
            st.plotly_chart(fig_category_value, use_container_width=True)

            # Stock status summary
            status_summary = df_inventory['Stock Status'].value_counts()

            col_normal, col_low = st.columns(2)
            with col_normal:
                st.metric(
                    label="Normal Stock Items",
                    value=status_summary.get('Normal', 0)
                )
            with col_low:
                st.metric(
                    label="Low Stock Items",
                    value=status_summary.get('Low', 0),
                    delta_color="inverse"
                )

        with col2:
            # Stock levels chart
            fig_stock = px.scatter(
                df_inventory,
                x='Safety Stock',
                y='Current Stock',
                color='Category',
                size='Total Value',
                hover_data=['Product Name', 'Location'],
                title="Current Stock vs Safety Stock Levels"
            )

            # Add diagonal line for reference
            max_val = max(df_inventory['Current Stock'].max(), df_inventory['Safety Stock'].max())
            fig_stock.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Safety Stock Line',
                line=dict(dash='dash', color='red')
            ))

            st.plotly_chart(fig_stock, use_container_width=True)

        # Low stock alerts
        low_stock_items = df_inventory[df_inventory['Stock Status'] == 'Low']
        if not low_stock_items.empty:
            st.markdown("### ⚠️ Low Stock Alerts")
            st.dataframe(
                low_stock_items[['Product Name', 'Category', 'Location', 'Current Stock', 'Safety Stock']],
                use_container_width=True
            )
    else:
        st.info("Inventory data not available")

def render_supplier_performance():
    """Render supplier performance section"""
    st.markdown("## 🤝 Supplier Performance")

    supplier_data = DashboardAPI.get_supplier_performance()

    if supplier_data:
        # Convert to DataFrame
        df_suppliers = pd.DataFrame([
            {
                'Supplier': supplier['supplier_name'],
                'On-Time Delivery': supplier['metrics']['on_time_delivery_rate'],
                'Quality Score': supplier['metrics']['quality_score'],
                'Lead Time (days)': supplier['metrics']['average_lead_time_days'],
                'Cost Competitiveness': supplier['metrics']['cost_competitiveness'],
                'Reliability': supplier['metrics']['reliability_score'],
                'Risk Level': supplier['risk_level']
            }
            for supplier in supplier_data['suppliers']
        ])

        col1, col2 = st.columns(2)

        with col1:
            # Supplier performance radar chart
            fig_radar = go.Figure()

            for idx, supplier in df_suppliers.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[
                        supplier['On-Time Delivery'],
                        supplier['Quality Score'] / 5,  # Normalize to 0-1
                        1 - (supplier['Lead Time (days)'] / 20),  # Inverse and normalize
                        supplier['Cost Competitiveness'],
                        supplier['Reliability']
                    ],
                    theta=[
                        'On-Time Delivery',
                        'Quality Score',
                        'Lead Time',
                        'Cost Competitiveness',
                        'Reliability'
                    ],
                    fill='toself',
                    name=supplier['Supplier']
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Supplier Performance Comparison",
                showlegend=True
            )

            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Risk level distribution
            risk_counts = df_suppliers['Risk Level'].value_counts()

            fig_risk = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Supplier Risk Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    'low': '#4CAF50',
                    'medium': '#FFC107',
                    'high': '#FF9800',
                    'critical': '#F44336'
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Top performing suppliers
            st.markdown("**Top Performing Suppliers:**")
            top_suppliers = df_suppliers.nlargest(3, 'On-Time Delivery')

            for idx, supplier in top_suppliers.iterrows():
                st.markdown(
                    f"🏆 {supplier['Supplier']}: "
                    f"{supplier['On-Time Delivery']:.1%} on-time delivery"
                )

        # Detailed supplier table
        st.markdown("### Detailed Supplier Metrics")
        st.dataframe(df_suppliers, use_container_width=True)

    else:
        st.info("Supplier performance data not available")

def render_navigation():
    """Render navigation sidebar"""
    st.sidebar.markdown("## 📋 Navigation")

    pages = {
        "🏠 Overview": "overview",
        "🌐 Market Intelligence": "market_intelligence", 
        "📈 Demand Forecasting": "demand_forecasting",
        "📦 Inventory Management": "inventory_management",
        "🤝 Supplier Performance": "supplier_performance"
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))
    return pages[selected_page]

def main():
    """Main dashboard application"""
    render_header()

    # Sidebar
    render_system_status()
    selected_page = render_navigation()

    # Auto-refresh option
    st.sidebar.markdown("## 🔄 Auto Refresh")
    auto_refresh = st.sidebar.checkbox("Enable Auto Refresh (30s)")

    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.experimental_rerun()

    # Main content area
    if selected_page == "overview":
        render_kpi_metrics()

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📊 Market Intelligence Summary", expanded=True):
                market_data = DashboardAPI.get_market_intelligence()
                if market_data:
                    social_media = market_data.get('social_media', {})
                    mentions = social_media.get('mentions_7d', 0)
                    sentiment = social_media.get('sentiment_positive', 0)
                    st.metric("Social Media Mentions", f"{mentions:,}")
                    st.metric("Positive Sentiment", f"{sentiment:.1%}")
                else:
                    st.info("Market intelligence data loading...")

        with col2:
            with st.expander("📈 Demand Forecast Summary", expanded=True):
                forecast_data = DashboardAPI.get_demand_forecast()
                if forecast_data and 'forecasts' in forecast_data:
                    forecasts = forecast_data['forecasts']
                    total_demand = sum(f['predicted_demand'] for f in forecasts[:30])
                    st.metric("30-Day Demand Forecast", f"{total_demand:,.0f}")

                    performance = forecast_data.get('model_performance', {})
                    r2_score = performance.get('r2_score', 0)
                    st.metric("Model Accuracy (R²)", f"{r2_score:.3f}")
                else:
                    st.info("Demand forecast data loading...")

    elif selected_page == "market_intelligence":
        render_market_intelligence()

    elif selected_page == "demand_forecasting":
        render_demand_forecasting()

    elif selected_page == "inventory_management":
        render_inventory_management()

    elif selected_page == "supplier_performance":
        render_supplier_performance()

if __name__ == "__main__":
    main()