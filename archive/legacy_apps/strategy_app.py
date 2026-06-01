import streamlit as pd_st
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Set page configuration
st.set_page_config(
    page_title="交易策略可视化分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# File paths
RECORD_DIR = 'record'
POSITION_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26持仓记录.csv')

@st.cache_data
def load_data():
    """Load and process position data."""
    try:
        df_pos = pd.read_csv(POSITION_FILE)
        df_pos['日期'] = pd.to_datetime(df_pos['日期'])
        # Create a dictionary of date -> held asset
        holding_map = df_pos.set_index('日期')['ETF名称'].to_dict()
        return holding_map, df_pos
    except Exception as e:
        st.error(f"Error loading position file: {e}")
        return {}, pd.DataFrame()

@st.cache_data
def get_history_files():
    """Map asset names to filenames and verify existence."""
    mapping = {
        '创业板': '159915.SZ_创业板ETF_history.csv',
        '南方原油': '501018.SH_南方原油(LOF)_history.csv',
        '上证180': '510180.SH_180ETF_history.csv',
        '30年国债': '511090.SH_30年国债ETF_history.csv',
        '港股科技': '513020.SH_港股科技ETF_history.csv',
        '纳指100': '513100.SH_纳指ETF_history.csv',
        '日经ETF': '513520.SH_日经ETF_history.csv',
        '黄金ETF': '518880.SH_黄金ETF_history.csv',
        '科创板': '588120.SH_科创板ETF_history.csv'
    }
    
    valid_mapping = {}
    for name, filename in mapping.items():
        if os.path.exists(filename):
            valid_mapping[name] = filename
            
    return valid_mapping

@st.cache_data
def load_asset_history(filename):
    """Load history data for a specific asset."""
    try:
        df_hist = pd.read_csv(filename)
        df_hist['trade_date'] = pd.to_datetime(df_hist['trade_date'], format='%Y%m%d')
        df_hist = df_hist.sort_values('trade_date')
        return df_hist
    except Exception as e:
        st.error(f"Error loading history file {filename}: {e}")
        return pd.DataFrame()

def create_chart(asset_name, df_hist, holding_map):
    """Create a plotly chart for a single asset."""
    dates = df_hist['trade_date'].tolist()
    prices = df_hist['close'].tolist()
    
    fig = go.Figure()
    
    # 1. Plot the full history in dim color (Background)
    fig.add_trace(go.Scatter(
        x=dates, 
        y=prices,
        mode='lines',
        name='非持有期',
        line=dict(color='lightgrey', width=1.5),
        hoverinfo='skip'
    ))
    
    # 2. Identify held segments
    held_indices = []
    for i, date in enumerate(dates):
        d = date.normalize()
        if holding_map.get(d) == asset_name:
            held_indices.append(i)
    
    # Calculate holding stats
    total_days = len(dates)
    held_days = len(held_indices)
    held_pct = (held_days / total_days * 100) if total_days > 0 else 0
    
    if held_indices:
        segments = []
        current_segment = [held_indices[0]]
        for i in range(1, len(held_indices)):
            if held_indices[i] == held_indices[i-1] + 1:
                current_segment.append(held_indices[i])
            else:
                segments.append(current_segment)
                current_segment = [held_indices[i]]
        segments.append(current_segment)
        
        held_x = []
        held_y = []
        
        for segment in segments:
            seg_dates = [dates[j] for j in segment]
            seg_prices = [prices[j] for j in segment]
            
            held_x.extend(seg_dates)
            held_y.extend(seg_prices)
            held_x.append(None) # Break line
            held_y.append(None)
            
        fig.add_trace(go.Scatter(
            x=held_x,
            y=held_y,
            mode='lines', # Removed markers for cleaner look, added hover logic below
            name='持有期',
            line=dict(color='#ff4b4b', width=2.5),
            hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>价格</b>: %{y:.2f}<extra></extra>'
        ))
        
        # Enable hover for the grey line too
        fig.data[0].hoverinfo = 'all'
        fig.data[0].hovertemplate = '<b>日期</b>: %{x|%Y-%m-%d}<br><b>价格</b>: %{y:.2f}<br><i>(非持有)</i><extra></extra>'

    fig.update_layout(
        title=dict(text=f'{asset_name}', x=0.5, xanchor='center'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#eee'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig, held_days, held_pct

def main():
    st.title("📊 策略持仓可视化看板")
    
    st.markdown("""
    本看板展示了 **9个主要标的** 的价格走势及策略持仓时段。
    - <span style='color:lightgrey'>灰色线条</span>：标的价格走势（空仓期）
    - <span style='color:#ff4b4b'>红色线条</span>：策略持有期
    """, unsafe_allow_html=True)
    
    # Load data
    holding_map, df_pos = load_data()
    if not holding_map:
        st.warning("未找到持仓数据，请检查 record 文件夹。")
        return
        
    asset_files = get_history_files()
    if not asset_files:
        st.warning("未找到标的历史数据文件，请检查根目录。")
        return

    # Sidebar controls
    st.sidebar.header("⚙️ 选项")
    
    # Asset selection
    all_assets = list(asset_files.keys())
    selected_assets = st.sidebar.multiselect(
        "选择展示标的",
        options=all_assets,
        default=all_assets
    )
    
    # Date range filter
    min_date = df_pos['日期'].min().date()
    max_date = df_pos['日期'].max().date()
    
    start_date, end_date = st.sidebar.slider(
        "选择时间范围",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # Layout: Grid
    st.subheader(f"📈 标的走势图 ({len(selected_assets)})")
    
    # Use columns for grid layout
    cols_per_row = 3
    cols = st.columns(cols_per_row)
    
    for idx, asset_name in enumerate(selected_assets):
        filename = asset_files[asset_name]
        
        # Load and filter history
        df_hist = load_asset_history(filename)
        if df_hist.empty:
            continue
            
        mask = (df_hist['trade_date'].dt.date >= start_date) & (df_hist['trade_date'].dt.date <= end_date)
        df_hist_filtered = df_hist.loc[mask].copy()
        
        if df_hist_filtered.empty:
            continue
            
        # Create chart
        fig, held_days, held_pct = create_chart(asset_name, df_hist_filtered, holding_map)
        
        # Display in grid
        with cols[idx % cols_per_row]:
            st.plotly_chart(fig, use_container_width=True)
            # Add small metric below chart
            st.caption(f"持有天数: {held_days} ({held_pct:.1f}%)")

    # Show raw data option
    with st.expander("查看原始持仓数据"):
        st.dataframe(df_pos.sort_values('日期', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
