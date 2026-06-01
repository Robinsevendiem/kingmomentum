import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Set page configuration
st.set_page_config(
    page_title="动量得分与熔断机制分析",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 动量得分深度分析仪表盘")

# ----------------- Data Loading & Calculation -----------------

# Remove cache to ensure we always load the latest data after updates in Home.py
def load_data():
    """Load history data for all assets"""
    mapping = {
        '创业板': 'data/159915.SZ_创业板ETF_history.csv',
        '南方原油': 'data/501018.SH_南方原油(LOF)_history.csv',
        '上证180': 'data/510180.SH_180ETF_history.csv',
        '30年国债': 'data/511090.SH_30年国债ETF_history.csv',
        '港股科技': 'data/513020.SH_港股科技ETF_history.csv',
        '纳指100': 'data/513100.SH_纳指ETF_history.csv',
        '日经ETF': 'data/513520.SH_日经ETF_history.csv',
        '黄金ETF': 'data/518880.SH_黄金ETF_history.csv',
        '科创板': 'data/588120.SH_科创板ETF_history.csv'
    }
    
    data = {}
    for name, filename in mapping.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                if 'trade_date' in df.columns:
                    try:
                        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    except:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').set_index('trade_date')
                data[name] = df
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
    return data

@st.cache_data
def calculate_scores_for_analysis(data, window):
    """
    Calculate Momentum Score (WLS) and Future Returns.
    """
    results = []
    
    for asset_name, df in data.items():
        # Prefer adjusted close for score calculation to handle splits
        if 'adj_close' in df.columns:
            prices = df['adj_close'].values
            price_col = 'adj_close'
        elif 'close' in df.columns:
            prices = df['close'].values
            price_col = 'close'
        else:
            continue
            
        dates = df.index
        
        # Vectorized-ish loop
        # We need to iterate
        x = np.arange(window)
        weights = 1 + (np.linspace(0, 1, window) ** 2)
        
        for i in range(window, len(df) - 20): # Ensure 20d future
            window_prices = prices[i-window : i]
            if len(window_prices) < window: continue
            
            # Skip if NaN
            if np.isnan(window_prices).any(): continue
            
            try:
                current_date = dates[i-1] 
                window_data = window_prices
                current_price = window_data[-1]
                
                y = np.log(window_data)
                
                coeffs = np.polyfit(x, y, 1, w=weights)
                slope = coeffs[0]
                
                y_pred = np.polyval(coeffs, x)
                sse = np.sum(weights * (y - y_pred)**2)
                sst = np.sum(weights * (y - np.average(y, weights=weights))**2)
                r2 = 1 - sse/sst if sst != 0 else 0
                
                score = (np.exp(slope * 252) - 1) * r2 * 100
                
                # Future Return: Next 5, 10, 15, 20 days
                res = {
                    'Asset': asset_name,
                    'Date': dates[i-1],
                    'Score': score,
                    'Close': current_price,
                }
                
                for days in [5, 10, 15, 20]:
                    future_idx = i - 1 + days
                    if future_idx < len(prices):
                        future_ret = prices[future_idx] / prices[i-1] - 1
                        res[f'Future_Return_{days}d'] = future_ret
                    else:
                        res[f'Future_Return_{days}d'] = np.nan
                        
                results.append(res)
            except:
                pass
            
    return pd.DataFrame(results)

# ----------------- UI Controls -----------------

data = load_data()

with st.sidebar:
    st.header("⚙️ 分析参数")
    
    if st.button("🔄 刷新数据 (清除缓存)"):
        st.cache_data.clear()
        st.rerun()

    window_choice = st.radio("动量窗口 (天)", [20, 25], index=1, help="对比不同窗口期对得分分布的影响")
    
    st.divider()
    selected_assets = st.multiselect("选择分析标的", list(data.keys()), default=list(data.keys()))

if not data:
    st.error("No data loaded.")
    st.stop()

# Calculate
with st.spinner(f"正在计算 {window_choice} 天窗口的动量得分..."):
    df_all = calculate_scores_for_analysis(data, window_choice)

if df_all.empty:
    st.warning("无足够数据进行分析。")
    st.stop()

# Filter assets
df_analysis = df_all[df_all['Asset'].isin(selected_assets)]

# ----------------- Tab 1: Overall Statistics -----------------

tab1, tab2, tab3 = st.tabs(["📊 整体统计分析", "📈 标的时间序列", "🔬 收益-得分关系"])

with tab1:
    st.subheader(f"窗口期: {window_choice} 天 - 得分分布统计")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Statistical Table instead of Box Plot
        st.markdown("**各标的得分分布统计表**")
        
        # Calculate stats per asset
        stats_df = df_analysis.groupby('Asset')['Score'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        stats_df = stats_df[['count', 'mean', 'std', '50%', '75%', '90%', '99%', 'max']]
        stats_df.columns = ['样本数', '均值', '标准差', '中位数', '75%分位', '90%分位', '99%分位', '最大值']
        
        # Style the dataframe
        st.dataframe(stats_df.style.format("{:.1f}"), use_container_width=True)
        
    with col2:
        # Stats Table (Overall)
        st.markdown("**整体极值频率**")
        total = len(df_analysis)
        gt_300 = len(df_analysis[df_analysis['Score'] > 300])
        gt_500 = len(df_analysis[df_analysis['Score'] > 500])
        gt_700 = len(df_analysis[df_analysis['Score'] > 700])
        gt_1000 = len(df_analysis[df_analysis['Score'] > 1000])
        
        st.metric("总样本数", total)
        st.metric("> 300分", f"{gt_300/total:.1%}", f"{gt_300}次")
        st.metric("> 500分", f"{gt_500/total:.1%}", f"{gt_500}次")
        st.metric("> 700分", f"{gt_700/total:.1%}", f"{gt_700}次")
        st.caption(f"其中 >1000分 的极端情况有 {gt_1000} 次 ({gt_1000/total:.2%})")
        
    # Heatmap instead of Histogram
    st.markdown("### 动量得分区间分布热力图 (Frequency Heatmap)")
    st.caption("颜色越深代表该标的落在该得分区间的频率越高。这能帮您识别不同标的的“性格”（如：纳指是否比国债更容易得高分？）")
    
    # Create bins for heatmap
    bins = list(range(0, 1100, 100)) + [float('inf')]
    labels = [f"{i}-{i+100}" for i in range(0, 1000, 100)] + [">1000"]
    
    df_heatmap = df_analysis.copy()
    df_heatmap['Score_Range'] = pd.cut(df_heatmap['Score'], bins=bins, labels=labels, right=False)
    
    # Pivot table: Index=Score_Range, Columns=Asset, Values=Count
    # Normalize by column (percentage of time for that asset)
    heatmap_data = pd.crosstab(df_heatmap['Score_Range'], df_heatmap['Asset'], normalize='columns') * 100
    
    # Plot Heatmap
    fig_hm = px.imshow(
        heatmap_data,
        labels=dict(x="标的资产", y="得分区间", color="频率(%)"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        text_auto=".1f",
        aspect="auto"
    )
    fig_hm.update_layout(height=500)
    st.plotly_chart(fig_hm, use_container_width=True)

# ----------------- Tab 2: Time Series Analysis -----------------

with tab2:
    st.subheader("价格 vs 动量得分双轴图")
    
    target_asset = st.selectbox("选择单一标的查看详情", selected_assets)
    
    if target_asset:
        df_single = df_analysis[df_analysis['Asset'] == target_asset].sort_values('Date')
        
        # Create dual-axis chart
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Since we prefer adj_close in calculation, the 'Close' column in df_single IS the adjusted close if available.
        # Let's label it clearly.
        price_label = "收盘价 (后复权)"
        
        # Price Line (Left Axis)
        fig_ts.add_trace(
            go.Scatter(x=df_single['Date'], y=df_single['Close'], name=price_label, 
                       line=dict(color='#1f77b4', width=2)),
            secondary_y=False
        )
        
        # Score Line (Right Axis)
        fig_ts.add_trace(
            go.Scatter(x=df_single['Date'], y=df_single['Score'], name="动量得分 (右轴)", 
                       line=dict(color='#d62728', width=1.5), opacity=0.7, fill='tozeroy'),
            secondary_y=True
        )
        
        # Add Threshold Lines based on user interest (0.4, 5, 10, etc.)
        fig_ts.add_hline(y=0.4, line_dash="dot", line_color="green", secondary_y=True, annotation_text="0.4分", opacity=0.5)
        fig_ts.add_hline(y=5, line_dash="dot", line_color="purple", secondary_y=True, annotation_text="5分", opacity=0.5)
        fig_ts.add_hline(y=10, line_dash="dot", line_color="blue", secondary_y=True, annotation_text="10分", opacity=0.5)
        fig_ts.add_hline(y=300, line_dash="dash", line_color="orange", secondary_y=True, annotation_text="300分 (熔断区)")
        fig_ts.add_hline(y=600, line_dash="dash", line_color="red", secondary_y=True, annotation_text="600分 (极度危险)")
        
        fig_ts.update_layout(
            title=f"{target_asset}: 价格与动量得分走势对比",
            hovermode="x unified",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Toggle for fixed range
        use_fixed_range = st.checkbox("锁定右轴范围 (0-1000)", value=False, help="勾选后，右轴将固定显示 0-1000，方便对比绝对热度；取消勾选则自动缩放，方便查看得分为低值（如 0.4, 5, 10分）时的波动细节。")
        
        # Optimize Y-Axes
        # Left Axis (Price) - Force auto range based on data
        min_price = df_single['Close'].min()
        max_price = df_single['Close'].max()
        padding = (max_price - min_price) * 0.1
        
        fig_ts.update_yaxes(
            title_text="<b>收盘价</b>", 
            title_font=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            secondary_y=False,
            showgrid=False,
            range=[min_price - padding, max_price + padding] # Explicitly set range
        )
        
        # Right Axis (Score)
        y2_config = dict(
            title_text="<b>动量得分</b>", 
            title_font=dict(color="#d62728"),
            tickfont=dict(color="#d62728"),
            secondary_y=True,
            showgrid=True,
        )
        
        # Interactive Range Slider for Score Axis
        st.markdown("**调整得分轴显示范围**")
        col_slider1, col_slider2 = st.columns([3, 1])
        with col_slider1:
            score_max_limit = float(df_single['Score'].max() * 1.1)
            # Slider to set max y-axis for score. This allows user to zoom into any specific score range easily.
            max_score_display = st.slider("最高显示得分", min_value=1.0, max_value=max(1000.0, score_max_limit), value=score_max_limit, step=10.0, help="拖动滑块可以放大查看特定得分区域（如 0-20, 0-50, 0-200）对应的价格曲线形态。")
        
        y2_config['range'] = [0, max_score_display]
        y2_config['rangemode'] = "tozero"
            
        fig_ts.update_yaxes(**y2_config)
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        st.info("""
        **🔍 动量得分与价格形态对应关系指南**:
        通过拖动上方滑块，您可以将视线聚焦在不同的得分层级：
        
        *   **[0 - 10分] 底部与试探期**：
            价格通常处于**长期横盘震荡**或**刚刚止跌企稳**。此时 R² 极低，均线缠绕。偶尔的脉冲式上涨会让得分突破 5 分，这是右侧潜伏的极佳观察点。
        *   **[10 - 50分] 趋势确立期**：
            价格开始走出**清晰的上升通道**，均线多头排列。此时是趋势最健康的阶段，策略在此区间内买入并持有的胜率和盈亏比极高。
        *   **[50 - 200分] 主升浪爆发期**：
            价格出现**加速上涨**，斜率变陡。这是动量策略获取超额收益的核心区域。但此时买入可能面临短期回调的风险。
        *   **[> 300分] 狂热与过热期**：
            价格呈**指数级飙升**，脱离了正常估值轨道。虽然可能继续疯涨，但随时面临均值回归的暴跌。这就是我们设置“熔断阈值”的原因。
        """)

# ----------------- Tab 3: Return Analysis (The Inverted U) -----------------

with tab3:
    st.subheader("🔬 动量分段收益统计 (0-2000分)")
    
    st.markdown("统计每一个标的在动量分数达到某一个数值区间后，未来 5/10/15/20 天的平均收益回报。")
    
    # 1. Configuration
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        target_asset_stats = st.selectbox("选择分析标的", selected_assets, key="stats_asset")
    
    # 2. Binning
    # Create bins 0 to 2000 with step 50
    bins = list(range(0, 2050, 50))
    # Labels: 0, 50, 100... (Use left edge or center for plot?) Let's use left edge string for category
    # But for plotting line chart, using numeric center is better.
    
    df_asset = df_analysis[df_analysis['Asset'] == target_asset_stats].copy()
    
    # Cut
    df_asset['Score_Bin'] = pd.cut(df_asset['Score'], bins=bins, labels=bins[:-1])
    
    # Groupby
    # We want mean return for each period
    cols = ['Future_Return_5d', 'Future_Return_10d', 'Future_Return_15d', 'Future_Return_20d']
    stats = df_asset.groupby('Score_Bin')[cols].mean()
    
    # Count samples to avoid noise
    counts = df_asset.groupby('Score_Bin')['Score'].count()
    stats['Sample_Count'] = counts
    
    # Filter out empty bins or bins with very few samples? 
    # User didn't ask, but it's good practice. For now show all non-empty.
    stats = stats.dropna(how='all')
    
    # Convert index to numeric for plotting
    stats.index = stats.index.astype(int)
    
    # 3. Visualization
    st.markdown(f"### {target_asset_stats}：不同动量分数的未来收益率曲线")
    
    fig = go.Figure()
    
    colors = ['#FF9F33', '#33C1FF', '#33FF57', '#FF3333'] # 5, 10, 15, 20
    
    for i, col in enumerate(cols):
        days = col.split('_')[2]
        fig.add_trace(go.Scatter(
            x=stats.index, 
            y=stats[col],
            mode='lines+markers',
            name=f'{days} 收益率',
            line=dict(width=2, color=colors[i]),
            connectgaps=True # Connect if some intermediate bins are empty? Maybe no.
        ))
    
    # Add Zero Line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    # Add Sample Count Bar on secondary axis? Or just hover.
    # Let's add bars for sample count at the bottom or secondary axis
    fig.add_trace(go.Bar(
        x=stats.index,
        y=stats['Sample_Count'],
        name='样本数量',
        marker_color='lightgray',
        opacity=0.3,
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="动量得分 vs 未来收益率 (及样本分布)",
        xaxis_title="动量得分 (Score)",
        yaxis_title="平均收益率",
        yaxis=dict(tickformat='.2%'),
        yaxis2=dict(
            title="样本数量",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. Data Table
    with st.expander("查看详细统计数据"):
        # Format for display
        display_stats = stats.copy()
        for c in cols:
            display_stats[c] = display_stats[c].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
        st.dataframe(display_stats)
        
    st.divider()
    
    # 5. Global Heatmap (All Assets, One Period)
    st.markdown("### 全市场对比：动量得分 vs 20日收益率")
    st.caption("横轴为动量得分区间，纵轴为不同标的。颜色代表**未来20天的平均收益率**。")
    
    # Global aggregation
    df_analysis['Score_Bin_Global'] = pd.cut(df_analysis['Score'], bins=bins, labels=bins[:-1])
    global_stats = df_analysis.groupby(['Asset', 'Score_Bin_Global'])['Future_Return_20d'].mean().unstack()
    
    # Filter columns (bins) that are mostly empty to keep chart clean?
    # Keep 0-1000 range mostly
    valid_cols = [c for c in global_stats.columns if c < 1200]
    global_stats_clean = global_stats[valid_cols]
    
    fig_hm_ret = px.imshow(
        global_stats_clean,
        labels=dict(x="动量得分区间", y="标的", color="20日收益率"),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto"
    )
    st.plotly_chart(fig_hm_ret, use_container_width=True)
