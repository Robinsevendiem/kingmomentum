import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="参数统计分析",
    page_icon="📊",
    layout="wide"
)

st.title("📊 策略参数多维度评估")

# Load Data
def load_opt_results():
    try:
        # Disable cache to avoid stale data issues
        df = pd.read_csv('data/optimization_results.csv')
        return df
    except:
        st.error("未找到优化结果文件 (data/optimization_results.csv)。请先运行 scripts/fine_tune_strategy.py")
        return pd.DataFrame()

df = load_opt_results()

if not df.empty:
    st.markdown("""
    当不同参数组合的总收益率接近时，单纯看收益率已不足以做出最佳决策。我们需要引入**风险调整后收益 (Sharpe/Calmar)**、**最大回撤**以及**交易频率**等多维度指标进行综合评估。
    """)

    # --- 1. Interactive Ranking ---
    st.header("🏆 参数排行榜 (交互式排序)")
    
    col_sort, col_top = st.columns([1, 3])
    with col_sort:
        sort_by = st.selectbox("排序依据", 
            ["return", "sharpe", "sortino", "calmar", "match_rate", "max_dd", "win_rate", "pl_ratio", "equity_r2"], 
            format_func=lambda x: {
                "return": "总收益率 (越高越好)",
                "sharpe": "夏普比率 (越高越好)",
                "sortino": "索提诺比率 (越高越好)",
                "calmar": "卡玛比率 (越高越好)",
                "match_rate": "还原度 (越高越好)",
                "max_dd": "最大回撤 (越接近0越好)",
                "win_rate": "胜率 (越高越好)",
                "pl_ratio": "盈亏比 (越高越好)",
                "equity_r2": "净值曲线平滑度 R² (越高越好)"
            }[x]
        )
        ascending = True if sort_by == "max_dd" else False
        
    sorted_df = df.sort_values(sort_by, ascending=ascending).head(10)
    
    # Format for display
    display_df = sorted_df.copy()
    display_df['return'] = display_df['return'].apply(lambda x: f"{x:.2%}")
    display_df['match_rate'] = display_df['match_rate'].apply(lambda x: f"{x:.2%}")
    display_df['max_dd'] = display_df['max_dd'].apply(lambda x: f"{x:.2%}")
    display_df['sharpe'] = display_df['sharpe'].apply(lambda x: f"{x:.2f}")
    display_df['sortino'] = display_df['sortino'].apply(lambda x: f"{x:.2f}")
    display_df['calmar'] = display_df['calmar'].apply(lambda x: f"{x:.2f}")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")
    display_df['pl_ratio'] = display_df['pl_ratio'].apply(lambda x: f"{x:.2f}")
    display_df['equity_r2'] = display_df['equity_r2'].apply(lambda x: f"{x:.4f}")
    
    # Rename columns
    display_df = display_df.rename(columns={
        'window': '窗口', 'cutoff': '熔断', 'buffer': '缓冲', 'crash_filter': '暴跌剔除',
        'match_rate': '还原度', 'return': '总收益', 'sharpe': '夏普', 'sortino': '索提诺',
        'max_dd': '最大回撤', 'calmar': '卡玛', 'trade_count': '交易次数',
        'win_rate': '胜率', 'pl_ratio': '盈亏比', 'equity_r2': '曲线R²'
    })
    
    st.dataframe(display_df, use_container_width=True)
    
    # --- 2. Risk-Return Scatter Plot ---
    st.header("⚖️ 风险-收益权衡图")
    st.markdown("横轴为**最大回撤**（风险），纵轴为**年化收益**（收益）。**越靠左上角**的点性价比越高（低风险高收益）。")
    
    # Calculate annualized return for plot (approx from total return)
    # Assuming ~8.5 years
    df['ann_return'] = (1 + df['return']) ** (1/8.5) - 1
    
    fig_scatter = px.scatter(df, x="max_dd", y="ann_return", 
                             color="cutoff", size="sharpe",
                             hover_data=["window", "buffer", "crash_filter", "return"],
                             labels={"max_dd": "最大回撤", "ann_return": "年化收益率", "cutoff": "熔断阈值"},
                             title="参数组合性价比分布 (气泡大小=夏普比率)")
    
    # Invert x axis? No, MaxDD is negative. -0.1 is right of -0.5. So standard axis works: right is better.
    # But usually risk on x axis increases to right. Here risk decreases (value increases) to right.
    # Let's keep it as is: Right = Low Drawdown (Good). Top = High Return (Good).
    # So Top-Right is the best quadrant.
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # --- 3. Parameter Stability (Heatmaps) ---
    st.header("🔬 参数平原分析 (稳定性)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("收益率热力图 (Window vs Cutoff)")
        # Pivot table for heatmap
        pivot_ret = df.pivot_table(index='cutoff', columns='window', values='return', aggfunc='mean')
        fig_hm_ret = px.imshow(pivot_ret, text_auto='.2%', aspect="auto",
                               color_continuous_scale="Viridis",
                               title="不同 窗口-熔断 组合的平均收益")
        st.plotly_chart(fig_hm_ret, use_container_width=True)
        
    with col2:
        st.subheader("还原度热力图 (Window vs Cutoff)")
        pivot_match = df.pivot_table(index='cutoff', columns='window', values='match_rate', aggfunc='mean')
        fig_hm_match = px.imshow(pivot_match, text_auto='.2%', aspect="auto",
                                 color_continuous_scale="Magma",
                                 title="不同 窗口-熔断 组合的还原度")
        st.plotly_chart(fig_hm_match, use_container_width=True)

    # --- 4. Conclusion ---
    st.divider()
    
    with st.expander("📚 指标详解指南 (Glossary)", expanded=False):
        st.markdown("""
        ### 1. 收益类指标
        - **总收益率 (Total Return)**: 策略在回测期间的累积盈亏百分比。
            - *计算*: `(期末净值 - 期初净值) / 期初净值`
            - *意义*: 越高越好，代表绝对赚钱能力。
        - **胜率 (Win Rate)**: 盈利交易次数占总卖出次数的比例。
            - *计算*: `盈利卖出次数 / 总卖出次数`
            - *意义*: >50% 为佳。高胜率策略对交易者心理压力较小。
        - **盈亏比 (Profit/Loss Ratio)**: 平均每次赚钱赚多少 vs 平均每次亏钱亏多少。
            - *计算*: `平均盈利金额 / 平均亏损金额`
            - *意义*: 通常 >1.5 为佳。如果胜率低，必须要有极高的盈亏比才能盈利。

        ### 2. 风险类指标
        - **最大回撤 (Max Drawdown)**: 历史上从最高点跌下来的最大幅度。
            - *计算*: `(当前净值 - 历史最高净值) / 历史最高净值` 的最小值。
            - *意义*: 越接近 0 越好。它代表了极端情况下的最大本金损失风险。
        
        ### 3. 性价比指标 (风险调整后收益)
        - **夏普比率 (Sharpe Ratio)**: 承受每单位总风险能带来的超额回报。
            - *计算*: `(年化收益率 - 无风险利率) / 年化波动率`
            - *意义*: >1 为优秀，>2 为卓越。衡量策略的“稳定性”。
        - **索提诺比率 (Sortino Ratio)**: 承受每单位**下跌风险**能带来的超额回报。
            - *计算*: `(年化收益率 - 无风险利率) / 下跌波动率`
            - *意义*: 比夏普更科学，因为它不惩罚上涨带来的波动。
        - **卡玛比率 (Calmar Ratio)**: 收益与最大回撤的比值。
            - *计算*: `年化收益率 / |最大回撤|`
            - *意义*: 衡量“回撤修复能力”。Calmar=2 意味着遭受 10% 的回撤只需半年就能涨回来。

        ### 4. 其他指标
        - **曲线平滑度 (Equity R²)**: 资金曲线与完美指数增长曲线的拟合度。
            - *意义*: 越接近 1.0，说明资金曲线越像一条直线，回撤越小，增长越稳。
        - **还原度 (Match Rate)**: 与原始策略持仓记录的重合天数比例。
            - *意义*: 越高说明该参数组合越接近原始策略的真实逻辑。
        """)

    st.markdown("""
    ### 📝 综合评价指南
    
    1. **看卡玛比率 (Calmar)**: 当收益率接近时，优先选择**卡玛比率更高**的参数。这意味着为了获得同样的收益，你需要忍受的回撤更小。
    2. **看参数平原**: 在热力图中，选择颜色斑块**连成一片**的区域，而不是孤立的亮点。这代表策略在该参数区间内具有**鲁棒性**。
    3. **看交易频率**: 在收益和风险都差不多的情况下，选择**交易次数更少**的参数（通常对应较大的 Buffer），可以减少滑点和手续费损耗。
    
    ### 💡 发现
    - **Window=25** 在还原度图表中呈现压倒性优势（深色区域）。
    - **Cutoff=300** 在收益率图表中呈现显著优势，但在还原度上较弱。
    - **Crash Filter** 开启后通常会导致点位向左下方移动（收益降低，回撤改善不明显），属于“负优化”。
    """)

st.divider()

st.header("🔄 归一化逻辑深度优化分析")

st.markdown("""
### 1. 为什么调整？(The Problem)
在原有的逻辑中，**过热标的**（得分远超熔断阈值，例如 Score=1000）虽然不会被买入，但它们**参与了归一化分数的计算**，作为分母中的 `Max` 值。

*   **后果**：
    *   **挤压效应**：一个 1000 分的异常值会将 0-100 的分数区间拉得极大。
    *   **失真**：导致其他正常上涨的标的（例如 Score=200）归一化后的分数被压缩得很低（可能只有 20 分），且彼此之间的分差极小。
    *   **迟钝**：策略对正常标的之间的强弱切换变得不敏感，错失轮动机会。

### 2. 优化方案 (The Solution)
**“归一化时剔除过热标的”**：在计算归一化分数之前，先将得分超过熔断阈值的标的从池中剔除。

*   **优势**：
    *   **还原真实分差**：剩余标的（Score=200）可能直接成为新的基准（100分）。
    *   **提升敏感度**：放大了正常标的之间的分差，使策略能更敏锐地捕捉到领涨资产的切换。
    *   **适应性更强**：在局部牛市中，策略不会因为某个龙头的极端暴涨而对其他板块“视而不见”。

### 3. 绩效对比与最优参数 (The Result)
基于新模式的全量网格搜索（2019-至今），我们得出了显著优于旧模式的参数组合。

#### 🏆 综合冠军组合
*   **配置**: Window=**20**, Cutoff=**600**, Buffer=**5**
*   **年化收益**: **56.0%**
*   **夏普比率**: **2.00**
*   **最大回撤**: **-17.6%**
*   **卡玛比率**: **3.18**

#### 🥈 稳健/低频组合
*   **配置**: Window=**25**, Cutoff=**400**, Buffer=**15**
*   **年化收益**: **55.5%**
*   **夏普比率**: **1.82**
*   **交易次数**: 429次 (比冠军组合少 22%)

> **核心发现**: 
> 1. **阈值上移 (300 -> 600)**：新模式消除了过热标的的挤压效应，允许我们在 300-600 的强势区间内持有更久，吃到更多主升浪。
> 2. **缓冲收窄 (Buffer 8 -> 5)**：分差被真实还原后，我们可以使用更小的缓冲区来提高轮动效率，而不用担心噪音干扰。
""")

