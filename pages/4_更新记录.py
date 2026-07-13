import streamlit as st


st.set_page_config(page_title="更新记录", page_icon="📝", layout="wide")


UPDATE_LOGS = [
    {
        "date": "2026-07-13",
        "category": "数据修复",
        "title": "修复 ETF 拆分导致的复权断层与异常成交记录",
        "details": [
            "修复 Tushare 基金数据在份额拆分场景下的复权处理问题，更新逻辑改为“原始价增量 + 基于复权因子全量重建前复权列”。",
            "新增统一数据更新模块 `data_update_utils.py`，同时接入 Home 页与本地批量更新脚本。",
            "已重建当前轮动池全部本地历史数据，并确认 `588120.SH` 在 2026-06-24 拆分前后的前复权价格序列恢复连续。",
            "过滤回测中由浮点残差产生的极小成交，避免交易记录与持仓明细出现 `shares/fee/amount=0` 的噪音记录。",
        ],
    },
    {
        "date": "2026-07-12",
        "category": "页面优化",
        "title": "优化交易时点对比页面展示",
        "details": [
            "将收益曲线与回撤曲线拆分为两张独立图表，展示方式与 Home 回测系统保持一致。",
            "保留净值曲线、相对净值曲线、交易记录、持仓明细和标的交易详情可视化。",
        ],
    },
    {
        "date": "2026-06-30",
        "category": "策略分析",
        "title": "增强交易时点对比能力",
        "details": [
            "新增独立页面“交易时点对比”，用于对比“信号日收盘成交”和“原始策略(T+1 开盘成交)”的绩效差异。",
            "补充绩效对比表、收益曲线、回撤曲线、相对净值曲线。",
            "增加两种执行方式下的交易记录、持仓明细，以及单标的买卖点复盘图。",
        ],
    },
    {
        "date": "2026-06-28",
        "category": "缓存修复",
        "title": "修复交易时点对比页面的数据滞后问题",
        "details": [
            "为历史 CSV 增加文件签名缓存键，签名包含文件路径、修改时间和大小。",
            "Home 页更新数据后，交易时点对比页面会自动失效缓存并读取最新历史数据。",
        ],
    },
    {
        "date": "2026-06-25",
        "category": "策略功能",
        "title": "扩展轮动策略持仓与资产池管理",
        "details": [
            "支持用户自由增减参与轮动的标的，并支持输入代码新增标的。",
            "默认保留 9 个内置标的，新增 `custom_assets.json` 和 `active_pool.json` 持久化配置。",
            "支持设置“持有前 N 名标的”等权轮动，并保留熔断阈值、归一化和换仓缓冲逻辑。",
        ],
    },
    {
        "date": "2026-06-22",
        "category": "项目整理",
        "title": "整理部署结构并适配 GitHub + Streamlit",
        "details": [
            "归档非线上运行必需的分析脚本与历史材料，保留最小线上运行目录。",
            "更新 README、`.gitignore` 和启动结构，便于 GitHub 与 Streamlit 部署。",
            "新增 Token 配置说明，支持本地 `.streamlit/secrets.toml` 与 Streamlit Cloud Secrets。",
        ],
    },
]


st.title("📝 更新记录")
st.caption("用于记录项目功能迭代、页面优化、数据修复与部署整理。")

all_categories = sorted({item["category"] for item in UPDATE_LOGS})
selected_categories = st.multiselect("筛选类型", options=all_categories, default=all_categories)

filtered_logs = [item for item in UPDATE_LOGS if item["category"] in selected_categories]

col1, col2, col3 = st.columns(3)
col1.metric("记录数", len(filtered_logs))
col2.metric("最新记录日期", filtered_logs[0]["date"] if filtered_logs else "-")
col3.metric("类型数", len({item["category"] for item in filtered_logs}) if filtered_logs else 0)

if not filtered_logs:
    st.info("当前筛选条件下暂无记录。")
else:
    for item in filtered_logs:
        with st.expander(f'{item["date"]} | {item["category"]} | {item["title"]}', expanded=item == filtered_logs[0]):
            for detail in item["details"]:
                st.markdown(f"- {detail}")

st.divider()
st.subheader("维护说明")
st.markdown(
    """
- 本页用于沉淀项目的重要更新、功能新增和问题修复。
- 后续新增记录时，直接在 `UPDATE_LOGS` 中按时间倒序追加即可。
"""
)
