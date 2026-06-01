# ETF Momentum Strategy Backtesting System

基于 `Streamlit` 的 ETF 动量轮动回测与分析系统，包含策略回测、最新持仓信号、动量分布分析和参数评估页面。

## 核心功能

- 策略回测：支持动量窗口、熔断阈值、归一化换仓缓冲、持有前 N 名等权等参数组合。
- 最新持仓：同步最新行情后，给出下一交易日建议持仓。
- 动量分析：观察不同动量得分对应的价格曲线与收益分布。
- 参数分析：读取预计算结果，展示多维绩效指标与稳健性结论。
- 标的池管理：支持用户自由增减参与轮动的标的，并可输入代码新增标的（自动下载历史数据并纳入轮动池）。

## 目录结构

```text
foolreveal/
├── Home.py                      # Streamlit 入口
├── pages/                       # 子页面
├── data/                        # 部署所需数据
│   ├── *_history.csv            # ETF 历史行情
│   ├── optimization_results.csv # 参数分析页依赖数据
│   ├── active_pool.json          # 当前轮动池（标的代码列表，用户配置）
│   └── custom_assets.json        # 自定义标的资产库（用户新增标的配置）
├── archive/                     # 非线上必需内容归档（默认建议不提交到仓库）
│   ├── analysis_results/         # 历史分析产物
│   ├── legacy_apps/              # 旧版存档
│   ├── scripts_research/         # 研究脚本
│   ├── scripts_tests/            # 旧测试脚本
│   ├── scripts_outputs/          # 脚本输出文件
│   ├── data_record/              # 回测导出的持仓/调仓记录（留痕）
│   └── data_reference/           # 原始策略对照数据（留痕）
├── scripts/                     # 离线分析/优化脚本（线上不调用）
├── docs/                        # 说明文档
├── .streamlit/
│   └── secrets.toml.example     # secrets 示例
├── requirements.txt
└── README.md
```

## 本地运行

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Streamlit Cloud 部署

1. 将仓库推送到 `GitHub`。
2. 在 [Streamlit Community Cloud](https://streamlit.io/cloud) 新建应用。
3. 仓库选择当前项目，`Main file path` 填写 `Home.py`。
4. 在应用后台 `Settings -> Secrets` 配置：

```toml
TUSHARE_TOKEN = "你的真实 Tushare Token"
```

5. 部署后应用会自动从 `st.secrets` 读取 Token；本地也可在侧边栏手动输入。

## 数据说明

- `data/*_history.csv`：网页运行和回测依赖的核心历史行情数据。
- `data/optimization_results.csv`：参数分析页读取的预计算结果。
- `data/active_pool.json`：当前轮动池（用户在侧边栏配置后持久化）。
- `data/custom_assets.json`：自定义标的资产库（用户输入代码新增标的后持久化）。
- `archive/`：历史分析产物与留痕数据归档，不影响主应用运行。

## 说明

- 仓库已移除代码中的硬编码 `Tushare Token`，适合直接公开到 `GitHub`。
- `scripts/` 中多数文件用于研究/离线分析/优化，不参与 `Streamlit` 线上页面启动。
