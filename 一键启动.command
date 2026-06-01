#!/bin/bash

# 获取脚本所在目录并进入
cd "$(dirname "$0")"

echo "正在启动 ETF 动量策略回测系统..."

# 尝试检测 streamlit 命令
if command -v streamlit &> /dev/null; then
    streamlit run Home.py
elif [ -f "/opt/anaconda3/bin/streamlit" ]; then
    # 如果系统路径没有，尝试使用 Anaconda 路径 (根据历史记录推断)
    /opt/anaconda3/bin/streamlit run Home.py
else
    echo "❌ 错误: 未找到 streamlit。"
    echo "请确保您已安装 streamlit (pip install streamlit) 或将其添加到系统路径中。"
    echo "按下任意键退出..."
    read -n 1
    exit 1
fi
