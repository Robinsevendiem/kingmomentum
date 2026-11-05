#!/bin/bash

# 自动准备项目虚拟环境并启动本地 Streamlit 页面。
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    BOOTSTRAP_PYTHON=""
    for CANDIDATE in \
        "$(command -v python3.11 2>/dev/null || true)" \
        "${HOME}/.local/bin/python3.11" \
        "/opt/homebrew/bin/python3.11" \
        "/usr/local/bin/python3.11"; do
        if [ -x "$CANDIDATE" ] && "$CANDIDATE" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' >/dev/null 2>&1; then
            BOOTSTRAP_PYTHON="$CANDIDATE"
            break
        fi
    done

    if [ -z "$BOOTSTRAP_PYTHON" ]; then
        echo "未找到 Python 3.11，无法创建项目虚拟环境。"
        echo "请安装 Python 3.11 后重新运行此启动器。"
        echo "按回车键关闭窗口。"
        read
        exit 1
    fi

    echo "正在创建项目虚拟环境……"
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
    echo "正在按 requirements.txt 安装项目依赖（首次运行需要联网）……"
    if ! "$PYTHON_BIN" -m pip install -r "$PROJECT_DIR/requirements.txt"; then
        echo "依赖安装失败。请检查网络后重新运行此启动器。"
        echo "按回车键关闭窗口。"
        read
        exit 1
    fi
fi

PORT=$("$PYTHON_BIN" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')

echo "正在启动 ETF 动量策略回测系统……"
echo "本地地址：http://127.0.0.1:${PORT}"
exec "$PYTHON_BIN" -m streamlit run "$PROJECT_DIR/Home.py" --server.address 127.0.0.1 --server.port "$PORT"
