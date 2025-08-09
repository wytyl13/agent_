#!/bin/bash

# 从命令行参数获取 PROJECT_ROOT，如果未提供，则使用现有方式

if [ -f ".env" ]; then
    CHAINLIT_PORT=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('CHAINLIT_PORT', '5002'))
")
    CONDA_ENVIRONMENT=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('CONDA_ENVIRONMENT', 'community_agent'))
")
    CONDA_ENV_PATH=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('CONDA_ENV_PATH', '/work/soft/anaconda3/bin/'))
")
else
    CHAINLIT_PORT=5002
    CONDA_ENVIRONMENT='community_agent'
    CONDA_ENV_PATH='/work/soft/anaconda3/bin/'
fi


if [ -n "$1" ]; then
    PROJECT_ROOT="$1"
else
    PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
fi

check_and_kill_port() {
    local port=$1
    local pid=$(sudo lsof -t -i :$port)

    if [ -n "$pid" ]; then
        echo "端口 $port 已被占用，正在终止进程 $pid"
        sudo kill $pid
    else
        echo "端口 $port 未被占用"
    fi
}

check_and_kill_port $CHAINLIT_PORT

# 激活虚拟环境
CONDA_ENV='/work/soft/anaconda3/bin/'
export PATH=$CONDA_ENV:$PATH
eval "$(conda shell.bash hook)"
conda init bash
conda activate $CONDA_ENVIRONMENT

timestamp=$(date +"%Y%m%d%H%M%S")
LOG_PATH=$PROJECT_ROOT/logs/chainlit
LOG_FILE="$LOG_PATH/${timestamp}.log"

if [ ! -d "$LOG_PATH" ]; then
    # 目录不存在，创建它
    mkdir -p "$LOG_PATH"
fi

echo "日志文件路径: $LOG_FILE"
cd "$PROJECT_ROOT" || { echo "无法切换到项目目录: $PROJECT_ROOT"; exit 1; }
BGE_MODEL_PATH=$PROJECT_ROOT/models/embedding
mkdir -p "$BGE_MODEL_PATH"
python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='$BGE_MODEL_PATH')
"
apt update && apt install git -y
nohup chainlit run chainlit.py --host 0.0.0.0 --port $CHAINLIT_PORT --ssl-cert $PROJECT_ROOT/cert/shunxikj.com.crt --ssl-key $PROJECT_ROOT/cert/shunxikj.com.key --headless > "$LOG_FILE" 2>&1 &
echo "检测脚本已在后台运行，输出日志位于: $LOG_FILE"