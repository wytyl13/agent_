#!/bin/bash

# PostgreSQL容器初始化脚本
# 使用方法: sudo bash postgres_init.sh <容器名称> <端口号>
# 例如: sudo bash postgres_init.sh postgres_app1 5432

set -e

# 检查参数
if [ $# -ne 2 ]; then
    echo "使用方法: $0 <容器名称> <端口号>"
    echo "例如: $0 postgres_app1 5432"
    exit 1
fi

CONTAINER_NAME=$1
PORT=$2
PASSWORD="123456"
DATABASE_NAME="postgres"
DATA_PATH="/mnt/data/$CONTAINER_NAME"

echo "正在初始化PostgreSQL容器..."
echo "容器名称: $CONTAINER_NAME"
echo "端口: $PORT"
echo "数据路径: $DATA_PATH"

# 检查端口是否被占用
if netstat -tuln | grep -q ":$PORT "; then
    echo "错误: 端口 $PORT 已被占用"
    exit 1
fi

# 检查容器名称是否已存在
if docker ps -a --format "table {{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
    echo "错误: 容器名称 '$CONTAINER_NAME' 已存在"
    exit 1
fi

# 创建数据目录
echo "创建数据目录: $DATA_PATH"
sudo mkdir -p $DATA_PATH
sudo chown -R 999:999 $DATA_PATH  # PostgreSQL容器使用999用户ID

echo "启动PostgreSQL容器..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart always \
    -e POSTGRES_PASSWORD=$PASSWORD \
    -e POSTGRES_DB=$DATABASE_NAME \
    -e PGPORT=$PORT \
    -v $DATA_PATH:/var/lib/postgresql/data \
    -p $PORT:$PORT \
    postgres:latest

# 等待容器健康检查通过
echo "等待PostgreSQL容器启动完成..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker exec $CONTAINER_NAME pg_isready -U postgres > /dev/null 2>&1; then
        echo "PostgreSQL容器已就绪"
        break
    fi
    echo "等待中... ($((attempt + 1))/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "错误: PostgreSQL容器启动超时"
    exit 1
fi

# 创建数据库表
echo "创建数据库表..."
docker exec -i $CONTAINER_NAME psql -U postgres -d $DATABASE_NAME << 'EOSQL'
-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL,
    "createdAt" TEXT
);

-- 创建线程表
CREATE TABLE IF NOT EXISTS threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

-- 创建步骤表
CREATE TABLE IF NOT EXISTS steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN DEFAULT FALSE,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

-- 创建元素表
CREATE TABLE IF NOT EXISTS elements (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

-- 创建反馈表
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" UUID PRIMARY KEY,
    "forId" UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

-- 显示创建的表
\dt
EOSQL

echo "=========================="
echo "PostgreSQL容器初始化完成！"
echo "容器名称: $CONTAINER_NAME"
echo "端口: $PORT"
echo "数据库连接信息:"
echo "  Host: localhost (或者你的服务器IP)"
echo "  Port: $PORT"
echo "  Database: $DATABASE_NAME"
echo "  Username: postgres"
echo "  Password: $PASSWORD"
echo ""
echo "连接字符串:"
echo "postgresql+asyncpg://postgres:$PASSWORD@localhost:$PORT/$DATABASE_NAME"
echo ""
echo "管理命令:"
echo "  停止容器: docker stop $CONTAINER_NAME"
echo "  启动容器: docker start $CONTAINER_NAME"
echo "  删除容器: docker rm -f $CONTAINER_NAME"
echo "  查看日志: docker logs $CONTAINER_NAME"
echo "  连接数据库: docker exec -it $CONTAINER_NAME psql -U postgres"
echo "=========================="
