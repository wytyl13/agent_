# Agent 核心库

🚀 基于Docker的智能Agent核心库，提供完整的后端API服务和前后端分离架构。

## 📋 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [架构设计](#架构设计)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [开发指南](#开发指南)

## ✨ 特性

- 🐳 **Docker容器化部署** - 完整的Docker部署方案，支持GPU加速
- 🔧 **模块化架构** - 基于Agent核心库的可扩展架构
- 🗄️ **多数据库支持** - 兼容PostgreSQL和MySQL数据库
- 🔌 **前后端分离** - RESTful API设计，支持多种参数传递方式
- ⚡ **GPU加速** - 内置CUDA支持，提升AI模型性能
- 🛠️ **工具基类** - 提供完整的工具开发基础框架

## 🚀 快速开始

### 前置要求

- Docker & Docker Compose
- NVIDIA GPU驱动（用于GPU加速）
- 8GB+ 内存推荐

### 部署步骤

#### 1. 准备环境

```bash
# 拉取NVIDIA CUDA基础镜像
docker pull nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 创建Docker网络
docker network create app-network
```

#### 2. 启动Agent容器

```bash
docker run -d \
  --name agent \
  --gpus all \
  --cpus="$(nproc)" \
  --memory="$(free -b | awk '/^Mem:/{printf "%.0f", $2*0.4}')" \
  --shm-size=16g \
  --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --privileged \
  -p 8891:8891 \
  -p 5004:5004 \
  -it \
  nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 \
  /bin/bash
```

#### 3. 启动PostgreSQL数据库

```bash
# 运行PostgreSQL初始化脚本
sudo bash postgres_docker_contanier_init.sh postgres_dev 5431
```

#### 4. 配置网络连接

```bash
# 将容器连接到网络
docker network connect app-network agent
docker network connect app-network postgres_dev
```

#### 5. 环境配置

```bash
# 进入Agent容器
docker exec -it agent bash

# 系统更新和依赖安装
apt update && apt install wget lsof git

# 创建工作目录
mkdir -p /work/soft
mkdir -p /work/ai

# 下载和安装Anaconda
cd /work/soft
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh

# 初始化conda环境
/work/soft/anaconda3/bin/conda init bash
source ~/.bashrc  # 可选
```

#### 6. 部署应用

```bash
# 克隆项目代码
cd /work/ai
git clone https://github.com/wytyl13/agent_.git

# 安装Python依赖
cd agent_
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/

# 启动服务
bash api.sh       # 启动API服务
bash chainlit.sh  # 启动Chainlit服务
```

### 🎯 访问服务

- **API服务**: http://localhost:8891
- **Chainlit界面**: http://localhost:5004

## 🏗️ 架构设计

### 核心原则

1. **Agent核心库驱动** - 所有功能基于Agent库构建，包含：
   - 🔧 工具基类
   - 🗄️ 数据库配置基类  
   - 🤖 大语言模型配置基类
   - 🛠️ 完整工具开发框架

2. **前后端分离架构** - 页面之外的所有功能以服务形式提供

3. **API服务设计** - 包含表结构信息和服务编写，支持：
   - URL参数传递
   - JSON请求体传递
   - 依赖项目核心库

4. **数据库抽象层** - SQL Provider基类特性：
   - 兼容PostgreSQL和MySQL
   - 统一deleted字段处理（整形存储，支持布尔值传递）
   - SmallInteger (PostgreSQL) / TINYINT (MySQL)

## 📁 项目结构

```
agent_/
├── agent/                 # Agent核心库
├── api/                   # API服务层
├── logs/                  # 日志文件
├── models/                # AI模型文件
├── retrieval_data/        # 检索数据
├── retrieval_storage/     # 检索存储
├── upload_dir/           # 上传文件目录
├── public/               # 静态资源
├── cert/                 # 证书文件
├── .env                  # 环境配置文件
├── requirements.txt      # Python依赖
├── api.sh               # API服务启动脚本
└── chainlit.sh          # Chainlit服务启动脚本
```

## ⚙️ 配置说明

所有动态配置都在根目录下的 `.env` 文件中进行管理。

### 环境变量示例

```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@postgres_dev:5432/dbname

# API配置
API_HOST=0.0.0.0
API_PORT=8891

# GPU配置
CUDA_VISIBLE_DEVICES=0

# 模型配置
MODEL_PATH=/work/ai/agent_/models
```

## 📚 API文档

API服务提供RESTful接口，支持多种参数传递方式：

### 请求方式
- **URL参数**: `GET /api/endpoint?param1=value1&param2=value2`
- **JSON请求体**: `POST /api/endpoint` with JSON payload

### 响应格式
```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

## 🔧 开发指南

### 添加新工具

1. 继承工具基类
2. 实现必要的接口方法
3. 在Agent核心库中注册

### 数据库操作

使用SQL Provider基类进行数据库操作：

```python
from agent.sql_provider import SQLProvider

# 创建数据库连接
provider = SQLProvider()

# 支持布尔值deleted字段
provider.create(table_name, data={'deleted': False})
```

### 自定义配置

在 `.env` 文件中添加新的环境变量，然后在代码中引用：

```python
import os
from dotenv import load_dotenv

load_dotenv()
custom_config = os.getenv('CUSTOM_CONFIG')
```

## 🤝 贡献

欢迎提交Pull Request和Issue来改进这个项目！

## 📄 许可证

此项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

⭐ 如果这个项目对你有帮助，请给它一个Star！