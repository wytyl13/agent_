### quick start
```
1、下载docker镜像nvidia/cuda（含有cuda的镜像）

2、创建Agent docker容器
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
  -it \
  nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 \
  /bin/bash

3、进入docker容器进行配置
docker exec -it agent bash
# 下载anaconda
apt update && apt install wget lsof git
mkdir -p /work/soft
mkdir -p /work/ai

cd /work/soft
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash bash Anaconda3-2024.06-1-Linux-x86_64.sh

/work/soft/anaconda3/bin/conda init bash (可选)
source ~/.bashrc

cd /work/ai
git clone 

# 
```



### 准则
```
1、以agent库为核心（包含所有基类：工具基类、数据库配置基类、大语言模型配置基类和所有工具的编写）
2、前后端分离（所有后端皆以服务呈现）
3、api服务只包含对应的表结构信息和服务编写（api服务不单独存在，依赖于项目其他核心），api接口兼容url传递参数和JSON请求体传递参数
4、sql_provider兼容支持postgresql、mysql两种数据库（两种数据库中表字段deleted统一使用整形（SmallInteger和TINYINT））,sql_provider基类兼容传递deleted字段为布尔值
```