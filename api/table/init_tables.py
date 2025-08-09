#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/09 09:38
@Author  : weiyutao
@File    : init_tables.py
"""
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.exc import OperationalError

from api.table.community_real_time_data import CommunityRealTimeData
from api.table.user_data import UserData

from agent.config.sql_config import SqlConfig
from api.table.base import Base

ROOT_DIRECTORY = Path(__file__).parent.parent.parent
SQL_CONFIG_PATH = str(ROOT_DIRECTORY / "agent" / "config" / "yaml" / "postgresql_config.yaml")

sql_config = SqlConfig.from_file(SQL_CONFIG_PATH)


async def check_tables_exist():
    """检查表是否已存在"""
    engine = create_async_engine(sql_config.sql_url)
    
    async with engine.begin() as conn:
        # 获取所有需要创建的表名
        table_names = [table.name for table in Base.metadata.tables.values()]
        
        # 检查每个表是否存在
        existing_tables = []
        for table_name in table_names:
            try:
                # 尝试查询表的存在性
                result = await conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
                existing_tables.append(table_name)
            except Exception:
                # 表不存在或查询失败
                pass
    
    await engine.dispose()
    return existing_tables


async def drop_all_tables():
    """删除所有表"""
    engine = create_async_engine(sql_config.sql_url)
    
    async with engine.begin() as conn:
        # 删除所有表
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()
    print("所有表已删除！")


async def create_all_tables():
    """异步创建所有表"""
    engine = create_async_engine(sql_config.sql_url)
    
    async with engine.begin() as conn:
        # 这会创建所有继承自Base的表
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("所有表创建完成！")


async def create_tables_with_check():
    """检查表是否存在，如果存在则询问用户是否删除重建"""
    existing_tables = await check_tables_exist()
    
    if existing_tables:
        print(f"检测到以下表已存在: {', '.join(existing_tables)}")
        
        while True:
            user_choice = input("是否要删除现有表并重新创建？(y/n): ").strip().lower()
            
            if user_choice in ['y', 'yes', '是']:
                print("正在删除现有表...")
                await drop_all_tables()
                print("正在重新创建表...")
                await create_all_tables()
                break
            elif user_choice in ['n', 'no', '否']:
                print("跳过表创建，使用现有表结构")
                break
            else:
                print("请输入 y(是) 或 n(否)")
    else:
        print("未检测到现有表，开始创建新表...")
        await create_all_tables()


async def create_default_users():
    """创建默认用户"""
    engine = create_async_engine(sql_config.sql_url)
    async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            # 检查用户是否已存在
            from sqlalchemy import select
            
            # 检查admin用户
            admin_result = await session.execute(
                select(UserData).where(UserData.username == "admin")
            )
            admin_user = admin_result.scalar_one_or_none()
            
            if not admin_user:
                # 创建admin用户
                admin_user = UserData(
                    username="admin",
                    password="admin",  # 注意：实际项目中应该加密密码
                    full_name="admin",
                    role="admin",
                    community="舜熙科技智慧养老社区",
                    status="active",
                    creator="system"
                )
                session.add(admin_user)
                print("创建admin用户成功")
            else:
                print("admin用户已存在")
            
            # 检查shunxikeji用户
            user_result = await session.execute(
                select(UserData).where(UserData.username == "shunxikeji")
            )
            regular_user = user_result.scalar_one_or_none()
            
            if not regular_user:
                # 创建shunxikeji用户
                regular_user = UserData(
                    username="shunxikeji",
                    password="shunxikeji",  # 注意：实际项目中应该加密密码
                    full_name="shunxikeji",
                    role="user",
                    community="舜熙科技智慧养老社区",
                    status="active",
                    creator="system"
                )
                session.add(regular_user)
                print("创建shunxikeji用户成功")
            else:
                print("shunxikeji用户已存在")
            
            # 提交事务
            await session.commit()
            print("默认用户创建完成！")
            
        except Exception as e:
            # 回滚事务
            await session.rollback()
            print(f"创建默认用户时发生错误: {e}")
        finally:
            await session.close()
    
    await engine.dispose()


async def init_database():
    """初始化数据库：创建表和默认用户"""
    print("开始初始化数据库...")
    
    try:
        # 1. 检查并创建表
        await create_tables_with_check()
        
        # 2. 创建默认用户
        await create_default_users()
        
        print("数据库初始化完成！")
    
    except Exception as e:
        print(f"数据库初始化过程中发生错误: {e}")
        raise
    
    
if __name__ == '__main__':
    import asyncio
    asyncio.run(init_database())