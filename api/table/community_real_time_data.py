#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异步PostgreSQL数据库操作示例
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, BigInteger, SmallInteger
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
import asyncio

from api.table.base import Base

class CommunityRealTimeData(Base):
    """
    社区实时数据表 - 异步版本
    """
    __tablename__ = 'community_real_time_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    type = Column(String(32), nullable=True, comment='类型')
    content = Column(Text, nullable=True, comment='内容')
    url = Column(String(255), nullable=True, comment='链接地址')
    creator = Column(String(64), nullable=True, comment='创建人')
    create_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新人')
    update_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment='更新时间')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')
    deleted = Column(SmallInteger, default=0, nullable=False, comment='是否删除')

