#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/24 11:35
@Author  : weiyutao
@File    : sql_provider.py
"""
import traceback
from sqlalchemy import select
from pydantic import BaseModel, model_validator, ValidationError
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Generic,
    TypeVar,
    Any,
    Type,
    List
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
from contextlib import contextmanager
import traceback
import urllib.parse
from sqlalchemy import and_
from datetime import datetime, date
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from contextlib import asynccontextmanager
import asyncio

from agent.provider.base_provider import BaseProvider
from agent.config.sql_config import SqlConfig

# 定义基类
class Base(DeclarativeBase):
    pass

# 定义泛型类型变量
ModelType = TypeVar("ModelType", bound=Base)


class SqlProvider(BaseProvider, Generic[ModelType]):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    sql_connection: Optional[sessionmaker] = None
    model: Type[ModelType] = None
    database_type: str = "mysql"
    
    def __init__(
        self, 
        model: Type[ModelType] = None,
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None
    ) -> None:
        super().__init__()
        self._init_param(sql_config_path, sql_config, model)
    
    
    def _init_param(self, sql_config_path: Optional[str] = None, sql_config: Optional[SqlConfig] = None, model : Type[ModelType] = None):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.sql_config = SqlConfig.from_file(self.sql_config_path) if self.sql_config is None and self.sql_config_path is not None else self.sql_config
        
        # 设置数据库类型
        if self.sql_config and hasattr(self.sql_config, 'database_type'):
            self.database_type = self.sql_config.database_type.lower()
        else:
            self.database_type = "mysql"  # 默认值，保证向后兼容
        
        # 标准化postgresql类型名称
        if self.database_type == "postgres":
            self.database_type = "postgresql"
        
        
        # 验证数据库类型
        if self.database_type not in ["mysql", "postgresql", "postgres"]:
            raise ValueError(f"Unsupported database type: {self.database_type}. Supported types: mysql, postgresql")
        
        # if self.sql_config is None and self.data is None:
        #     raise ValueError("config config_path and data must not be null!")
        self.sql_connection = self.get_sql_connection() if self.sql_connection is None else self.sql_connection
        self.model = model
        if self.model is None:
            raise ValueError("model must not be null!")

    
    def get_sql_connection(self):
        try:
            sql_info = self.sql_config
            username = sql_info.username
            database = sql_info.database
            password = sql_info.password
            host = sql_info.host
            port = sql_info.port
        except Exception as e:
            raise ValueError(f"fail to init the sql connect information!\n{self.sql_config}") from e
        # 因为url中的密码可能存在冲突的字符串，因此需要在进行数据库连接前对其进行编码
        # urllib.parse.quote_plus() 函数将特殊字符替换为其 URL 编码的对应项。例如，! 变为 %21，@ 变为 %40。这确保了密码被视为单个字符串，并且不会破坏 URL 语法。
        encoded_password = urllib.parse.quote_plus(password)
        # 根据数据库类型构建不同的连接字符串
        if self.database_type == "mysql":
                # MySQL 异步驱动
            database_url = f"mysql+aiomysql://{username}:{encoded_password}@{host}:{port}/{database}"
            # 或者使用 asyncmy：
            # database_url = f"mysql+asyncmy://{username}:{encoded_password}@{host}:{port}/{database}"
        elif self.database_type == "postgresql":
            database_url = f"postgresql+asyncpg://{username}:{encoded_password}@{host}:{port}/{database}"
        
        try:
            # 两种数据库都使用异步引擎
            engine = create_async_engine(
                database_url, 
                pool_size=10, 
                max_overflow=20,
                pool_pre_ping=True
            )
            SessionLocal = async_sessionmaker(bind=engine)
            return SessionLocal
        except Exception as e:
            raise ValueError("fail to create the sql connector engine!") from e
    
    
    def set_model(self, model: Type[ModelType] = None):
        """reset model"""
        if model is None:
            raise ValueError('model must not be null!')
        self.model = model
    
    
    @asynccontextmanager
    async def get_db_session(self):
        """提供数据库会话的上下文管理器"""
        if not self.sql_connection:
            raise ValueError("Database connection not initialized")
        
        # 统一异步处理，不区分数据库类型
        async with self.sql_connection() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
    
    
    async def add_record(self, data: Dict[str, Any]) -> int:
        """添加记录"""
        async with self.get_db_session() as session:
            try:
                record = self.model(**data)
                session.add(record)
                session.flush()  # 刷新以获取ID
                record_id = record.id
                return record_id
            except Exception as e:
                error_info = f"Failed to add record: {e}"
                self.logger.error(error_info)
                self.logger.error(traceback.print_exc())
                raise ValueError(error_info) from e
    
    
    async def bulk_insert_with_update(self, data_list: List[Dict[str, Any]]) -> int:
        """批量插入，遇到重复数据时覆盖旧数据"""
        if not data_list:
            return 0
        
        try:
            from sqlalchemy import text
            
            table_name = self.model.__tablename__
            sample_data = data_list[0]
            columns = [col for col in sample_data.keys() if col != 'id']
            columns_str = ', '.join(columns)
            
            success_count = 0
            
            async with self.get_db_session() as session:
                for data in data_list:
                    try:
                        clean_data = {k: v for k, v in data.items() if k != 'id'}
                        
                        # 构建单条插入SQL（使用新语法）
                        placeholders = ', '.join([f':{col}' for col in columns])
                        updates = ', '.join([f'{col} = VALUES({col})' for col in columns])
                        
                        # 兼容不同MySQL版本的写法
                        if self.database_type == "mysql":
                            # MySQL 语法
                            sql = f"""
                            INSERT INTO {table_name} ({columns_str})
                            VALUES ({placeholders})
                            ON DUPLICATE KEY UPDATE {updates}
                            """
                        elif self.database_type == "postgresql":
                            # PostgreSQL 语法 (需要指定冲突字段，假设是主键 id)
                            updates_pg = ', '.join([f'{col} = EXCLUDED.{col}' for col in columns])
                            sql = f"""
                            INSERT INTO {table_name} ({columns_str})
                            VALUES ({placeholders})
                            ON CONFLICT (id) DO UPDATE SET {updates_pg}
                            """
                        
                        session.execute(text(sql), clean_data)
                        success_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"插入失败: {e}, 数据: {clean_data}")
                        continue
                
                session.commit()
            
            self.logger.info(f"批量插入/更新完成: {success_count}/{len(data_list)} 条成功")
            return success_count
            
        except Exception as e:
            self.logger.error(f"批量插入/更新失败: {e}")
            return 0
    
    
    
    async def bulk_insert_with_update_bake(self, data_list: List[Dict[str, Any]]) -> int:
        """批量插入，遇到重复数据时覆盖旧数据"""
        if not data_list:
            return 0
        
        try:
            from sqlalchemy import text
            
            # 获取表名
            table_name = self.model.__tablename__
            
            # 构建字段列表（排除自增主键id）
            sample_data = data_list[0]
            columns = [col for col in sample_data.keys() if col != 'id']
            columns_str = ', '.join(columns)
            
            # 构建VALUES占位符
            values_placeholder = ', '.join([f':{col}' for col in columns])
            
            # 构建UPDATE部分（覆盖所有字段）
            update_assignments = []
            for col in columns:
                update_assignments.append(f'{col} = VALUES({col})')
            update_str = ', '.join(update_assignments)
            
            # 构建完整SQL
            sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({values_placeholder})
            ON DUPLICATE KEY UPDATE {update_str}
            """
            
            success_count = 0
            async with self.get_db_session() as session:
                for data in data_list:
                    try:
                        # 移除id字段（如果存在）
                        clean_data = {k: v for k, v in data.items() if k != 'id'}
                        session.execute(text(sql), clean_data)
                        success_count += 1
                    except Exception as e:
                        self.logger.error(f"插入失败: {e}")
                        continue
                
                session.commit()
            
            self.logger.info(f"批量插入/更新完成: {success_count}/{len(data_list)} 条成功")
            return success_count
            
        except Exception as e:
            self.logger.error(f"批量插入/更新失败: {e}")
            return 0
    
    
    async def delete_record(self, record_id: int, hard_delete: bool = False) -> bool:
        """软删除记录"""
        async with self.get_db_session() as session:
            try:
                stmt = select(self.model).where(
                    self.model.id == record_id,
                    self.model.deleted == False
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                return {key: value for key, value in record.__dict__.items() 
                        if key != '_sa_instance_state'} if record else None
            except Exception as e:
                error_info = f"Failed to get record by id: {record_id}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    
    def update_record(self, record_id: int, data: Dict[str, Any]) -> bool:
        """更新记录"""
        with self.get_db_session() as session:
            try:
                result = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).update(data)
                return result > 0
            except Exception as e:
                error_info = f"Failed to update record {record_id} with data: {data}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e


    async def update_record_enhanced(self, record_id: int, data: Dict[str, Any], return_updated: bool = True) -> Optional[Dict[str, Any]]:
        """
        增强版更新记录函数
        
        Args:
            record_id (int): 要更新的记录ID
            data (Dict[str, Any]): 包含要更新字段的字典
            return_updated (bool): 是否返回更新后的记录，默认为True
            
        Returns:
            Optional[Dict[str, Any]]: 如果return_updated为True，返回更新后的记录字典；否则返回None
            
        Raises:
            ValueError: 当记录不存在、数据为空或更新失败时抛出
        """
        async with self.get_db_session() as session:
            try:
                if not data:
                    raise ValueError("更新数据不能为空")
                
                # 查询要更新的记录
                record = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).first()
                
                if not record:
                    raise ValueError(f"ID为 {record_id} 的记录不存在或已被删除")
                
                # 过滤掉不存在的字段
                valid_data = {}
                invalid_fields = []
                
                for key, value in data.items():
                    if hasattr(self.model, key):
                        # 跳过主键字段
                        if key != 'id':
                            valid_data[key] = value
                    else:
                        invalid_fields.append(key)
                
                if invalid_fields:
                    self.logger.warning(f"以下字段在模型中不存在，将被忽略: {invalid_fields}")
                
                if not valid_data:
                    raise ValueError("没有有效的字段需要更新")
                
                # 执行更新操作
                result = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).update(valid_data)
                
                if result == 0:
                    raise ValueError(f"更新失败，记录ID {record_id} 不存在")
                
                # 如果需要返回更新后的记录
                if return_updated:
                    session.commit()
                    updated_record = session.query(self.model).filter(
                        self.model.id == record_id
                    ).first()
                    
                    if updated_record:
                        return {
                            key: value for key, value in updated_record.__dict__.items() 
                            if key != '_sa_instance_state'
                        }
                
                return None
                
            except Exception as e:
                session.rollback()
                error_info = f"更新记录失败 ID: {record_id}, 数据: {data}, 错误: {str(e)}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e


    async def upsert_record_by_unique_field(
        self, 
        unique_field: Union[str, List[str]] = None, 
        data: Dict[str, Any] = None,
        db_model: Type[Base] = None
    ) -> Dict[str, Any]:

        db_model = self.model if db_model is None else db_model
        """
        根据唯一字段进行记录的更新或插入
        
        Args:
            unique_field (str): 用于判断记录唯一性的字段名
            data (Dict[str, Any]): 要插入或更新的数据字典
            db_model (Type[Base]): 数据库模型类
        
        Returns:
            Dict[str, Any]: 插入或更新后的记录
        """
        
        def convert_numpy_types(value):
            """转换numpy数据类型为Python原生类型"""
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, np.bool_):
                return bool(value)
            return value
        
        
        async with self.get_db_session() as session:
            try:
                
                if not data:
                    raise ValueError("Empty data dictionary provided")
                
                # 转换数据类型
                converted_data = {
                    key: convert_numpy_types(value)
                    for key, value in data.items()
                }

                # 将单个字段转换为列表，统一处理
                unique_fields = [unique_field] if isinstance(unique_field, str) else unique_field
                
                # 检查唯一字段是否存在于模型中
                for field in unique_fields:
                    if not hasattr(db_model, field):
                        raise ValueError(f"Unique field {field} not found in model")
                
                # 构建唯一键的查询条件
                filter_conditions = []
                for field in unique_fields:
                    field_value = converted_data.get(field)
                    if field_value is None:
                        raise ValueError(f"Unique field {field} value is None")
                    filter_conditions.append(getattr(db_model, field) == field_value)
                
                # 添加未删除条件
                filter_conditions.append(db_model.deleted == False)
                
                # 查询是否存在记录
                existing_record = session.query(db_model).filter(
                    and_(*filter_conditions)
                ).first()
                
                # 构建要更新的数据字典
                valid_data = {
                    key: value 
                    for key, value in converted_data.items() 
                    if hasattr(db_model, key) and key != 'id'  # 排除id和不存在的字段
                }
                if not valid_data:
                    raise ValueError("No valid fields to update")
                
                # 如果记录已存在，更新记录
                if existing_record:
                    for key, value in valid_data.items():
                        setattr(existing_record, key, value)
                    record = existing_record
                
                # 如果记录不存在，创建新记录
                else:
                    # 移除可能的id字段，防止主键冲突
                    record = db_model(**valid_data)
                    session.add(record)
                
                # 提交事务
                session.commit()
                session.refresh(record)
                
                # 转换为字典返回
                result = {}
                for key in valid_data.keys():
                    value = getattr(record, key)
                    # 处理SQLAlchemy对象关系
                    if hasattr(value, '__table__'):
                        continue  # 跳过关联对象
                    result[key] = value
                
                return result
            except Exception as e:
                session.rollback()
                if isinstance(unique_field, str):
                    error_info = f"Failed to upsert record with {unique_field}={data.get(unique_field)}"
                else:
                    unique_values = {field: data.get(field) for field in unique_field}
                    error_info = f"Failed to upsert record with unique fields: {unique_values}"
                self.logger.error(f"{error_info}. Error: {str(e)}")
                raise ValueError(error_info) from e
    

    async def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """根据ID查询记录"""
        async with self.get_db_session() as session:
            try:
                record = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).first()
                return record.__dict__ if record else None
            except Exception as e:
                error_info = f"Failed to get record by id: {record_id}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    
    async def get_record_by_condition_bake(
        self, 
        condition: Optional[Dict[str, Any]],
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        async with self.get_db_session() as session:
            try:
                
                # 获取模型的所有字段
                all_fields = [column.key for column in self.model.__table__.columns]
                
                if fields:
                    # 如果指定了字段，只查询指定字段
                    query_fields = fields
                else:
                    query_fields = all_fields
                
                # 排除不需要的字段
                if exclude_fields:
                    query_fields = [f for f in query_fields if f not in exclude_fields]
                    
                # 构建查询条件
                query = session.query(*[getattr(self.model, field) for field in query_fields])
                
                # 添加未删除条件
                query = query.filter(self.model.deleted == False)

                # Apply filters based on the provided condition
                if condition:
                    for key, value in condition.items():
                        # Assuming that keys in condition match the model's attributes
                        query = query.filter(getattr(self.model, key) == value)

                # Apply date range filter
                if date_range:
                    date_field = date_range.get('date_field')
                    start_date_str = date_range.get('start_date')
                    end_date_str = date_range.get('end_date')

                    if not date_field:
                        raise ValueError("date_field must be specified for date range filtering")

                    # Convert string dates to datetime objects
                    try:
                        if start_date_str:
                            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                            query = query.filter(getattr(self.model, date_field) >= start_date)
                        
                        if end_date_str:
                            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                            query = query.filter(getattr(self.model, date_field) <= end_date)
                    except ValueError as e:
                        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. {str(e)}")


                # 执行查询
                records = query.all()

                # 处理查询结果
                if not records:
                    return []
                
                # 返回查询结果
                return [dict(zip(query_fields, record)) for record in records]
                # if fields:
                #     # 如果指定了字段，返回包含指定字段的字典列表
                #     return [dict(zip(fields, record)) for record in records]
                # else:
                #     return [{
                #         key: value 
                #         for key, value in record.__dict__.items() 
                #         if key != '_sa_instance_state'
                #         } for record in records]
            except Exception as e:
                error_info = f"Failed to get records by condition: {condition}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    
    async def get_record_by_condition(
        self, 
        condition: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        增强版条件查询函数 - 支持精确到秒的时间查询
        
        Args:
            condition: 查询条件字典 {'device_id': 'DEV001', 'state': '呼吸暂停'}
            fields: 指定返回字段列表 ['timestamp', 'state', 'heart_bpm']
            exclude_fields: 排除字段列表 ['id', 'create_time']
            date_range: 日期范围查询
                {
                    'date_field': 'timestamp',  # 日期字段名
                    'start_date': '2025-06-27 15:30:45',  # 开始时间
                    'end_date': '2025-06-27 16:30:45'     # 结束时间
                }
        
        支持的时间格式：
            - '2025-06-27 15:30:45' (精确到秒)
            - '2025-06-27 15:30' (精确到分钟)
            - '2025-06-27' (整天范围)
            - '1751011266.382772' (时间戳)
        
        Returns:
            List[Dict]: 查询结果列表
        """
        async with self.get_db_session() as session:
            try:
                # 获取模型的所有字段
                all_fields = [column.key for column in self.model.__table__.columns]
                
                if fields:
                    # 如果指定了字段，只查询指定字段
                    query_fields = fields
                else:
                    query_fields = all_fields
                
                # 排除不需要的字段
                if exclude_fields:
                    query_fields = [f for f in query_fields if f not in exclude_fields]
                    
                # 构建查询条件
                query = session.query(*[getattr(self.model, field) for field in query_fields])
                
                # 添加未删除条件
                # query = query.filter(self.model.deleted == False)

                # 应用基础查询条件
                if condition:
                    for key, value in condition.items():
                        if key == 'deleted' and isinstance(value, bool):
                            value = 1 if value else 0
                        # 支持范围查询
                        if isinstance(value, dict) and ('min' in value or 'max' in value):
                            field_attr = getattr(self.model, key)
                            if 'min' in value:
                                query = query.filter(field_attr >= value['min'])
                            if 'max' in value:
                                query = query.filter(field_attr <= value['max'])
                        # 支持列表查询 (IN 操作)
                        elif isinstance(value, (list, tuple)):
                            query = query.filter(getattr(self.model, key).in_(value))
                        # 普通等值查询
                        else:
                            query = query.filter(getattr(self.model, key) == value)

                # 🔧 增强版日期范围过滤 - 支持精确到秒
                if date_range:
                    date_field = date_range.get('date_field')
                    start_date_str = date_range.get('start_date')
                    end_date_str = date_range.get('end_date')

                    if not date_field:
                        import traceback
                        raise ValueError("date_field must be specified for date range filtering \n{traceback.format_exc()}")

                    try:
                        if start_date_str:
                            start_date = self._parse_datetime_unified(start_date_str, is_end_date=False)
                            query = query.filter(getattr(self.model, date_field) >= start_date)
                        
                        if end_date_str:
                            end_date = self._parse_datetime_unified(end_date_str, is_end_date=True)
                            query = query.filter(getattr(self.model, date_field) <= end_date)
                    except ValueError as e:
                        import traceback
                        raise ValueError(f"Invalid date format. {str(e)} \n{traceback.format_exc()}")

                # 执行查询
                records = query.all()

                # 处理查询结果
                if not records:
                    return []
                
                # 返回查询结果
                return [dict(zip(query_fields, record)) for record in records]
                
            except Exception as e:
                import traceback
                error_info = f"Failed to get records by condition: {condition}, \n {traceback.format_exc()}"
                self.logger.error(error_info)
                raise ValueError(f"{error_info}") from e


    def _parse_datetime_unified(self, datetime_str: str, is_end_date: bool = False) -> datetime:
        """
        统一的日期时间解析方法，支持多种格式
        
        支持的格式：
        - '2025-06-27' → 2025-06-27 00:00:00 (开始) 或 2025-06-27 23:59:59 (结束)
        - '2025-06-27 15:30:45' → 2025-06-27 15:30:45
        - '2025-06-27 15:30' → 2025-06-27 15:30:00
        - '1751011266.382772' → 时间戳转换
        - '1751011266' → 整数时间戳转换
        
        Args:
            datetime_str: 时间字符串
            is_end_date: 是否为结束时间（影响只有日期时的处理）
            
        Returns:
            datetime: 解析后的datetime对象
        """
        # 尝试解析时间戳（浮点数）
        try:
            timestamp = float(datetime_str)
            return datetime.fromtimestamp(timestamp)
        except ValueError:
            pass
        
        # 尝试解析整数时间戳
        try:
            timestamp = int(datetime_str)
            return datetime.fromtimestamp(timestamp)
        except ValueError:
            pass
        
        # 定义支持的日期格式（按精确度排序）
        formats = [
            '%Y-%m-%d %H:%M:%S.%f',  # 2025-06-27 15:30:45.123456
            '%Y-%m-%d %H:%M:%S',     # 2025-06-27 15:30:45
            '%Y-%m-%d %H:%M',        # 2025-06-27 15:30
            '%Y-%m-%d',              # 2025-06-27
            '%Y/%m/%d %H:%M:%S',     # 2025/06/27 15:30:45
            '%Y/%m/%d %H:%M',        # 2025/06/27 15:30
            '%Y/%m/%d',              # 2025/06/27
        ]
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(datetime_str, fmt)
                
                # 如果只有日期，需要特殊处理
                if fmt in ['%Y-%m-%d', '%Y/%m/%d']:
                    if is_end_date:
                        # 结束日期：设置为当天的23:59:59.999999
                        parsed_date = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                    # 开始日期：保持00:00:00（默认）
                
                return parsed_date
                
            except ValueError:
                continue
        
        # 所有格式都失败
        raise ValueError(
            f"Unsupported datetime format: '{datetime_str}'. "
            f"Supported formats: 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD HH:MM:SS', "
            f"'YYYY-MM-DD HH:MM:SS.fff', 'YYYY/MM/DD...', or timestamp"
        )

    
    def get_field_names_and_descriptions(self) -> Dict[str, str]:
        field_info = {}
        # 获取模型的所有字段
        for column in self.model.__table__.columns:
            # 假设中文描述存储在列的 doc 属性中
            # 如果没有中文描述，可以使用其他方法来获取
            field_info[column.name] = column.comment  if column.comment else "无描述"
        return field_info
    
    
    
    async def update_deep_health_advice_by_id(self, record_id: int, new_health_advice) -> Optional[Dict[str, Any]]:
        async with self.get_db_session() as session:
            try:
                # 查询要更新的记录
                record = session.query(self.model).filter(self.model.id == record_id, self.model.deleted == False).one_or_none()
                
                if record is None:
                    error_info = f"Record with ID {record_id} not found."
                    self.logger.error(error_info)
                    raise ValueError(error_info)

                # 更新 rank 字段
                record.deep_health_advice = new_health_advice
                
                # 提交更改
                session.commit()
                
                # 返回更新后的记录（可选）
                return {key: value for key, value in record.__dict__.items() if key != '_sa_instance_state'}
            
            except Exception as e:
                error_info = f"Failed to update health advice for record ID {record_id}: {str(e)}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    
    
    
    async def delete_records_by_condition(self, condition: Dict[str, Any]) -> int:
        """
        按照指定条件硬删除多条记录（永久从数据库中删除）
        
        Args:
            condition (Dict[str, Any]): 删除条件，格式为 {字段名: 值}
        
        Returns:
            int: 成功删除的记录数量
        """
        async with self.get_db_session() as session:
            try:
                query = session.query(self.model)
                
                # 添加条件过滤，忽略不存在的字段
                valid_conditions = {}
                for key, value in condition.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)
                        valid_conditions[key] = value
                    else:
                        self.logger.warning(f"Field '{key}' not found in model, ignoring this condition")
                
                if not valid_conditions:
                    self.logger.warning("No valid conditions found, no records will be deleted")
                    return 0
                    
                # 获取要删除的记录数量
                count_to_delete = query.count()
                
                # 执行硬删除操作
                query.delete(synchronize_session=False)
                
                return count_to_delete
            except Exception as e:
                error_info = f"Failed to delete records by condition: {condition}"
                self.logger.error(error_info)
                self.logger.error(traceback.format_exc())
                raise ValueError(error_info) from e
    
    
    def exec_sql(self, query: Optional[str] = None):
        """query words check data"""
        with self.sql_connection() as db:
            try:
                result = db.execute(text(query)).fetchall()
                db.commit()
            except Exception as e:
                db.rollback()        
                error_info = f"Failed to execute SQL query: {query}!"
                self.logger.error(f"{traceback.print_exc()}\n{error_info}")
                raise ValueError(error_info) from e
            if result is not None:
                # 将 RowProxy 转换为列表，然后再转换为 NumPy 数组
                numpy_array = np.array(result)
                return numpy_array
            return result