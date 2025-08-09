#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/06 14:34
@Author  : weiyutao
@File    : community_real_time_data_server.py
"""


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import logging
from fastapi.encoders import jsonable_encoder
import asyncio
from pathlib import Path

from api.table.community_real_time_data import CommunityRealTimeData
from agent.config.sql_config import SqlConfig
from agent.provider.sql_provider import SqlProvider


ROOT_DIRECTORY = Path(__file__).parent.parent
SQL_CONFIG_PATH = str(ROOT_DIRECTORY / "agent" / "config" / "yaml" / "postgresql_config.yaml")



class CommunityRealTimeInfo(BaseModel):
    type: str
    content: str
    username: Optional[str] = ""


class ListCommunityRealTimeInfo(BaseModel):
    type: Optional[str] = None
    username: Optional[str] = None


class CommunityRealTimeDataServer:
    """社区服务类"""
    
    def __init__(self, sql_config_path: str):
        self.sql_config_path = sql_config_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    
    def register_routes(self, app: FastAPI):
        """注册社区相关的路由"""
        app.post("/api/community_real_time_data/save")(self.save_community_data)
        app.post("/api/community_real_time_data")(self.post_community_data)
        app.get("/api/community_real_time_data")(self.get_community_data)
    
    
    async def save_community_data(self, community_info: CommunityRealTimeInfo):
        """
        POST请求 - 保存社区管理员上传的通告或时讯消息
        Examples:
        - POST /api/community_real_time_data/save  {"username": "JOHN", "type": "通告", "content": "今天放假！"} ->  新增一条通告：今天放假，通告添加者为JOHN
        - POST /api/community_real_time_data/save  {"username": "JOHN", "type": ""} -> 不合法，必须提供类型
        - POST /api/community_real_time_data  {"username": "JOHN", "type": "时讯消息", "content": ""} -> 不合法，必须提供具体内容
        - POST /api/community_real_time_data  {} -> username为必填项，不合法的传参
        """
        try:
            self.logger.info(f"收到社区数据保存请求: {community_info}")
            
            if not community_info.type:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "请提供内容类型", "timestamp": datetime.now().isoformat()}
                )
            
            if not community_info.content:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "请提供具体内容", "timestamp": datetime.now().isoformat()}
                )
            
            sql_provider = SqlProvider(model=CommunityRealTimeData, sql_config_path=self.sql_config_path)
            insert_data = {
                "type": community_info.type,
                "content": community_info.content,
                "creator": community_info.username,
                "updater": community_info.username,
                "create_time": datetime.now(),
                "update_time": datetime.now()
            }
            
            result = await sql_provider.add_record(insert_data)
            
            if result:
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "message": "保存成功", "data": str(result), "timestamp": datetime.now().isoformat()}
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": "数据库添加记录失败", "timestamp": datetime.now().isoformat()}
                )
                
        except Exception as e:
            self.logger.error(f"保存社区数据失败: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"保存失败: {str(e)}", "timestamp": datetime.now().isoformat()}
            )
    
    
    async def get_community_data(
        self, 
        username: Optional[str] = None,
        type: Optional[str] = None
    ):
        """
        GET请求 - 支持传递username参数获取该用户上传的最近几条的社区时讯消息或通告，支持在JSON请求体中传递username, type等参数过滤设备信息
        Examples:
        - GET /api/community_real_time_data?username=JOHN ->  获取john用户的时讯消息和通告
        - GET /api/community_real_time_data?username=JOHN&type=通告 -> 获取john用户的通告
        - GET /api/community_real_time_data?username=JOHN&type=时讯消息 -> 获取john用户的通告
        - GET /api/community_real_time_data  -> username为必填项，不合法的传参
        """
        if not username:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "用户名不能为空", "timestamp": datetime.now().isoformat()}
            )
        try:
            sql_provider = SqlProvider(
                model=CommunityRealTimeData, 
                sql_config_path=self.sql_config_path
            )
            condition = {"creator": username, "deleted": False}
            
            if type:
                condition["type"] = type
            
            result = await sql_provider.get_record_by_condition(
                condition=condition,
                fields=["id", "type", "content", "create_time"]
            )
            
            # 只返回最新的5条
            result = result[-5:] if len(result) > 5 else result
            json_compatible_result = jsonable_encoder(result)
            return JSONResponse(
                status_code=200,
                content={"success": True, "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"获取数据失败: {str(e)}", "timestamp": datetime.now().isoformat()}
            )
        finally:
            if sql_provider:
                await sql_provider.close()
                # 等待一小段时间确保连接完全关闭
                await asyncio.sleep(0.1)
    
    
    async def post_community_data(self, list_info: ListCommunityRealTimeInfo):
        """
        POST请求 - 支持传递username参数获取该用户上传的最近几条的社区时讯消息或通告，支持在JSON请求体中传递username, type等参数过滤设备信息
        Examples:
        - POST /api/community_real_time_data  {"username": "JOHN"} ->  获取john用户的时讯消息和通告
        - POST /api/community_real_time_data  {"username": "JOHN", "type": "通告"} -> 获取john用户的通告
        - POST /api/community_real_time_data  {"username": "JOHN", "type": "时讯消息"} -> 获取john用户的通告
        - POST /api/community_real_time_data  {} -> username为必填项，不合法的传参
        """
        # 无效代码-----------------------------------------------------------------------------------------
        try:
            username = list_info.username
            type = list_info.type
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"传参错误！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
        # 无效代码-----------------------------------------------------------------------------------------
        
        try:
            result = await self.get_community_data(username=username, type=type)
            return result
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"获取数据失败: {str(e)}", "timestamp": datetime.now().isoformat()}
            )
    
    
if __name__ == '__main__':
    ROOT_DIRECTORY = Path(__file__).parent.parent.parent
    SQL_CONFIG_PATH = str(ROOT_DIRECTORY / "agent" / "config" / "yaml" / "postgresql_config.yaml")
    community_server = CommunityRealTimeDataServer(sql_config_path=SQL_CONFIG_PATH)
    community_info = CommunityRealTimeInfo(type="时讯消息", content="2025年8月10日放假", username="admin")


    async def main():
        result = await community_server.save_community_data(community_info=community_info)
        print(f"保存结果: {result}")

    asyncio.run(main())

