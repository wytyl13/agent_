#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/06 11:22
@Author  : weiyutao
@File    : user_data_server.py
"""



from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
from pydantic import BaseModel
from typing import (
    Optional
)
from fastapi.encoders import jsonable_encoder
import asyncio


from api.table.user_data import UserData
from agent.provider.sql_provider import SqlProvider


class ListUserData(BaseModel):
    username: Optional[str] = None


class UserDataServer:
    """用户服务类"""
    
    def __init__(self, sql_config_path: str):
        self.sql_config_path = sql_config_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    
    def register_routes(self, app: FastAPI):
        """注册用户相关的路由"""
        app.get("/api/user_data")(self.get_user_data)
        app.post("/api/user_data")(self.post_user_data)
    
    
    async def get_user_data(
        self,
        username: Optional[str] = None,
    ):
        """
        GET请求 - 支持获取所有用户（不传递任何参数）信息，支持获取指定用户（在url中传递username参数）信息
        Examples:
        - GET /api/user_data -> 获取所有用户信息
        - GET /api/user_data?username=john -> 获取john用户的设备信息
        """
        condition = {}
        if username is not None:
            condition["username"] = username
        try:
            sql_provider = SqlProvider(model=UserData, sql_config_path=self.sql_config_path)
            result = await sql_provider.get_record_by_condition(
                condition=condition,
                fields=["id", "username", "password", "full_name", "gender", "age", "address", "phone", "email", "status", "create_time", "tenant_id", "role", "community"]
            )
            json_compatible_result = jsonable_encoder(result)
            
            return JSONResponse(
                status_code=200,
                content={"success": True, "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"获取用户数据失败: {str(e)}", "timestamp": datetime.now().isoformat()}
            )
        finally:
            if sql_provider:
                await sql_provider.close()
                # 等待一小段时间确保连接完全关闭
                await asyncio.sleep(0.1)


    async def post_user_data(
        self,
        list_user_data: ListUserData,
    ):
        """
        POST请求 - 支持获取所有用户（使用JSON空白请求体）信息，支持获取指定用户（在JSON请求体中传递username参数）信息
        Examples:
        - POST /api/user_data {} -> 获取所有用户信息
        - POST /api/user_data {"username": "JOHN"} -> 获取JOHN用户的设备信息
        """
        # 无效代码-----------------------------------------------------------------------------------------
        try:
            username = list_user_data.username
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"传参错误！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
        # 无效代码-----------------------------------------------------------------------------------------
        try:
            result = await self.get_user_data(username)
            return result
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"获取用户数据失败: {str(e)}", "timestamp": datetime.now().isoformat()}
            )
    