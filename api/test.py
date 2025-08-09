#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/08 15:32
@Author  : weiyutao
@File    : test.py
"""


from agent.config.sql_config import SqlConfig
from agent.provider.sql_provider import SqlProvider
from api.table.user_data import UserData
from api.table.community_real_time_data import CommunityRealTimeData


if __name__ == '__main__':
    import asyncio
    
    async def main():
        sql_provider = None
        # try:
        #     sql_provider = SqlProvider(
        #         model=CommunityRealTimeData, 
        #         sql_config_path="/work/ai/Agent/agent/config/yaml/sql_config.yaml"
        #     )
        #     result = await sql_provider.get_record_by_condition({})
        #     print(result)
        # finally:
        #     if sql_provider:
        #         await sql_provider.close()
        #         # 等待一小段时间确保连接完全关闭
        #         await asyncio.sleep(0.1)
        try:
            sql_provider = SqlProvider(
                model=UserData, 
                sql_config_path="/work/ai/Agent/agent/config/yaml/postgresql_config.yaml"
            )
            result = await sql_provider.get_record_by_condition({})
            print(result)
        finally:
            if sql_provider:
                await sql_provider.close()
                # 等待一小段时间确保连接完全关闭
                await asyncio.sleep(0.1)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())