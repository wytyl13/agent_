from agent.base.tool import tool
from pydantic import BaseModel, Field
from typing import (
    List,
    Type
)

class TestSchema(BaseModel):
    question: str = Field(
        ...,
        description="Test name"
    )

@tool
class Test:
    args_schema: Type[BaseModel] = TestSchema
    name="test_name"
    description="tesrt"
    def __init__(self):
        pass
    
    async def execute(
        self, question: str, 
    ):
        
        print(question)
        
        
if __name__ == '__main__':
    tests = Test()
    print(tests.get_simple_tool_description())
    