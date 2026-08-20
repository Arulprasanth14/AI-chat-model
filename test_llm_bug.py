import asyncio
import os
import json
from app.domain.llm.openai_provider import OpenAIProvider
from app.core.config import settings
from app.project_profiles.base_profile import BaseProfile
from app.domain.llm.prompt_builder import PromptBuilder
import app.domain.llm.tool_schema as tool_schema
from pathlib import Path

async def main():
    llm = OpenAIProvider(api_key=settings.openai_api_key, model='gpt-4o')
    
    profile_dir = Path("app/project_profiles/picasso_fusion")
    yaml_path = profile_dir / "profile.yaml"
    profile = BaseProfile.from_yaml(yaml_path)
    
    builder = PromptBuilder()
    missing_fields = [] # just empty for this test
    sys_prompt_msgs = builder.build(profile, [], [], missing_fields, 5, False, None)
    
    tools = tool_schema.get_extract_tool_schema()
    
    msg = "our main message is affordability and trust. and we want a 20% increase in engagement."
    sys_prompt_msgs.append({'role': 'user', 'content': msg})
    
    print('Calling LLM Phase A...')
    res = ""
    async for token in llm.stream_tool_call(
        messages=sys_prompt_msgs,
        tool_schema=tools,
        temperature=0.1
    ):
        res += token
        
    print('RAW LLM JSON:')
    print(res)

if __name__ == '__main__':
    asyncio.run(main())
