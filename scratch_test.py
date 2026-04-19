import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

async def test_kimi():
    client = ChatOpenAI(
        model="moonshotai/kimi-k2.5",
        api_key=os.getenv("KIMI_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.7,
    )
    
    system_content = (
        "You are Nova, an Elite AI Reasoning Tutor. "
        "\n\n*** CRITICAL SYSTEM INSTRUCTION ***\n"
        "You MUST output your internal reasoning before answering.\n"
        "You MUST write your reasoning inside a markdown code block named `thought`.\n"
        "Example:\n"
        "```thought\n"
        "The user said hii. I should say hello.\n"
        "```\n"
        "Hello! How can I help you today?"
    )
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content="hii")
    ]
    
    full_content = ""
    async for chunk in client.astream(messages):
        content = chunk.content if chunk.content else ""
        full_content += content
    print("\n\nFULL CONTENT:\n", full_content)

asyncio.run(test_kimi())
