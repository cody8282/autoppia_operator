"""
llm_agent – A minimal LLM agent with ReAct (Reason + Act) loop.

Usage::

    from llm_agent import LLMAgent, AgentConfig, LLMConfig
    from llm_agent.tools import ToolRegistry

    # 1. Configure
    llm_config = LLMConfig(provider_type="openai", api_key="sk-...", model_name="gpt-4o")

    # 2. (Optional) register tools
    tools = ToolRegistry()
    tools.register_function(
        name="get_weather",
        description="Get weather for a city",
        function=lambda city: f"Sunny in {city}",
        parameters={"city": {"type": "string", "required": True}},
    )

    # 3. Create & run
    agent = LLMAgent(
        name="assistant",
        system_prompt="You are a helpful assistant.",
        llm_config=llm_config,
        tools=tools,
    )
    agent.start()
    response = agent.act("What's the weather in London?")
    print(response.content)
    agent.stop()
"""

from .agent import LLMAgent
from .config import AgentConfig, LLMConfig
from .models import AgentResponse, TaskEnvelope, TaskResult, TaskStatus

__all__ = [
    "LLMAgent",
    "AgentConfig",
    "LLMConfig",
    "AgentResponse",
    "TaskEnvelope",
    "TaskResult",
    "TaskStatus",
]
