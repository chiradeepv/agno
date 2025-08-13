import pytest

from agno.agent.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.response import RunEvent
from agno.tools.yfinance import YFinanceTools
from agno.app.agui.app import AGUIApp
from agno.app.agui.utils import async_stream_agno_response_as_agui_events


def test_tool_use_tool_call_limit():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        tool_call_limit=1,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = agent.run("Find me the current price of TSLA and APPL.")

    # Verify tool usage, should only call the first tool
    assert len(response.tools) == 1
    assert response.tools[0].tool_name == "get_current_stock_price"
    assert response.tools[0].tool_args == {"symbol": "TSLA"}
    assert response.tools[0].result is not None
    assert response.content is not None


def test_tool_use_tool_call_limit_stream():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        tool_call_limit=1,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response_stream = agent.run("Find me the current price of TSLA and APPL.", stream=True)

    tools = []
    for chunk in response_stream:
        if chunk.event == RunEvent.tool_call_completed:
            tools.append(chunk.tool)

    assert len(tools) == 1
    assert tools[0].tool_name == "get_current_stock_price"
    assert tools[0].tool_args == {"symbol": "TSLA"}
    assert tools[0].result is not None


@pytest.mark.asyncio
async def test_tool_use_tool_call_limit_async():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        tool_call_limit=1,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response = await agent.arun("Find me the current price of TSLA and APPL.")

    # Verify tool usage, should only call the first tool
    assert len(response.tools) == 1
    assert response.tools[0].tool_name == "get_current_stock_price"
    assert response.tools[0].tool_args == {"symbol": "TSLA"}
    assert response.tools[0].result is not None
    assert response.content is not None


@pytest.mark.asyncio
async def test_tool_use_tool_call_limit_stream_async():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        tool_call_limit=4,
        markdown=True,
        telemetry=False,
        monitoring=False,
    )

    response_stream = await agent.arun("Find me the current price of TSLA, AAPL, and MSFT.", stream=True)

    tools = []
    async for chunk in response_stream:
        print(chunk.event)
        if chunk.event == RunEvent.tool_call_completed:
            print(chunk.tool)
            tools.append(chunk.tool)

    assert len(tools) == 3
    assert tools[0].tool_name == "get_current_stock_price"
    assert tools[0].tool_args == {"symbol": "TSLA"}
    assert tools[0].result is not None

@pytest.mark.asyncio
async def test_tool_ag_ui_integration_async():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(cache_results=True)],
        tool_call_limit=4,
        markdown=True,
        telemetry=False,
        monitoring=False,
        debug_mode=True
    )

    agui_app = AGUIApp(
        agent=agent,
        name="Investment Analyst",
        app_id="investment_analyst",
        description="An investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    )

    response_stream = await agent.arun("Find me the current price of TSLA, AAPL, and MSFT.", stream=True)
    async for event in async_stream_agno_response_as_agui_events(response_stream, "thread_1", "run_1"):
        print(event)
