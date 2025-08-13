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
        if chunk.event == RunEvent.tool_call_completed:
            tools.append(chunk.tool)

    assert len(tools) == 3
    assert tools[0].tool_name == "get_current_stock_price"
    assert tools[0].tool_args == {"symbol": "TSLA"}
    assert tools[0].result is not None

@pytest.mark.asyncio
async def test_tool_ag_ui_integration_async():
    from ag_ui.core import EventType
    
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
    
    events = []
    async for event in async_stream_agno_response_as_agui_events(response_stream, "thread_1", "run_1"):
        events.append(event)
    
    # Verify we got events
    assert len(events) > 0, "Should have received AG-UI events"
    
    # Extract event types
    event_types = [event.type for event in events]
    
    # Verify expected event types are present
    assert EventType.TEXT_MESSAGE_START in event_types, "Should have message start event"
    assert EventType.TEXT_MESSAGE_CONTENT in event_types, "Should have message content events"
    assert EventType.TEXT_MESSAGE_END in event_types, "Should have message end event"
    
    # Verify tool call events (should have 3 tool calls for TSLA, AAPL, MSFT)
    tool_call_starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
    assert len(tool_call_starts) == 3, f"Should have 3 tool call starts, got {len(tool_call_starts)}"
    
    tool_call_ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
    assert len(tool_call_ends) == 3, f"Should have 3 tool call ends, got {len(tool_call_ends)}"
    
    tool_call_results = [e for e in events if e.type == EventType.TOOL_CALL_RESULT]
    assert len(tool_call_results) == 3, f"Should have 3 tool call results, got {len(tool_call_results)}"
    
    # Verify run completion
    assert EventType.RUN_FINISHED in event_types, "Should have run finished event"
    
    # Verify tool call IDs are unique
    tool_call_ids = set()
    for event in events:
        if hasattr(event, 'tool_call_id'):
            tool_call_ids.add(event.tool_call_id)
    assert len(tool_call_ids) == 3, f"Should have 3 unique tool call IDs, got {len(tool_call_ids)}"
    
    # Verify content contains stock symbols
    content_events = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
    all_content = ' '.join([e.delta for e in content_events if hasattr(e, 'delta')])
    
    # Check that stock prices are mentioned in the response
    assert any(symbol in all_content.upper() for symbol in ['TSLA', 'AAPL', 'MSFT']), \
        "Response should mention at least one of the requested stock symbols"
