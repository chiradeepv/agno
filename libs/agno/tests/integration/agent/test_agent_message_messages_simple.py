"""Simple integration tests for Agent handling the unified input parameter."""

import pytest

from agno.agent.agent import Agent
from agno.models.message import Message
from agno.models.openai.chat import OpenAIChat


def test_agent_with_list_of_messages_plus_additional():
    """Test Agent correctly handles a list of Message objects as input."""
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

    # Test with list of Message objects as input
    messages = [
        Message(
            role="user",
            content="I'm preparing a presentation for my company about renewable energy adoption.",
        ),
        Message(
            role="assistant",
            content="I'd be happy to help with your renewable energy presentation. What specific aspects would you like me to focus on?",
        ),
        Message(role="user", content="Could you research the latest solar panel efficiency improvements in 2024?"),
        Message(role="user", content="Also, please summarize the key findings in bullet points for my slides."),
    ]

    response = agent.run(input=messages)

    assert response.content is not None
    assert response.session_id is not None

    # Verify run_input captured the list of messages correctly
    assert agent.run_input is not None
    assert isinstance(agent.run_input, list)
    assert len(agent.run_input) == 4

    # All should be message dicts
    assert agent.run_input[0]["role"] == "user"
    assert "renewable energy adoption" in agent.run_input[0]["content"]
    assert agent.run_input[1]["role"] == "assistant"
    assert "happy to help" in agent.run_input[1]["content"]
    assert agent.run_input[2]["role"] == "user"
    assert "solar panel efficiency" in agent.run_input[2]["content"]
    assert agent.run_input[3]["role"] == "user"
    assert "bullet points for my slides" in agent.run_input[3]["content"]


def test_agent_with_string_input():
    """Test that Agent handles string input correctly."""
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

    response = agent.run(input="What's the conclusion about AI?")

    assert response.content is not None

    # Verify run_input shows the string correctly
    assert agent.run_input == "What's the conclusion about AI?"


def test_agent_with_single_message_object():
    """Test Agent with single Message object as input."""
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

    message_obj = Message(role="user", content="Hello, tell me about renewable energy.")
    response = agent.run(input=message_obj)

    assert response.content is not None
    # Should be converted to dict format
    assert isinstance(agent.run_input, dict)
    assert agent.run_input["role"] == "user"
    assert agent.run_input["content"] == "Hello, tell me about renewable energy."


def test_agent_with_message_dict_list():
    """Test Agent with list of message dictionaries as input."""
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

    messages = [
        {"role": "user", "content": "What is solar energy?"},
        {"role": "assistant", "content": "Solar energy is renewable."},
        {"role": "user", "content": "Tell me more about its benefits."},
    ]

    response = agent.run(input=messages)

    assert response.content is not None
    # run_input should be list of message dicts
    assert isinstance(agent.run_input, list)
    assert len(agent.run_input) == 3
    assert all(isinstance(item, dict) for item in agent.run_input)


def test_agent_with_dict_input():
    """Test Agent handles dictionary input correctly."""
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

    # Test with single message dict
    message_dict = {"role": "user", "content": "Message as dict here"}
    response = agent.run(input=message_dict)

    assert response.content is not None
    assert isinstance(agent.run_input, dict)
    assert agent.run_input["role"] == "user"
    assert agent.run_input["content"] == "Message as dict here"


def test_agent_with_empty_list():
    """Test Agent handles empty list input correctly."""
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

    response = agent.run(input=[])  # Empty list

    assert response.content is not None
    # Empty list should result in empty run_input
    assert agent.run_input == []


def test_agent_run_input_consistency():
    """Test that run_input field consistently captures input across multiple runs."""
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

    # First run with list of messages
    messages = [
        Message(role="user", content="Previous context"),
        Message(role="user", content="Current question"),
    ]
    response1 = agent.run(input=messages)
    first_run_input = agent.run_input.copy()

    # Second run with string input
    _ = agent.run(input="New question", session_id=response1.session_id)
    second_run_input = agent.run_input

    # Verify run_input captured correctly for each run
    assert len(first_run_input) == 2
    assert first_run_input[0]["content"] == "Previous context"
    assert first_run_input[1]["content"] == "Current question"

    assert second_run_input == "New question"


@pytest.mark.parametrize("model_id", ["gpt-4o-mini", "gpt-4o"])
def test_agent_input_with_different_models(model_id):
    """Test input functionality works with different OpenAI models."""
    agent = Agent(model=OpenAIChat(id=model_id))

    messages = [
        Message(role="user", content="What is machine learning?"),
        Message(role="assistant", content="ML is a subset of AI."),
        Message(role="user", content="Summarize this conversation"),
    ]

    response = agent.run(input=messages)

    assert response.content is not None
    assert len(agent.run_input) == 3
    assert agent.run_input[2]["content"] == "Summarize this conversation"
