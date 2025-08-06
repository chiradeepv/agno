import pytest

from agno.agent.agent import Agent
from agno.models.openai.chat import OpenAIChat


def chat_agent_factory(shared_db, session_id, session_state):
    """Create an agent with storage and memory for testing."""
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        db=shared_db,
        session_id=session_id,
        session_state=session_state,
    )


def test_agent_default_state(shared_db):
    session_id = "session_1"
    session_state = {"test_key": "test_value"}
    chat_agent = chat_agent_factory(shared_db, session_id, session_state)

    response = chat_agent.run("Hello, how are you?")
    
    assert response.run_id is not None
    
    assert chat_agent.session_id == session_id
    assert chat_agent.session_state == session_state
    assert chat_agent.session_name is None
    
    session_from_storage = shared_db.read(session_id=session_id)
    assert session_from_storage is not None
    assert session_from_storage.session_id == session_id
    assert session_from_storage.session_data["session_state"] == {
        "current_session_id": session_id,
        "test_key": "test_value",
    }


def test_agent_session_state_switch_session_id(chat_agent):
    session_id_1 = "session_1"
    session_id_2 = "session_2"

    chat_agent.session_name = "my_test_session"
    chat_agent.session_state = {"test_key": "test_value"}

    # First run with a session ID (reset should not happen)
    response = chat_agent.run("What can you do?", session_id=session_id_1)
    assert response.run_id is not None
    assert chat_agent.session_id == session_id_1
    assert chat_agent.session_name == "my_test_session"
    assert chat_agent.session_state == {"test_key": "test_value"}

    # Second run with different session ID
    response = chat_agent.run("What can you do?", session_id=session_id_2)
    assert response.run_id is not None
    assert chat_agent.session_id == session_id_2
    assert chat_agent.session_name is None
    assert chat_agent.session_state == {}

    # Third run with the original session ID
    response = chat_agent.run("What can you do?", session_id=session_id_1)
    assert response.run_id is not None
    assert chat_agent.session_id == session_id_1
    assert chat_agent.session_name == "my_test_session"
    assert chat_agent.session_state == {"test_key": "test_value"}


def test_agent_session_state_on_run(chat_agent):
    session_id_1 = "session_1"
    session_id_2 = "session_2"

    chat_agent.session_name = "my_test_session"

    # First run with a different session ID
    response = chat_agent.run("What can you do?", session_id=session_id_1, session_state={"test_key": "test_value"})
    assert response.run_id is not None
    assert chat_agent.session_id == session_id_1
    assert chat_agent.session_state == {"test_key": "test_value"}

    # Second run with different session ID
    response = chat_agent.run("What can you do?", session_id=session_id_2)
    assert response.run_id is not None
    assert chat_agent.session_id == session_id_2
    assert chat_agent.session_state == {}

    # Third run with the original session ID
    response = chat_agent.run(
        "What can you do?", session_id=session_id_1, session_state={"something_else": "other_value"}
    )
    assert response.run_id is not None
    assert chat_agent.session_id == session_id_1
    assert chat_agent.session_state == {"test_key": "test_value", "something_else": "other_value"}, (
        "Merging session state should work"
    )
