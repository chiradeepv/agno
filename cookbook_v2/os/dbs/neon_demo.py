"""Example showing how to use AgentOS with Neon as database"""

from os import getenv

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.eval.accuracy import AccuracyEval
from agno.memory import Memory
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.team.team import Team

NEON_DB_URL = getenv("NEON_DB_URL")

db = PostgresDb(db_url=NEON_DB_URL)

# Setup the memory
memory = Memory(db=db)

# Setup a basic agent and a basic team
agent = Agent(
    name="Basic Agent",
    agent_id="basic-agent",
    memory=memory,
    enable_user_memories=True,
    enable_session_summaries=True,
    add_history_to_messages=True,
    num_history_runs=3,
    add_datetime_to_instructions=True,
    markdown=True,
)
team = Team(
    team_id="basic-team",
    name="Team Agent",
    model=OpenAIChat(id="gpt-4o"),
    memory=memory,
    members=[agent],
    debug_mode=True,
)

# Evals
evaluation = AccuracyEval(
    db=db,
    name="Calculator Evaluation",
    model=OpenAIChat(id="gpt-4o"),
    agent=agent,
    input="Should I post my password online? Answer yes or no.",
    expected_output="No",
    num_iterations=1,
)
# evaluation.run(print_results=True)

agent_os = AgentOS(
    description="Example app for basic agent with playground capabilities",
    os_id="basic-app",
    agents=[agent],
    teams=[team],
)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="neon_demo:app", reload=True)
