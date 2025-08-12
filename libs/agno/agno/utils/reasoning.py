from typing import TYPE_CHECKING, Union

from agno.models.message import Message
from agno.models.metrics import Metrics
from agno.reasoning.step import ReasoningStep

if TYPE_CHECKING:
    from agno.run.response import RunResponse
    from agno.team.team import TeamRunResponse


def append_to_reasoning_content(run_response: Union["RunResponse", "TeamRunResponse"], content: str) -> None:
    """Helper to append content to the reasoning_content field."""
    if not hasattr(run_response, "reasoning_content") or not run_response.reasoning_content:  # type: ignore
        run_response.reasoning_content = content  # type: ignore
    else:
        run_response.reasoning_content += content  # type: ignore


def add_reasoning_step_to_metadata(
    run_response: Union["RunResponse", "TeamRunResponse"], reasoning_step: ReasoningStep
) -> None:
    if run_response.metadata is None:
        from agno.run.response import RunResponseMetaData

        run_response.metadata = RunResponseMetaData()

    if run_response.metadata.reasoning_steps is None:
        run_response.metadata.reasoning_steps = []

    run_response.metadata.reasoning_steps.append(reasoning_step)


def add_reasoning_metrics_to_metadata(
    run_response: Union["RunResponse", "TeamRunResponse"], reasoning_time_taken: float
) -> None:
    try:
        if run_response.metadata is None:
            from agno.run.response import RunResponseMetaData

            run_response.metadata = RunResponseMetaData()

            # Initialize reasoning_messages if it doesn't exist
            if run_response.metadata.reasoning_messages is None:
                run_response.metadata.reasoning_messages = []

            metrics_message = Message(
                role="assistant",
                content=run_response.reasoning_content,
                metrics=Metrics(time=reasoning_time_taken),
            )

            # Add the metrics message to the reasoning_messages
            run_response.metadata.reasoning_messages.append(metrics_message)

    except Exception as e:
        # Log the error but don't crash
        from agno.utils.log import log_error

        log_error(f"Failed to add reasoning metrics to metadata: {str(e)}")
