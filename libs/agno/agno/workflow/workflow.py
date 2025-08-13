import asyncio
from dataclasses import dataclass
from datetime import datetime
from os import getenv
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
    overload,
)
from uuid import uuid4

from fastapi import WebSocket
from pydantic import BaseModel

from agno.agent.agent import Agent
from agno.db.base import BaseDb, SessionType
from agno.media import Audio, AudioArtifact, Image, ImageArtifact, Video, VideoArtifact
from agno.run.base import RunStatus
from agno.run.workflow import (
    ConditionExecutionCompletedEvent,
    ConditionExecutionStartedEvent,
    LoopExecutionCompletedEvent,
    LoopExecutionStartedEvent,
    LoopIterationCompletedEvent,
    LoopIterationStartedEvent,
    ParallelExecutionCompletedEvent,
    ParallelExecutionStartedEvent,
    RouterExecutionCompletedEvent,
    RouterExecutionStartedEvent,
    StepCompletedEvent,
    StepOutputEvent,
    StepsExecutionCompletedEvent,
    StepsExecutionStartedEvent,
    StepStartedEvent,
    WorkflowCompletedEvent,
    WorkflowRunEvent,
    WorkflowRunResponse,
    WorkflowRunResponseEvent,
    WorkflowStartedEvent,
)
from agno.session.workflow import WorkflowSession
from agno.team.team import Team
from agno.utils.log import (
    log_debug,
    logger,
    set_log_level_to_debug,
    set_log_level_to_info,
    use_workflow_logger,
)
from agno.workflow.condition import Condition
from agno.workflow.loop import Loop
from agno.workflow.parallel import Parallel
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.steps import Steps
from agno.workflow.types import (
    StepInput,
    StepMetrics,
    StepOutput,
    StepType,
    WebSocketHandler,
    WorkflowExecutionInput,
    WorkflowMetrics,
)

# TODO: This is used to infer the Step type in the main to_dict() method. Instead, each step should have a type field
STEP_TYPE_MAPPING = {
    Step: StepType.STEP,
    Steps: StepType.STEPS,
    Loop: StepType.LOOP,
    Parallel: StepType.PARALLEL,
    Condition: StepType.CONDITION,
    Router: StepType.ROUTER,
}

WorkflowSteps = Union[
    Callable[
        ["Workflow", WorkflowExecutionInput],
        Union[StepOutput, Awaitable[StepOutput], Iterator[StepOutput], AsyncIterator[StepOutput], Any],
    ],
    Steps,
    List[
        Union[
            Callable[
                [StepInput], Union[StepOutput, Awaitable[StepOutput], Iterator[StepOutput], AsyncIterator[StepOutput]]
            ],
            Step,
            Steps,
            Loop,
            Parallel,
            Condition,
            Router,
        ]
    ],
]


@dataclass
class Workflow:
    """Pipeline-based workflow execution"""

    # Workflow identification - make name optional with default
    name: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None

    # Workflow configuration
    steps: Optional[WorkflowSteps] = None

    db: Optional[BaseDb] = None

    # Session management
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    user_id: Optional[str] = None
    workflow_session_id: Optional[str] = None
    workflow_session_state: Optional[Dict[str, Any]] = None

    # Runtime state
    run_id: Optional[str] = None
    run_response: Optional[WorkflowRunResponse] = None

    # Workflow session for storage
    workflow_session: Optional[WorkflowSession] = None
    debug_mode: Optional[bool] = False

    # --- Workflow Streaming ---
    # Stream the response from the Workflow
    stream: Optional[bool] = None
    # Stream the intermediate steps from the Workflow
    stream_intermediate_steps: bool = False

    store_events: bool = False
    events_to_skip: Optional[List[WorkflowRunEvent]] = None

    os_id: Optional[str] = None

    websocket_handler: Optional[WebSocketHandler] = None

    input_schema: Optional[Type[BaseModel]] = None

    # Control whether to store executor responses (agent/team responses) in flattened runs
    store_executor_responses: bool = True

    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        db: Optional[BaseDb] = None,
        steps: Optional[WorkflowSteps] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        workflow_session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = False,
        stream: Optional[bool] = None,
        stream_intermediate_steps: bool = False,
        store_events: bool = False,
        events_to_skip: Optional[List[WorkflowRunEvent]] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        store_executor_responses: bool = True,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.db = db
        self.steps = steps
        self.session_id = session_id
        self.session_name = session_name
        self.workflow_session_state = workflow_session_state
        self.user_id = user_id
        self.debug_mode = debug_mode
        self.store_events = store_events
        self.events_to_skip = events_to_skip or []
        self.stream = stream
        self.stream_intermediate_steps = stream_intermediate_steps
        self.store_executor_responses = store_executor_responses
        self.input_schema = input_schema

    def _validate_input(
        self, message: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]]
    ) -> Optional[BaseModel]:
        """Parse and validate input against input_schema if provided"""
        if self.input_schema is None:
            return None

        if message is None:
            raise ValueError("Input required when input_schema is set")

        # Case 1: Message is already a BaseModel instance
        if isinstance(message, BaseModel):
            if isinstance(message, self.input_schema):
                try:
                    # Re-validate to catch any field validation errors
                    message.model_validate(message.model_dump())
                    return message
                except Exception as e:
                    raise ValueError(f"BaseModel validation failed: {str(e)}")
            else:
                # Different BaseModel types
                raise ValueError(f"Expected {self.input_schema.__name__} but got {type(message).__name__}")

        # Case 2: Message is a dict
        elif isinstance(message, dict):
            try:
                validated_model = self.input_schema(**message)
                return validated_model
            except Exception as e:
                raise ValueError(f"Failed to parse dict into {self.input_schema.__name__}: {str(e)}")

        # Case 3: Other types not supported for structured input
        else:
            raise ValueError(
                f"Cannot validate {type(message)} against input_schema. Expected dict or {self.input_schema.__name__} instance."
            )

    @property
    def run_parameters(self) -> Dict[str, Any]:
        """Get the run parameters for the workflow"""

        if self.steps is None:
            return {}

        parameters = {}

        if self.steps and callable(self.steps):
            from inspect import Parameter, signature

            sig = signature(self.steps)  # type: ignore

            for param_name, param in sig.parameters.items():
                if param_name not in ["workflow", "execution_input", "self"]:
                    parameters[param_name] = {
                        "name": param_name,
                        "default": param.default.default
                        if hasattr(param.default, "__class__") and param.default.__class__.__name__ == "FieldInfo"
                        else (param.default if param.default is not Parameter.empty else None),
                        "annotation": (
                            param.annotation.__name__
                            if hasattr(param.annotation, "__name__")
                            else (
                                str(param.annotation).replace("typing.Optional[", "").replace("]", "")
                                if "typing.Optional" in str(param.annotation)
                                else str(param.annotation)
                            )
                        )
                        if param.annotation is not Parameter.empty
                        else None,
                        "required": param.default is Parameter.empty,
                    }
        else:
            parameters = {
                "message": {
                    "name": "message",
                    "default": None,
                    "annotation": "str",
                    "required": True,
                },
            }

        return parameters

    def initialize_workflow(self):
        if self.id is None:
            self.id = str(uuid4())
            log_debug(f"Generated new workflow_id: {self.id}")

        if self.session_id is None:
            self.session_id = str(uuid4())
            log_debug(f"Generated new session_id: {self.session_id}")

        self._update_workflow_session_state()

    def _generate_workflow_session_name(self, session_id: str) -> str:
        """Generate a name for the workflow session"""

        if self.session_id is None:
            return f"Workflow Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_session_name = f"Workflow Session-{datetime_str}"

        if self.description:
            truncated_desc = self.description[:40] + "-" if len(self.description) > 40 else self.description
            new_session_name = f"{truncated_desc} - {datetime_str}"
        return new_session_name

    def set_session_name(
        self, session_id: Optional[str] = None, autogenerate: bool = False, session_name: Optional[str] = None
    ):
        """Set the session name and save to storage"""
        if self.session_id is None and session_id is None:
            raise Exception("Session ID is not set")
        session_id = session_id or self.session_id

        if autogenerate:
            # -*- Generate name for session
            session_name = self._generate_workflow_session_name(session_id=session_id)
            log_debug(f"Generated Workflow Session Name: {session_name}")
        elif session_name is None:
            raise Exception("Session name is not set")

        self.get_workflow_session()
        self.session_name = session_name
        # -*- Save to storage
        self.write_to_storage()

    def delete_session(self, session_id: str):
        """Delete the current session and save to storage"""
        if self.db is None:
            return
        # -*- Delete session
        self.db.delete_session(session_id=session_id)

    def _handle_event(
        self,
        event: "WorkflowRunResponseEvent",
        workflow_run_response: WorkflowRunResponse,
        websocket_handler: Optional[WebSocketHandler] = None,
    ) -> "WorkflowRunResponseEvent":
        """Handle workflow events for storage - similar to Team._handle_event"""
        if self.store_events:
            # Check if this event type should be skipped
            if self.events_to_skip:
                event_type = event.event
                for skip_event in self.events_to_skip:
                    if isinstance(skip_event, str):
                        if event_type == skip_event:
                            return event
                    else:
                        # It's a WorkflowRunEvent enum
                        if event_type == skip_event.value:
                            return event

            # Store the event
            if workflow_run_response.events is None:
                workflow_run_response.events = []

            workflow_run_response.events.append(event)

        # Broadcast to WebSocket if available (async context only)
        if websocket_handler:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                if loop:
                    asyncio.create_task(websocket_handler.handle_event(event))
            except RuntimeError:
                pass

        return event

    def _transform_step_output_to_event(
        self, step_output: StepOutput, workflow_run_response: WorkflowRunResponse, step_index: Optional[int] = None
    ) -> StepOutputEvent:
        """Transform a StepOutput object into a StepOutputEvent for consistent streaming interface"""
        return StepOutputEvent(
            step_output=step_output,
            run_id=workflow_run_response.run_id or "",
            workflow_name=workflow_run_response.workflow_name,
            workflow_id=workflow_run_response.workflow_id,
            session_id=workflow_run_response.session_id,
            step_name=step_output.step_name,
            step_index=step_index,
        )

    def _set_debug_mode(self) -> None:
        """Set debug mode and configure logging"""
        if self.debug_mode or getenv("AGNO_DEBUG", "false").lower() == "true":
            use_workflow_logger()

            self.debug_mode = True
            set_log_level_to_debug(source_type="workflow")

            # Propagate to steps - only if steps is iterable (not callable)
            if self.steps and not callable(self.steps):
                if isinstance(self.steps, Steps):
                    steps_to_iterate = self.steps.steps
                else:
                    steps_to_iterate = self.steps

                for step in steps_to_iterate:
                    self._propagate_debug_to_step(step)
        else:
            set_log_level_to_info(source_type="workflow")

    def _propagate_debug_to_step(self, step):
        """Recursively propagate debug mode to steps and nested primitives"""
        # Handle direct Step objects
        if hasattr(step, "active_executor") and step.active_executor:
            executor = step.active_executor
            if hasattr(executor, "debug_mode"):
                executor.debug_mode = True

            # If it's a team, propagate to all members
            if hasattr(executor, "members"):
                for member in executor.members:
                    if hasattr(member, "debug_mode"):
                        member.debug_mode = True

        # Handle nested primitives - check both 'steps' and 'choices' attributes
        for attr_name in ["steps", "choices"]:
            if hasattr(step, attr_name):
                attr_value = getattr(step, attr_name)
                if attr_value and isinstance(attr_value, list):
                    for nested_step in attr_value:
                        self._propagate_debug_to_step(nested_step)

    def _create_step_input(
        self,
        execution_input: WorkflowExecutionInput,
        previous_step_outputs: Optional[Dict[str, StepOutput]] = None,
        shared_images: Optional[List[ImageArtifact]] = None,
        shared_videos: Optional[List[VideoArtifact]] = None,
        shared_audio: Optional[List[AudioArtifact]] = None,
    ) -> StepInput:
        """Helper method to create StepInput with enhanced data flow support"""

        previous_step_content = None
        if previous_step_outputs:
            last_output = list(previous_step_outputs.values())[-1]
            previous_step_content = last_output.content if last_output else None
            log_debug(f"Using previous step content from: {list(previous_step_outputs.keys())[-1]}")

        return StepInput(
            input=execution_input.input,
            previous_step_content=previous_step_content,
            previous_step_outputs=previous_step_outputs,
            additional_data=execution_input.additional_data,
            images=shared_images or [],
            videos=shared_videos or [],
            audio=shared_audio or [],
        )

    def _get_step_count(self) -> int:
        """Get the number of steps in the workflow"""
        if self.steps is None:
            return 0
        elif callable(self.steps):
            return 1  # Callable function counts as 1 step
        else:
            # Handle Steps wrapper
            if isinstance(self.steps, Steps):
                return len(self.steps.steps)
            else:
                return len(self.steps)

    def _aggregate_workflow_metrics(self, step_results: List[Union[StepOutput, List[StepOutput]]]) -> WorkflowMetrics:
        """Aggregate metrics from all step responses into structured workflow metrics"""
        steps_dict = {}
        total_steps = 0

        def process_step_output(step_output: StepOutput):
            """Process a single step output for metrics"""
            nonlocal total_steps
            total_steps += 1

            # Add step-specific metrics
            if step_output.step_name and step_output.metrics:
                if step_output.parallel_step_outputs:
                    # This is a parallel step - create nested step metrics for each sub-step
                    parallel_step_metrics = {}
                    for sub_step_name, sub_step_output in step_output.parallel_step_outputs.items():
                        if sub_step_output.metrics:
                            parallel_step_metrics[sub_step_name] = StepMetrics(
                                step_name=sub_step_name,
                                executor_type=sub_step_output.executor_type or "unknown",
                                executor_name=sub_step_output.executor_name or "unknown",
                                metrics=sub_step_output.metrics,
                                parallel_steps=None,
                            )

                    # Create a StepMetrics for the parallel container
                    step_metrics = StepMetrics(
                        step_name=step_output.step_name,
                        executor_type="parallel",
                        executor_name=step_output.step_name or "Parallel",
                        metrics=step_output.metrics,
                        parallel_steps=parallel_step_metrics if parallel_step_metrics else None,
                    )
                else:
                    # Regular step
                    step_metrics = StepMetrics(
                        step_name=step_output.step_name,
                        executor_type=step_output.executor_type or "unknown",
                        executor_name=step_output.executor_name or "unknown",
                        metrics=step_output.metrics,
                        parallel_steps=None,
                    )

                steps_dict[step_output.step_name] = step_metrics

        # Process all step results
        for step_result in step_results:
            if isinstance(step_result, list):
                # Handle List[StepOutput] from workflow components
                for sub_step_output in step_result:
                    process_step_output(sub_step_output)
            else:
                # Handle single StepOutput
                process_step_output(step_result)

        return WorkflowMetrics(
            total_steps=total_steps,
            steps=steps_dict,
        )

    def _call_custom_function(
        self, func: Callable, workflow: "Workflow", execution_input: WorkflowExecutionInput, **kwargs: Any
    ) -> Any:
        """Call custom function with only the parameters it expects"""
        from inspect import signature

        sig = signature(func)

        # Build arguments based on what the function actually accepts
        call_kwargs = {}

        # Only add workflow and execution_input if the function expects them
        if "workflow" in sig.parameters:  # type: ignore
            call_kwargs["workflow"] = self
        if "execution_input" in sig.parameters:
            call_kwargs["execution_input"] = execution_input  # type: ignore

        # Add any other kwargs that the function expects
        for param_name in kwargs:
            if param_name in sig.parameters:  # type: ignore
                call_kwargs[param_name] = kwargs[param_name]

        # If function has **kwargs parameter, pass all remaining kwargs
        for param in sig.parameters.values():  # type: ignore
            if param.kind == param.VAR_KEYWORD:
                call_kwargs.update(kwargs)
                break

        try:
            return func(**call_kwargs)
        except TypeError as e:
            # If signature inspection fails, fall back to original method
            logger.warning(
                f"Async function signature inspection failed: {e}. Falling back to original calling convention."
            )
            return func(workflow, execution_input, **kwargs)

    def _execute(
        self, execution_input: WorkflowExecutionInput, workflow_run_response: WorkflowRunResponse, **kwargs: Any
    ) -> WorkflowRunResponse:
        """Execute a specific pipeline by name synchronously"""
        from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction

        workflow_run_response.status = RunStatus.running

        if callable(self.steps):
            if iscoroutinefunction(self.steps) or isasyncgenfunction(self.steps):
                raise ValueError("Cannot use async function with synchronous execution")
            elif isgeneratorfunction(self.steps):
                content = ""
                for chunk in self.steps(self, execution_input, **kwargs):
                    if hasattr(chunk, "content") and chunk.content is not None and isinstance(chunk.content, str):
                        content += chunk.content
                    else:
                        content += str(chunk)
                workflow_run_response.content = content
            else:
                # Execute the workflow with the custom executor
                workflow_run_response.content = self._call_custom_function(self.steps, self, execution_input, **kwargs)  # type: ignore[arg-type]

            workflow_run_response.status = RunStatus.completed
        else:
            try:
                # Track outputs from each step for enhanced data flow
                collected_step_outputs: List[Union[StepOutput, List[StepOutput]]] = []
                previous_step_outputs: Dict[str, StepOutput] = {}

                shared_images: List[ImageArtifact] = execution_input.images or []
                output_images: List[ImageArtifact] = (execution_input.images or []).copy()  # Start with input images
                shared_videos: List[VideoArtifact] = execution_input.videos or []
                output_videos: List[VideoArtifact] = (execution_input.videos or []).copy()  # Start with input videos
                shared_audio: List[AudioArtifact] = execution_input.audio or []
                output_audio: List[AudioArtifact] = (execution_input.audio or []).copy()  # Start with input audio

                for i, step in enumerate(self.steps):  # type: ignore[arg-type]
                    step_name = getattr(step, "name", f"step_{i + 1}")
                    log_debug(f"Executing step {i + 1}/{self._get_step_count()}: {step_name}")

                    # Create enhanced StepInput
                    step_input = self._create_step_input(
                        execution_input=execution_input,
                        previous_step_outputs=previous_step_outputs,
                        shared_images=shared_images,
                        shared_videos=shared_videos,
                        shared_audio=shared_audio,
                    )

                    step_output = step.execute(
                        step_input,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        workflow_run_response=workflow_run_response,
                        store_executor_responses=self.store_executor_responses,
                    )  # type: ignore[union-attr]

                    # Update the workflow-level previous_step_outputs dictionary
                    if isinstance(step_output, list):
                        log_debug(f"Step returned {len(step_output)} outputs")
                        # For multiple outputs (from Loop, Condition, etc.), store the last one
                        if step_output:
                            previous_step_outputs[step_name] = step_output[-1]
                            if any(output.stop for output in step_output):
                                logger.info(f"Early termination requested by step {step_name}")
                                break
                    else:
                        # Single output
                        previous_step_outputs[step_name] = step_output
                        if step_output.stop:
                            logger.info(f"Early termination requested by step {step_name}")
                            break

                    # Update shared media for next step
                    if isinstance(step_output, list):
                        for output in step_output:
                            shared_images.extend(output.images or [])
                            shared_videos.extend(output.videos or [])
                            shared_audio.extend(output.audio or [])
                            output_images.extend(output.images or [])
                            output_videos.extend(output.videos or [])
                            output_audio.extend(output.audio or [])
                    else:
                        shared_images.extend(step_output.images or [])
                        shared_videos.extend(step_output.videos or [])
                        shared_audio.extend(step_output.audio or [])
                        output_images.extend(step_output.images or [])
                        output_videos.extend(step_output.videos or [])
                        output_audio.extend(step_output.audio or [])

                    collected_step_outputs.append(step_output)

                    self._collect_workflow_session_state_from_agents_and_teams()

                # Update the workflow_run_response with completion data
                if collected_step_outputs:
                    workflow_run_response.workflow_metrics = self._aggregate_workflow_metrics(collected_step_outputs)
                    last_output = collected_step_outputs[-1]
                    if isinstance(last_output, list) and last_output:
                        # If it's a list (from Condition/Loop/etc.), use the last one
                        workflow_run_response.content = last_output[-1].content
                    elif not isinstance(last_output, list):
                        # Single StepOutput
                        workflow_run_response.content = last_output.content
                else:
                    workflow_run_response.content = "No steps executed"

                workflow_run_response.step_results = collected_step_outputs
                workflow_run_response.images = output_images
                workflow_run_response.videos = output_videos
                workflow_run_response.audio = output_audio
                workflow_run_response.status = RunStatus.completed

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Workflow execution failed: {e}")
                # Store error response
                workflow_run_response.status = RunStatus.error
                workflow_run_response.content = f"Workflow execution failed: {e}"

            finally:
                self._save_run_to_storage(workflow_run_response)

        return workflow_run_response

    def _execute_stream(
        self,
        execution_input: WorkflowExecutionInput,
        workflow_run_response: WorkflowRunResponse,
        stream_intermediate_steps: bool = False,
        **kwargs: Any,
    ) -> Iterator[WorkflowRunResponseEvent]:
        """Execute a specific pipeline by name with event streaming"""
        from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction

        workflow_run_response.status = RunStatus.running

        workflow_started_event = WorkflowStartedEvent(
            run_id=workflow_run_response.run_id or "",
            workflow_name=workflow_run_response.workflow_name,
            workflow_id=workflow_run_response.workflow_id,
            session_id=workflow_run_response.session_id,
        )
        yield self._handle_event(workflow_started_event, workflow_run_response)

        if callable(self.steps):
            if iscoroutinefunction(self.steps) or isasyncgenfunction(self.steps):
                raise ValueError("Cannot use async function with synchronous execution")
            elif isgeneratorfunction(self.steps):
                content = ""
                for chunk in self._call_custom_function(self.steps, self, execution_input, **kwargs):  # type: ignore[arg-type]
                    # Update the run_response with the content from the result
                    if hasattr(chunk, "content") and chunk.content is not None and isinstance(chunk.content, str):
                        content += chunk.content
                        yield chunk
                    else:
                        content += str(chunk)
                workflow_run_response.content = content
            else:
                workflow_run_response.content = self._call_custom_function(self.steps, self, execution_input, **kwargs)
            workflow_run_response.status = RunStatus.completed

        else:
            try:
                # Track outputs from each step for enhanced data flow
                collected_step_outputs: List[Union[StepOutput, List[StepOutput]]] = []
                previous_step_outputs: Dict[str, StepOutput] = {}

                shared_images: List[ImageArtifact] = execution_input.images or []
                output_images: List[ImageArtifact] = (execution_input.images or []).copy()  # Start with input images
                shared_videos: List[VideoArtifact] = execution_input.videos or []
                output_videos: List[VideoArtifact] = (execution_input.videos or []).copy()  # Start with input videos
                shared_audio: List[AudioArtifact] = execution_input.audio or []
                output_audio: List[AudioArtifact] = (execution_input.audio or []).copy()  # Start with input audio

                early_termination = False

                for i, step in enumerate(self.steps):  # type: ignore[arg-type]
                    step_name = getattr(step, "name", f"step_{i + 1}")
                    log_debug(f"Streaming step {i + 1}/{self._get_step_count()}: {step_name}")

                    # Create enhanced StepInput
                    step_input = self._create_step_input(
                        execution_input=execution_input,
                        previous_step_outputs=previous_step_outputs,
                        shared_images=shared_images,
                        shared_videos=shared_videos,
                        shared_audio=shared_audio,
                    )

                    # Execute step with streaming and yield all events
                    for event in step.execute_stream(  # type: ignore[union-attr]
                        step_input,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        stream_intermediate_steps=stream_intermediate_steps,
                        workflow_run_response=workflow_run_response,
                        step_index=i,
                        store_executor_responses=self.store_executor_responses,
                    ):
                        # Handle events
                        if isinstance(event, StepOutput):
                            step_output = event
                            collected_step_outputs.append(step_output)

                            # Update the workflow-level previous_step_outputs dictionary
                            previous_step_outputs[step_name] = step_output

                            # Transform StepOutput to StepOutputEvent for consistent streaming interface
                            step_output_event = self._transform_step_output_to_event(
                                step_output, workflow_run_response, step_index=i
                            )

                            if step_output.stop:
                                logger.info(f"Early termination requested by step {step_name}")
                                # Update shared media for next step
                                shared_images.extend(step_output.images or [])
                                shared_videos.extend(step_output.videos or [])
                                shared_audio.extend(step_output.audio or [])
                                output_images.extend(step_output.images or [])
                                output_videos.extend(step_output.videos or [])
                                output_audio.extend(step_output.audio or [])

                                # Only yield StepOutputEvent for function executors, not for agents/teams
                                if getattr(step, "executor_type", None) == "function":
                                    yield step_output_event

                                # Break out of the step loop
                                early_termination = True
                                break

                            # Update shared media for next step
                            shared_images.extend(step_output.images or [])
                            shared_videos.extend(step_output.videos or [])
                            shared_audio.extend(step_output.audio or [])
                            output_images.extend(step_output.images or [])
                            output_videos.extend(step_output.videos or [])
                            output_audio.extend(step_output.audio or [])

                            # Only yield StepOutputEvent for generator functions, not for agents/teams
                            if getattr(step, "executor_type", None) == "function":
                                yield step_output_event

                        elif isinstance(event, WorkflowRunResponseEvent):  # type: ignore
                            yield self._handle_event(event, workflow_run_response)  # type: ignore

                        else:
                            # Yield other internal events
                            yield self._handle_event(event, workflow_run_response)  # type: ignore

                    # Break out of main step loop if early termination was requested
                    if "early_termination" in locals() and early_termination:
                        break

                    self._collect_workflow_session_state_from_agents_and_teams()

                # Update the workflow_run_response with completion data
                if collected_step_outputs:
                    workflow_run_response.workflow_metrics = self._aggregate_workflow_metrics(collected_step_outputs)
                    last_output = collected_step_outputs[-1]
                    if isinstance(last_output, list) and last_output:
                        # If it's a list (from Condition/Loop/etc.), use the last one
                        workflow_run_response.content = last_output[-1].content
                    elif not isinstance(last_output, list):
                        # Single StepOutput
                        workflow_run_response.content = last_output.content
                else:
                    workflow_run_response.content = "No steps executed"

                workflow_run_response.step_results = collected_step_outputs
                workflow_run_response.images = output_images
                workflow_run_response.videos = output_videos
                workflow_run_response.audio = output_audio
                workflow_run_response.status = RunStatus.completed

            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")

                from agno.run.workflow import WorkflowErrorEvent

                error_event = WorkflowErrorEvent(
                    run_id=self.run_id or "",
                    workflow_id=self.id,
                    workflow_name=self.name,
                    session_id=self.session_id,
                    error=str(e),
                )

                yield error_event

                # Update workflow_run_response with error
                workflow_run_response.content = error_event.error
                workflow_run_response.status = RunStatus.error

        # Yield workflow completed event
        workflow_completed_event = WorkflowCompletedEvent(
            run_id=workflow_run_response.run_id or "",
            content=workflow_run_response.content,
            workflow_name=workflow_run_response.workflow_name,
            workflow_id=workflow_run_response.workflow_id,
            session_id=workflow_run_response.session_id,
            step_results=workflow_run_response.step_results,  # type: ignore
            metadata=workflow_run_response.metadata,
        )
        yield self._handle_event(workflow_completed_event, workflow_run_response)

        # Store the completed workflow response
        self._save_run_to_storage(workflow_run_response)

    async def _acall_custom_function(
        self, func: Callable, workflow: "Workflow", execution_input: WorkflowExecutionInput, **kwargs: Any
    ) -> Any:
        """Call custom function with only the parameters it expects - handles both async functions and async generators"""
        from inspect import isasyncgenfunction, signature

        sig = signature(func)

        # Build arguments based on what the function actually accepts
        call_kwargs = {}

        # Only add workflow and execution_input if the function expects them
        if "workflow" in sig.parameters:  # type: ignore
            call_kwargs["workflow"] = self
        if "execution_input" in sig.parameters:
            call_kwargs["execution_input"] = execution_input  # type: ignore

        # Add any other kwargs that the function expects
        for param_name in kwargs:
            if param_name in sig.parameters:  # type: ignore
                call_kwargs[param_name] = kwargs[param_name]

        # If function has **kwargs parameter, pass all remaining kwargs
        for param in sig.parameters.values():  # type: ignore
            if param.kind == param.VAR_KEYWORD:
                call_kwargs.update(kwargs)
                break

        try:
            # Check if it's an async generator function
            if isasyncgenfunction(func):
                # For async generators, call the function and return the async generator directly
                return func(**call_kwargs)  # type: ignore
            else:
                # For regular async functions, await the result
                return await func(**call_kwargs)  # type: ignore
        except TypeError as e:
            # If signature inspection fails, fall back to original method
            logger.warning(
                f"Async function signature inspection failed: {e}. Falling back to original calling convention."
            )
            if isasyncgenfunction(func):
                # For async generators, use the same signature inspection logic in fallback
                return func(**call_kwargs)  # type: ignore
            else:
                # For regular async functions, use the same signature inspection logic in fallback
                return await func(**call_kwargs)  # type: ignore

    async def _aexecute(
        self, execution_input: WorkflowExecutionInput, workflow_run_response: WorkflowRunResponse, **kwargs: Any
    ) -> WorkflowRunResponse:
        """Execute a specific pipeline by name asynchronously"""
        from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction

        workflow_run_response.status = RunStatus.running

        if callable(self.steps):
            # Execute the workflow with the custom executor
            content = ""

            if iscoroutinefunction(self.steps):  # type: ignore
                workflow_run_response.content = await self._acall_custom_function(
                    self.steps, self, execution_input, **kwargs
                )
            elif isgeneratorfunction(self.steps):
                for chunk in self.steps(self, execution_input, **kwargs):  # type: ignore[arg-type]
                    if hasattr(chunk, "content") and chunk.content is not None and isinstance(chunk.content, str):
                        content += chunk.content
                    else:
                        content += str(chunk)
                workflow_run_response.content = content
            elif isasyncgenfunction(self.steps):  # type: ignore
                async_gen = await self._acall_custom_function(self.steps, self, execution_input, **kwargs)
                async for chunk in async_gen:
                    if hasattr(chunk, "content") and chunk.content is not None and isinstance(chunk.content, str):
                        content += chunk.content
                    else:
                        content += str(chunk)
                workflow_run_response.content = content
            else:
                workflow_run_response.content = self._call_custom_function(self.steps, self, execution_input, **kwargs)
            workflow_run_response.status = RunStatus.completed

        else:
            try:
                # Track outputs from each step for enhanced data flow
                collected_step_outputs: List[Union[StepOutput, List[StepOutput]]] = []
                previous_step_outputs: Dict[str, StepOutput] = {}

                shared_images: List[ImageArtifact] = execution_input.images or []
                output_images: List[ImageArtifact] = (execution_input.images or []).copy()  # Start with input images
                shared_videos: List[VideoArtifact] = execution_input.videos or []
                output_videos: List[VideoArtifact] = (execution_input.videos or []).copy()  # Start with input videos
                shared_audio: List[AudioArtifact] = execution_input.audio or []
                output_audio: List[AudioArtifact] = (execution_input.audio or []).copy()  # Start with input audio

                for i, step in enumerate(self.steps):  # type: ignore[arg-type]
                    step_name = getattr(step, "name", f"step_{i + 1}")
                    log_debug(f"Async Executing step {i + 1}/{self._get_step_count()}: {step_name}")

                    # Create enhanced StepInput
                    step_input = self._create_step_input(
                        execution_input=execution_input,
                        previous_step_outputs=previous_step_outputs,
                        shared_images=shared_images,
                        shared_videos=shared_videos,
                        shared_audio=shared_audio,
                    )

                    step_output = await step.aexecute(
                        step_input,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        workflow_run_response=workflow_run_response,
                        store_executor_responses=self.store_executor_responses,
                    )  # type: ignore[union-attr]

                    # Update the workflow-level previous_step_outputs dictionary
                    if isinstance(step_output, list):
                        # For multiple outputs (from Loop, Condition, etc.), store the last one
                        if step_output:
                            previous_step_outputs[step_name] = step_output[-1]
                            if any(output.stop for output in step_output):
                                logger.info(f"Early termination requested by step {step_name}")
                                break
                    else:
                        # Single output
                        previous_step_outputs[step_name] = step_output
                        if step_output.stop:
                            logger.info(f"Early termination requested by step {step_name}")
                            break

                    # Update shared media for next step
                    if isinstance(step_output, list):
                        for output in step_output:
                            shared_images.extend(output.images or [])
                            shared_videos.extend(output.videos or [])
                            shared_audio.extend(output.audio or [])
                            output_images.extend(output.images or [])
                            output_videos.extend(output.videos or [])
                            output_audio.extend(output.audio or [])
                    else:
                        shared_images.extend(step_output.images or [])
                        shared_videos.extend(step_output.videos or [])
                        shared_audio.extend(step_output.audio or [])
                        output_images.extend(step_output.images or [])
                        output_videos.extend(step_output.videos or [])
                        output_audio.extend(step_output.audio or [])

                    collected_step_outputs.append(step_output)

                    self._collect_workflow_session_state_from_agents_and_teams()

                # Update the workflow_run_response with completion data
                if collected_step_outputs:
                    workflow_run_response.workflow_metrics = self._aggregate_workflow_metrics(collected_step_outputs)
                    last_output = collected_step_outputs[-1]
                    if isinstance(last_output, list) and last_output:
                        # If it's a list (from Condition/Loop/etc.), use the last one
                        workflow_run_response.content = last_output[-1].content
                    elif not isinstance(last_output, list):
                        # Single StepOutput
                        workflow_run_response.content = last_output.content
                else:
                    workflow_run_response.content = "No steps executed"

                workflow_run_response.step_results = collected_step_outputs
                workflow_run_response.images = output_images
                workflow_run_response.videos = output_videos
                workflow_run_response.audio = output_audio
                workflow_run_response.status = RunStatus.completed

            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                workflow_run_response.status = RunStatus.error
                workflow_run_response.content = f"Workflow execution failed: {e}"

        # Store error response
        self._save_run_to_storage(workflow_run_response)

        return workflow_run_response

    async def _aexecute_stream(
        self,
        execution_input: WorkflowExecutionInput,
        workflow_run_response: WorkflowRunResponse,
        stream_intermediate_steps: bool = False,
        websocket_handler: Optional[WebSocketHandler] = None,
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowRunResponseEvent]:
        """Execute a specific pipeline by name with event streaming"""
        from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction

        workflow_run_response.status = RunStatus.running
        workflow_started_event = WorkflowStartedEvent(
            run_id=workflow_run_response.run_id or "",
            workflow_name=workflow_run_response.workflow_name,
            workflow_id=workflow_run_response.workflow_id,
            session_id=workflow_run_response.session_id,
        )
        yield self._handle_event(workflow_started_event, workflow_run_response, websocket_handler=websocket_handler)

        if callable(self.steps):
            if iscoroutinefunction(self.steps):  # type: ignore
                workflow_run_response.content = await self._acall_custom_function(
                    self.steps, self, execution_input, **kwargs
                )
            elif isgeneratorfunction(self.steps):
                content = ""
                for chunk in self.steps(self, execution_input, **kwargs):  # type: ignore[arg-type]
                    if hasattr(chunk, "content") and chunk.content is not None and isinstance(chunk.content, str):
                        content += chunk.content
                        yield chunk
                    else:
                        content += str(chunk)
                workflow_run_response.content = content
            elif isasyncgenfunction(self.steps):  # type: ignore
                content = ""
                async_gen = await self._acall_custom_function(self.steps, self, execution_input, **kwargs)
                async for chunk in async_gen:
                    if hasattr(chunk, "content") and chunk.content is not None and isinstance(chunk.content, str):
                        content += chunk.content
                        yield chunk
                    else:
                        content += str(chunk)
                workflow_run_response.content = content
            else:
                workflow_run_response.content = self.steps(self, execution_input, **kwargs)
            workflow_run_response.status = RunStatus.completed

        else:
            try:
                # Track outputs from each step for enhanced data flow
                collected_step_outputs: List[Union[StepOutput, List[StepOutput]]] = []
                previous_step_outputs: Dict[str, StepOutput] = {}

                shared_images: List[ImageArtifact] = execution_input.images or []
                output_images: List[ImageArtifact] = (execution_input.images or []).copy()  # Start with input images
                shared_videos: List[VideoArtifact] = execution_input.videos or []
                output_videos: List[VideoArtifact] = (execution_input.videos or []).copy()  # Start with input videos
                shared_audio: List[AudioArtifact] = execution_input.audio or []
                output_audio: List[AudioArtifact] = (execution_input.audio or []).copy()  # Start with input audio

                early_termination = False

                for i, step in enumerate(self.steps):  # type: ignore[arg-type]
                    step_name = getattr(step, "name", f"step_{i + 1}")
                    log_debug(f"Async streaming step {i + 1}/{self._get_step_count()}: {step_name}")

                    # Create enhanced StepInput
                    step_input = self._create_step_input(
                        execution_input=execution_input,
                        previous_step_outputs=previous_step_outputs,
                        shared_images=shared_images,
                        shared_videos=shared_videos,
                        shared_audio=shared_audio,
                    )

                    # Execute step with streaming and yield all events
                    async for event in step.aexecute_stream(  # type: ignore[union-attr]
                        step_input,
                        session_id=self.session_id,
                        user_id=self.user_id,
                        stream_intermediate_steps=stream_intermediate_steps,
                        workflow_run_response=workflow_run_response,
                        step_index=i,
                        store_executor_responses=self.store_executor_responses,
                    ):
                        if isinstance(event, StepOutput):
                            step_output = event
                            collected_step_outputs.append(step_output)

                            # Update the workflow-level previous_step_outputs dictionary
                            previous_step_outputs[step_name] = step_output

                            # Transform StepOutput to StepOutputEvent for consistent streaming interface
                            step_output_event = self._transform_step_output_to_event(
                                step_output, workflow_run_response, step_index=i
                            )

                            if step_output.stop:
                                logger.info(f"Early termination requested by step {step_name}")
                                # Update shared media for next step
                                shared_images.extend(step_output.images or [])
                                shared_videos.extend(step_output.videos or [])
                                shared_audio.extend(step_output.audio or [])
                                output_images.extend(step_output.images or [])
                                output_videos.extend(step_output.videos or [])
                                output_audio.extend(step_output.audio or [])

                                if getattr(step, "executor_type", None) == "function":
                                    yield step_output_event

                                # Break out of the step loop
                                early_termination = True
                                break

                            # Update shared media for next step
                            shared_images.extend(step_output.images or [])
                            shared_videos.extend(step_output.videos or [])
                            shared_audio.extend(step_output.audio or [])
                            output_images.extend(step_output.images or [])
                            output_videos.extend(step_output.videos or [])
                            output_audio.extend(step_output.audio or [])

                            # Only yield StepOutputEvent for generator functions, not for agents/teams
                            if getattr(step, "executor_type", None) == "function":
                                yield step_output_event

                        elif isinstance(event, WorkflowRunResponseEvent):  # type: ignore
                            yield self._handle_event(event, workflow_run_response, websocket_handler=websocket_handler)  # type: ignore

                        else:
                            # Yield other internal events
                            yield self._handle_event(event, workflow_run_response, websocket_handler=websocket_handler)  # type: ignore

                    # Break out of main step loop if early termination was requested
                    if "early_termination" in locals() and early_termination:
                        break

                    self._collect_workflow_session_state_from_agents_and_teams()

                # Update the workflow_run_response with completion data
                if collected_step_outputs:
                    workflow_run_response.workflow_metrics = self._aggregate_workflow_metrics(collected_step_outputs)
                    last_output = collected_step_outputs[-1]
                    if isinstance(last_output, list) and last_output:
                        # If it's a list (from Condition/Loop/etc.), use the last one
                        workflow_run_response.content = last_output[-1].content
                    elif not isinstance(last_output, list):
                        # Single StepOutput
                        workflow_run_response.content = last_output.content
                else:
                    workflow_run_response.content = "No steps executed"

                workflow_run_response.step_results = collected_step_outputs
                workflow_run_response.images = output_images
                workflow_run_response.videos = output_videos
                workflow_run_response.audio = output_audio
                workflow_run_response.status = RunStatus.completed

            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")

                from agno.run.workflow import WorkflowErrorEvent

                error_event = WorkflowErrorEvent(
                    run_id=self.run_id or "",
                    workflow_id=self.id,
                    workflow_name=self.name,
                    session_id=self.session_id,
                    error=str(e),
                )

                yield error_event

                # Update workflow_run_response with error
                workflow_run_response.content = error_event.error
                workflow_run_response.status = RunStatus.error

        # Yield workflow completed event
        workflow_completed_event = WorkflowCompletedEvent(
            run_id=workflow_run_response.run_id or "",
            content=workflow_run_response.content,
            workflow_name=workflow_run_response.workflow_name,
            workflow_id=workflow_run_response.workflow_id,
            session_id=workflow_run_response.session_id,
            step_results=workflow_run_response.step_results,  # type: ignore[arg-type]
            metadata=workflow_run_response.metadata,
        )
        yield self._handle_event(workflow_completed_event, workflow_run_response, websocket_handler=websocket_handler)

        # Store the completed workflow response
        self._save_run_to_storage(workflow_run_response)

    def _update_workflow_session_state(self):
        if not self.workflow_session_state:
            self.workflow_session_state = {}

        self.workflow_session_state.update(
            {
                "workflow_id": self.id,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "session_name": self.session_name,
            }
        )
        if self.name:
            self.workflow_session_state["workflow_name"] = self.name

        return self.workflow_session_state

    async def _arun_background(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        **kwargs: Any,
    ) -> WorkflowRunResponse:
        """Execute workflow in background using asyncio.create_task()"""

        if user_id is not None:
            self.user_id = user_id
        if session_id is not None:
            self.session_id = session_id

        if self.session_id is None:
            self.session_id = str(uuid4())

        self.run_id = str(uuid4())

        self.initialize_workflow()
        self.load_session()
        self._prepare_steps()

        # Create workflow run response with PENDING status
        workflow_run_response = WorkflowRunResponse(
            run_id=self.run_id,
            session_id=self.session_id,
            workflow_id=self.id,
            workflow_name=self.name,
            created_at=int(datetime.now().timestamp()),
            status=RunStatus.pending,
        )

        # Store PENDING response immediately
        self._save_run_to_storage(workflow_run_response)

        # Prepare execution input
        inputs = WorkflowExecutionInput(
            input=input,
            additional_data=additional_data,
            audio=audio,  # type: ignore
            images=images,  # type: ignore
            videos=videos,  # type: ignore
        )

        self.update_agents_and_teams_session_info()

        async def execute_workflow_background():
            """Simple background execution"""
            try:
                # Update status to RUNNING and save
                workflow_run_response.status = RunStatus.running
                self._save_run_to_storage(workflow_run_response)

                await self._aexecute(execution_input=inputs, workflow_run_response=workflow_run_response, **kwargs)

                self._save_run_to_storage(workflow_run_response)

                log_debug(f"Background execution completed with status: {workflow_run_response.status}")

            except Exception as e:
                logger.error(f"Background workflow execution failed: {e}")
                workflow_run_response.status = RunStatus.error
                workflow_run_response.content = f"Background execution failed: {str(e)}"
                self._save_run_to_storage(workflow_run_response)

        # Create and start asyncio task
        loop = asyncio.get_running_loop()
        loop.create_task(execute_workflow_background())

        # Return SAME object that will be updated by background execution
        return workflow_run_response

    async def _arun_background_stream(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream_intermediate_steps: bool = False,
        websocket_handler: Optional[WebSocketHandler] = None,
        **kwargs: Any,
    ) -> WorkflowRunResponse:
        """Execute workflow in background with streaming and WebSocket broadcasting"""

        if user_id is not None:
            self.user_id = user_id
        if session_id is not None:
            self.session_id = session_id

        if self.session_id is None:
            self.session_id = str(uuid4())

        self.run_id = str(uuid4())

        self.initialize_workflow()
        self.load_session()
        self._prepare_steps()

        # Create workflow run response with PENDING status
        workflow_run_response = WorkflowRunResponse(
            run_id=self.run_id,
            session_id=self.session_id,
            workflow_id=self.id,
            workflow_name=self.name,
            created_at=int(datetime.now().timestamp()),
            status=RunStatus.pending,
        )

        # Store PENDING response immediately
        self._save_run_to_storage(workflow_run_response)

        # Prepare execution input
        inputs = WorkflowExecutionInput(
            input=input,
            additional_data=additional_data,
            audio=audio,  # type: ignore
            images=images,  # type: ignore
            videos=videos,  # type: ignore
        )

        self.update_agents_and_teams_session_info()

        async def execute_workflow_background_stream():
            """Background execution with streaming and WebSocket broadcasting"""
            try:
                # Update status to RUNNING and save
                workflow_run_response.status = RunStatus.running
                self._save_run_to_storage(workflow_run_response)

                # Execute with streaming - consume all events (they're auto-broadcast via _handle_event)
                async for event in self._aexecute_stream(
                    execution_input=inputs,
                    workflow_run_response=workflow_run_response,
                    stream_intermediate_steps=stream_intermediate_steps,
                    websocket_handler=websocket_handler,
                    **kwargs,
                ):
                    # Events are automatically broadcast by _handle_event
                    # We just consume them here to drive the execution
                    pass

                self._save_run_to_storage(workflow_run_response)

                log_debug(f"Background streaming execution completed with status: {workflow_run_response.status}")

            except Exception as e:
                logger.error(f"Background streaming workflow execution failed: {e}")
                workflow_run_response.status = RunStatus.error
                workflow_run_response.content = f"Background streaming execution failed: {str(e)}"
                self._save_run_to_storage(workflow_run_response)

        # Create and start asyncio task for background streaming execution
        loop = asyncio.get_running_loop()
        loop.create_task(execute_workflow_background_stream())

        # Return SAME object that will be updated by background execution
        return workflow_run_response

    def get_run(self, run_id: str) -> Optional[WorkflowRunResponse]:
        """Get the status and details of a background workflow run - SIMPLIFIED"""
        if self.db is not None and self.session_id is not None:
            session = self.db.get_session(session_id=self.session_id, session_type=SessionType.WORKFLOW)
            if session and isinstance(session, WorkflowSession) and session.runs:
                # Find the run by ID
                for run in session.runs:
                    if run.run_id == run_id:
                        return run

        return None

    @overload
    def run(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: Literal[False] = False,
        stream_intermediate_steps: Optional[bool] = None,
        background: Optional[bool] = False,
    ) -> WorkflowRunResponse: ...

    @overload
    def run(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: Literal[True] = True,
        stream_intermediate_steps: Optional[bool] = None,
        background: Optional[bool] = False,
    ) -> Iterator[WorkflowRunResponseEvent]: ...

    def run(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: bool = False,
        stream_intermediate_steps: Optional[bool] = None,
        background: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[WorkflowRunResponse, Iterator[WorkflowRunResponseEvent]]:
        """Execute the workflow synchronously with optional streaming"""

        validated_input = self._validate_input(input)
        if validated_input is not None:
            input = validated_input

        if background:
            raise RuntimeError("Background execution is not supported for sync run()")

        self._set_debug_mode()

        log_debug(f"Workflow Run Start: {self.name}", center=True)

        # Use simple defaults
        stream = stream or self.stream or False
        stream_intermediate_steps = stream_intermediate_steps or self.stream_intermediate_steps or False

        # Can't have stream_intermediate_steps if stream is False
        if not stream:
            stream_intermediate_steps = False

        log_debug(f"Stream: {stream}")
        log_debug(f"Total steps: {self._get_step_count()}")

        if user_id is not None:
            self.user_id = user_id
            log_debug(f"User ID: {user_id}")
        if session_id is not None:
            self.session_id = session_id
            log_debug(f"Session ID: {session_id}")

        if self.session_id is None:
            self.session_id = str(uuid4())

        self.run_id = str(uuid4())

        self.initialize_workflow()

        # Load or create session
        self.load_session()

        # Prepare steps
        self._prepare_steps()

        # Create workflow run response that will be updated by reference
        workflow_run_response = WorkflowRunResponse(
            run_id=self.run_id,
            session_id=self.session_id,
            workflow_id=self.id,
            workflow_name=self.name,
            created_at=int(datetime.now().timestamp()),
        )
        self.run_response = workflow_run_response

        inputs = WorkflowExecutionInput(
            input=input,
            additional_data=additional_data,
            audio=audio,  # type: ignore
            images=images,  # type: ignore
            videos=videos,  # type: ignore
        )
        log_debug(
            f"Created pipeline input with session state keys: {list(self.workflow_session_state.keys()) if self.workflow_session_state else 'None'}"
        )

        self.update_agents_and_teams_session_info()

        if stream:
            return self._execute_stream(
                execution_input=inputs,  # type: ignore[arg-type]
                workflow_run_response=workflow_run_response,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )
        else:
            return self._execute(execution_input=inputs, workflow_run_response=workflow_run_response, **kwargs)

    @overload
    async def arun(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: Literal[False] = False,
        stream_intermediate_steps: Optional[bool] = None,
        background: Optional[bool] = False,
        websocket: Optional[WebSocket] = None,
    ) -> WorkflowRunResponse: ...

    @overload
    async def arun(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: Literal[True] = True,
        stream_intermediate_steps: Optional[bool] = None,
        background: Optional[bool] = False,
        websocket: Optional[WebSocket] = None,
    ) -> AsyncIterator[WorkflowRunResponseEvent]: ...

    async def arun(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: bool = False,
        stream_intermediate_steps: Optional[bool] = False,
        background: Optional[bool] = False,
        websocket: Optional[WebSocket] = None,
        **kwargs: Any,
    ) -> Union[WorkflowRunResponse, AsyncIterator[WorkflowRunResponseEvent]]:
        """Execute the workflow synchronously with optional streaming"""

        validated_input = self._validate_input(input)
        if validated_input is not None:
            input = validated_input

        websocket_handler = None
        if websocket:
            from agno.workflow.types import WebSocketHandler

            websocket_handler = WebSocketHandler(websocket=websocket)

        if background:
            if stream and websocket:
                # Background + Streaming + WebSocket = Real-time events
                return await self._arun_background_stream(
                    input=input,
                    additional_data=additional_data,
                    user_id=user_id,
                    session_id=session_id,
                    audio=audio,
                    images=images,
                    videos=videos,
                    stream_intermediate_steps=stream_intermediate_steps or False,
                    websocket_handler=websocket_handler,
                    **kwargs,
                )
            elif stream and not websocket:
                # Background + Streaming but no WebSocket = Not supported
                raise ValueError("Background streaming execution requires a WebSocket for real-time events")
            else:
                # Background + Non-streaming = Polling (existing)
                return await self._arun_background(
                    input=input,
                    additional_data=additional_data,
                    user_id=user_id,
                    session_id=session_id,
                    audio=audio,
                    images=images,
                    videos=videos,
                    **kwargs,
                )

        self._set_debug_mode()

        log_debug(f"Async Workflow Run Start: {self.name}", center=True)

        # Use simple defaults
        stream = stream or self.stream or False
        stream_intermediate_steps = stream_intermediate_steps or self.stream_intermediate_steps or False

        # Can't have stream_intermediate_steps if stream is False
        if not stream:
            stream_intermediate_steps = False

        log_debug(f"Stream: {stream}")

        # Set user_id and session_id if provided
        if user_id is not None:
            self.user_id = user_id
            log_debug(f"User ID: {user_id}")
        if session_id is not None:
            self.session_id = session_id
            log_debug(f"Session ID: {session_id}")

        if self.session_id is None:
            self.session_id = str(uuid4())

        self.run_id = str(uuid4())

        self.initialize_workflow()

        # Load or create session
        self.load_session()

        # Prepare steps
        self._prepare_steps()

        # Create workflow run response that will be updated by reference
        workflow_run_response = WorkflowRunResponse(
            run_id=self.run_id,
            session_id=self.session_id,
            workflow_id=self.id,
            workflow_name=self.name,
            created_at=int(datetime.now().timestamp()),
        )
        self.run_response = workflow_run_response

        inputs = WorkflowExecutionInput(
            input=input,
            additional_data=additional_data,
            audio=audio,  # type: ignore
            images=images,  # type: ignore
            videos=videos,  # type: ignore
        )
        log_debug(
            f"Created async pipeline input with session state keys: {list(self.workflow_session_state.keys()) if self.workflow_session_state else 'None'}"
        )

        self.update_agents_and_teams_session_info()

        if stream:
            return self._aexecute_stream(
                execution_input=inputs,
                workflow_run_response=workflow_run_response,
                stream_intermediate_steps=stream_intermediate_steps,
                websocket=websocket,
                **kwargs,
            )
        else:
            return await self._aexecute(
                execution_input=inputs, workflow_run_response=workflow_run_response, websocket=websocket, **kwargs
            )

    def _prepare_steps(self):
        """Prepare the steps for execution"""
        if not callable(self.steps) and self.steps is not None:
            prepared_steps: List[Union[Step, Steps, Loop, Parallel, Condition, Router]] = []
            for i, step in enumerate(self.steps):  # type: ignore
                if callable(step) and hasattr(step, "__name__"):
                    step_name = step.__name__
                    log_debug(f"Step {i + 1}: Wrapping callable function '{step_name}'")
                    prepared_steps.append(Step(name=step_name, description="User-defined callable step", executor=step))
                elif isinstance(step, Agent):
                    step_name = step.name or f"step_{i + 1}"
                    log_debug(f"Step {i + 1}: Agent '{step_name}'")
                    prepared_steps.append(Step(name=step_name, description=step.description, agent=step))
                elif isinstance(step, Team):
                    step_name = step.name or f"step_{i + 1}"
                    log_debug(f"Step {i + 1}: Team '{step_name}' with {len(step.members)} members")
                    prepared_steps.append(Step(name=step_name, description=step.description, team=step))
                elif isinstance(step, (Step, Steps, Loop, Parallel, Condition, Router)):
                    step_type = type(step).__name__
                    step_name = getattr(step, "name", f"unnamed_{step_type.lower()}")
                    log_debug(f"Step {i + 1}: {step_type} '{step_name}'")
                    prepared_steps.append(step)
                else:
                    raise ValueError(f"Invalid step type: {type(step).__name__}")

            self.steps = prepared_steps  # type: ignore
            log_debug("Step preparation completed")

    def get_workflow_session(self) -> WorkflowSession:
        """Get a WorkflowSession object for storage"""
        workflow_data = {}

        if self.steps and not callable(self.steps):
            steps_dict = []
            for step in self.steps:  # type: ignore
                # TODO: The step should have a type field
                step_type = STEP_TYPE_MAPPING[type(step)]
                step_dict = {
                    "name": step.name if hasattr(step, "name") else step.__name__,
                    "description": step.description if hasattr(step, "description") else "User-defined callable step",
                    "type": step_type.value,
                }
                steps_dict.append(step_dict)

            workflow_data["steps"] = steps_dict

        elif callable(self.steps):
            workflow_data["steps"] = [
                {
                    "name": "Custom Function",
                    "description": "User-defined callable workflow",
                    "type": "Callable",
                }
            ]

        if self.session_id is None:
            raise ValueError("Session ID is required")

        return WorkflowSession(
            session_id=self.session_id,
            user_id=self.user_id,
            workflow_id=self.id,
            workflow_name=self.name,
            runs=self.workflow_session.runs if self.workflow_session else [],
            workflow_data=workflow_data,
            session_data=self.workflow_session_state,
        )

    def load_workflow_session(self, session: WorkflowSession):
        """Load workflow session from storage"""
        if self.id is None and session.workflow_id is not None:
            self.id = session.workflow_id
        if self.user_id is None and session.user_id is not None:
            self.user_id = session.user_id
        if self.session_id is None and session.session_id is not None:
            self.session_id = session.session_id
        if self.name is None and session.workflow_name is not None:
            self.name = session.workflow_name

        self.workflow_session = session
        log_debug(f"Loaded WorkflowSession: {session.session_id}")

    def read_from_storage(self) -> Optional[WorkflowSession]:
        """Load the WorkflowSession from storage"""
        if self.db is not None and self.session_id is not None:
            session = self.db.get_session(session_id=self.session_id, session_type=SessionType.WORKFLOW)
            if session and isinstance(session, WorkflowSession):
                self.load_workflow_session(session)
                return session
        return None

    def write_to_storage(self) -> Optional[WorkflowSession]:
        """Save the WorkflowSession to storage"""
        if self.db is not None:
            session_to_save = self.get_workflow_session()
            saved_session = self.db.upsert_session(session=session_to_save)
            if saved_session and isinstance(saved_session, WorkflowSession):
                self.workflow_session = saved_session
                return saved_session
        return None

    def load_session(self, force: bool = False) -> Optional[str]:
        """Load an existing session from storage or create a new one"""
        if self.workflow_session is not None and not force:
            if self.session_id is not None and self.workflow_session.session_id == self.session_id:
                log_debug("Using existing workflow session")
                return self.workflow_session.session_id

        if self.db is not None:
            # Try to load existing session
            existing_session = self.read_from_storage()

            # Create new session if it doesn't exist
            if existing_session is None:
                log_debug("Creating new WorkflowSession")

                # Ensure we have a session_id
                if self.session_id is None:
                    self.session_id = str(uuid4())

                self.workflow_session = WorkflowSession(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    workflow_id=self.id,
                    workflow_name=self.name,
                )
                saved_session = self.write_to_storage()
                if saved_session is None:
                    raise Exception("Failed to create new WorkflowSession in storage")
                log_debug(f"Created WorkflowSession: {saved_session.session_id}")

        return self.session_id

    def new_session(self) -> None:
        """Create a new workflow session"""
        log_debug("Creating new workflow session")

        self.workflow_session = None
        self.session_id = str(uuid4())

        log_debug(f"New session ID: {self.session_id}")
        self.load_session(force=True)

    def _format_step_content_for_display(self, step_output: StepOutput) -> str:
        """Format content for display, handling structured outputs. Works for both raw content and StepOutput objects."""
        # If it's a StepOutput, extract the content
        if hasattr(step_output, "content"):
            actual_content = step_output.content
        else:
            actual_content = step_output

        if not actual_content:
            return ""

        # If it's already a string, return as-is
        if isinstance(actual_content, str):
            return actual_content

        # If it's a structured output (BaseModel or dict), format it nicely
        if isinstance(actual_content, BaseModel):
            return (
                f"**Structured Output:**\n\n```json\n{actual_content.model_dump_json(indent=2, exclude_none=True)}\n```"
            )
        elif isinstance(actual_content, (dict, list)):
            import json

            return f"**Structured Output:**\n\n```json\n{json.dumps(actual_content, indent=2, default=str)}\n```"
        else:
            # Fallback to string conversion
            return str(actual_content)

    def print_response(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: bool = False,
        stream_intermediate_steps: bool = False,
        markdown: bool = True,
        show_time: bool = True,
        show_step_details: bool = True,
        console: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Print workflow execution with rich formatting and optional streaming

        Args:
            message: The main query/input for the workflow
            message_data: Attached message data to the input
            user_id: User ID
            session_id: Session ID
            audio: Audio input
            images: Image input
            videos: Video input
            stream: Whether to stream the response content
            stream_intermediate_steps: Whether to stream intermediate steps
            markdown: Whether to render content as markdown
            show_time: Whether to show execution time
            show_step_details: Whether to show individual step outputs
            console: Rich console instance (optional)
        """

        stream_intermediate_steps = stream_intermediate_steps or self.stream_intermediate_steps or False
        stream = stream or self.stream or False

        if stream:
            self._print_response_stream(
                input=input,
                user_id=user_id,
                session_id=session_id,
                additional_data=additional_data,
                audio=audio,
                images=images,
                videos=videos,
                stream_intermediate_steps=stream_intermediate_steps,
                markdown=markdown,
                show_time=show_time,
                show_step_details=show_step_details,
                console=console,
                **kwargs,
            )
        else:
            self._print_response(
                input=input,
                user_id=user_id,
                session_id=session_id,
                additional_data=additional_data,
                audio=audio,
                images=images,
                videos=videos,
                markdown=markdown,
                show_time=show_time,
                show_step_details=show_step_details,
                console=console,
                **kwargs,
            )

    def _print_response(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        markdown: bool = True,
        show_time: bool = True,
        show_step_details: bool = True,
        console: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Print workflow execution with rich formatting (non-streaming)"""
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.status import Status
        from rich.text import Text

        from agno.utils.response import create_panel
        from agno.utils.timer import Timer

        if console is None:
            from agno.cli.console import console

        # Show workflow info
        media_info = []
        if audio:
            media_info.append(f"Audio files: {len(audio)}")
        if images:
            media_info.append(f"Images: {len(images)}")
        if videos:
            media_info.append(f"Videos: {len(videos)}")

        workflow_info = f"""**Workflow:** {self.name}"""
        if self.description:
            workflow_info += f"""\n\n**Description:** {self.description}"""
        workflow_info += f"""\n\n**Steps:** {self._get_step_count()} steps"""
        if message:
            if isinstance(message, str):
                workflow_info += f"""\n\n**Message:** {message}"""
            else:
                # Handle structured input message
                if isinstance(message, BaseModel):
                    data_display = message.model_dump_json(indent=2, exclude_none=True)
                elif isinstance(message, (dict, list)):
                    import json

                    data_display = json.dumps(message, indent=2, default=str)
                else:
                    data_display = str(message)
                workflow_info += f"""\n\n**Structured Input:**\n```json\n{data_display}\n```"""
        if user_id:
            workflow_info += f"""\n\n**User ID:** {user_id}"""
        if session_id:
            workflow_info += f"""\n\n**Session ID:** {session_id}"""
        workflow_info = workflow_info.strip()

        workflow_panel = create_panel(
            content=Markdown(workflow_info) if markdown else workflow_info,
            title="Workflow Information",
            border_style="cyan",
        )
        console.print(workflow_panel)  # type: ignore

        # Start timer
        response_timer = Timer()
        response_timer.start()

        with Live(console=console) as live_log:
            status = Status("Starting workflow...", spinner="dots")
            live_log.update(status)

            try:
                # Execute workflow and get the response directly
                workflow_response: WorkflowRunResponse = self.run(
                    input=input,
                    user_id=user_id,
                    session_id=session_id,
                    additional_data=additional_data,
                    audio=audio,
                    images=images,
                    videos=videos,
                    **kwargs,
                )  # type: ignore

                response_timer.stop()

                if show_step_details and workflow_response.step_results:
                    for i, step_output in enumerate(workflow_response.step_results):
                        # Handle both single StepOutput and List[StepOutput] (from loop/parallel steps)
                        if isinstance(step_output, list):
                            # This is a loop or parallel step with multiple outputs
                            for j, sub_step_output in enumerate(step_output):
                                if sub_step_output.content:
                                    formatted_content = self._format_step_content_for_display(sub_step_output)
                                    step_panel = create_panel(
                                        content=Markdown(formatted_content) if markdown else formatted_content,
                                        title=f"Step {i + 1}.{j + 1}: {sub_step_output.step_name} (Completed)",
                                        border_style="orange3",
                                    )
                                    console.print(step_panel)  # type: ignore
                        else:
                            # This is a regular single step
                            if step_output.content:
                                formatted_content = self._format_step_content_for_display(step_output)
                                step_panel = create_panel(
                                    content=Markdown(formatted_content) if markdown else formatted_content,
                                    title=f"Step {i + 1}: {step_output.step_name} (Completed)",
                                    border_style="orange3",
                                )
                                console.print(step_panel)  # type: ignore

                # For callable functions, show the content directly since there are no step_results
                elif show_step_details and callable(self.steps) and workflow_response.content:
                    step_panel = create_panel(
                        content=Markdown(workflow_response.content) if markdown else workflow_response.content,  # type: ignore
                        title="Custom Function (Completed)",
                        border_style="orange3",
                    )
                    console.print(step_panel)  # type: ignore

                # Show final summary
                if workflow_response.metadata:
                    status = workflow_response.status.value  # type: ignore
                    summary_content = ""
                    summary_content += f"""\n\n**Status:** {status}"""
                    summary_content += f"""\n\n**Steps Completed:** {len(workflow_response.step_results) if workflow_response.step_results else 0}"""
                    summary_content = summary_content.strip()

                    summary_panel = create_panel(
                        content=Markdown(summary_content) if markdown else summary_content,
                        title="Execution Summary",
                        border_style="blue",
                    )
                    console.print(summary_panel)  # type: ignore

                live_log.update("")

                # Final completion message
                if show_time:
                    completion_text = Text(f"Completed in {response_timer.elapsed:.1f}s", style="bold green")
                    console.print(completion_text)  # type: ignore

            except Exception as e:
                import traceback

                traceback.print_exc()
                response_timer.stop()
                error_panel = create_panel(
                    content=f"Workflow execution failed: {str(e)}", title="Execution Error", border_style="red"
                )
                console.print(error_panel)  # type: ignore

    def _print_response_stream(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream_intermediate_steps: bool = False,
        markdown: bool = True,
        show_time: bool = True,
        show_step_details: bool = True,
        console: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Print workflow execution with clean streaming"""
        from rich.console import Group
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.status import Status
        from rich.text import Text

        from agno.utils.response import create_panel
        from agno.utils.timer import Timer

        if console is None:
            from agno.cli.console import console

        stream_intermediate_steps = True  # With streaming print response, we need to stream intermediate steps

        # Show workflow info (same as before)
        media_info = []
        if audio:
            media_info.append(f"Audio files: {len(audio)}")
        if images:
            media_info.append(f"Images: {len(images)}")
        if videos:
            media_info.append(f"Videos: {len(videos)}")

        workflow_info = f"""**Workflow:** {self.name}"""
        if self.description:
            workflow_info += f"""\n\n**Description:** {self.description}"""
        workflow_info += f"""\n\n**Steps:** {self._get_step_count()} steps"""
        if message:
            if isinstance(message, str):
                workflow_info += f"""\n\n**Message:** {message}"""
            else:
                # Handle structured input message
                if isinstance(message, BaseModel):
                    data_display = message.model_dump_json(indent=2, exclude_none=True)
                elif isinstance(message, (dict, list)):
                    import json

                    data_display = json.dumps(message, indent=2, default=str)
                else:
                    data_display = str(message)
                workflow_info += f"""\n\n**Structured Input:**\n```json\n{data_display}\n```"""
        if user_id:
            workflow_info += f"""\n\n**User ID:** {user_id}"""
        if session_id:
            workflow_info += f"""\n\n**Session ID:** {session_id}"""
        workflow_info = workflow_info.strip()

        workflow_panel = create_panel(
            content=Markdown(workflow_info) if markdown else workflow_info,
            title="Workflow Information",
            border_style="cyan",
        )
        console.print(workflow_panel)  # type: ignore

        # Start timer
        response_timer = Timer()
        response_timer.start()

        # Streaming execution variables with smart step tracking
        current_step_content = ""
        current_step_name = ""
        current_step_index = 0
        step_results = []
        step_started_printed = False
        is_callable_function = callable(self.steps)

        # Smart step hierarchy tracking
        current_primitive_context = None  # Current primitive being executed (parallel, loop, etc.)
        step_display_cache = {}  # type: ignore

        def get_step_display_number(step_index: Union[int, tuple], step_name: str = "") -> str:
            """Generate clean two-level step numbering: x.y format only"""

            # Handle tuple format for child steps
            if isinstance(step_index, tuple):
                if len(step_index) >= 2:
                    parent_idx, sub_idx = step_index[0], step_index[1]

                    # Extract base parent index if it's nested
                    if isinstance(parent_idx, tuple):
                        base_idx = parent_idx[0] if len(parent_idx) > 0 else 0
                        while isinstance(base_idx, tuple) and len(base_idx) > 0:
                            base_idx = base_idx[0]
                    else:
                        base_idx = parent_idx

                    # Check context for parallel special case
                    if current_primitive_context and current_primitive_context["type"] == "parallel":
                        # For parallel child steps, all get the same number based on their actual step_index
                        return f"Step {base_idx + 1}.{sub_idx + 1}"
                    elif current_primitive_context and current_primitive_context["type"] == "loop":
                        iteration = current_primitive_context.get("current_iteration", 1)
                        return f"Step {base_idx + 1}.{sub_idx + 1} (Iteration {iteration})"
                    else:
                        # Regular child step numbering
                        return f"Step {base_idx + 1}.{sub_idx + 1}"  # type: ignore
                else:
                    # Single element tuple - treat as main step
                    return f"Step {step_index[0] + 1}"

            # Handle integer step_index - main step
            if not current_primitive_context:
                # Regular main step
                return f"Step {step_index + 1}"
            else:
                # This shouldn't happen with the new logic, but fallback
                return f"Step {step_index + 1}"

        with Live(console=console, refresh_per_second=10) as live_log:
            status = Status("Starting workflow...", spinner="dots")
            live_log.update(status)

            try:
                for response in self.run(
                    input=input,
                    user_id=user_id,
                    session_id=session_id,
                    additional_data=additional_data,
                    audio=audio,
                    images=images,
                    videos=videos,
                    stream=True,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                ):  # type: ignore
                    # Handle the new event types
                    if isinstance(response, WorkflowStartedEvent):
                        status.update("Workflow started...")
                        if is_callable_function:
                            current_step_name = "Custom Function"
                            current_step_index = 0
                        live_log.update(status)

                    elif isinstance(response, StepStartedEvent):
                        current_step_name = response.step_name or "Unknown"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Generate smart step number
                        step_display = get_step_display_number(current_step_index, current_step_name)
                        status.update(f"Starting {step_display}: {current_step_name}...")
                        live_log.update(status)

                    elif isinstance(response, StepCompletedEvent):
                        step_name = response.step_name or "Unknown"
                        step_index = response.step_index or 0

                        # Generate smart step number for completion (will use cached value)
                        step_display = get_step_display_number(step_index, step_name)
                        status.update(f"Completed {step_display}: {step_name}")

                        if response.content:
                            step_results.append(
                                {
                                    "step_name": step_name,
                                    "step_index": step_index,
                                    "content": response.content,
                                    "event": response.event,
                                }
                            )

                        # Print the final step result in orange (only once)
                        if show_step_details and current_step_content and not step_started_printed:
                            live_log.update(status, refresh=True)

                            final_step_panel = create_panel(
                                content=Markdown(current_step_content) if markdown else current_step_content,
                                title=f"{step_display}: {step_name} (Completed)",
                                border_style="orange3",
                            )
                            console.print(final_step_panel)  # type: ignore
                            step_started_printed = True

                    elif isinstance(response, LoopExecutionStartedEvent):
                        current_step_name = response.step_name or "Loop"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up loop context
                        current_primitive_context = {
                            "type": "loop",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "current_iteration": 1,
                            "max_iterations": response.max_iterations,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        status.update(
                            f"Starting loop: {current_step_name} (max {response.max_iterations} iterations)..."
                        )
                        live_log.update(status)

                    elif isinstance(response, LoopIterationStartedEvent):
                        if current_primitive_context and current_primitive_context["type"] == "loop":
                            current_primitive_context["current_iteration"] = response.iteration
                            current_primitive_context["sub_step_counter"] = 0  # Reset for new iteration
                            # Clear cache for new iteration
                            step_display_cache.clear()

                        status.update(
                            f"Loop iteration {response.iteration}/{response.max_iterations}: {response.step_name}..."
                        )
                        live_log.update(status)

                    elif isinstance(response, LoopIterationCompletedEvent):
                        status.update(
                            f"Completed iteration {response.iteration}/{response.max_iterations}: {response.step_name}"
                        )

                    elif isinstance(response, LoopExecutionCompletedEvent):
                        step_name = response.step_name or "Loop"
                        step_index = response.step_index or 0

                        status.update(f"Completed loop: {step_name} ({response.total_iterations} iterations)")
                        live_log.update(status, refresh=True)

                        # Print loop summary
                        if show_step_details:
                            summary_content = "**Loop Summary:**\n\n"
                            summary_content += (
                                f"- Total iterations: {response.total_iterations}/{response.max_iterations}\n"
                            )
                            summary_content += (
                                f"- Total steps executed: {sum(len(iteration) for iteration in response.all_results)}\n"
                            )

                            loop_summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title=f"Loop {step_name} (Completed)",
                                border_style="yellow",
                            )
                            console.print(loop_summary_panel)  # type: ignore

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()
                        step_started_printed = True

                    elif isinstance(response, ParallelExecutionStartedEvent):
                        current_step_name = response.step_name or "Parallel Steps"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up parallel context
                        current_primitive_context = {
                            "type": "parallel",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "total_steps": response.parallel_step_count,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        # Print parallel execution summary panel
                        live_log.update(status, refresh=True)
                        parallel_summary = f"**Parallel Steps:** {response.parallel_step_count}"
                        # Use get_step_display_number for consistent numbering
                        step_display = get_step_display_number(current_step_index, current_step_name)
                        parallel_panel = create_panel(
                            content=Markdown(parallel_summary) if markdown else parallel_summary,
                            title=f"{step_display}: {current_step_name}",
                            border_style="cyan",
                        )
                        console.print(parallel_panel)  # type: ignore

                        status.update(
                            f"Starting parallel execution: {current_step_name} ({response.parallel_step_count} steps)..."
                        )
                        live_log.update(status)

                    elif isinstance(response, ParallelExecutionCompletedEvent):
                        step_name = response.step_name or "Parallel Steps"
                        step_index = response.step_index or 0

                        status.update(f"Completed parallel execution: {step_name}")

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()

                    elif isinstance(response, ConditionExecutionStartedEvent):
                        current_step_name = response.step_name or "Condition"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up condition context
                        current_primitive_context = {
                            "type": "condition",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "condition_result": response.condition_result,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        condition_text = "met" if response.condition_result else "not met"
                        status.update(f"Starting condition: {current_step_name} (condition {condition_text})...")
                        live_log.update(status)

                    elif isinstance(response, ConditionExecutionCompletedEvent):
                        step_name = response.step_name or "Condition"
                        step_index = response.step_index or 0

                        status.update(f"Completed condition: {step_name}")

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()

                    elif isinstance(response, RouterExecutionStartedEvent):
                        current_step_name = response.step_name or "Router"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up router context
                        current_primitive_context = {
                            "type": "router",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "selected_steps": response.selected_steps,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        selected_steps_text = ", ".join(response.selected_steps) if response.selected_steps else "none"
                        status.update(f"Starting router: {current_step_name} (selected: {selected_steps_text})...")
                        live_log.update(status)

                    elif isinstance(response, RouterExecutionCompletedEvent):
                        step_name = response.step_name or "Router"
                        step_index = response.step_index or 0

                        status.update(f"Completed router: {step_name}")

                        # Print router summary
                        if show_step_details:
                            selected_steps_text = (
                                ", ".join(response.selected_steps) if response.selected_steps else "none"
                            )
                            summary_content = "**Router Summary:**\n\n"
                            summary_content += f"- Selected steps: {selected_steps_text}\n"
                            summary_content += f"- Executed steps: {response.executed_steps or 0}\n"

                            router_summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title=f"Router {step_name} (Completed)",
                                border_style="purple",
                            )
                            console.print(router_summary_panel)  # type: ignore

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()
                        step_started_printed = True

                    elif isinstance(response, StepsExecutionStartedEvent):
                        current_step_name = response.step_name or "Steps"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False
                        status.update(f"Starting steps: {current_step_name} ({response.steps_count} steps)...")
                        live_log.update(status)

                    elif isinstance(response, StepsExecutionCompletedEvent):
                        step_name = response.step_name or "Steps"
                        step_index = response.step_index or 0

                        status.update(f"Completed steps: {step_name}")

                        # Add results from executed steps to step_results
                        if response.step_results:
                            for i, step_result in enumerate(response.step_results):
                                # Use the same numbering system as other primitives
                                step_display_number = get_step_display_number(step_index, step_result.step_name or "")
                                step_results.append(
                                    {
                                        "step_name": f"{step_display_number}: {step_result.step_name}",
                                        "step_index": step_index,
                                        "content": step_result.content,
                                        "event": "StepsStepResult",
                                    }
                                )

                        # Print steps summary
                        if show_step_details:
                            summary_content = "**Steps Summary:**\n\n"
                            summary_content += f"- Total steps: {response.steps_count or 0}\n"
                            summary_content += f"- Executed steps: {response.executed_steps or 0}\n"

                            steps_summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title=f"Steps {step_name} (Completed)",
                                border_style="yellow",
                            )
                            console.print(steps_summary_panel)  # type: ignore

                        step_started_printed = True

                    elif isinstance(response, WorkflowCompletedEvent):
                        status.update("Workflow completed!")

                        # For callable functions, print the final content block here since there are no step events
                        if (
                            is_callable_function
                            and show_step_details
                            and current_step_content
                            and not step_started_printed
                        ):
                            final_step_panel = create_panel(
                                content=Markdown(current_step_content) if markdown else current_step_content,
                                title="Custom Function (Completed)",
                                border_style="orange3",
                            )
                            console.print(final_step_panel)  # type: ignore
                            step_started_printed = True

                        live_log.update(status, refresh=True)

                        # Show final summary
                        if response.metadata:
                            status = response.status
                            summary_content = ""
                            summary_content += f"""\n\n**Status:** {status}"""
                            summary_content += f"""\n\n**Steps Completed:** {len(response.step_results) if response.step_results else 0}"""
                            summary_content = summary_content.strip()

                            summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title="Execution Summary",
                                border_style="blue",
                            )
                            console.print(summary_panel)  # type: ignore

                    else:
                        # Handle streaming content
                        if isinstance(response, str):
                            response_str = response
                        elif isinstance(response, StepOutputEvent):
                            response_str = response.content or ""  # type: ignore
                        else:
                            from agno.run.response import RunResponseContentEvent
                            from agno.run.team import RunResponseContentEvent as TeamRunResponseContentEvent

                            current_step_executor_type = None
                            # Handle both integer and tuple step indices for parallel execution
                            actual_step_index = current_step_index
                            if isinstance(current_step_index, tuple):
                                # For tuple indices, use the first element (parent step index)
                                actual_step_index = current_step_index[0]
                                # If it's nested tuple, keep extracting until we get an integer
                                while isinstance(actual_step_index, tuple) and len(actual_step_index) > 0:
                                    actual_step_index = actual_step_index[0]

                            if not is_callable_function and self.steps and actual_step_index < len(self.steps):  # type: ignore
                                step = self.steps[actual_step_index]  # type: ignore
                                if hasattr(step, "executor_type"):
                                    current_step_executor_type = step.executor_type

                            # Check if this is a streaming content event from agent or team
                            if isinstance(response, (TeamRunResponseContentEvent, WorkflowRunResponseEvent)):  # type: ignore
                                # Check if this is a team's final structured output
                                is_structured_output = (
                                    isinstance(response, TeamRunResponseContentEvent)
                                    and hasattr(response, "content_type")
                                    and response.content_type != "str"
                                    and response.content_type != ""
                                )
                                response_str = response.content  # type: ignore
                            elif isinstance(response, RunResponseContentEvent) and current_step_executor_type != "team":
                                response_str = response.content  # type: ignore
                            else:
                                continue

                        # Use the unified formatting function for consistency
                        response_str = self._format_step_content_for_display(response_str)  # type: ignore

                        # Filter out empty responses and add to current step content
                        if response_str and response_str.strip():
                            # If it's a structured output from a team, replace the content instead of appending
                            if "is_structured_output" in locals() and is_structured_output:
                                current_step_content = response_str
                            else:
                                current_step_content += response_str

                            # Live update the step panel with streaming content
                            if show_step_details and not step_started_printed:
                                # Generate smart step number for streaming title (will use cached value)
                                step_display = get_step_display_number(current_step_index, current_step_name)
                                title = f"{step_display}: {current_step_name} (Streaming...)"
                                if is_callable_function:
                                    title = "Custom Function (Streaming...)"

                                # Show the streaming content live in orange panel
                                live_step_panel = create_panel(
                                    content=Markdown(current_step_content) if markdown else current_step_content,
                                    title=title,
                                    border_style="orange3",
                                )

                                # Create group with status and current step content
                                group = Group(status, live_step_panel)
                                live_log.update(group)

                response_timer.stop()

                live_log.update("")

                # Final completion message
                if show_time:
                    completion_text = Text(f"Completed in {response_timer.elapsed:.1f}s", style="bold green")
                    console.print(completion_text)  # type: ignore

            except Exception as e:
                import traceback

                traceback.print_exc()
                response_timer.stop()
                error_panel = create_panel(
                    content=f"Workflow execution failed: {str(e)}", title="Execution Error", border_style="red"
                )
                console.print(error_panel)  # type: ignore

    async def aprint_response(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream: bool = False,
        stream_intermediate_steps: bool = False,
        markdown: bool = True,
        show_time: bool = True,
        show_step_details: bool = True,
        console: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Print workflow execution with rich formatting and optional streaming

        Args:
            message: The main message/input for the workflow
            message_data: Attached message data to the input
            user_id: User ID
            session_id: Session ID
            audio: Audio input
            images: Image input
            videos: Video input
            stream_intermediate_steps: Whether to stream intermediate steps
            stream: Whether to stream the response content
            markdown: Whether to render content as markdown
            show_time: Whether to show execution time
            show_step_details: Whether to show individual step outputs
            console: Rich console instance (optional)
        """
        if stream:
            await self._aprint_response_stream(
                input=input,
                additional_data=additional_data,
                user_id=user_id,
                session_id=session_id,
                audio=audio,
                images=images,
                videos=videos,
                stream_intermediate_steps=stream_intermediate_steps,
                markdown=markdown,
                show_time=show_time,
                show_step_details=show_step_details,
                console=console,
                **kwargs,
            )
        else:
            await self._aprint_response(
                input=input,
                additional_data=additional_data,
                user_id=user_id,
                session_id=session_id,
                audio=audio,
                images=images,
                videos=videos,
                markdown=markdown,
                show_time=show_time,
                show_step_details=show_step_details,
                console=console,
                **kwargs,
            )

    async def _aprint_response(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        markdown: bool = True,
        show_time: bool = True,
        show_step_details: bool = True,
        console: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Print workflow execution with rich formatting (non-streaming)"""
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.status import Status
        from rich.text import Text

        from agno.utils.response import create_panel
        from agno.utils.timer import Timer

        if console is None:
            from agno.cli.console import console

        # Show workflow info
        media_info = []
        if audio:
            media_info.append(f"Audio files: {len(audio)}")
        if images:
            media_info.append(f"Images: {len(images)}")
        if videos:
            media_info.append(f"Videos: {len(videos)}")

        workflow_info = f"""**Workflow:** {self.name}"""
        if self.description:
            workflow_info += f"""\n\n**Description:** {self.description}"""
        workflow_info += f"""\n\n**Steps:** {self._get_step_count()} steps"""
        if message:
            if isinstance(message, str):
                workflow_info += f"""\n\n**Message:** {message}"""
            else:
                # Handle structured input message
                if isinstance(message, BaseModel):
                    data_display = message.model_dump_json(indent=2, exclude_none=True)
                elif isinstance(message, (dict, list)):
                    import json

                    data_display = json.dumps(message, indent=2, default=str)
                else:
                    data_display = str(message)
                workflow_info += f"""\n\n**Structured Input:**\n```json\n{data_display}\n```"""
        if user_id:
            workflow_info += f"""\n\n**User ID:** {user_id}"""
        if session_id:
            workflow_info += f"""\n\n**Session ID:** {session_id}"""
        workflow_info = workflow_info.strip()

        workflow_panel = create_panel(
            content=Markdown(workflow_info) if markdown else workflow_info,
            title="Workflow Information",
            border_style="cyan",
        )
        console.print(workflow_panel)  # type: ignore

        # Start timer
        response_timer = Timer()
        response_timer.start()

        with Live(console=console) as live_log:
            status = Status("Starting async workflow...\n", spinner="dots")
            live_log.update(status)

            try:
                # Execute workflow and get the response directly
                workflow_response: WorkflowRunResponse = await self.arun(
                    input=input,
                    additional_data=additional_data,
                    user_id=user_id,
                    session_id=session_id,
                    audio=audio,
                    images=images,
                    videos=videos,
                    **kwargs,
                )  # type: ignore

                response_timer.stop()

                # Show individual step responses if available
                if show_step_details and workflow_response.step_results:
                    for i, step_output in enumerate(workflow_response.step_results):
                        # Handle both single StepOutput and List[StepOutput] (from loop/parallel steps)
                        if isinstance(step_output, list):
                            # This is a loop or parallel step with multiple outputs
                            for j, sub_step_output in enumerate(step_output):
                                if sub_step_output.content:
                                    formatted_content = self._format_step_content_for_display(sub_step_output)
                                    step_panel = create_panel(
                                        content=Markdown(formatted_content) if markdown else formatted_content,
                                        title=f"Step {i + 1}.{j + 1}: {sub_step_output.step_name} (Completed)",
                                        border_style="orange3",
                                    )
                                    console.print(step_panel)  # type: ignore
                        else:
                            # This is a regular single step
                            if step_output.content:
                                formatted_content = self._format_step_content_for_display(step_output)
                                step_panel = create_panel(
                                    content=Markdown(formatted_content) if markdown else formatted_content,
                                    title=f"Step {i + 1}: {step_output.step_name} (Completed)",
                                    border_style="orange3",
                                )
                                console.print(step_panel)  # type: ignore

                # For callable functions, show the content directly since there are no step_results
                elif show_step_details and callable(self.steps) and workflow_response.content:
                    step_panel = create_panel(
                        content=Markdown(workflow_response.content) if markdown else workflow_response.content,  # type: ignore
                        title="Custom Function (Completed)",
                        border_style="orange3",
                    )
                    console.print(step_panel)  # type: ignore

                # Show final summary
                if workflow_response.metadata:
                    status = workflow_response.status.value  # type: ignore
                    summary_content = ""
                    summary_content += f"""\n\n**Status:** {status}"""
                    summary_content += f"""\n\n**Steps Completed:** {len(workflow_response.step_results) if workflow_response.step_results else 0}"""
                    summary_content = summary_content.strip()

                    summary_panel = create_panel(
                        content=Markdown(summary_content) if markdown else summary_content,
                        title="Execution Summary",
                        border_style="blue",
                    )
                    console.print(summary_panel)  # type: ignore

                live_log.update("")

                # Final completion message
                if show_time:
                    completion_text = Text(f"Completed in {response_timer.elapsed:.1f}s", style="bold green")
                    console.print(completion_text)  # type: ignore

            except Exception as e:
                import traceback

                traceback.print_exc()
                response_timer.stop()
                error_panel = create_panel(
                    content=f"Workflow execution failed: {str(e)}", title="Execution Error", border_style="red"
                )
                console.print(error_panel)  # type: ignore

    async def _aprint_response_stream(
        self,
        input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[List[Audio]] = None,
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        stream_intermediate_steps: bool = False,
        markdown: bool = True,
        show_time: bool = True,
        show_step_details: bool = True,
        console: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Print workflow execution with clean streaming - orange step blocks displayed once"""
        from rich.console import Group
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.status import Status
        from rich.text import Text

        from agno.utils.response import create_panel
        from agno.utils.timer import Timer

        if console is None:
            from agno.cli.console import console

        stream_intermediate_steps = True  # With streaming print response, we need to stream intermediate steps

        # Show workflow info (same as before)
        media_info = []
        if audio:
            media_info.append(f"Audio files: {len(audio)}")
        if images:
            media_info.append(f"Images: {len(images)}")
        if videos:
            media_info.append(f"Videos: {len(videos)}")

        workflow_info = f"""**Workflow:** {self.name}"""
        if self.description:
            workflow_info += f"""\n\n**Description:** {self.description}"""
        workflow_info += f"""\n\n**Steps:** {self._get_step_count()} steps"""
        if message:
            if isinstance(message, str):
                workflow_info += f"""\n\n**Message:** {message}"""
            else:
                # Handle structured input message
                if isinstance(message, BaseModel):
                    data_display = message.model_dump_json(indent=2, exclude_none=True)
                elif isinstance(message, (dict, list)):
                    import json

                    data_display = json.dumps(message, indent=2, default=str)
                else:
                    data_display = str(message)
                workflow_info += f"""\n\n**Structured Input:**\n```json\n{data_display}\n```"""
        if user_id:
            workflow_info += f"""\n\n**User ID:** {user_id}"""
        if session_id:
            workflow_info += f"""\n\n**Session ID:** {session_id}"""
        workflow_info = workflow_info.strip()

        workflow_panel = create_panel(
            content=Markdown(workflow_info) if markdown else workflow_info,
            title="Workflow Information",
            border_style="cyan",
        )
        console.print(workflow_panel)  # type: ignore

        # Start timer
        response_timer = Timer()
        response_timer.start()

        # Streaming execution variables
        current_step_content = ""
        current_step_name = ""
        current_step_index = 0
        step_results = []
        step_started_printed = False
        is_callable_function = callable(self.steps)

        # Smart step hierarchy tracking
        current_primitive_context = None  # Current primitive being executed (parallel, loop, etc.)
        step_display_cache = {}  # type: ignore

        def get_step_display_number(step_index: Union[int, tuple], step_name: str = "") -> str:
            """Generate clean two-level step numbering: x.y format only"""

            # Handle tuple format for child steps
            if isinstance(step_index, tuple):
                if len(step_index) >= 2:
                    parent_idx, sub_idx = step_index[0], step_index[1]

                    # Extract base parent index if it's nested
                    if isinstance(parent_idx, tuple):
                        base_idx = parent_idx[0] if len(parent_idx) > 0 else 0
                        while isinstance(base_idx, tuple) and len(base_idx) > 0:
                            base_idx = base_idx[0]
                    else:
                        base_idx = parent_idx

                    # Check context for parallel special case
                    if current_primitive_context and current_primitive_context["type"] == "parallel":
                        # For parallel child steps, all get the same number based on their actual step_index
                        return f"Step {base_idx + 1}.{sub_idx + 1}"
                    elif current_primitive_context and current_primitive_context["type"] == "loop":
                        iteration = current_primitive_context.get("current_iteration", 1)
                        return f"Step {base_idx + 1}.{sub_idx + 1} (Iteration {iteration})"
                    else:
                        # Regular child step numbering
                        return f"Step {base_idx + 1}.{sub_idx + 1}"  # type: ignore
                else:
                    # Single element tuple - treat as main step
                    return f"Step {step_index[0] + 1}"

            # Handle integer step_index - main step
            if not current_primitive_context:
                # Regular main step
                return f"Step {step_index + 1}"
            else:
                # This shouldn't happen with the new logic, but fallback
                return f"Step {step_index + 1}"

        with Live(console=console, refresh_per_second=10) as live_log:
            status = Status("Starting async workflow...", spinner="dots")
            live_log.update(status)

            try:
                async for response in await self.arun(
                    input=input,
                    additional_data=additional_data,
                    user_id=user_id,
                    session_id=session_id,
                    audio=audio,
                    images=images,
                    videos=videos,
                    stream=True,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                ):  # type: ignore
                    # Handle the new event types
                    if isinstance(response, WorkflowStartedEvent):
                        status.update("Workflow started...")
                        if is_callable_function:
                            current_step_name = "Custom Function"
                            current_step_index = 0
                        live_log.update(status)

                    elif isinstance(response, StepStartedEvent):
                        current_step_name = response.step_name or "Unknown"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Generate smart step number
                        step_display = get_step_display_number(current_step_index, current_step_name)
                        status.update(f"Starting {step_display}: {current_step_name}...")
                        live_log.update(status)

                    elif isinstance(response, StepCompletedEvent):
                        step_name = response.step_name or "Unknown"
                        step_index = response.step_index or 0

                        # Generate smart step number for completion (will use cached value)
                        step_display = get_step_display_number(step_index, step_name)
                        status.update(f"Completed {step_display}: {step_name}")

                        if response.content:
                            step_results.append(
                                {
                                    "step_name": step_name,
                                    "step_index": step_index,
                                    "content": response.content,
                                    "event": response.event,
                                }
                            )

                        # Print the final step result in orange (only once)
                        if show_step_details and current_step_content and not step_started_printed:
                            live_log.update(status, refresh=True)

                            final_step_panel = create_panel(
                                content=Markdown(current_step_content) if markdown else current_step_content,
                                title=f"{step_display}: {step_name} (Completed)",
                                border_style="orange3",
                            )
                            console.print(final_step_panel)  # type: ignore
                            step_started_printed = True

                    elif isinstance(response, LoopExecutionStartedEvent):
                        current_step_name = response.step_name or "Loop"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up loop context
                        current_primitive_context = {
                            "type": "loop",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "current_iteration": 1,
                            "max_iterations": response.max_iterations,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        status.update(
                            f"Starting loop: {current_step_name} (max {response.max_iterations} iterations)..."
                        )
                        live_log.update(status)

                    elif isinstance(response, LoopIterationStartedEvent):
                        if current_primitive_context and current_primitive_context["type"] == "loop":
                            current_primitive_context["current_iteration"] = response.iteration
                            current_primitive_context["sub_step_counter"] = 0  # Reset for new iteration
                            # Clear cache for new iteration
                            step_display_cache.clear()

                        status.update(
                            f"Loop iteration {response.iteration}/{response.max_iterations}: {response.step_name}..."
                        )
                        live_log.update(status)

                    elif isinstance(response, LoopIterationCompletedEvent):
                        status.update(
                            f"Completed iteration {response.iteration}/{response.max_iterations}: {response.step_name}"
                        )

                    elif isinstance(response, LoopExecutionCompletedEvent):
                        step_name = response.step_name or "Loop"
                        step_index = response.step_index or 0

                        status.update(f"Completed loop: {step_name} ({response.total_iterations} iterations)")
                        live_log.update(status, refresh=True)

                        # Print loop summary
                        if show_step_details:
                            summary_content = "**Loop Summary:**\n\n"
                            summary_content += (
                                f"- Total iterations: {response.total_iterations}/{response.max_iterations}\n"
                            )
                            summary_content += (
                                f"- Total steps executed: {sum(len(iteration) for iteration in response.all_results)}\n"
                            )

                            loop_summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title=f"Loop {step_name} (Completed)",
                                border_style="yellow",
                            )
                            console.print(loop_summary_panel)  # type: ignore

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()
                        step_started_printed = True

                    elif isinstance(response, ParallelExecutionStartedEvent):
                        current_step_name = response.step_name or "Parallel Steps"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up parallel context
                        current_primitive_context = {
                            "type": "parallel",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "total_steps": response.parallel_step_count,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        # Print parallel execution summary panel
                        live_log.update(status, refresh=True)
                        parallel_summary = f"**Parallel Steps:** {response.parallel_step_count}"
                        # Use get_step_display_number for consistent numbering
                        step_display = get_step_display_number(current_step_index, current_step_name)
                        parallel_panel = create_panel(
                            content=Markdown(parallel_summary) if markdown else parallel_summary,
                            title=f"{step_display}: {current_step_name}",
                            border_style="cyan",
                        )
                        console.print(parallel_panel)  # type: ignore

                        status.update(
                            f"Starting parallel execution: {current_step_name} ({response.parallel_step_count} steps)..."
                        )
                        live_log.update(status)

                    elif isinstance(response, ParallelExecutionCompletedEvent):
                        step_name = response.step_name or "Parallel Steps"
                        step_index = response.step_index or 0

                        status.update(f"Completed parallel execution: {step_name}")

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()

                    elif isinstance(response, ConditionExecutionStartedEvent):
                        current_step_name = response.step_name or "Condition"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up condition context
                        current_primitive_context = {
                            "type": "condition",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "condition_result": response.condition_result,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        condition_text = "met" if response.condition_result else "not met"
                        status.update(f"Starting condition: {current_step_name} (condition {condition_text})...")
                        live_log.update(status)

                    elif isinstance(response, ConditionExecutionCompletedEvent):
                        step_name = response.step_name or "Condition"
                        step_index = response.step_index or 0

                        status.update(f"Completed condition: {step_name}")

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()

                    elif isinstance(response, RouterExecutionStartedEvent):
                        current_step_name = response.step_name or "Router"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False

                        # Set up router context
                        current_primitive_context = {
                            "type": "router",
                            "step_index": current_step_index,
                            "sub_step_counter": 0,
                            "selected_steps": response.selected_steps,
                        }

                        # Clear cache for this primitive's sub-steps
                        step_display_cache.clear()

                        selected_steps_text = ", ".join(response.selected_steps) if response.selected_steps else "none"
                        status.update(f"Starting router: {current_step_name} (selected: {selected_steps_text})...")
                        live_log.update(status)

                    elif isinstance(response, RouterExecutionCompletedEvent):
                        step_name = response.step_name or "Router"
                        step_index = response.step_index or 0

                        status.update(f"Completed router: {step_name}")

                        # Print router summary
                        if show_step_details:
                            selected_steps_text = (
                                ", ".join(response.selected_steps) if response.selected_steps else "none"
                            )
                            summary_content = "**Router Summary:**\n\n"
                            summary_content += f"- Selected steps: {selected_steps_text}\n"
                            summary_content += f"- Executed steps: {response.executed_steps or 0}\n"

                            router_summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title=f"Router {step_name} (Completed)",
                                border_style="purple",
                            )
                            console.print(router_summary_panel)  # type: ignore

                        # Reset context
                        current_primitive_context = None
                        step_display_cache.clear()
                        step_started_printed = True

                    elif isinstance(response, StepsExecutionStartedEvent):
                        current_step_name = response.step_name or "Steps"
                        current_step_index = response.step_index or 0  # type: ignore
                        current_step_content = ""
                        step_started_printed = False
                        status.update(f"Starting steps: {current_step_name} ({response.steps_count} steps)...")
                        live_log.update(status)

                    elif isinstance(response, StepsExecutionCompletedEvent):
                        step_name = response.step_name or "Steps"
                        step_index = response.step_index or 0

                        status.update(f"Completed steps: {step_name}")

                        # Add results from executed steps to step_results
                        if response.step_results:
                            for i, step_result in enumerate(response.step_results):
                                # Use the same numbering system as other primitives
                                step_display_number = get_step_display_number(step_index, step_result.step_name or "")
                                step_results.append(
                                    {
                                        "step_name": f"{step_display_number}: {step_result.step_name}",
                                        "step_index": step_index,
                                        "content": step_result.content,
                                        "event": "StepsStepResult",
                                    }
                                )

                        # Print steps summary
                        if show_step_details:
                            summary_content = "**Steps Summary:**\n\n"
                            summary_content += f"- Total steps: {response.steps_count or 0}\n"
                            summary_content += f"- Executed steps: {response.executed_steps or 0}\n"

                            steps_summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title=f"Steps {step_name} (Completed)",
                                border_style="yellow",
                            )
                            console.print(steps_summary_panel)  # type: ignore

                        step_started_printed = True

                    elif isinstance(response, WorkflowCompletedEvent):
                        status.update("Workflow completed!")

                        # For callable functions, print the final content block here since there are no step events
                        if (
                            is_callable_function
                            and show_step_details
                            and current_step_content
                            and not step_started_printed
                        ):
                            final_step_panel = create_panel(
                                content=Markdown(current_step_content) if markdown else current_step_content,
                                title="Custom Function (Completed)",
                                border_style="orange3",
                            )
                            console.print(final_step_panel)  # type: ignore
                            step_started_printed = True

                        live_log.update(status, refresh=True)

                        # Show final summary
                        if response.metadata:
                            status = response.status
                            summary_content = ""
                            summary_content += f"""\n\n**Status:** {status}"""
                            summary_content += f"""\n\n**Steps Completed:** {len(response.step_results) if response.step_results else 0}"""
                            summary_content = summary_content.strip()

                            summary_panel = create_panel(
                                content=Markdown(summary_content) if markdown else summary_content,
                                title="Execution Summary",
                                border_style="blue",
                            )
                            console.print(summary_panel)  # type: ignore

                    else:
                        if isinstance(response, str):
                            response_str = response
                        elif isinstance(response, StepOutputEvent):
                            # Handle StepOutputEvent objects yielded from workflow
                            response_str = response.content or ""  # type: ignore
                        else:
                            from agno.run.response import RunResponseContentEvent
                            from agno.run.team import RunResponseContentEvent as TeamRunResponseContentEvent

                            current_step_executor_type = None
                            # Handle both integer and tuple step indices for parallel execution
                            actual_step_index = current_step_index
                            if isinstance(current_step_index, tuple):
                                # For tuple indices, use the first element (parent step index)
                                actual_step_index = current_step_index[0]
                                # If it's nested tuple, keep extracting until we get an integer
                                while isinstance(actual_step_index, tuple) and len(actual_step_index) > 0:
                                    actual_step_index = actual_step_index[0]

                            # Check if this is a streaming content event from agent or team
                            if isinstance(
                                response,
                                (RunResponseContentEvent, TeamRunResponseContentEvent, WorkflowRunResponseEvent),  # type: ignore
                            ):  # type: ignore
                                # Extract the content from the streaming event
                                response_str = response.content  # type: ignore

                                # Check if this is a team's final structured output
                                is_structured_output = (
                                    isinstance(response, TeamRunResponseContentEvent)
                                    and hasattr(response, "content_type")
                                    and response.content_type != "str"
                                    and response.content_type != ""
                                )
                            elif isinstance(response, RunResponseContentEvent) and current_step_executor_type != "team":
                                response_str = response.content  # type: ignore
                            else:
                                continue

                        # Use the unified formatting function for consistency
                        response_str = self._format_step_content_for_display(response_str)  # type: ignore

                        # Filter out empty responses and add to current step content
                        if response_str and response_str.strip():
                            # If it's a structured output from a team, replace the content instead of appending
                            if "is_structured_output" in locals() and is_structured_output:
                                current_step_content = response_str
                            else:
                                current_step_content += response_str

                            # Live update the step panel with streaming content
                            if show_step_details and not step_started_printed:
                                # Generate smart step number for streaming title (will use cached value)
                                step_display = get_step_display_number(current_step_index, current_step_name)
                                title = f"{step_display}: {current_step_name} (Streaming...)"
                                if is_callable_function:
                                    title = "Custom Function (Streaming...)"

                                # Show the streaming content live in orange panel
                                live_step_panel = create_panel(
                                    content=Markdown(current_step_content) if markdown else current_step_content,
                                    title=title,
                                    border_style="orange3",
                                )

                                # Create group with status and current step content
                                group = Group(status, live_step_panel)
                                live_log.update(group)

                response_timer.stop()

                live_log.update("")

                # Final completion message
                if show_time:
                    completion_text = Text(f"Completed in {response_timer.elapsed:.1f}s", style="bold green")
                    console.print(completion_text)  # type: ignore

            except Exception as e:
                import traceback

                traceback.print_exc()
                response_timer.stop()
                error_panel = create_panel(
                    content=f"Workflow execution failed: {str(e)}", title="Execution Error", border_style="red"
                )
                console.print(error_panel)  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation"""
        # TODO: Handle nested
        if self.steps is None or callable(self.steps):
            steps_list = []
        elif isinstance(self.steps, Steps):
            steps_list = self.steps.steps
        else:
            steps_list = self.steps

        return {
            "name": self.name,
            "workflow_id": self.id,
            "description": self.description,
            "steps": [
                # TODO: this should be encapsulated in step.to_dict() - for all step classes
                {
                    "name": s.name if hasattr(s, "name") else s.__name__,
                    "description": s.description if hasattr(s, "description") else "User-defined callable step",
                    # TODO: The step should have a type field
                    "type": STEP_TYPE_MAPPING[type(s)].value,  # type: ignore
                    "agent": s.agent if hasattr(s, "agent") else None,  # type: ignore
                    "team": s.team if hasattr(s, "team") else None,  # type: ignore
                }
                for s in steps_list
            ],
            "session_id": self.session_id,
        }

    def _collect_workflow_session_state_from_agents_and_teams(self):
        """Collect updated workflow_session_state from agents after step execution"""
        if self.workflow_session_state is None:
            self.workflow_session_state = {}

        # Collect state from all agents in all steps
        if self.steps and not callable(self.steps):
            steps_list = self.steps.steps if isinstance(self.steps, Steps) else self.steps
            for step in steps_list:
                if isinstance(step, Step):
                    executor = step.active_executor
                    if hasattr(executor, "workflow_session_state") and executor.workflow_session_state:
                        # Merge the agent's session state back into workflow session state
                        from agno.utils.merge_dict import merge_dictionaries

                        merge_dictionaries(self.workflow_session_state, executor.workflow_session_state)

                    # If it's a team, collect from all members
                    if hasattr(executor, "members"):
                        for member in executor.members:
                            if hasattr(member, "workflow_session_state") and member.workflow_session_state:
                                merge_dictionaries(self.workflow_session_state, member.workflow_session_state)

    def _update_executor_workflow_session_state(self, executor) -> None:
        """Update executor with workflow_session_state"""
        if self.workflow_session_state is not None:
            # Update session_state with workflow_session_state
            executor.workflow_session_state = self.workflow_session_state

    def _save_run_to_storage(self, workflow_run_response: WorkflowRunResponse) -> None:
        """Helper method to save workflow run response to storage"""
        if self.workflow_session:
            self.workflow_session.upsert_run(workflow_run_response)
            self.write_to_storage()

    def update_agents_and_teams_session_info(self):
        """Update agents and teams with workflow session information"""
        log_debug("Updating agents and teams with session information")
        # Initialize steps - only if steps is iterable (not callable)
        if self.steps and not callable(self.steps):
            steps_list = self.steps.steps if isinstance(self.steps, Steps) else self.steps
            for step in steps_list:
                # TODO: Handle properly steps inside other primitives
                if isinstance(step, Step):
                    active_executor = step.active_executor

                    if hasattr(active_executor, "workflow_session_id"):
                        active_executor.workflow_session_id = self.session_id
                    if hasattr(active_executor, "workflow_id"):
                        active_executor.workflow_id = self.id

                    # Set workflow_session_state on agents and teams
                    self._update_executor_workflow_session_state(active_executor)

                    # If it's a team, update all members
                    if hasattr(active_executor, "members"):
                        for member in active_executor.members:
                            if hasattr(member, "workflow_session_id"):
                                member.workflow_session_id = self.session_id
                            if hasattr(member, "workflow_id"):
                                member.workflow_id = self.id

                            # Set workflow_session_state on team members
                            self._update_executor_workflow_session_state(member)
