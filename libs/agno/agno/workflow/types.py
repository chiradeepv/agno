from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import WebSocket
from pydantic import BaseModel

from agno.media import AudioArtifact, ImageArtifact, VideoArtifact
from agno.models.metrics import Metrics
from agno.utils.log import log_warning


@dataclass
class WorkflowExecutionInput:
    """Input data for a step execution"""

    input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None

    additional_data: Optional[Dict[str, Any]] = None

    # Media inputs
    images: Optional[List[ImageArtifact]] = None
    videos: Optional[List[VideoArtifact]] = None
    audio: Optional[List[AudioArtifact]] = None

    def get_input_as_string(self) -> Optional[str]:
        """Convert input to string representation"""
        if self.input is None:
            return None

        if isinstance(self.input, str):
            return self.input
        elif isinstance(self.input, BaseModel):
            return self.input.model_dump_json(indent=2, exclude_none=True)
        elif isinstance(self.input, (dict, list)):
            import json

            return json.dumps(self.input, indent=2, default=str)
        else:
            return str(self.input)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        input_dict: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        if self.input is not None:
            if isinstance(self.input, BaseModel):
                input_dict = self.input.model_dump(exclude_none=True)
            elif isinstance(self.input, (dict, list)):
                input_dict = self.input
            else:
                input_dict = str(self.input)

        return {
            "input": input_dict,
            "additional_data": self.additional_data,
            "images": [img.to_dict() for img in self.images] if self.images else None,
            "videos": [vid.to_dict() for vid in self.videos] if self.videos else None,
            "audio": [aud.to_dict() for aud in self.audio] if self.audio else None,
        }


@dataclass
class StepInput:
    """Input data for a step execution"""

    input: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None

    previous_step_content: Optional[Any] = None
    previous_step_outputs: Optional[Dict[str, "StepOutput"]] = None

    additional_data: Optional[Dict[str, Any]] = None

    # Media inputs
    images: Optional[List[ImageArtifact]] = None
    videos: Optional[List[VideoArtifact]] = None
    audio: Optional[List[AudioArtifact]] = None

    def get_input_as_string(self) -> Optional[str]:
        """Convert input to string representation"""
        if self.input is None:
            return None

        if isinstance(self.input, str):
            return self.input
        elif isinstance(self.input, BaseModel):
            return self.input.model_dump_json(indent=2, exclude_none=True)
        elif isinstance(self.input, (dict, list)):
            import json

            return json.dumps(self.input, indent=2, default=str)
        else:
            return str(self.input)

    def get_step_output(self, step_name: str) -> Optional["StepOutput"]:
        """Get output from a specific previous step by name"""
        if not self.previous_step_outputs:
            return None
        return self.previous_step_outputs.get(step_name)

    def get_step_content(self, step_name: str) -> Optional[Union[str, Dict[str, str]]]:
        """Get content from a specific previous step by name

        For parallel steps, if you ask for the parallel step name, returns a dict
        with {step_name: content} for each sub-step.
        """
        step_output = self.get_step_output(step_name)
        if not step_output:
            return None

        # If this is a parallel step with sub-outputs, return structured dict
        if step_output.parallel_step_outputs:
            return {
                sub_step_name: sub_output.content  # type: ignore[misc]
                for sub_step_name, sub_output in step_output.parallel_step_outputs.items()
                if sub_output.content
            }

        # Regular step, return content directly
        return step_output.content  # type: ignore[return-value]

    def get_all_previous_content(self) -> str:
        """Get concatenated content from all previous steps"""
        if not self.previous_step_outputs:
            return ""

        content_parts = []
        for step_name, output in self.previous_step_outputs.items():
            if output.content:
                content_parts.append(f"=== {step_name} ===\n{output.content}")

        return "\n\n".join(content_parts)

    def get_last_step_content(self) -> Optional[str]:
        """Get content from the most recent step (for backward compatibility)"""
        if not self.previous_step_outputs:
            return None

        last_output = list(self.previous_step_outputs.values())[-1] if self.previous_step_outputs else None
        return last_output.content if last_output else None  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Handle the unified message field
        input_dict: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        if self.input is not None:
            if isinstance(self.input, BaseModel):
                input_dict = self.input.model_dump(exclude_none=True)
            elif isinstance(self.input, (dict, list)):
                input_dict = self.input
            else:
                input_dict = str(self.input)

        previous_step_content_str: Optional[str] = None
        # Handle previous_step_content (keep existing logic)
        if isinstance(self.previous_step_content, BaseModel):
            previous_step_content_str = self.previous_step_content.model_dump_json(indent=2, exclude_none=True)
        elif isinstance(self.previous_step_content, dict):
            import json

            previous_step_content_str = json.dumps(self.previous_step_content, indent=2, default=str)
        elif self.previous_step_content:
            previous_step_content_str = str(self.previous_step_content)

        # Convert previous_step_outputs to serializable format (keep existing logic)
        previous_steps_dict = {}
        if self.previous_step_outputs:
            for step_name, output in self.previous_step_outputs.items():
                previous_steps_dict[step_name] = output.to_dict()

        return {
            "input": input_dict,
            "previous_step_outputs": previous_steps_dict,
            "previous_step_content": previous_step_content_str,
            "additional_data": self.additional_data,
            "images": [img.to_dict() for img in self.images] if self.images else None,
            "videos": [vid.to_dict() for vid in self.videos] if self.videos else None,
            "audio": [aud.to_dict() for aud in self.audio] if self.audio else None,
        }


@dataclass
class StepOutput:
    """Output data from a step execution"""

    step_name: Optional[str] = None
    step_id: Optional[str] = None
    executor_type: Optional[str] = None
    executor_name: Optional[str] = None
    # Primary output
    content: Optional[Union[str, Dict[str, Any], List[Any], BaseModel, Any]] = None

    # For parallel steps: store individual step outputs
    parallel_step_outputs: Optional[Dict[str, "StepOutput"]] = None

    # Link to the run ID of the step execution
    step_run_id: Optional[str] = None

    # Media outputs
    images: Optional[List[ImageArtifact]] = None
    videos: Optional[List[VideoArtifact]] = None
    audio: Optional[List[AudioArtifact]] = None

    # Metrics for this step execution
    metrics: Optional[Metrics] = None

    success: bool = True
    error: Optional[str] = None

    stop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Handle the unified content field
        content_dict: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        if self.content is not None:
            if isinstance(self.content, BaseModel):
                content_dict = self.content.model_dump(exclude_none=True)
            elif isinstance(self.content, (dict, list)):
                content_dict = self.content
            else:
                content_dict = str(self.content)

        return {
            "content": content_dict,
            "step_name": self.step_name,
            "step_id": self.step_id,
            "executor_type": self.executor_type,
            "executor_name": self.executor_name,
            "step_run_id": self.step_run_id,
            "images": [img.to_dict() for img in self.images] if self.images else None,
            "videos": [vid.to_dict() for vid in self.videos] if self.videos else None,
            "audio": [aud.to_dict() for aud in self.audio] if self.audio else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "success": self.success,
            "error": self.error,
            "stop": self.stop,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepOutput":
        """Create StepOutput from dictionary"""
        # Reconstruct media artifacts
        images = data.get("images")
        if images:
            images = [ImageArtifact.model_validate(img) for img in images]

        videos = data.get("videos")
        if videos:
            videos = [VideoArtifact.model_validate(vid) for vid in videos]

        audio = data.get("audio")
        if audio:
            audio = [AudioArtifact.model_validate(aud) for aud in audio]

        return cls(
            step_name=data.get("step_name"),
            step_id=data.get("step_id"),
            executor_type=data.get("executor_type"),
            executor_name=data.get("executor_name"),
            content=data.get("content"),
            step_run_id=data.get("step_run_id"),
            images=images,
            videos=videos,
            audio=audio,
            metrics=data.get("metrics"),
            success=data.get("success", True),
            error=data.get("error"),
            stop=data.get("stop", False),
        )


@dataclass
class StepMetrics:
    """Metrics for a single step execution"""

    step_name: str
    executor_type: str  # "agent", "team", etc.
    executor_name: str

    # For regular steps: our generic metrics data
    metrics: Optional[Metrics] = None

    # For parallel steps: nested step metrics
    parallel_steps: Optional[Dict[str, "StepMetrics"]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - only include relevant fields"""
        result = {
            "step_name": self.step_name,
            "executor_type": self.executor_type,
            "executor_name": self.executor_name,
        }

        # Only include the relevant field based on executor type
        if self.executor_type == "parallel" and self.parallel_steps:
            result["parallel_steps"] = {name: step.to_dict() for name, step in self.parallel_steps.items()}  # type: ignore[assignment]
        elif self.executor_type != "parallel":
            # For non-parallel steps, include metrics (even if None)
            result["metrics"] = self.metrics.to_dict() if self.metrics else None  # type: ignore[assignment]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepMetrics":
        """Create StepMetrics from dictionary"""

        # Parse nested parallel steps if they exist
        parallel_steps = None
        if "parallel_steps" in data and data["parallel_steps"] is not None:
            parallel_steps = {name: cls.from_dict(step_data) for name, step_data in data["parallel_steps"].items()}

        return cls(
            step_name=data["step_name"],
            executor_type=data["executor_type"],
            executor_name=data["executor_name"],
            metrics=data.get("metrics") if data.get("executor_type") != "parallel" else None,
            parallel_steps=parallel_steps,
        )


@dataclass
class WorkflowMetrics:
    """Complete metrics for a workflow execution"""

    total_steps: int
    steps: Dict[str, StepMetrics]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_steps": self.total_steps,
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowMetrics":
        """Create WorkflowMetrics from dictionary"""
        steps = {name: StepMetrics.from_dict(step_data) for name, step_data in data["steps"].items()}

        return cls(
            total_steps=data["total_steps"],
            steps=steps,
        )


@dataclass
class WebSocketHandler:
    """Generic WebSocket handler for real-time workflow events"""

    websocket: Optional[WebSocket] = None

    async def handle_event(self, event: Any) -> None:
        """Handle an event object - serializes and sends via WebSocket"""
        if not self.websocket:
            return

        try:
            if hasattr(event, "to_dict"):
                data = event.to_dict()
            elif hasattr(event, "__dict__"):
                data = event.__dict__
            elif isinstance(event, dict):
                data = event
            else:
                data = {"type": "message", "content": str(event)}

            import json

            await self.websocket.send_text(json.dumps(data))

        except Exception as e:
            log_warning(f"Failed to handle WebSocket event: {e}")

    async def handle_text(self, message: str) -> None:
        """Handle a plain text message"""
        if not self.websocket:
            return

        try:
            await self.websocket.send_text(message)
        except Exception as e:
            log_warning(f"Failed to send WebSocket text: {e}")

    async def handle_dict(self, data: Dict[str, Any]) -> None:
        """Handle a dictionary directly"""
        if not self.websocket:
            return

        try:
            import json

            await self.websocket.send_text(json.dumps(data))
        except Exception as e:
            log_warning(f"Failed to send WebSocket dict: {e}")


class StepType(str, Enum):
    STEP = "Step"
    STEPS = "Steps"
    LOOP = "Loop"
    PARALLEL = "Parallel"
    CONDITION = "Condition"
    ROUTER = "Router"
