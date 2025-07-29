from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from agno.models.base import Model
from agno.run.response import Message
from agno.utils.log import log_debug, log_warning

# TODO: Look into moving all managers into a separate dir
if TYPE_CHECKING:
    from agno.session import Session


@dataclass
class SessionSummary:
    """Model for Session Summary."""

    summary: str
    topics: Optional[List[str]] = None
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        _dict = {
            "summary": self.summary,
            "topics": self.topics,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
        return {k: v for k, v in _dict.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSummary":
        last_updated = data.get("last_updated")
        if last_updated:
            data["last_updated"] = datetime.fromisoformat(last_updated)
        return cls(**data)


class SessionSummaryResponse(BaseModel):
    """Model for Session Summary."""

    summary: str = Field(
        ...,
        description="Summary of the session. Be concise and focus on only important information. Do not make anything up.",
    )
    topics: Optional[List[str]] = Field(None, description="Topics discussed in the session.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True, indent=2)


@dataclass
class SessionSummaryManager:
    """Session Summary Manager"""

    # Model used for session summary generation
    model: Optional[Model] = None

    # Prompt used for session summary generation
    session_summary_prompt: Optional[str] = None

    # Whether session summaries were created in the last run
    summaries_updated: bool = False

    def get_response_format(self, model: "Model") -> Union[Dict[str, Any], Type[BaseModel]]:  # type: ignore
        if model.supports_native_structured_outputs:
            return SessionSummaryResponse

        elif model.supports_json_schema_outputs:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": SessionSummaryResponse.__name__,
                    "schema": SessionSummaryResponse.model_json_schema(),
                },
            }
        else:
            return {"type": "json_object"}

    def get_system_message(
        self,
        conversation: List[Message],
        response_format: Union[Dict[str, Any], Type[BaseModel]],
    ) -> Message:
        if self.session_summary_prompt is not None:
            return Message(role="system", content=self.session_summary_prompt)

        system_prompt = dedent("""\
        Analyze the following conversation between a user and an assistant, and extract the following details:
          - Summary (str): Provide a concise summary of the session, focusing on important information that would be helpful for future interactions.
          - Topics (Optional[List[str]]): List the topics discussed in the session.
        Keep the summary concise and to the point. Only include relevant information.

        <conversation>
        """)
        conversation_messages = []
        for message in conversation:
            if message.role == "user":
                conversation_messages.append(f"User: {message.content}")
            elif message.role in ["assistant", "model"]:
                conversation_messages.append(f"Assistant: {message.content}\n")
        system_prompt += "\n".join(conversation_messages)
        system_prompt += "</conversation>"

        if response_format == {"type": "json_object"}:
            from agno.utils.prompts import get_json_output_prompt

            system_prompt += "\n" + get_json_output_prompt(SessionSummaryResponse)  # type: ignore

        return Message(role="system", content=system_prompt)

    def _prepare_summary_messages(
        self,
        session: Optional["Session"] = None,
    ) -> List[Message]:
        """Prepare messages for session summary generation"""
        response_format = self.get_response_format(self.model)

        return [
            self.get_system_message(
                conversation=session.get_messages_for_session(),
                response_format=response_format,
            ),
            Message(role="user", content="Provide the summary of the conversation."),
        ]

    def _process_summary_response(self, summary_response, session_summary_model: "Model") -> Optional[SessionSummary]:  # type: ignore
        """Process the model response into a SessionSummary"""
        from datetime import datetime

        if summary_response is None:
            return None

        # Handle native structured outputs
        if (
            session_summary_model.supports_native_structured_outputs
            and summary_response.parsed is not None
            and isinstance(summary_response.parsed, SessionSummaryResponse)
        ):
            session_summary = SessionSummary(
                summary=summary_response.parsed.summary,
                topics=summary_response.parsed.topics,
                last_updated=datetime.now(),
            )
            self.summary = session_summary
            log_debug("Session summary created", center=True)
            return session_summary

        # Handle string responses
        if isinstance(summary_response.content, str):
            try:
                from agno.utils.string import parse_response_model_str

                parsed_summary = parse_response_model_str(summary_response.content, SessionSummaryResponse)

                if parsed_summary is not None:
                    session_summary = SessionSummary(
                        summary=parsed_summary.summary, topics=parsed_summary.topics, last_updated=datetime.now()
                    )
                    self.summary = session_summary
                    log_debug("Session summary created", center=True)
                    return session_summary
                else:
                    log_warning("Failed to parse session summary response")

            except Exception as e:
                log_warning(f"Failed to parse session summary response: {e}")

        return None

    def create_session_summary(
        self,
        session: Optional["Session"] = None,
    ) -> Optional[SessionSummary]:
        """Creates a summary of the session"""
        log_debug("Creating session summary", center=True)
        if self.model is None:
            return None

        messages = self._prepare_summary_messages(session)
        response_format = self.get_response_format(self.model)

        summary_response = self.model.response(messages=messages, response_format=response_format)
        session_summary = self._process_summary_response(summary_response, self.model)

        if session_summary is not None:
            session.summary = session_summary
            self.summaries_updated = True

        return session_summary

    async def acreate_session_summary(
        self,
        session: Optional["Session"] = None,
    ) -> Optional[SessionSummary]:
        """Creates a summary of the session"""
        log_debug("Creating session summary", center=True)
        if self.model is None:
            return None

        messages = self._prepare_summary_messages(session)
        response_format = self.get_response_format(self.model)

        summary_response = await self.model.aresponse(messages=messages, response_format=response_format)
        session_summary = self._process_summary_response(summary_response, self.model)

        if session_summary is not None:
            session.summary = session_summary
            self.summaries_updated = True

        return session_summary
