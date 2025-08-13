import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union, cast
from uuid import uuid4

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agno.agent.agent import Agent
from agno.db.base import SessionType
from agno.media import Audio, Image, Video
from agno.media import File as FileMedia
from agno.os.apps.utils import PaginatedResponse, PaginationInfo, SortOrder
from agno.os.auth import get_authentication_dependency
from agno.os.schema import (
    AgentResponse,
    AgentSessionDetailSchema,
    AgentSummaryResponse,
    AppsResponse,
    ConfigResponse,
    InterfaceResponse,
    ManagerResponse,
    Model,
    RunSchema,
    SessionSchema,
    TeamResponse,
    TeamRunSchema,
    TeamSessionDetailSchema,
    TeamSummaryResponse,
    WorkflowResponse,
    WorkflowRunSchema,
    WorkflowSessionDetailSchema,
    WorkflowSummaryResponse,
)
from agno.os.settings import AgnoAPISettings
from agno.os.utils import (
    get_agent_by_id,
    get_team_by_id,
    get_workflow_by_id,
    get_workflow_input_schema_dict,
    process_audio,
    process_document,
    process_image,
    process_video,
)
from agno.run.response import RunResponse, RunResponseErrorEvent
from agno.run.team import RunResponseErrorEvent as TeamRunResponseErrorEvent
from agno.run.workflow import WorkflowErrorEvent
from agno.team.team import Team
from agno.utils.log import log_debug, log_error, log_warning, logger
from agno.workflow.workflow import Workflow

if TYPE_CHECKING:
    from agno.os.app import AgentOS


from agno.os.utils import get_component_memory_app


class WebSocketManager:
    """Manages WebSocket connections for workflow runs"""

    active_connections: Dict[str, WebSocket]  # {run_id: websocket}

    def __init__(
        self,
        active_connections: Optional[Dict[str, WebSocket]] = None,
    ):
        # Store active connections: {run_id: websocket}
        self.active_connections: Dict[str, WebSocket] = active_connections or {}

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        logger.debug("WebSocket connected")

        # Send connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "event": "connected",
                    "message": "Connected to workflow events",
                }
            )
        )

    async def register_workflow_websocket(self, run_id: str, websocket: WebSocket):
        """Register a workflow run with its WebSocket connection"""
        self.active_connections[run_id] = websocket
        logger.debug(f"Registered WebSocket for run_id: {run_id}")

    async def disconnect_by_run_id(self, run_id: str):
        """Remove WebSocket connection by run_id"""
        if run_id in self.active_connections:
            del self.active_connections[run_id]
            logger.debug(f"WebSocket disconnected for run_id: {run_id}")

    async def get_websocket_for_run(self, run_id: str) -> Optional[WebSocket]:
        """Get WebSocket connection for a workflow run"""
        return self.active_connections.get(run_id)


# Global manager instance
websocket_manager = WebSocketManager(
    active_connections={},
)


async def agent_response_streamer(
    agent: Agent,
    message: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    images: Optional[List[Image]] = None,
    audio: Optional[List[Audio]] = None,
    videos: Optional[List[Video]] = None,
    files: Optional[List[FileMedia]] = None,
) -> AsyncGenerator:
    try:
        run_response = agent.arun(
            message,
            session_id=session_id,
            user_id=user_id,
            images=images,
            audio=audio,
            videos=videos,
            files=files,
            stream=True,
            stream_intermediate_steps=True,
        )
        async for run_response_chunk in run_response:
            yield run_response_chunk.to_json()
    except Exception as e:
        import traceback

        traceback.print_exc(limit=3)
        error_response = RunResponseErrorEvent(
            content=str(e),
        )
        yield error_response.to_json()


async def agent_continue_response_streamer(
    agent: Agent,
    run_id: Optional[str] = None,
    updated_tools: Optional[List] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncGenerator:
    try:
        continue_response = agent.acontinue_run(
            run_id=run_id,
            updated_tools=updated_tools,
            session_id=session_id,
            user_id=user_id,
            stream=True,
            stream_intermediate_steps=True,
        )
        async for run_response_chunk in continue_response:
            yield run_response_chunk.to_json()
    except Exception as e:
        import traceback

        traceback.print_exc(limit=3)
        error_response = RunResponseErrorEvent(
            content=str(e),
        )
        yield error_response.to_json()
        return


async def team_response_streamer(
    team: Team,
    message: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    images: Optional[List[Image]] = None,
    audio: Optional[List[Audio]] = None,
    videos: Optional[List[Video]] = None,
    files: Optional[List[FileMedia]] = None,
) -> AsyncGenerator:
    """Run the given team asynchronously and yield its response"""
    try:
        run_response = team.arun(
            message,
            session_id=session_id,
            user_id=user_id,
            images=images,
            audio=audio,
            videos=videos,
            files=files,
            stream=True,
            stream_intermediate_steps=True,
        )
        async for run_response_chunk in run_response:
            yield run_response_chunk.to_json()
    except Exception as e:
        import traceback

        traceback.print_exc()
        error_response = TeamRunResponseErrorEvent(
            content=str(e),
        )
        yield error_response.to_json()
        return


async def handle_workflow_via_websocket(websocket: WebSocket, message: dict, os: "AgentOS"):
    """Handle workflow execution directly via WebSocket"""
    try:
        workflow_id = message.get("workflow_id")
        session_id = message.get("session_id")
        user_message = message.get("message", "")
        user_id = message.get("user_id")

        if not workflow_id:
            await websocket.send_text(json.dumps({"event": "error", "error": "workflow_id is required"}))
            return

        # Get workflow from OS
        workflow = get_workflow_by_id(workflow_id, os.workflows)
        if not workflow:
            await websocket.send_text(json.dumps({"event": "error", "error": f"Workflow {workflow_id} not found"}))
            return

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid4())

        # Execute workflow in background with streaming
        await workflow.arun(
            message=user_message,
            session_id=session_id,
            user_id=user_id,
            stream=True,
            stream_intermediate_steps=True,
            background=True,
            websocket=websocket,
        )

    except Exception as e:
        logger.error(f"Error executing workflow via WebSocket: {e}")
        await websocket.send_text(json.dumps({"event": "error", "error": str(e)}))


async def workflow_response_streamer(
    workflow: Workflow,
    message: Optional[Union[str, Dict[str, Any], List[Any], BaseModel]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs: Any,
) -> AsyncGenerator:
    try:
        run_response = await workflow.arun(
            message,
            session_id=session_id,
            user_id=user_id,
            stream=True,
            stream_intermediate_steps=True,
            **kwargs,
        )

        async for run_response_chunk in run_response:
            yield run_response_chunk.to_json()
    except Exception as e:
        import traceback

        traceback.print_exc()
        error_response = WorkflowErrorEvent(
            error=str(e),
        )
        yield error_response.to_json()
        return


def get_base_router(
    os: "AgentOS",
    settings: AgnoAPISettings = AgnoAPISettings(),
) -> APIRouter:
    router = APIRouter(tags=["Core"], dependencies=[Depends(get_authentication_dependency(settings))])

    # -- Main Routes ---
    @router.get("/health")
    async def health_check():
        return JSONResponse(content={"status": "ok"})

    @router.get(
        "/config",
        response_model=ConfigResponse,
        response_model_exclude_none=True,
    )
    async def config() -> ConfigResponse:
        apps_response = AppsResponse(
            session=[
                ManagerResponse(
                    type=app.type,
                    name=app.display_name or "Sessions app",
                    version=app.version,
                    route=app.router_prefix,
                )
                for app in os.apps
                if app.type == "session"
            ],
            knowledge=[
                ManagerResponse(
                    type=app.type,
                    name=app.display_name or "Knowledge app",
                    version=app.version,
                    route=app.router_prefix,
                )
                for app in os.apps
                if app.type == "knowledge"
            ],
            memory=[
                ManagerResponse(
                    type=app.type,
                    name=app.display_name or "Memory app",
                    version=app.version,
                    route=app.router_prefix,
                )
                for app in os.apps
                if app.type == "memory"
            ],
            eval=[
                ManagerResponse(
                    type=app.type,
                    name=app.display_name or "Evals app",
                    version=app.version,
                    route=app.router_prefix,
                )
                for app in os.apps
                if app.type == "eval"
            ],
            metrics=[
                ManagerResponse(
                    type=app.type,
                    name=app.display_name or "Metrics app",
                    version=app.version,
                    route=app.router_prefix,
                )
                for app in os.apps
                if app.type == "metrics"
            ],
        )

        apps_response.session = apps_response.session or None
        apps_response.knowledge = apps_response.knowledge or None
        apps_response.memory = apps_response.memory or None
        apps_response.eval = apps_response.eval or None
        apps_response.metrics = apps_response.metrics or None

        return ConfigResponse(
            os_id=os.os_id or "Unnamed OS",
            description=os.description,
            interfaces=[
                InterfaceResponse(type=interface.type, version=interface.version, route=interface.router_prefix)
                for interface in os.interfaces
            ],
            apps=apps_response,
            agents=[
                AgentSummaryResponse(agent_id=agent.id, name=agent.name, description=agent.description)
                for agent in os.agents
            ]
            if os.agents
            else [],
            teams=[
                TeamSummaryResponse(team_id=team.id, name=team.name, description=team.description) for team in os.teams
            ]
            if os.teams
            else [],
            workflows=[
                WorkflowSummaryResponse(workflow_id=workflow.id, name=workflow.name, description=workflow.description)
                for workflow in os.workflows
            ]
            if os.workflows
            else [],
        )

    @router.get(
        "/models",
        response_model=List[Model],
        response_model_exclude_none=True,
    )
    async def get_models():
        """Return the list of all models used by agents and teams in the contextual OS"""
        all_components: List[Union[Agent, Team]] = []
        if os.agents:
            all_components.extend(os.agents)
        if os.teams:
            all_components.extend(os.teams)

        unique_models = {}
        for item in all_components:
            model = cast(Model, item.model)
            if model.id is not None and model.provider is not None:
                key = (model.id, model.provider)
                if key not in unique_models:
                    unique_models[key] = Model(id=model.id, provider=model.provider)

        return list(unique_models.values())

    # -- Agent routes ---

    @router.post("/agents/{agent_id}/runs")
    async def create_agent_run(
        agent_id: str,
        message: str = Form(...),
        stream: bool = Form(False),
        session_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
        files: Optional[List[UploadFile]] = File(None),
    ):
        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        if session_id is None or session_id == "":
            log_debug("Creating new session")
            session_id = str(uuid4())

        base64_images: List[Image] = []
        base64_audios: List[Audio] = []
        base64_videos: List[Video] = []
        input_files: List[FileMedia] = []

        if files:
            for file in files:
                if file.content_type in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
                    try:
                        base64_image = process_image(file)
                        base64_images.append(base64_image)
                    except Exception as e:
                        log_error(f"Error processing image {file.filename}: {e}")
                        continue
                elif file.content_type in ["audio/wav", "audio/mp3", "audio/mpeg"]:
                    try:
                        base64_audio = process_audio(file)
                        base64_audios.append(base64_audio)
                    except Exception as e:
                        log_error(f"Error processing audio {file.filename}: {e}")
                        continue
                elif file.content_type in [
                    "video/x-flv",
                    "video/quicktime",
                    "video/mpeg",
                    "video/mpegs",
                    "video/mpgs",
                    "video/mpg",
                    "video/mpg",
                    "video/mp4",
                    "video/webm",
                    "video/wmv",
                    "video/3gpp",
                ]:
                    try:
                        base64_video = process_video(file)
                        base64_videos.append(base64_video)
                    except Exception as e:
                        log_error(f"Error processing video {file.filename}: {e}")
                        continue
                elif file.content_type in [
                    "application/pdf",
                    "text/csv",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "text/plain",
                    "application/json",
                ]:
                    # Process document files
                    try:
                        file_content = await file.read()
                        input_files.append(FileMedia(content=file_content))
                    except Exception as e:
                        log_error(f"Error processing file {file.filename}: {e}")
                        continue
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type")

        if stream:
            return StreamingResponse(
                agent_response_streamer(
                    agent,
                    message,
                    session_id=session_id,
                    user_id=user_id,
                    images=base64_images if base64_images else None,
                    audio=base64_audios if base64_audios else None,
                    videos=base64_videos if base64_videos else None,
                    files=input_files if input_files else None,
                ),
                media_type="text/event-stream",
            )
        else:
            run_response = cast(
                RunResponse,
                await agent.arun(
                    message=message,
                    session_id=session_id,
                    user_id=user_id,
                    images=base64_images if base64_images else None,
                    audio=base64_audios if base64_audios else None,
                    videos=base64_videos if base64_videos else None,
                    files=input_files if input_files else None,
                    stream=False,
                ),
            )
            return run_response.to_dict()

    @router.post(
        "/agents/{agent_id}/runs/{run_id}/continue",
    )
    async def continue_agent_run(
        agent_id: str,
        run_id: str,
        tools: str = Form(...),  # JSON string of tools
        session_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
        stream: bool = Form(True),
    ):
        # Parse the JSON string manually
        try:
            tools_data = json.loads(tools) if tools else None
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in tools field")

        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        if session_id is None or session_id == "":
            log_warning(
                "Continuing run without session_id. This might lead to unexpected behavior if session context is important."
            )

        # Convert tools dict to ToolExecution objects if provided
        updated_tools = None
        if tools_data:
            try:
                from agno.models.response import ToolExecution

                updated_tools = [ToolExecution.from_dict(tool) for tool in tools_data]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid structure or content for tools: {str(e)}")

        if stream:
            return StreamingResponse(
                agent_continue_response_streamer(
                    agent,
                    run_id=run_id,  # run_id from path
                    updated_tools=updated_tools,
                    session_id=session_id,
                    user_id=user_id,
                ),
                media_type="text/event-stream",
            )
        else:
            run_response_obj = cast(
                RunResponse,
                await agent.acontinue_run(
                    run_id=run_id,  # run_id from path
                    updated_tools=updated_tools,
                    session_id=session_id,
                    user_id=user_id,
                    stream=False,
                ),
            )
            return run_response_obj.to_dict()

    @router.delete(
        "/agents/{agent_id}/sessions/{session_id}",
        status_code=204,
    )
    async def delete_agent_session(agent_id: str, session_id: str) -> None:
        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.db is None:
            raise HTTPException(status_code=404, detail="Agent has no database. Sessions are unavailable.")

        agent.db.delete_session(session_id=session_id)

    @router.get(
        "/agents",
        response_model=List[AgentResponse],
        response_model_exclude_none=True,
    )
    async def get_agents():
        """Return the list of all Agents present in the contextual OS"""
        if os.agents is None:
            return []

        agents = []
        for agent in os.agents:
            agent_memory_app = get_component_memory_app(component=agent, os_apps=os.apps)
            agents.append(AgentResponse.from_agent(agent=agent, memory_app=agent_memory_app))

        return agents

    @router.get(
        "/agents/{agent_id}/sessions",
        response_model=PaginatedResponse[SessionSchema],
        status_code=200,
    )
    async def get_agent_sessions(
        agent_id: str,
        user_id: Optional[str] = Query(default=None, description="Filter sessions by user ID"),
        limit: Optional[int] = Query(default=20, description="Number of sessions to return"),
        page: Optional[int] = Query(default=1, description="Page number"),
        sort_by: Optional[str] = Query(default="created_at", description="Field to sort by"),
        sort_order: Optional[SortOrder] = Query(default="desc", description="Sort order (asc or desc)"),
    ) -> PaginatedResponse[SessionSchema]:
        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent with id {agent_id} not found")
        if agent.db is None:
            raise HTTPException(status_code=404, detail="Agent has no database. Sessions are unavailable.")

        sessions, total_count = agent.db.get_sessions(
            session_type=SessionType.AGENT,
            component_id=agent_id,
            user_id=user_id,
            limit=limit,
            page=page,
            sort_by=sort_by,
            sort_order=sort_order,
            deserialize=False,
        )

        return PaginatedResponse(
            data=[SessionSchema.from_dict(session) for session in sessions],  # type: ignore
            meta=PaginationInfo(
                page=page,
                limit=limit,
                total_count=total_count,  # type: ignore
                total_pages=(total_count + limit - 1) // limit if limit is not None and limit > 0 else 0,  # type: ignore
            ),
        )

    @router.get(
        "/agents/{agent_id}/sessions/{session_id}",
        response_model=AgentSessionDetailSchema,
        status_code=200,
    )
    async def get_agent_session_by_id(
        agent_id: str,
        session_id: str,
    ) -> AgentSessionDetailSchema:
        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent with id {agent_id} not found")

        if agent.db is None:
            raise HTTPException(status_code=404, detail="Agent has no database. Sessions are unavailable.")

        session = agent.db.get_session(session_type=SessionType.AGENT, session_id=session_id)  # type: ignore
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        return AgentSessionDetailSchema.from_session(session)  # type: ignore

    @router.get(
        "/agents/{agent_id}/sessions/{session_id}/runs",
        response_model=List[RunSchema],
        status_code=200,
    )
    async def get_agent_session_runs(
        agent_id: str,
        session_id: str,
    ) -> List[RunSchema]:
        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent with id {agent_id} not found")

        if agent.db is None:
            raise HTTPException(status_code=404, detail="Agent has no database. Runs are unavailable.")

        session = agent.db.get_session(session_type=SessionType.AGENT, session_id=session_id, deserialize=False)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        return [RunSchema.from_dict(run) for run in session["runs"]]  # type: ignore

    @router.get(
        "/agents/{agent_id}",
        response_model=AgentResponse,
    )
    async def get_agent(agent_id: str):
        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        return AgentResponse.from_agent(agent)

    @router.post(
        "/agents/{agent_id}/sessions/{session_id}/rename",
        response_model=AgentSessionDetailSchema,
    )
    async def rename_agent_session(
        agent_id: str,
        session_id: str,
        session_name: str = Body(embed=True),
    ):
        agent = get_agent_by_id(agent_id, os.agents)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.db is None:
            raise HTTPException(status_code=404, detail="Agent has no database. Sessions are unavailable.")

        session = agent.set_session_name(session_id=session_id, session_name=session_name)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        return AgentSessionDetailSchema.from_session(session)  # type: ignore

    # -- Team routes ---

    @router.post("/teams/{team_id}/runs")
    async def create_team_run(
        team_id: str,
        message: str = Form(...),
        stream: bool = Form(True),
        monitor: bool = Form(True),
        session_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
        files: Optional[List[UploadFile]] = File(None),
    ):
        logger.debug(f"Creating team run: {message} {session_id} {monitor} {user_id} {team_id} {files}")
        team = get_team_by_id(team_id, os.teams)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")

        if session_id is not None and session_id != "":
            logger.debug(f"Continuing session: {session_id}")
        else:
            logger.debug("Creating new session")
            session_id = str(uuid4())

        base64_images: List[Image] = []
        base64_audios: List[Audio] = []
        base64_videos: List[Video] = []
        document_files: List[FileMedia] = []

        if files:
            for file in files:
                if file.content_type in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
                    try:
                        base64_image = process_image(file)
                        base64_images.append(base64_image)
                    except Exception as e:
                        logger.error(f"Error processing image {file.filename}: {e}")
                        continue
                elif file.content_type in ["audio/wav", "audio/mp3", "audio/mpeg"]:
                    try:
                        base64_audio = process_audio(file)
                        base64_audios.append(base64_audio)
                    except Exception as e:
                        logger.error(f"Error processing audio {file.filename}: {e}")
                        continue
                elif file.content_type in [
                    "video/x-flv",
                    "video/quicktime",
                    "video/mpeg",
                    "video/mpegs",
                    "video/mpgs",
                    "video/mpg",
                    "video/mpg",
                    "video/mp4",
                    "video/webm",
                    "video/wmv",
                    "video/3gpp",
                ]:
                    try:
                        base64_video = process_video(file)
                        base64_videos.append(base64_video)
                    except Exception as e:
                        logger.error(f"Error processing video {file.filename}: {e}")
                        continue
                elif file.content_type in [
                    "application/pdf",
                    "text/csv",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "text/plain",
                    "application/json",
                ]:
                    document_file = process_document(file)
                    if document_file is not None:
                        document_files.append(document_file)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type")

        if stream:
            return StreamingResponse(
                team_response_streamer(
                    team,
                    message,
                    session_id=session_id,
                    user_id=user_id,
                    images=base64_images if base64_images else None,
                    audio=base64_audios if base64_audios else None,
                    videos=base64_videos if base64_videos else None,
                    files=document_files if document_files else None,
                ),
                media_type="text/event-stream",
            )
        else:
            run_response = await team.arun(
                message=message,
                session_id=session_id,
                user_id=user_id,
                images=base64_images if base64_images else None,
                audio=base64_audios if base64_audios else None,
                videos=base64_videos if base64_videos else None,
                files=document_files if document_files else None,
                stream=False,
            )
            return run_response.to_dict()

    @router.delete(
        "/teams/{team_id}/sessions/{session_id}",
        status_code=204,
    )
    async def delete_team_session(team_id: str, session_id: str) -> None:
        team = get_team_by_id(team_id, os.teams)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
        if team.db is None:
            raise HTTPException(status_code=404, detail="Team has no database. Sessions are unavailable.")

        team.db.delete_session(session_id=session_id)

    @router.get(
        "/teams",
        response_model=List[TeamResponse],
        response_model_exclude_none=True,
    )
    async def get_teams():
        """Return the list of all Teams present in the contextual OS"""
        if os.teams is None:
            return []

        teams = []
        for team in os.teams:
            team_memory_app = get_component_memory_app(component=team, os_apps=os.apps)
            teams.append(TeamResponse.from_team(team=team, memory_app=team_memory_app))

        return teams

    @router.get(
        "/teams/{team_id}/sessions",
        response_model=PaginatedResponse[SessionSchema],
        status_code=200,
    )
    async def get_team_sessions(
        team_id: str,
        user_id: Optional[str] = Query(default=None, description="Filter sessions by user ID"),
        session_name: Optional[str] = Query(default=None, description="Filter sessions by name"),
        limit: Optional[int] = Query(default=20, description="Number of sessions to return"),
        page: Optional[int] = Query(default=1, description="Page number"),
        sort_by: Optional[str] = Query(default="created_at", description="Field to sort by"),
        sort_order: Optional[SortOrder] = Query(default="desc", description="Sort order (asc or desc)"),
    ) -> PaginatedResponse[SessionSchema]:
        team = get_team_by_id(team_id, os.teams)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")

        if team.db is None:
            raise HTTPException(status_code=404, detail="Team has no associated database. Sessions are unavailable.")

        sessions, total_count = team.db.get_sessions(
            session_type=SessionType.TEAM,
            component_id=team_id,
            user_id=user_id,
            session_name=session_name,
            limit=limit,
            page=page,
            sort_by=sort_by,
            sort_order=sort_order,
            deserialize=False,
        )

        return PaginatedResponse(
            data=[SessionSchema.from_dict(session) for session in sessions],  # type: ignore
            meta=PaginationInfo(
                page=page,
                limit=limit,
                total_count=total_count,  # type: ignore
                total_pages=(total_count + limit - 1) // limit if limit is not None and limit > 0 else 0,  # type: ignore
            ),
        )

    @router.get(
        "/teams/{team_id}/sessions/{session_id}",
        response_model=TeamSessionDetailSchema,
        status_code=200,
    )
    async def get_team_session_by_id(
        team_id: str,
        session_id: str,
    ) -> TeamSessionDetailSchema:
        team = get_team_by_id(team_id, os.teams)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")

        if team.db is None:
            raise HTTPException(status_code=404, detail="Team has no associated database. Sessions are unavailable.")

        session = team.db.get_session(session_type=SessionType.TEAM, session_id=session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        return TeamSessionDetailSchema.from_session(session)  # type: ignore

    @router.get(
        "/teams/{team_id}/sessions/{session_id}/runs",
        response_model=List[TeamRunSchema],
        status_code=200,
    )
    async def get_team_session_runs(
        team_id: str,
        session_id: str,
    ) -> List[TeamRunSchema]:
        team = get_team_by_id(team_id, os.teams)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")

        if team.db is None:
            raise HTTPException(status_code=404, detail="Team has no associated database. Runs are unavailable.")

        session = team.db.get_session(session_type=SessionType.TEAM, session_id=session_id, deserialize=False)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        runs = session.get("runs")  # type: ignore
        if not runs:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} has no runs")

        return [TeamRunSchema.from_dict(run) for run in runs]

    @router.get(
        "/teams/{team_id}",
        response_model=TeamResponse,
    )
    async def get_team(team_id: str):
        team = get_team_by_id(team_id, os.teams)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")

        return TeamResponse.from_team(team)

    @router.post(
        "/teams/{team_id}/sessions/{session_id}/rename",
        response_model=TeamSessionDetailSchema,
    )
    async def rename_team_session(
        team_id: str,
        session_id: str,
        session_name: str = Body(embed=True),
    ) -> TeamSessionDetailSchema:
        team = get_team_by_id(team_id, os.teams)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
        if team.db is None:
            raise HTTPException(status_code=404, detail="Team has no database. Sessions are unavailable.")

        session = team.set_session_name(session_id=session_id, session_name=session_name)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        return TeamSessionDetailSchema.from_session(session)  # type: ignore

    # -- Workflow routes ---

    @router.websocket("/workflows/ws")
    async def workflow_websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for receiving real-time workflow events"""
        await websocket_manager.connect(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                action = message.get("action")

                if action == "ping":
                    await websocket.send_text(json.dumps({"event": "pong"}))

                elif action == "start-workflow":
                    # Handle workflow execution directly via WebSocket
                    await handle_workflow_via_websocket(websocket, message, os)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            # Clean up any run_ids associated with this websocket
            runs_to_remove = [run_id for run_id, ws in websocket_manager.active_connections.items() if ws == websocket]
            for run_id in runs_to_remove:
                await websocket_manager.disconnect_by_run_id(run_id)

    @router.get(
        "/workflows/",
        response_model=List[WorkflowResponse],
        response_model_exclude_none=True,
    )
    async def get_workflows():
        if os.workflows is None:
            return []

        return [
            WorkflowResponse(
                id=str(workflow.id),
                name=workflow.name,
                description=workflow.description,
                input_schema=get_workflow_input_schema_dict(workflow),
            )
            for workflow in os.workflows
        ]

    @router.get(
        "/workflows/{workflow_id}/",
        response_model=WorkflowResponse,
    )
    async def get_workflow(workflow_id: str):
        workflow = get_workflow_by_id(workflow_id, os.workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return WorkflowResponse.from_workflow(workflow)

    @router.post("/workflows/{workflow_id}/runs")
    async def create_workflow_run(
        workflow_id: str,
        message: str = Form(...),
        stream: bool = Form(True),
        session_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
        **kwargs: Any,
    ):
        # Retrieve the workflow by ID
        workflow = get_workflow_by_id(workflow_id, os.workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if session_id:
            logger.debug(f"Continuing session: {session_id}")
        else:
            logger.debug("Creating new session")
            session_id = str(uuid4())

        # Return based on stream parameter
        try:
            if stream:
                return StreamingResponse(
                    workflow_response_streamer(
                        workflow,
                        message=message,
                        session_id=session_id,
                        user_id=user_id,
                        **kwargs,
                    ),
                    media_type="text/event-stream",
                )
            else:
                run_response = await workflow.arun(
                    message=message,
                    session_id=session_id,
                    user_id=user_id,
                    stream=False,
                    **kwargs,
                )
                return run_response.to_dict()
        except Exception as e:
            # Handle unexpected runtime errors
            raise HTTPException(status_code=500, detail=f"Error running workflow: {str(e)}")

    @router.get(
        "/workflows/{workflow_id}/sessions",
        response_model=PaginatedResponse[SessionSchema],
        response_model_exclude_none=True,
    )
    async def get_workflow_sessions(
        workflow_id: str,
        user_id: Optional[str] = Query(default=None, description="Filter sessions by user ID"),
        limit: Optional[int] = Query(default=20, description="Number of sessions to return"),
        page: Optional[int] = Query(default=1, description="Page number"),
        sort_by: Optional[str] = Query(default="created_at", description="Field to sort by"),
        sort_order: Optional[SortOrder] = Query(default="desc", description="Sort order (asc or desc)"),
    ):
        workflow = get_workflow_by_id(workflow_id, os.workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if workflow.db is None:
            raise HTTPException(status_code=404, detail="Workflow has no database. Sessions are unavailable.")

        sessions, total_count = workflow.db.get_sessions(
            session_type=SessionType.WORKFLOW,
            component_id=workflow_id,
            limit=limit,
            page=page,
            user_id=user_id,
            sort_by=sort_by,
            sort_order=sort_order,
            deserialize=False,
        )

        return PaginatedResponse(
            data=[SessionSchema.from_dict(session) for session in sessions],  # type: ignore
            meta=PaginationInfo(
                page=page,
                limit=limit,
                total_count=total_count,  # type: ignore
                total_pages=(total_count + limit - 1) // limit if limit is not None and limit > 0 else 0,  # type: ignore
            ),
        )

    @router.get(
        "/workflows/{workflow_id}/sessions/{session_id}",
        response_model=WorkflowSessionDetailSchema,
        response_model_exclude_none=True,
    )
    async def get_workflow_session(workflow_id: str, session_id: str):
        workflow = get_workflow_by_id(workflow_id, os.workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if workflow.db is None:
            raise HTTPException(status_code=404, detail="Workflow has no database. Sessions are unavailable.")

        session = workflow.db.get_session(session_type=SessionType.WORKFLOW, session_id=session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        return WorkflowSessionDetailSchema.from_session(session)  # type: ignore

    @router.get(
        "/workflows/{workflow_id}/sessions/{session_id}/runs",
        response_model=List[WorkflowRunSchema],
        response_model_exclude_none=True,
    )
    async def get_workflow_session_runs(workflow_id: str, session_id: str):
        workflow = get_workflow_by_id(workflow_id, os.workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if workflow.db is None:
            raise HTTPException(status_code=404, detail="Workflow has no database. Runs are unavailable.")

        session = workflow.db.get_session(session_type=SessionType.WORKFLOW, session_id=session_id, deserialize=False)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")

        return [WorkflowRunSchema.from_dict(run) for run in session["runs"]]  # type: ignore

    return router
