from typing import Optional

from pydantic import BaseModel


class ContainerContext(BaseModel):
    """ContainerContext is a context object passed when creating containers."""

    # Workspace name
    workspace_name: str
    # Path to the workspace directory inside the container
    workspace_root: str
    # Path to the workspace parent directory inside the container
    workspace_parent: str
    # Path to the requirements.txt file relative to the workspace_root
    requirements_file: Optional[str] = None
