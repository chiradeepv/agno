from typing import Dict

from httpx import AsyncClient as HttpxAsyncClient
from httpx import Client as HttpxClient
from httpx import Response

from agno.cli.settings import agno_cli_settings


class Api:
    def __init__(self):
        self.headers: Dict[str, str] = {
            "user-agent": f"{agno_cli_settings.app_name}/{agno_cli_settings.app_version}",
            "Content-Type": "application/json",
        }

    def Client(self) -> HttpxClient:
        return HttpxClient(
            base_url=agno_cli_settings.api_url,
            headers=self.headers,
            timeout=60,
        )

    def AsyncClient(self) -> HttpxAsyncClient:
        return HttpxAsyncClient(
            base_url=agno_cli_settings.api_url,
            headers=self.headers,
            timeout=60,
        )

api = Api()


def invalid_response(r: Response) -> bool:
    """Returns true if the response is invalid"""

    if r.status_code >= 400:
        return True
    return False
