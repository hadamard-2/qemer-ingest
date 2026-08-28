import io
import re
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx

from qemer_ingest.models import RepositoryRef

GITHUB_API_URL = "https://api.github.com"
COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


def parse_repository_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    path_parts = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or len(path_parts) != 3
        or not path_parts[1]
        or not path_parts[2]
    ):
        raise ValueError("repository URL must be a public GitHub HTTPS repository")

    owner, repository = path_parts[1:]
    if repository.endswith(".git"):
        repository = repository.removesuffix(".git")
    if not repository:
        raise ValueError("repository URL must include a repository name")
    return owner, repository


class GitHubClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def resolve(self, url: str, ref: str) -> RepositoryRef:
        owner, repository = parse_repository_url(url)
        response = await self._get(f"/repos/{owner}/{repository}/commits/{ref}")
        sha = response.json().get("sha")
        if not isinstance(sha, str) or not COMMIT_SHA.fullmatch(sha):
            raise ValueError("GitHub commit response did not include a full SHA")
        return RepositoryRef(url, owner, repository, ref, sha)

    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        if not isinstance(source.commit_sha, str) or not COMMIT_SHA.fullmatch(
            source.commit_sha
        ):
            raise ValueError("repository source must include a full SHA")
        response = await self._get(
            f"/repos/{source.owner}/{source.repository}/tarball/{source.commit_sha}"
        )
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
            members = archive.getmembers()
            top_level = self._top_level_directory(members)
            for member in members:
                self._validate_member(member)
            archive.extractall(destination, members=members, filter="data")
        extracted = destination / top_level
        if not extracted.is_dir():
            raise ValueError("archive must contain one top-level directory")
        return extracted

    async def _get(self, path: str) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=GITHUB_API_URL,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            response = await client.get(path)
            response.raise_for_status()
            return response

    @staticmethod
    def _top_level_directory(members: list[tarfile.TarInfo]) -> str:
        top_levels = {
            PurePosixPath(member.name).parts[0] for member in members if member.name
        }
        if len(top_levels) != 1:
            raise ValueError("archive must contain one top-level directory")
        return top_levels.pop()

    @staticmethod
    def _validate_member(member: tarfile.TarInfo) -> None:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("archive contains unsafe member path")
