import io
import tarfile
from pathlib import Path

import httpx
import pytest

from qemer_ingest.github import GitHubClient, parse_repository_url
from qemer_ingest.models import RepositoryRef


def test_parse_repository_url_accepts_public_github_https_url() -> None:
    assert parse_repository_url("https://github.com/numpy/numpy.git") == ("numpy", "numpy")


@pytest.mark.parametrize(
    "url",
    (
        "git@github.com:numpy/numpy.git",
        "https://github.com/numpy/numpy/tree/main",
        "https://gitlab.com/numpy/numpy",
    ),
)
def test_parse_repository_url_rejects_non_repository_urls(url: str) -> None:
    with pytest.raises(ValueError):
        parse_repository_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "git@github.com:numpy/numpy.git",
        "https://github.com/numpy/numpy/tree/main",
        "https://gitlab.com/numpy/numpy",
    ),
)
async def test_resolve_rejects_non_repository_urls_before_request(url: str) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid URLs must not trigger a request")

    client = GitHubClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError):
        await client.resolve(url, "v2.3.0")


@pytest.mark.asyncio
async def test_resolve_returns_immutable_commit_sha() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/repos/numpy/numpy/commits/v2.3.0"
        return httpx.Response(200, json={"sha": "b" * 40})

    client = GitHubClient(transport=httpx.MockTransport(handler))
    resolved = await client.resolve("https://github.com/numpy/numpy", "v2.3.0")

    assert (resolved.owner, resolved.repository, resolved.commit_sha) == (
        "numpy",
        "numpy",
        "b" * 40,
    )


@pytest.mark.asyncio
async def test_resolve_rejects_a_non_immutable_sha() -> None:
    client = GitHubClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"sha": "not-a-full-commit"})
        )
    )

    with pytest.raises(ValueError, match="full SHA"):
        await client.resolve("https://github.com/numpy/numpy", "v2.3.0")


def build_archive(member_name: str) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        contents = b"NumPy documentation"
        member = tarfile.TarInfo(member_name)
        member.size = len(contents)
        tar.addfile(member, io.BytesIO(contents))
    return archive.getvalue()


@pytest.mark.asyncio
async def test_download_archive_extracts_and_returns_its_single_top_level_directory(
    tmp_path: Path,
) -> None:
    source = RepositoryRef(
        url="https://github.com/numpy/numpy",
        owner="numpy",
        repository="numpy",
        requested_ref="v2.3.0",
        commit_sha="b" * 40,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/repos/numpy/numpy/tarball/{source.commit_sha}"
        return httpx.Response(200, content=build_archive(f"numpy-{source.commit_sha}/README.md"))

    client = GitHubClient(transport=httpx.MockTransport(handler))

    extracted = await client.download_archive(source, tmp_path)

    assert extracted == tmp_path / f"numpy-{source.commit_sha}"
    assert (extracted / "README.md").read_bytes() == b"NumPy documentation"


@pytest.mark.asyncio
async def test_download_archive_rejects_non_immutable_source_before_request(
    tmp_path: Path,
) -> None:
    source = RepositoryRef(
        url="https://github.com/numpy/numpy",
        owner="numpy",
        repository="numpy",
        requested_ref="main",
        commit_sha="main",
    )

    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("a mutable source must not request an archive")

    client = GitHubClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError, match="full SHA"):
        await client.download_archive(source, tmp_path)


@pytest.mark.asyncio
async def test_download_archive_rejects_an_archive_without_a_top_level_directory(
    tmp_path: Path,
) -> None:
    source = RepositoryRef(
        url="https://github.com/numpy/numpy",
        owner="numpy",
        repository="numpy",
        requested_ref="v2.3.0",
        commit_sha="b" * 40,
    )
    client = GitHubClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=build_archive("README.md"))
        )
    )

    with pytest.raises(ValueError, match="top-level directory"):
        await client.download_archive(source, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("member_name", ("/README.md", "numpy-archive/../README.md"))
async def test_download_archive_rejects_unsafe_member_paths(
    tmp_path: Path, member_name: str
) -> None:
    source = RepositoryRef(
        url="https://github.com/numpy/numpy",
        owner="numpy",
        repository="numpy",
        requested_ref="v2.3.0",
        commit_sha="b" * 40,
    )
    client = GitHubClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=build_archive(member_name))
        )
    )

    with pytest.raises(ValueError, match="unsafe"):
        await client.download_archive(source, tmp_path)
