import json

import httpx
import pytest

from qemer_ingest.tokenization import TokenizationClient


@pytest.mark.asyncio
async def test_preflight_requires_tokenize_and_detokenize_endpoints() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": ""})
        return httpx.Response(404)

    client = TokenizationClient(
        "http://embeddings.test/", transport=httpx.MockTransport(handler)
    )

    await client.preflight()

    assert [request.url.path for request in requests] == ["/tokenize", "/detokenize"]
    assert [json.loads(request.content) for request in requests] == [
        {"content": "", "add_special": True},
        {"tokens": []},
    ]


@pytest.mark.asyncio
async def test_tokenize_returns_a_tuple_of_integers() -> None:
    client = TokenizationClient(
        "http://embeddings.test/",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"tokens": [1, 2, 3]})
        ),
    )

    tokens = await client.tokenize("text", add_special=False)

    assert tokens == (1, 2, 3)


@pytest.mark.asyncio
async def test_detokenize_returns_a_string() -> None:
    client = TokenizationClient(
        "http://embeddings.test/",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"content": "text"})
        ),
    )

    content = await client.detokenize((1, 2))

    assert content == "text"


@pytest.mark.asyncio
async def test_tokenize_rejects_malformed_tokens() -> None:
    client = TokenizationClient(
        "http://embeddings.test/",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"tokens": ["1"]})
        ),
    )

    with pytest.raises(ValueError, match="tokenize response is invalid"):
        await client.tokenize("text", add_special=False)


@pytest.mark.asyncio
async def test_tokenize_rejects_boolean_token_ids() -> None:
    client = TokenizationClient(
        "http://embeddings.test/",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"tokens": [True, False]})
        ),
    )

    with pytest.raises(ValueError, match="tokenize response is invalid"):
        await client.tokenize("text", add_special=False)


@pytest.mark.asyncio
async def test_detokenize_rejects_malformed_content() -> None:
    client = TokenizationClient(
        "http://embeddings.test/",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"content": 1})
        ),
    )

    with pytest.raises(ValueError, match="detokenize response is invalid"):
        await client.detokenize((1, 2))


@pytest.mark.asyncio
async def test_tokenize_names_the_failed_endpoint() -> None:
    client = TokenizationClient(
        "http://embeddings.test/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, request=request)
        ),
    )

    with pytest.raises(ValueError, match="/tokenize"):
        await client.tokenize("text", add_special=False)


@pytest.mark.asyncio
async def test_detokenize_names_the_failed_endpoint() -> None:
    client = TokenizationClient(
        "http://embeddings.test/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, request=request)
        ),
    )

    with pytest.raises(ValueError, match="/detokenize"):
        await client.detokenize((1, 2))
