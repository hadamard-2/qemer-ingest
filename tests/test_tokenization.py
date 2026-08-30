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

    async with client:
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

    async with client:
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

    async with client:
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
        async with client:
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
        async with client:
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
        async with client:
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
        async with client:
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
        async with client:
            await client.detokenize((1, 2))


@pytest.mark.asyncio
async def test_token_client_reuses_one_http_client_and_closes_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(200, json={"content": "text"})

    tokenizer = TokenizationClient(
        "http://embeddings.test", transport=httpx.MockTransport(handler)
    )

    async with tokenizer:
        http_client = tokenizer._client
        assert http_client is not None
        await tokenizer.tokenize("text", add_special=True)
        await tokenizer.detokenize((1,))
        assert tokenizer._client is http_client
        assert not http_client.is_closed

    assert http_client.is_closed
    assert tokenizer._client is None


@pytest.mark.asyncio
async def test_token_client_closes_after_an_endpoint_failure() -> None:
    tokenizer = TokenizationClient(
        "http://embeddings.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, request=request)
        ),
    )

    with pytest.raises(ValueError, match="/tokenize"):
        async with tokenizer:
            http_client = tokenizer._client
            assert http_client is not None
            await tokenizer.tokenize("text", add_special=True)

    assert http_client.is_closed
    assert tokenizer._client is None


@pytest.mark.asyncio
async def test_token_client_rejects_requests_outside_its_context() -> None:
    tokenizer = TokenizationClient("http://embeddings.test")

    with pytest.raises(RuntimeError, match="async context manager"):
        await tokenizer.tokenize("text", add_special=True)
