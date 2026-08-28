import json

import httpx
import pytest

from qemer_ingest.models import DocumentUnit


def make_units() -> tuple[DocumentUnit, ...]:
    return (
        DocumentUnit(
            "numpy-2.3-001",
            "prose",
            "Overview",
            "https://github.com/numpy/numpy/blob/commit/README.md",
            "NumPy is for arrays.",
        ),
        DocumentUnit(
            "numpy-2.3-002",
            "code",
            "Example",
            "https://github.com/numpy/numpy/blob/commit/docs/example.md",
            "import numpy as np",
        ),
    )


@pytest.mark.asyncio
async def test_embed_all_posts_one_request_per_unit_and_preserves_order() -> None:
    from qemer_ingest.embedding import EmbeddingClient

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0, 3.0]}]})

    units = make_units()
    client = EmbeddingClient(
        "http://embeddings.test/",
        "nomic-embed-text-v1.5",
        3,
        transport=httpx.MockTransport(handler),
    )

    embedded = await client.embed_all(units)

    assert [request.method for request in requests] == ["POST", "POST"]
    assert [request.url.path for request in requests] == ["/v1/embeddings"] * 2
    assert [json.loads(request.content) for request in requests] == [
        {"input": "NumPy is for arrays.", "model": "nomic-embed-text-v1.5"},
        {"input": "import numpy as np", "model": "nomic-embed-text-v1.5"},
    ]
    assert [item.unit for item in embedded] == list(units)
    assert [item.vector for item in embedded] == [[1.0, 2.0, 3.0]] * 2


@pytest.mark.asyncio
async def test_embed_all_rejects_an_empty_response_with_the_source_url() -> None:
    from qemer_ingest.embedding import EmbeddingClient

    unit = make_units()[0]
    client = EmbeddingClient(
        "http://embeddings.test",
        "nomic-embed-text-v1.5",
        3,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []})),
    )

    with pytest.raises(ValueError, match=unit.source_url):
        await client.embed_all((unit,))


@pytest.mark.asyncio
async def test_embed_all_rejects_wrong_width_with_the_source_url() -> None:
    from qemer_ingest.embedding import EmbeddingClient

    unit = make_units()[1]
    client = EmbeddingClient(
        "http://embeddings.test",
        "nomic-embed-text-v1.5",
        3,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]}]})
        ),
    )

    with pytest.raises(ValueError, match=unit.source_url):
        await client.embed_all((unit,))


def test_embedding_client_rejects_a_non_positive_dimension() -> None:
    from qemer_ingest.embedding import EmbeddingClient

    with pytest.raises(ValueError, match="dimension"):
        EmbeddingClient("http://embeddings.test", "nomic-embed-text-v1.5", 0)
