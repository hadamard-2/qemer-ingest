import httpx

from qemer_ingest.models import DocumentUnit, EmbeddedUnit


class EmbeddingClient:
    """Embed document units through an already-running compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int,
        document_prefix: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = dimension
        self.document_prefix = document_prefix
        self._transport = transport

    async def embed_all(
        self, units: tuple[DocumentUnit, ...]
    ) -> tuple[EmbeddedUnit, ...]:
        embedded: list[EmbeddedUnit] = []
        async with httpx.AsyncClient(transport=self._transport) as client:
            for unit in units:
                vector = await self._embed_one(client, unit)
                embedded.append(EmbeddedUnit(unit, vector))
        return tuple(embedded)

    async def _embed_one(
        self, client: httpx.AsyncClient, unit: DocumentUnit
    ) -> list[float]:
        try:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": self.document_prefix + unit.text, "model": self.model},
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ValueError(
                f"embedding request failed for {unit.source_url}: {error}"
            ) from error

        try:
            data = response.json()["data"]
            vector = [float(value) for value in data[0]["embedding"]]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"embedding response is invalid for {unit.source_url}"
            ) from error

        if len(vector) != self.dimension:
            raise ValueError(
                f"embedding dimension for {unit.source_url} is {len(vector)}, "
                f"expected {self.dimension}"
            )
        return vector
