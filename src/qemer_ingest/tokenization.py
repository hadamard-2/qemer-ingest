import httpx


class TokenizationClient:
    """Call token-service endpoints exposed by the embedding server."""

    def __init__(
        self,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport

    async def preflight(self) -> None:
        await self.tokenize("", add_special=True)
        await self.detokenize(())

    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        response = await self._post(
            "/tokenize", {"content": content, "add_special": add_special}
        )
        try:
            tokens = response.json()["tokens"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("tokenize response is invalid") from error
        if not isinstance(tokens, list) or any(
            type(token) is not int for token in tokens
        ):
            raise ValueError("tokenize response is invalid")
        return tuple(tokens)

    async def detokenize(self, tokens: tuple[int, ...]) -> str:
        response = await self._post("/detokenize", {"tokens": list(tokens)})
        try:
            content = response.json()["content"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("detokenize response is invalid") from error
        if not isinstance(content, str):
            raise ValueError("detokenize response is invalid")  # noqa: TRY004
        return content

    async def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ValueError(f"token endpoint {path} failed: {error}") from error
        return response
