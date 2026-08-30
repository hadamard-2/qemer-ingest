# Token-aware chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound every embedding request by a 2048-token default using required `/tokenize` and `/detokenize` endpoints, while keeping corpus rows source-oriented and supporting an explicit document prefix.

**Architecture:** A new token-service client owns the backend wire contract. The chunker replaces LangChain character splitting with async source-token slicing and checks each prefixed final payload against the authoritative token endpoint. The CLI preflights token services before GitHub resolution, and the embedding client applies the configured prefix only at request time.

**Tech Stack:** Python 3.14, httpx, Typer, pytest, pytest-asyncio, Ruff, pyarrow, llama-server-compatible token endpoints.

**Spec:** `docs/superpowers/specs/2026-08-30-token-aware-chunking-design.md`

## Global Constraints

- Remove `--chunk-size` and `--chunk-overlap`; do not retain aliases or character-based fallback behavior.
- `build` defaults are exactly `--chunk-size-tokens 2048`, `--chunk-overlap-tokens 0`, and `--document-prefix ""`.
- Every embedding backend must provide `POST /tokenize` and `POST /detokenize` at the same base URL as `/v1/embeddings`; no `transformers`, model download, remote tokenizer code, or local tokenizer fallback is allowed.
- `/tokenize` with `add_special: true` is authoritative for every final embedding payload and the prefix counts toward the 2048-token budget.
- Keep Markdown, RST, and TXT parsers authoritative for semantic structure, source URLs, titles, kinds, parser skips, parent IDs, and selected files.
- Unsplit units retain their original IDs and text; split child IDs remain `<parent-snippet-id>-<kind>-<three-digit-child-ordinal>` with ordinals starting at `001`.
- Child artifact text is detokenized source-derived text without `document_prefix`; only the embedding request receives `document_prefix + unit.text`.
- Replace report fields `chunk_size` and `chunk_overlap` with integer `chunk_size_tokens` and `chunk_overlap_tokens`, and add exact-string `document_prefix`.
- Token-endpoint preflight must fail before GitHub resolution or any embedding request; a failed build must not publish an artifact.
- Do not modify `inspect`, Qemer invocation, publication, private-repository support, or embedding batching.
- Do not commit or push unless the user separately authorizes it for this implementation.

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` / `uv.lock` | Remove the obsolete `langchain-text-splitters` dependency. |
| `src/qemer_ingest/tokenization.py` | Implement the required `/tokenize` and `/detokenize` client plus endpoint preflight and response validation. |
| `src/qemer_ingest/chunking.py` | Convert parsed units into token-bounded source-derived units with token overlap and stable child identities. |
| `src/qemer_ingest/embedding.py` | Apply the optional document prefix to outgoing embedding payloads only. |
| `src/qemer_ingest/models.py` | Replace character policy report fields with token policy and prefix fields. |
| `src/qemer_ingest/artifact.py` | Serialize the new report policy fields. |
| `src/qemer_ingest/cli.py` | Replace CLI policy options, preflight token services, run token-aware chunking, and wire request prefixing. |
| `tests/test_tokenization.py` | Verify endpoint request shape, preflight, malformed payload handling, and source-specific errors. |
| `tests/test_chunking.py` | Verify exact token boundaries, splitting, prefix-inclusive budget, token overlap, text/metadata preservation, and child IDs. |
| `tests/test_embedding.py` | Verify embedding payload prefixing without altering returned units. |
| `tests/test_cli_build.py` | Verify default/help behavior, removal of character flags, early endpoint failure, real build integration, report values, and no published output on token failure. |
| `tests/test_artifact.py` | Verify new persisted report fields and source-oriented corpus rows. |
| `README.md` | Document required token endpoints, 2048-token default, explicit prefix option, and reproducibility fields. |

### Task 1: Add the required token-service client

**Files:**

- Create: `src/qemer_ingest/tokenization.py`
- Create: `tests/test_tokenization.py`

**Interfaces:**

- Produces `TokenizationClient(base_url: str, transport: httpx.AsyncBaseTransport | None = None)`.
- Produces `async def preflight(self) -> None`, `async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]`, and `async def detokenize(self, tokens: tuple[int, ...]) -> str`.
- Later tasks pass a `TokenizationClient` to `chunk_units` and call `preflight` before constructing a GitHub client.

- [ ] **Step 1: Write failing token-service contract tests.**

Create `tests/test_tokenization.py` with a `httpx.MockTransport` handler that records requests and returns the two documented JSON response shapes. Assert the preflight request sequence, exact paths, and exact JSON bodies:

```python
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
```

Add focused tests that assert `tokenize("text", add_special=False)` returns a tuple of integers, `detokenize((1, 2))` returns a string, malformed `{"tokens": ["1"]}` is rejected, malformed `{"content": 1}` is rejected, and a failed request raises `ValueError` naming `/tokenize` or `/detokenize`.

- [ ] **Step 2: Run the tests to verify RED.**

Run: `UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_tokenization.py -q`

Expected: collection fails because `qemer_ingest.tokenization` does not yet exist.

- [ ] **Step 3: Implement the token-service client.**

Create `src/qemer_ingest/tokenization.py`. Normalize the base URL with `rstrip("/")`. Use an `httpx.AsyncClient` with the optional transport for every request. `preflight` must call `tokenize("", add_special=True)` followed by `detokenize(())`; it must not special-case or swallow endpoint failures.

Implement the parsing and request shape exactly as follows:

```python
async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
    response = await self._post(
        "/tokenize", {"content": content, "add_special": add_special}
    )
    try:
        tokens = response.json()["tokens"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("tokenize response is invalid") from error
    if not isinstance(tokens, list) or any(
        not isinstance(token, int) for token in tokens
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
        raise ValueError("detokenize response is invalid")
    return content
```

`_post` must call `raise_for_status()` and convert `httpx.HTTPError` to `ValueError(f"token endpoint {path} failed: {error}")`. Do not add retries, authentication, a tokenizer dependency, or a fallback path.

- [ ] **Step 4: Run focused verification.**

Run: `UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_tokenization.py -q && UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff check src/qemer_ingest/tokenization.py tests/test_tokenization.py && UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff format --check src/qemer_ingest/tokenization.py tests/test_tokenization.py`

Expected: all token-service tests pass and Ruff reports no changes.

### Task 2: Replace character chunking with exact token slicing

**Files:**

- Modify: `src/qemer_ingest/chunking.py`
- Modify: `tests/test_chunking.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**

- Consumes `TokenizationClient` from `qemer_ingest.tokenization`.
- Replaces the public interface with `async def chunk_units(units: tuple[DocumentUnit, ...], *, tokenizer: TokenizationClient, chunk_size_tokens: int, chunk_overlap_tokens: int, document_prefix: str) -> tuple[DocumentUnit, ...]`.
- Produces source-oriented `DocumentUnit` values; later tasks pass the output unchanged to `EmbeddingClient`.

- [ ] **Step 1: Remove the obsolete dependency and write failing token-chunking tests.**

Remove `langchain-text-splitters` from `pyproject.toml` with `uv remove langchain-text-splitters`, retaining no direct LangChain dependency in the lock file.

Replace `tests/test_chunking.py` with an async fake codec that makes every source character one token and prepends a sentinel token when `add_special=True`:

```python
class CharacterTokenizationClient:
    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        special = (-1,) if add_special else ()
        return special + tuple(ord(character) for character in content)

    async def detokenize(self, tokens: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in tokens)
```

Add these behavioral tests before implementation:

```python
@pytest.mark.asyncio
async def test_chunk_units_keeps_a_payload_at_exact_token_limit() -> None:
    original = unit(text="a" * 2047)
    chunks = await chunk_units(
        (original,),
        tokenizer=CharacterTokenizationClient(),
        chunk_size_tokens=2048,
        chunk_overlap_tokens=0,
        document_prefix="",
    )
    assert chunks == (original,)


@pytest.mark.asyncio
async def test_chunk_units_splits_a_2049_token_payload_with_stable_metadata() -> None:
    chunks = await chunk_units(
        (unit(text="a" * 2048),),
        tokenizer=CharacterTokenizationClient(),
        chunk_size_tokens=2048,
        chunk_overlap_tokens=0,
        document_prefix="",
    )
    assert [chunk.snippet_id for chunk in chunks] == [
        "numpy-2.3-abc-prose-001",
        "numpy-2.3-abc-prose-002",
    ]
    assert [chunk.text for chunk in chunks] == ["a" * 2047, "a"]
```

Add a prefix-budget test with `chunk_size_tokens=6`, `document_prefix="p:"`, and source text `"abcdef"`; assert every `await tokenizer.tokenize("p:" + chunk.text, add_special=True)` has at most six tokens, while every artifact chunk text omits `"p:"`. Add a token-overlap test at budget six and overlap two that asserts the second child begins with the previous child’s final two source-token characters. Add invalid-option tests for non-positive size and negative or equal overlap, and add a prefix-only-overflow test that raises `ValueError` before any child is emitted.

- [ ] **Step 2: Run the tests to verify RED.**

Run: `UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_chunking.py -q`

Expected: tests fail because the old synchronous character API cannot accept `tokenizer`, token-sized parameters, or `document_prefix`.

- [ ] **Step 3: Implement token-aware source slicing.**

Remove `RecursiveCharacterTextSplitter`, `_SEPARATORS`, and every character-size helper. Validate `chunk_size_tokens > 0` and `0 <= chunk_overlap_tokens < chunk_size_tokens` with messages referring to token size and token overlap.

For each unit, count `document_prefix + unit.text` using `tokenizer.tokenize(..., add_special=True)`. Return `(unit,)` unchanged when the count is at most the budget. For oversized text, tokenize the raw source text with `add_special=False`, then use an async binary search to find the longest nonempty raw token slice whose detokenized text, after prefixing and tokenizing with `add_special=True`, fits the budget. Construct each child from that detokenized raw source slice and retain the existing kind/title/source URL/ordinal-ID behavior.

Advance the next raw-token start by `end - chunk_overlap_tokens`. If that would not advance beyond the preceding start, raise `ValueError("token overlap leaves no room for new source tokens")` rather than looping forever. Before splitting, count the prefix alone with `add_special=True`; if it is already at or above the selected budget, raise `ValueError("document prefix leaves no room for source tokens")`.

Every final child must be rechecked with `tokenizer.tokenize(document_prefix + child.text, add_special=True)` and must be at or below the budget. Do not persist the prefix in `DocumentUnit.text`, and do not introduce a character fallback.

- [ ] **Step 4: Run focused token-chunking verification.**

Run: `UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_chunking.py -q && UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff check src/qemer_ingest/chunking.py tests/test_chunking.py && UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff format --check src/qemer_ingest/chunking.py tests/test_chunking.py && rg -n "langchain|RecursiveCharacterTextSplitter" pyproject.toml uv.lock src tests`

Expected: the test suite and Ruff pass; the final search returns no matches.

### Task 3: Wire token policy, prefixing, reporting, and documentation into build

**Files:**

- Modify: `src/qemer_ingest/embedding.py`
- Modify: `src/qemer_ingest/models.py`
- Modify: `src/qemer_ingest/artifact.py`
- Modify: `src/qemer_ingest/cli.py`
- Modify: `tests/test_embedding.py`
- Modify: `tests/test_artifact.py`
- Modify: `tests/test_cli_build.py`
- Modify: `README.md`

**Interfaces:**

- Consumes token-aware `chunk_units`, `TokenizationClient`, and `EmbeddingClient(..., document_prefix: str = "")`.
- Produces the token-policy CLI, source-oriented Parquet rows, prefixed HTTP embedding payloads, and token-policy report JSON.

- [ ] **Step 1: Write failing embedding, report, and CLI tests.**

In `tests/test_embedding.py`, add a prefix request test that constructs `EmbeddingClient(..., document_prefix="search_document: ")`, runs `embed_all(make_units())`, and asserts the two JSON bodies are exactly:

```python
[
    {
        "input": "search_document: NumPy is for arrays.",
        "model": "nomic-embed-text-v1.5",
    },
    {"input": "search_document: import numpy as np", "model": "nomic-embed-text-v1.5"},
]
```

Also assert `item.unit` remains each original, unprefixed unit.

In `tests/test_artifact.py`, replace every `BuildReport(chunk_size=..., chunk_overlap=...)` construction and expected JSON payload with `chunk_size_tokens=2048`, `chunk_overlap_tokens=0`, and `document_prefix="search_document: "`. Assert no legacy report keys remain.

In `tests/test_cli_build.py`, add a `FakeTokenizationClient` with `preflight`, `tokenize`, and `detokenize` methods following the `CharacterTokenizationClient` test semantics. Monkeypatch it wherever the build test already monkeypatches GitHub and embedding clients. Update expected default report payloads to:

```python
"chunk_size_tokens": 2048,
"chunk_overlap_tokens": 0,
"document_prefix": "",
```

Replace the character-option help regression with assertions that `--chunk-size-tokens`, `--chunk-overlap-tokens`, `--document-prefix`, `2048`, and `0` appear, while `--chunk-size` and `--chunk-overlap` do not. Replace invalid character-option tests with invalid token-option tests. Add a preflight test that makes `FakeTokenizationClient.preflight` raise `ValueError("token endpoint /detokenize failed")` and makes `GitHubClient` raise if constructed; assert the endpoint error is displayed, the GitHub client is not constructed, and the output path does not exist. Add a configured build test at `--chunk-size-tokens 8 --chunk-overlap-tokens 2 --document-prefix 'search_document: '` that verifies child IDs, report values, and that `RecordingEmbeddingClient` receives the prefix configuration while its captured units remain unprefixed.

- [ ] **Step 2: Run affected tests to verify RED.**

Run: `UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_embedding.py tests/test_artifact.py tests/test_cli_build.py -q`

Expected: failures because the model/report fields, token CLI options, token-service preflight, and embedding prefix parameter do not exist.

- [ ] **Step 3: Implement the public build contract.**

In `embedding.py`, add `document_prefix: str = ""` after `dimension`, store it, and post `{"input": self.document_prefix + unit.text, "model": self.model}`. Keep the existing vector validation and source-URL error messages.

In `models.py`, replace `BuildReport.chunk_size` and `BuildReport.chunk_overlap` with `chunk_size_tokens`, `chunk_overlap_tokens`, and `document_prefix`. In `artifact.py`, serialize those exact names and remove the old names.

In `cli.py`, replace the two module-level character option declarations with:

```python
_CHUNK_SIZE_TOKENS_OPTION = typer.Option(2048, "--chunk-size-tokens")
_CHUNK_OVERLAP_TOKENS_OPTION = typer.Option(0, "--chunk-overlap-tokens")
_DOCUMENT_PREFIX_OPTION = typer.Option("", "--document-prefix")
```

Replace build parameters, validation hints, and validation messages with the token names. After validating scalar CLI values and confirming output absence, construct `TokenizationClient(embedding_url)` and call `asyncio.run(tokenizer.preflight())` before constructing `GitHubClient`. After parsing units, call `asyncio.run(chunk_units(...))` with the token client, token size, token overlap, and prefix. Construct `EmbeddingClient(embedding_url, embedding_model, embedding_dim, document_prefix=document_prefix)`, and pass all three new report policy values into `BuildReport`. Do not alter `inspect`.

- [ ] **Step 4: Document the required backend and public options.**

Replace the README overflow-chunking section with a token-aware section. State that `build` requires `/tokenize` and `/detokenize` at the embedding URL, uses a 2048-token default and zero token overlap, and accepts `--document-prefix` to prepend exact caller-selected text only to embedding requests. State that `build-report.json` records `chunk_size_tokens`, `chunk_overlap_tokens`, and `document_prefix`, while corpus text remains source-oriented.

Use this command example, keeping every prose paragraph on one line:

```sh
qemer-ingest build https://github.com/numpy/numpy --ref v2.5.2 --library numpy --version 2.5.2 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --chunk-size-tokens 2048 --chunk-overlap-tokens 0 --document-prefix 'search_document: ' --output ./qemer-corpora/numpy-2.5.2
```

- [ ] **Step 5: Run integration and complete-project verification.**

Run: `UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest -q && UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff check . && UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff format --check . && UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv build && git diff --check && git status --short`

Expected: all tests pass, Ruff reports no changes, package build produces an sdist and wheel, the diff has no whitespace errors, and status contains only files within this plan plus any pre-existing user changes.

## Plan self-review

**Spec coverage:** Task 1 establishes the mandatory service contract and early failure behavior. Task 2 replaces character splitting with final-payload token budgeting, token overlap, source-derived artifact chunks, child identities, and dependency removal. Task 3 exposes the new public CLI, applies optional request-only prefixing, replaces report fields, documents the backend contract, and verifies the whole project.

**Placeholder scan:** The plan has no TBD/TODO markers. Each task names its exact files, interfaces, test commands, expected RED behavior, concrete option names, request payloads, and verification commands.

**Type consistency:** `TokenizationClient` is defined in Task 1 and consumed by the async `chunk_units` interface in Task 2. Task 3 constructs the same client, passes its result through `chunk_units`, and gives the same `document_prefix` to report serialization and `EmbeddingClient` request construction.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-30-token-aware-chunking.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task and review between tasks.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, with checkpoints.

Which approach?
