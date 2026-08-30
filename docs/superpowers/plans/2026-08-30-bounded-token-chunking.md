# Bounded token chunking implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for each behavior change and `superpowers:verification-before-completion` before reporting completion. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace exhaustive longest-slice probing with deterministic bounded refinement and reuse one token-service HTTP client for the full build, while preserving exact embedding-payload token limits and every existing public contract.

**Architecture:** `chunk_units` will tokenize the document prefix once, begin each oversized child with a budget-sized raw-token window, and shrink an oversized candidate through proportional plus geometric refinement capped at nine authoritative checks. `TokenizationClient` will own one `httpx.AsyncClient` as an async context manager, and Typer's synchronous `build` command will perform local validation before entering one `asyncio.run` orchestration that keeps preflight, GitHub I/O, chunking, and embedding in the same event loop.

**Tech Stack:** Python 3.14, httpx, Typer, pytest, pytest-asyncio, Ruff, uv, llama-server-compatible `/tokenize`, `/detokenize`, and `/v1/embeddings` endpoints.

**Spec:** `docs/superpowers/specs/2026-08-30-bounded-token-chunking-design.md`

## Global constraints

- Preserve the public CLI names, defaults, artifact schema, document-prefix semantics, exact source-token overlap, child IDs and metadata, endpoint preflight, and failure-before-publication behavior already implemented by the token-aware chunking work.
- Keep stock llama-server endpoints only: `POST /tokenize`, `POST /detokenize`, and `POST /v1/embeddings`; do not add `/chunk`, patch llama.cpp, introduce a local tokenizer, or download tokenizer assets.
- `/tokenize` with `add_special: true` over `document_prefix + child.text` remains the authoritative final budget check; bounded refinement relaxes only the mathematical-longest-slice guarantee.
- Preserve original `DocumentUnit` identity for unsplit units and source-derived, unprefixed text for all artifact rows.
- Do not change Markdown, RST, or TXT parsing, Sphinx diagnostic handling, embedding batching, Qemer invocation, artifact publication, or private-repository support.
- Reuse one token-service connection pool for preflight and chunking, and close it on both success and failure.
- Keep synchronous scalar and output-path validation ahead of `asyncio.run` and ahead of all client construction.
- Do not commit or push unless the user separately authorizes it.

## File structure

| File | Responsibility |
| --- | --- |
| `src/qemer_ingest/chunking.py` | Replace exhaustive descending slice search with budget-sized initial candidates and bounded proportional/geometric refinement. |
| `tests/test_chunking.py` | Lock down deterministic shorter-slice behavior, large-unit first-probe size, nine-check ceiling, exact budgets, overlap, metadata, prefix handling, and removal of redundant final tokenization. |
| `src/qemer_ingest/tokenization.py` | Make `TokenizationClient` an async context manager owning one reusable `httpx.AsyncClient`. |
| `tests/test_tokenization.py` | Verify one client lifecycle, request reuse, response validation, and closure after success or endpoint failure. |
| `src/qemer_ingest/cli.py` | Move networked build work under one async orchestration and one token-client context without changing the command contract. |
| `tests/test_cli_build.py` | Preserve validation/preflight ordering and prove the token client exits on successful and failed builds. |
| `README.md` | Describe deterministic near-maximum bounded chunk selection instead of mathematical-longest selection. |

### Task 1: Implement bounded candidate refinement

**Files:**

- Modify: `src/qemer_ingest/chunking.py`
- Modify: `tests/test_chunking.py`

**Interfaces:**

- Preserve `async def chunk_units(units: tuple[DocumentUnit, ...], *, tokenizer: TokenizationClient, chunk_size_tokens: int, chunk_overlap_tokens: int, document_prefix: str) -> tuple[DocumentUnit, ...]`.
- Replace `_longest_fitting_slice` with a private bounded selector whose accepted return remains `(end: int, text: str)` and whose failure return remains `(start, "")`.
- Add a constant `_MAX_REFINEMENT_ATTEMPTS = 8`; the optional one-token fallback makes the absolute ceiling nine candidate checks per emitted child.

- [ ] **Step 1: Replace the mathematical-longest regression with deterministic bounded behavior.**

Rename `test_chunk_units_selects_longest_nonmonotonic_fitting_slice` to `test_chunk_units_accepts_a_deterministic_bounded_slice_for_nonmonotonic_counts` and change its expected output to show that the algorithm is allowed to miss the unprobed five-character fit:

```python
@pytest.mark.asyncio
async def test_chunk_units_accepts_a_deterministic_bounded_slice_for_nonmonotonic_counts() -> (
    None
):
    chunks = await chunk_units(
        (unit(text="abcdef"),),
        tokenizer=NonmonotonicTokenizationClient(),
        chunk_size_tokens=4,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert [chunk.text for chunk in chunks] == ["ab", "cd", "ef"]
```

The existing nonmonotonic fake deliberately reports lengths three and four as oversized while length five fits. A bounded selector starts at the budget-sized length four, shrinks to three and then two, accepts two, and never performs an exhaustive upward search for five.

- [ ] **Step 2: Add a counting codec and the 20,680-token first-probe regression.**

Add this fake next to the existing test codecs:

```python
class RecordingCharacterTokenizationClient(CharacterTokenizationClient):
    def __init__(self) -> None:
        self.detokenized_lengths: list[int] = []
        self.authoritative_contents: list[str] = []

    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        if add_special:
            self.authoritative_contents.append(content)
        return await super().tokenize(content, add_special=add_special)

    async def detokenize(self, tokens: tuple[int, ...]) -> str:
        self.detokenized_lengths.append(len(tokens))
        return await super().detokenize(tokens)
```

Add the measured NumPy-sized regression:

```python
@pytest.mark.asyncio
async def test_chunk_units_starts_a_large_unit_with_a_budget_sized_candidate() -> None:
    tokenizer = RecordingCharacterTokenizationClient()
    chunks = await chunk_units(
        (unit(text="a" * 20_680),),
        tokenizer=tokenizer,
        chunk_size_tokens=2_048,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert tokenizer.detokenized_lengths[0] == 2_047
    assert max(tokenizer.detokenized_lengths) <= 2_047
    assert len(tokenizer.detokenized_lengths) == len(chunks)
    assert all(
        len(await tokenizer.tokenize(chunk.text, add_special=True)) <= 2_048
        for chunk in chunks
    )
```

This test proves the first detokenization is derived from the remaining payload budget, including the one special token returned by the fake, rather than the 20,680-token document length. Each candidate fits on its first attempt, so the detokenization count must equal the emitted child count.

- [ ] **Step 3: Add forced-refinement and accepted-candidate request-count tests.**

Add a fake that makes only the initial seven-source-token boundary appear oversized under an eight-token budget:

```python
class BoundaryInflatingTokenizationClient(RecordingCharacterTokenizationClient):
    async def tokenize(self, content: str, *, add_special: bool) -> tuple[int, ...]:
        if add_special:
            self.authoritative_contents.append(content)
            if len(content) == 7:
                return tuple(range(9))
            return (-1,) + tuple(ord(character) for character in content)
        return tuple(ord(character) for character in content)
```

Add these regressions:

```python
@pytest.mark.asyncio
async def test_chunk_units_refines_an_oversized_boundary_with_bounded_checks() -> None:
    tokenizer = BoundaryInflatingTokenizationClient()
    chunks = await chunk_units(
        (unit(text="abcdefghi"),),
        tokenizer=tokenizer,
        chunk_size_tokens=8,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert tokenizer.detokenized_lengths[:2] == [7, 6]
    assert [chunk.text for chunk in chunks] == ["abcdef", "ghi"]
    assert len(tokenizer.detokenized_lengths) <= len(chunks) * 9


@pytest.mark.asyncio
async def test_chunk_units_does_not_retokenize_an_accepted_child() -> None:
    tokenizer = RecordingCharacterTokenizationClient()
    chunks = await chunk_units(
        (unit(text="abcdefghi"),),
        tokenizer=tokenizer,
        chunk_size_tokens=7,
        chunk_overlap_tokens=0,
        document_prefix="",
    )

    assert [chunk.text for chunk in chunks] == ["abcdef", "ghi"]
    assert tokenizer.authoritative_contents.count("abcdef") == 1
    assert tokenizer.authoritative_contents.count("ghi") == 1
```

The full-unit check records `"abcdefghi"` once; each accepted child must be recorded only by the successful candidate verification. A second final verification would make the assertions fail.

- [ ] **Step 4: Run the focused tests to verify RED.**

Run:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_chunking.py -q
```

Expected: the nonmonotonic expectation, large-unit first-probe assertion, forced-refinement sequence, or accepted-candidate count fails because the current implementation begins at the complete remaining document, decrements one raw token at a time, and re-tokenizes the accepted child.

- [ ] **Step 5: Tokenize the prefix once and pass its count through oversized-unit processing.**

In `chunk_units`, keep option validation first, then perform one authoritative prefix tokenization before the unit loop:

```python
prefix_tokens = await tokenizer.tokenize(document_prefix, add_special=True)
if len(prefix_tokens) >= chunk_size_tokens:
    raise ValueError("document prefix leaves no room for source tokens")

for unit in units:
    try:
        chunked.extend(
            await _chunk_unit(
                unit,
                tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                document_prefix=document_prefix,
                prefix_token_count=len(prefix_tokens),
            )
        )
    except ValueError as error:
        raise ValueError(f"{error} for {unit.source_url}") from error
```

Add `prefix_token_count: int` to `_chunk_unit`. Preserve the existing full prefixed-unit tokenization and return the original unit unchanged when it fits. For an oversized unit, tokenize raw source once with `add_special=False` and calculate the initial maximum source length with `max(1, chunk_size_tokens - prefix_token_count)`.

- [ ] **Step 6: Implement the bounded selector and strict shrink formula.**

Replace `_longest_fitting_slice` with these helpers:

```python
_MAX_REFINEMENT_ATTEMPTS = 8


def _next_candidate_length(
    current_length: int,
    *,
    chunk_size_tokens: int,
    observed_payload_tokens: int,
) -> int:
    proportional = current_length * chunk_size_tokens // observed_payload_tokens
    geometric = current_length - max(1, (current_length + 7) // 8)
    return max(1, min(current_length - 1, proportional, geometric))


async def _bounded_fitting_slice(
    source_tokens: tuple[int, ...],
    start: int,
    *,
    tokenizer: TokenizationClient,
    chunk_size_tokens: int,
    max_source_tokens: int,
    document_prefix: str,
) -> tuple[int, str]:
    remaining_source_tokens = len(source_tokens) - start
    candidate_length = min(remaining_source_tokens, max_source_tokens)
    last_tested_length = 0

    for _ in range(_MAX_REFINEMENT_ATTEMPTS):
        last_tested_length = candidate_length
        end = start + candidate_length
        text = await tokenizer.detokenize(source_tokens[start:end])
        payload_tokens = await tokenizer.tokenize(
            document_prefix + text, add_special=True
        )
        if len(payload_tokens) <= chunk_size_tokens:
            return end, text
        if candidate_length == 1:
            return start, ""
        candidate_length = _next_candidate_length(
            candidate_length,
            chunk_size_tokens=chunk_size_tokens,
            observed_payload_tokens=len(payload_tokens),
        )

    if last_tested_length != 1:
        end = start + 1
        text = await tokenizer.detokenize(source_tokens[start:end])
        payload_tokens = await tokenizer.tokenize(
            document_prefix + text, add_special=True
        )
        if len(payload_tokens) <= chunk_size_tokens:
            return end, text

    return start, ""
```

Call `_bounded_fitting_slice` with `max_source_tokens=max(1, chunk_size_tokens - prefix_token_count)`. If it returns the starting index, preserve `ValueError("token budget leaves no room for source tokens")`. Construct the child immediately from the accepted text and remove the redundant `tokenizer.tokenize(document_prefix + child.text, add_special=True)` block because the successful selector call is already the authoritative verification.

Keep the existing advancement rule, overlap-only tail suppression, and non-progress guard unchanged:

```python
next_start = end - chunk_overlap_tokens
if next_start <= start:
    raise ValueError("token overlap leaves no room for new source tokens")
start = next_start
```

- [ ] **Step 7: Run focused verification and check for exhaustive-search remnants.**

Run:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_chunking.py -q
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff check src/qemer_ingest/chunking.py tests/test_chunking.py
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff format --check src/qemer_ingest/chunking.py tests/test_chunking.py
rg -n "_longest_fitting_slice|range\(len\(source_tokens\), start, -1\)|final_payload_tokens" src/qemer_ingest/chunking.py
```

Expected: all focused tests and Ruff checks pass; the final search returns no matches.

### Task 2: Reuse one token-service client per build

**Files:**

- Modify: `src/qemer_ingest/tokenization.py`
- Modify: `tests/test_tokenization.py`

**Interfaces:**

- Preserve `TokenizationClient(base_url: str, transport: httpx.AsyncBaseTransport | None = None)` and the existing public `preflight`, `tokenize`, and `detokenize` methods.
- Add `async def __aenter__(self) -> Self` and `async def __aexit__(...) -> None`.
- Require requests to occur inside the async context; `_post` raises `RuntimeError("tokenization client must be used as an async context manager")` otherwise.

- [ ] **Step 1: Convert existing tokenization contract tests to use the lifecycle.**

Wrap every request-bearing assertion in `tests/test_tokenization.py` with the client context:

```python
client = TokenizationClient(
    "http://embeddings.test/", transport=httpx.MockTransport(handler)
)

async with client:
    await client.preflight()
```

Apply the same pattern to `tokenize`, `detokenize`, malformed-response, and endpoint-failure tests so they continue testing the wire contract rather than the new usage guard.

- [ ] **Step 2: Add reuse, closure, and outside-context tests.**

Add these tests using `MockTransport` so no network is involved:

```python
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
```

These lifecycle tests intentionally inspect `_client` because connection-pool ownership is the internal behavior under test and no new production inspection API is justified.

- [ ] **Step 3: Run focused tests to verify RED.**

Run:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_tokenization.py -q
```

Expected: lifecycle tests fail because the current class creates and closes a new `httpx.AsyncClient` inside every `_post` call and does not implement async context entry or exit.

- [ ] **Step 4: Implement async context ownership.**

Add the required imports and client field:

```python
from types import TracebackType
from typing import Self


class TokenizationClient:
    def __init__(
        self,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        if self._client is not None:
            raise RuntimeError("tokenization client is already open")
        self._client = httpx.AsyncClient(transport=self._transport)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
```

Replace the per-request client block in `_post` with the owned client:

```python
client = self._client
if client is None:
    raise RuntimeError("tokenization client must be used as an async context manager")
try:
    response = await client.post(f"{self.base_url}{path}", json=payload)
    response.raise_for_status()
except httpx.HTTPError as error:
    raise ValueError(f"token endpoint {path} failed: {error}") from error
return response
```

Do not change endpoint paths, JSON request bodies, response validation, preflight order, timeouts, retries, or authentication behavior in this task.

- [ ] **Step 5: Update real-client chunking tests to enter the token context.**

In the two `tests/test_chunking.py` tests that instantiate the real `TokenizationClient` for HTTP and malformed-response failures, wrap the `chunk_units` call:

```python
with pytest.raises(ValueError) as error:
    async with tokenizer:
        await chunk_units(
            (document,),
            tokenizer=tokenizer,
            chunk_size_tokens=4,
            chunk_overlap_tokens=0,
            document_prefix="",
        )
```

Keep the fake codecs unchanged because `chunk_units` consumes a tokenizer-like object but does not own its lifecycle.

- [ ] **Step 6: Run token-client and chunking verification.**

Run:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_tokenization.py tests/test_chunking.py -q
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff check src/qemer_ingest/tokenization.py tests/test_tokenization.py tests/test_chunking.py
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff format --check src/qemer_ingest/tokenization.py tests/test_tokenization.py tests/test_chunking.py
```

Expected: all tokenization and chunking tests pass and Ruff reports no changes.

### Task 3: Run the build in one async orchestration

**Files:**

- Modify: `src/qemer_ingest/cli.py`
- Modify: `tests/test_cli_build.py`
- Modify: `README.md`

**Interfaces:**

- Preserve the synchronous Typer `build` command and every public option, default, validation message, progress line, and result line.
- Add one private async build helper returning the three values required by the synchronous success output: resolved commit SHA, selected file count, and emitted row count.
- Enter exactly one `TokenizationClient` context inside that helper and keep token preflight, prefix validation, GitHub operations, chunking, embedding, and artifact construction within the same `asyncio.run` call.

- [ ] **Step 1: Make the fake token client observable as an async context manager.**

Extend `FakeTokenizationClient` in `tests/test_cli_build.py` without changing its tokenize/detokenize semantics:

```python
class FakeTokenizationClient:
    entered = 0
    exited = 0

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    async def __aenter__(self) -> "FakeTokenizationClient":
        FakeTokenizationClient.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        FakeTokenizationClient.exited += 1
```

Add an autouse fixture so assertions do not leak between CLI tests:

```python
@pytest.fixture(autouse=True)
def reset_fake_tokenization_lifecycle() -> None:
    FakeTokenizationClient.entered = 0
    FakeTokenizationClient.exited = 0
```

- [ ] **Step 2: Add success and failure lifecycle assertions while preserving ordering tests.**

In the successful `build` integration test, assert:

```python
assert FakeTokenizationClient.entered == 1
assert FakeTokenizationClient.exited == 1
```

In the existing token-preflight-failure test, use a failing subclass or the existing failure switch and assert the same one-entry, one-exit counts, that GitHub was not constructed, and that the output path does not exist. Keep the existing scalar-validation and occupied-output tests asserting `FakeTokenizationClient.entered == 0`, because local validation must happen before `asyncio.run` or client construction.

In the prefix-only-overflow test, assert one context entry and exit, no GitHub construction, no embedding call, and no published output. These checks distinguish proper exception-safe context exit from merely passing existing error messages.

- [ ] **Step 3: Run focused CLI tests to verify RED.**

Run:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_cli_build.py -q
```

Expected: fake lifecycle assertions fail because the current command does not enter the token client and uses several independent `asyncio.run` calls.

- [ ] **Step 4: Extract all networked build work into one private async helper.**

Add a private result type near the CLI helpers:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _BuildResult:
    commit_sha: str
    selected_files: int
    emitted_rows: int
```

Add an async helper with the validated values currently available in `build`:

```python
async def _run_build(
    *,
    repository_url: str,
    ref: str,
    library: str,
    version: str,
    embedding_url: str,
    embedding_model: str,
    embedding_dim: int,
    output: Path,
    include: tuple[str, ...],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    document_prefix: str,
) -> _BuildResult:
```

Move the existing post-validation build body into this function without changing discovery, parser, report, embedding, or publication logic. Structure the ownership and ordering exactly as follows:

```python
async with TokenizationClient(embedding_url) as tokenizer:
    try:
        await tokenizer.preflight()
        prefix_tokens = (
            await tokenizer.tokenize(document_prefix, add_special=True)
            if document_prefix
            else ()
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--embedding-url") from error
    if len(prefix_tokens) >= chunk_size_tokens:
        raise typer.BadParameter(
            "document prefix leaves no room for source tokens",
            param_hint="--document-prefix",
        )

    github = GitHubClient()
    source = await github.resolve(repository_url, ref)

    with TemporaryDirectory() as temporary_directory:
        repository_root = await github.download_archive(
            source, Path(temporary_directory)
        )
        discovery = discover(repository_root, include)
        if not discovery.selected:
            raise typer.BadParameter("no documentation files selected")

        parsed_documents = tuple(
            parse_document_with_report(
                repository_root / relative_path,
                source,
                library,
                version,
                repository_root=repository_root,
            )
            for relative_path in discovery.selected
        )
        parsed_units = tuple(
            unit for parsed in parsed_documents for unit in parsed.units
        )
        units = await chunk_units(
            parsed_units,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            document_prefix=document_prefix,
        )
        if not units:
            raise typer.BadParameter("selected documentation produced no rows")

        embedding = EmbeddingClient(
            embedding_url,
            embedding_model,
            embedding_dim,
            document_prefix=document_prefix,
        )
        embedded = await embedding.embed_all(units)

        report = BuildReport(
            repository_url=source.url,
            requested_ref=source.requested_ref,
            resolved_commit=source.commit_sha,
            selected_files=discovery.selected,
            skipped_files=discovery.skipped,
            explicitly_included_files=discovery.explicitly_included,
            parser_skips=tuple(
                skip for parsed in parsed_documents for skip in parsed.skipped
            ),
            prose_rows=sum(unit.kind == "prose" for unit in units),
            code_rows=sum(unit.kind == "code" for unit in units),
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            document_prefix=document_prefix,
        )
        build_artifact(
            output,
            library,
            version,
            embedding_model,
            embedding_dim,
            embedded,
            report,
        )

return _BuildResult(
    commit_sha=source.commit_sha,
    selected_files=len(discovery.selected),
    emitted_rows=len(embedded),
)
```

The required behavioral order is local validation, token context entry, token endpoint preflight, prefix capacity validation, GitHub work, synchronous parsing, async chunking, async embedding, and final artifact publication. Do not add, remove, or reinterpret report fields.

The temporary directory remains correct here because it is private staging for the downloaded GitHub archive and is automatically cleaned after artifact construction; the requested artifact is still written to the user-selected `output` path.

- [ ] **Step 5: Replace multiple event-loop entries with one call from the Typer command.**

Keep all current scalar validation and the `os.path.lexists(output)` check in synchronous `build`. After they pass, invoke only the helper:

```python
result = asyncio.run(
    _run_build(
        repository_url=repository_url,
        ref=ref,
        library=library,
        version=version,
        embedding_url=embedding_url,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        output=output,
        include=tuple(include),
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        document_prefix=document_prefix,
    )
)
```

Render the existing success output from `result.commit_sha`, `result.selected_files`, and `result.emitted_rows`. Remove every other `asyncio.run` call from `build`; do not make the Typer callback itself async.

- [ ] **Step 6: Update README wording to match the bounded guarantee.**

Replace any claim that chunking finds the longest fitting slice with one single-line paragraph:

```markdown
Chunk selection is deterministic and near-maximum: qemer-ingest starts with a budget-sized raw-token window and performs at most eight refinement checks plus a one-token fallback, while `/tokenize` remains authoritative for every accepted prefixed payload. It does not exhaustively search for the mathematically longest fitting slice.
```

Keep the existing endpoint, prefix, CLI option, artifact, and reproducibility documentation unchanged.

- [ ] **Step 7: Run focused CLI and documentation verification.**

Run:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest tests/test_cli_build.py tests/test_tokenization.py tests/test_chunking.py -q
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff check src/qemer_ingest/cli.py src/qemer_ingest/tokenization.py src/qemer_ingest/chunking.py tests/test_cli_build.py tests/test_tokenization.py tests/test_chunking.py
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff format --check src/qemer_ingest/cli.py src/qemer_ingest/tokenization.py src/qemer_ingest/chunking.py tests/test_cli_build.py tests/test_tokenization.py tests/test_chunking.py
test "$(sed -n '/^def build(/,/^def main/p' src/qemer_ingest/cli.py | rg -n "asyncio\.run" | wc -l)" -eq 1
rg -n "near-maximum|mathematically longest" README.md
```

Expected: focused tests and Ruff pass, the synchronous `build` block contains exactly one `asyncio.run`, and README contains the bounded-guarantee wording. The unrelated `inspect` command keeps its existing event-loop calls.

### Task 4: Verify the complete implementation and run a live smoke build

**Files:**

- Verify only; do not make unrelated changes while responding to failures.

- [ ] **Step 1: Run the complete unit suite and static checks.**

Run each command independently so a failure is visible at its source:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv build
```

Expected: the full suite passes, Ruff reports no issues or formatting changes, and both sdist and wheel build successfully.

- [ ] **Step 2: Audit the implementation against the performance and scope contracts.**

Run:

```bash
rg -n "_MAX_REFINEMENT_ATTEMPTS|_next_candidate_length|_bounded_fitting_slice" src/qemer_ingest/chunking.py tests/test_chunking.py
rg -n "AsyncClient\(" src/qemer_ingest/tokenization.py
sed -n '/^def build(/,/^def main/p' src/qemer_ingest/cli.py | rg -n "asyncio\.run"
rg -n "POST /chunk|/chunk|transformers|AutoTokenizer" src tests pyproject.toml README.md
git diff --check
```

Expected: the bounded helpers and their tests are present; `tokenization.py` contains one `AsyncClient` construction in `__aenter__`; the synchronous `build` block contains one `asyncio.run`; the forbidden endpoint/tokenizer search returns no matches; `git diff --check` is silent.

- [ ] **Step 3: Run a stock llama-server health check.**

With the user's embedding server listening on `http://127.0.0.1:8080`, run:

```bash
curl --fail --silent --show-error http://127.0.0.1:8080/health
```

Expected: llama-server reports a healthy or ready status. If the server is not running, report that environmental blocker rather than changing code or starting a different model.

- [ ] **Step 4: Run an end-to-end small-repository build.**

First choose a new output path that does not exist; `/tmp/qemer-bounded-hello-world` is the default for this verification. Confirm it is absent before running the build:

```bash
test ! -e /tmp/qemer-bounded-hello-world
```

Then run:

```bash
UV_CACHE_DIR=/tmp/qemer-ingest-uv-cache uv run qemer-ingest build \
  https://github.com/octocat/Hello-World \
  --ref master \
  --library hello-world \
  --version master \
  --embedding-url http://127.0.0.1:8080 \
  --embedding-model nomic-embed-text-v1.5 \
  --embedding-dim 768 \
  --output /tmp/qemer-bounded-hello-world
```

Expected: the command resolves the repository, selects and parses its documentation, embeds all emitted rows, publishes `/tmp/qemer-bounded-hello-world`, and prints its resolved commit, selected-file count, and emitted-row count. If the proposed path already exists, choose another explicit unused `/tmp/qemer-bounded-hello-world-<number>` path; do not delete or overwrite the existing artifact.

- [ ] **Step 5: Inspect the produced artifact metadata and report measured evidence.**

Run:

```bash
find /tmp/qemer-bounded-hello-world -maxdepth 1 -type f -print
sed -n '1,240p' /tmp/qemer-bounded-hello-world/build-report.json
sed -n '1,240p' /tmp/qemer-bounded-hello-world/manifest.json
```

Expected: the artifact contains its compressed corpus, build report, and manifest; the JSON records the resolved commit, selected files, token chunk policy, row count, embedding model, and dimension. In the completion report, include the full test count, Ruff/build results, the live repository commit, selected files, emitted rows, and output path; do not claim NumPy-scale performance from the small smoke test.

## Plan self-review

- [ ] **Spec coverage:** Confirm the plan covers budget-sized first probes, deterministic bounded refinement, the eight-plus-one check ceiling, exact final authoritative counts, no redundant accepted-child check, overlap and identity preservation, one reusable token client, exception-safe closure, one event loop, validation and preflight order, unchanged publication semantics, README wording, and a live stock llama-server smoke build.
- [ ] **Placeholder scan:** Run `rg -n "TO[D]O|TB[D]|FIXM[E]|fill i[n]|as neede[d]|similar t[o]" docs/superpowers/plans/2026-08-30-bounded-token-chunking.md` and expect no matches.
- [ ] **Type and signature consistency:** Confirm test fakes satisfy every method they exercise, `TokenizationClient` retains its constructor and public endpoint methods, `chunk_units` retains its public signature, and `_run_build` receives every validated CLI value without changing public option types.
- [ ] **Scope check:** Confirm the diff contains no parser changes, no embedding batching changes, no llama.cpp changes, no new endpoint or tokenizer dependency, no artifact schema changes, and no automatic commit or push.

## Execution handoff

The plan supports either execution mode:

1. **Subagent-driven execution:** Use `superpowers:subagent-driven-development` in this session, dispatching one fresh implementer per task and applying the required spec-compliance and code-quality reviews after each task.
2. **Inline execution:** Use `superpowers:executing-plans` and implement the checkboxes sequentially in this worktree with review checkpoints between tasks.

The user has already selected subagent-driven execution for this feature, so option 1 is the default for the next implementation turn. Do not begin implementation as part of this plan-writing task.
