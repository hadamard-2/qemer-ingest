# LangChain overflow chunking implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound every embedding request by splitting only structurally valid prose/code units that exceed a configurable character limit.

**Architecture:** Existing Markdown, RST, and TXT parsing continues to establish semantic units and provenance. A new small module uses LangChain's standalone `RecursiveCharacterTextSplitter` to divide only oversized `DocumentUnit` values, then the `build` command applies it before embedding and writes the selected policy into `build-report.json`.

**Tech Stack:** Python 3.14, uv, Typer, `langchain-text-splitters`, pytest, pytest-asyncio, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-29-langchain-overflow-chunking-design.md`

## Global Constraints

- Add only the standalone `langchain-text-splitters` dependency; do not add the `langchain` framework or a model/tokenizer dependency.
- Keep the current Markdown/RST/TXT structural parsers as the authority for titles, prose/code classification, parser skips, parent IDs, and pinned source URLs.
- Use `RecursiveCharacterTextSplitter` only after parsing a `DocumentUnit` whose text exceeds the configured character limit.
- `build` defaults are exactly `--chunk-size 8000` and `--chunk-overlap 0`; require `chunk_size > 0` and `0 <= chunk_overlap < chunk_size` before creating any GitHub or embedding client.
- Unsplit units retain their existing IDs. Split child IDs are exactly `<parent-snippet-id>-<kind>-<three-digit-child-ordinal>` with ordinals starting at `001`.
- Child units retain their parent `kind`, `title`, and pinned `source_url`; use explicit separators `("\n\n", "\n", " ", "")` and apply the configured overlap to both prose and code.
- `build-report.json` records `chunk_size` and `chunk_overlap`; prose/code row counts are calculated after splitting.
- Do not add token-counting, semantic/LLM chunking, embedding-runtime changes, Qemer invocation, publication, private-repository access, commits, or pushes.

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Declare the standalone LangChain text-splitter dependency. |
| `src/qemer_ingest/chunking.py` | Deterministically split oversized parsed units and preserve their metadata. |
| `src/qemer_ingest/models.py` | Record chunk policy values in `BuildReport`. |
| `src/qemer_ingest/cli.py` | Expose/validate chunk CLI options and apply chunking before embeddings. |
| `src/qemer_ingest/artifact.py` | Serialize chunk policy values in `build-report.json`. |
| `tests/test_chunking.py` | Unit-test LangChain overflow behavior, overlap, and child identities. |
| `tests/test_cli_build.py` | Exercise defaults, configured chunking, and preflight validation through the real CLI pipeline. |
| `tests/test_artifact.py` | Verify persisted report fields. |
| `README.md` | Document chunk options, defaults, and reproducibility reporting. |

### Task 1: Add the LangChain overflow chunker

**Files:**

- Modify: `pyproject.toml`
- Create: `src/qemer_ingest/chunking.py`
- Create: `tests/test_chunking.py`

**Interfaces:**

- Consumes: `DocumentUnit` from `qemer_ingest.models`.
- Produces: `chunk_units(units: tuple[DocumentUnit, ...], *, chunk_size: int, chunk_overlap: int) -> tuple[DocumentUnit, ...]`.
- Later tasks call `chunk_units` after flattening parsed units and before `EmbeddingClient.embed_all`.

- [ ] **Step 1: Add the standalone runtime dependency.**

Run: `uv add langchain-text-splitters`

Expected: `pyproject.toml` lists `langchain-text-splitters` in `[project].dependencies`; no `langchain` package is added.

- [ ] **Step 2: Write failing unit tests for unchanged and oversized units.**

Create `tests/test_chunking.py` with the following helpers and tests:

```python
from qemer_ingest.chunking import chunk_units
from qemer_ingest.models import DocumentUnit


def unit(kind: str = "prose", text: str = "short text") -> DocumentUnit:
    return DocumentUnit("numpy-2.3-abc", kind, "Arrays", "https://example.test/README.md", text)


def test_chunk_units_keeps_a_unit_at_or_below_the_limit() -> None:
    original = unit(text="abcdefgh")
    assert chunk_units((original,), chunk_size=8, chunk_overlap=0) == (original,)


def test_chunk_units_splits_oversized_units_with_stable_child_metadata() -> None:
    chunks = chunk_units((unit(text="abcdefghijklmnopqrst"),), chunk_size=8, chunk_overlap=2)
    assert [(chunk.snippet_id, chunk.kind, chunk.title, chunk.source_url, chunk.text) for chunk in chunks] == [
        ("numpy-2.3-abc-prose-001", "prose", "Arrays", "https://example.test/README.md", "abcdefgh"),
        ("numpy-2.3-abc-prose-002", "prose", "Arrays", "https://example.test/README.md", "ghijklmn"),
        ("numpy-2.3-abc-prose-003", "prose", "Arrays", "https://example.test/README.md", "mnop"),
    ]
```

Add a third test with a prose and code unit sharing parent ID `numpy-2.3-abc`; split both at eight characters and assert the resulting IDs use `-prose-001` and `-code-001`, preventing cross-kind collisions.

- [ ] **Step 3: Run the unit tests to confirm the intended RED failure.**

Run: `uv run pytest tests/test_chunking.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'qemer_ingest.chunking'`.

- [ ] **Step 4: Implement validation and deterministic splitting.**

Create `src/qemer_ingest/chunking.py` with this public shape:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qemer_ingest.models import DocumentUnit

_SEPARATORS = ("\n\n", "\n", " ", "")


def chunk_units(
    units: tuple[DocumentUnit, ...], *, chunk_size: int, chunk_overlap: int
) -> tuple[DocumentUnit, ...]:
    _validate_options(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        separators=list(_SEPARATORS),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    output: list[DocumentUnit] = []
    for unit in units:
        output.extend(_chunk_unit(unit, splitter, chunk_size))
    return tuple(output)
```

Implement `_validate_options` to raise `ValueError("chunk size must be positive")` for a non-positive limit and `ValueError("chunk overlap must be non-negative and smaller than chunk size")` when overlap is negative or not smaller than the limit. `_chunk_unit` must return `(unit,)` when `len(unit.text) <= chunk_size`; otherwise call `splitter.split_text(unit.text)`, ignore blank returned strings, and construct one `DocumentUnit` per chunk with `f"{unit.snippet_id}-{unit.kind}-{ordinal:03d}"`.

- [ ] **Step 5: Run focused chunking verification.**

Run: `uv run pytest tests/test_chunking.py -q && uv run ruff check src/qemer_ingest/chunking.py tests/test_chunking.py && uv run ruff format --check src/qemer_ingest/chunking.py tests/test_chunking.py`

Expected: PASS with no warnings; the test proves unchanged units preserve IDs and oversized prose/code units preserve metadata while receiving stable, kind-qualified child IDs.

### Task 2: Wire chunk policy into build reports and the CLI

**Files:**

- Modify: `src/qemer_ingest/models.py`
- Modify: `src/qemer_ingest/artifact.py`
- Modify: `src/qemer_ingest/cli.py`
- Modify: `tests/test_artifact.py`
- Modify: `tests/test_cli_build.py`

**Interfaces:**

- Consumes: `chunk_units` from `qemer_ingest.chunking` and the existing `BuildReport` construction in `cli.build`.
- Produces: `BuildReport(repository_url: str, requested_ref: str, resolved_commit: str, selected_files: tuple[Path, ...], skipped_files: dict[str, str], explicitly_included_files: tuple[Path, ...], parser_skips: tuple[ParserSkip, ...], prose_rows: int, code_rows: int, chunk_size: int, chunk_overlap: int)` and `build --chunk-size <int> --chunk-overlap <int>`.
- `artifact._report_payload` serializes both fields exactly as JSON integers.

- [ ] **Step 1: Write failing report serialization and build CLI tests.**

Extend the expected `build-report.json` payloads in `tests/test_cli_build.py` so each includes:

```python
"chunk_size": 8000,
"chunk_overlap": 0,
```

Add `from hashlib import sha256` to `tests/test_cli_build.py`, implement this fake beside `FakeGitHubClient`, and add the following end-to-end test using `RecordingEmbeddingClient`:

```python
class LongReadmeGitHubClient(FakeGitHubClient):
    async def download_archive(self, source: RepositoryRef, destination: Path) -> Path:
        repository_root = destination / "numpy-a"
        repository_root.mkdir()
        (repository_root / "README.md").write_text(
            "# Overflow\n\nabcdefghijklmnopqrst\n", encoding="utf-8"
        )
        return repository_root


def test_build_chunks_before_embedding_and_records_the_requested_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "GitHubClient", LongReadmeGitHubClient)
    monkeypatch.setattr(cli, "EmbeddingClient", RecordingEmbeddingClient)
    RecordingEmbeddingClient.calls.clear()
    output = tmp_path / "numpy-chunked"

    result = CliRunner().invoke(
        cli.app,
        build_arguments(output) + ["--chunk-size", "8", "--chunk-overlap", "2"],
    )

    assert result.exit_code == 0, result.output
    embedded = RecordingEmbeddingClient.calls[0]
    digest = sha256(("numpy\0" "2.3.0\0" "README.md\0" "1").encode()).hexdigest()[:16]
    parent_id = f"numpy-2.3.0-{digest}"
    assert [unit.snippet_id for unit in embedded] == [
        f"{parent_id}-prose-001",
        f"{parent_id}-prose-002",
        f"{parent_id}-prose-003",
    ]
    assert json.loads((output / "build-report.json").read_text())["chunk_size"] == 8
    assert json.loads((output / "build-report.json").read_text())["chunk_overlap"] == 2
```

Also add the help regression test before the CLI implementation:

```python
def test_build_help_describes_overflow_chunking_options() -> None:
    result = CliRunner().invoke(cli.app, ["build", "--help"])
    assert result.exit_code == 0
    assert "--chunk-size" in result.output
    assert "--chunk-overlap" in result.output
    assert "8000" in result.output
    assert "0" in result.output
```

Add this complete preflight test alongside the current invalid option tests:

```python
@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("--chunk-size", "0", "must be positive"),
        ("--chunk-overlap", "-1", "must be non-negative"),
    ),
)
def test_build_rejects_invalid_chunk_options_before_external_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
) -> None:
    def unexpected_client() -> None:
        raise AssertionError("invalid chunk option constructed a GitHub client")

    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    arguments = build_arguments(tmp_path / "output") + [option, value]
    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code != 0
    assert message in result.output
```

Add this exact relation-validation test, also before implementation:

```python
def test_build_rejects_overlap_equal_to_chunk_size_before_external_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_client() -> None:
        raise AssertionError("invalid chunk option constructed a GitHub client")

    monkeypatch.setattr(cli, "GitHubClient", unexpected_client)
    result = CliRunner().invoke(
        cli.app,
        build_arguments(tmp_path / "output")
        + ["--chunk-size", "8", "--chunk-overlap", "8"],
    )

    assert result.exit_code != 0
    assert "must be smaller than chunk size" in result.output
```

- [ ] **Step 2: Run the affected tests to verify RED.**

Run: `uv run pytest tests/test_cli_build.py tests/test_artifact.py -q`

Expected: FAIL because `BuildReport` has no chunk fields, `build` does not accept the two options, and report JSON lacks the new keys.

- [ ] **Step 3: Extend the report model and serializer.**

Add these frozen dataclass fields after `code_rows` in `BuildReport`:

```python
chunk_size: int
chunk_overlap: int
```

Add exact integer fields to `artifact._report_payload`:

```python
"chunk_size": report.chunk_size,
"chunk_overlap": report.chunk_overlap,
```

Update every `BuildReport` constructor in test fixtures and production code with explicit values so no test depends on an implicit policy.

- [ ] **Step 4: Add CLI options, validate them early, and apply the splitter.**

At module level in `cli.py`, define:

```python
_CHUNK_SIZE_OPTION = typer.Option(8000, "--chunk-size")
_CHUNK_OVERLAP_OPTION = typer.Option(0, "--chunk-overlap")
```

Add `chunk_size: int = _CHUNK_SIZE_OPTION` and `chunk_overlap: int = _CHUNK_OVERLAP_OPTION` to `build`. After the existing embedding dimension check and before `os.path.lexists(output)`, reject `chunk_size <= 0`, `chunk_overlap < 0`, and `chunk_overlap >= chunk_size` with the messages asserted in tests.

After flattening `parsed_documents` and before the empty-unit check, replace the direct tuple with:

```python
parsed_units = tuple(unit for parsed in parsed_documents for unit in parsed.units)
units = chunk_units(
    parsed_units,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
```

Use `units` for the empty-result guard, embedding request, row counts, and `BuildReport`. Pass `chunk_size` and `chunk_overlap` into the report constructor. Import only `chunk_units` from the new module; do not change `inspect`.

- [ ] **Step 5: Run the integration and serialization tests.**

Run: `uv run pytest tests/test_chunking.py tests/test_cli_build.py tests/test_artifact.py -q && uv run ruff check src tests && uv run ruff format --check src tests`

Expected: PASS. The configured build sends child units to the fake embedding client in splitter order, report JSON retains the caller's policy, default builds record `8000`/`0`, and invalid policy never constructs an external client.

### Task 3: Document the policy and verify the complete project

**Files:**

- Modify: `README.md`

**Interfaces:**

- Consumes: completed `build --chunk-size` and `--chunk-overlap` options plus serialized `BuildReport` values.
- Produces: user-facing build documentation consistent with the tested CLI contract.

- [ ] **Step 1: Document bounded overflow chunking.**

Add a README section immediately after the build example with this content:

````markdown
### Overflow chunking

`build` preserves Markdown/RST/TXT structure first, then splits only prose or code units longer than `--chunk-size`. The default is `8000` characters with no overlap. Set `--chunk-overlap` to repeat trailing characters at the start of the next chunk; it must be smaller than the selected chunk size.

```sh
qemer-ingest build https://github.com/numpy/numpy --ref v2.3.0 --library numpy --version 2.3.0 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --chunk-size 12000 --chunk-overlap 400 --output ./qemer-corpora/numpy-2.3.0
```

`build-report.json` records `chunk_size` and `chunk_overlap`, so the artifact records the policy used to create its embedding rows.
````

Keep every prose paragraph on one line; do not alter the existing public-GitHub or user-managed-embedding boundary text.

- [ ] **Step 2: Run final verification.**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv build && git diff --check`

Expected: all tests pass, Ruff reports no lint or formatting changes, package builds an sdist and wheel, and the diff has no whitespace errors.

- [ ] **Step 3: Confirm local-only state.**

Run: `git status --short`

Expected: only implementation/docs/test changes for this plan are present; no push or publication occurs.

## Plan self-review

**Spec coverage:** Task 1 introduces only the standalone LangChain overflow engine and exact child identity behavior. Task 2 exposes the approved CLI policy, validates it before external work, applies splitting before embedding, and serializes it for reproducibility. Task 3 documents the feature and runs the required full project verification.

**Placeholder scan:** The plan contains no TBD/TODO markers and every test, validation message, option name, interface, and command is explicit.

**Type consistency:** `chunk_units` consumes/returns `tuple[DocumentUnit, ...]`; `cli.build` creates the `BuildReport` with the two new integer fields; `artifact._report_payload` serializes those same field names as `chunk_size` and `chunk_overlap`.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-29-langchain-overflow-chunking.md`. The plan is ready for sub-agent-driven execution or inline execution; no commit or push has been made.
