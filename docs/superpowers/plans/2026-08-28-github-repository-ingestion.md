# GitHub Repository Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python `qemer-ingest` CLI that inspects and packages documentation from a pinned public GitHub revision as a local Qemer corpus artifact.

**Architecture:** The CLI resolves and downloads one public GitHub revision, applies conservative path discovery, parses documentation into Qemer prose/code units, embeds those units through a user-supplied HTTP endpoint, and atomically writes a local manifest/artifact/report directory. The project shares only Qemer's documented artifact contract; it never imports Qemer code or invokes the Qemer executable.

**Tech Stack:** Python 3.14, uv, Typer, httpx, markdown-it-py, docutils, PyArrow, zstandard, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-28-github-repository-ingestion-design.md`

## Global Constraints

- Accept only public `https://github.com/<owner>/<repository>` source URLs and require `--ref`.
- Resolve the requested ref to a full Git commit SHA before discovering or parsing files.
- Default discovery includes only root README files and Markdown/RST/TXT files beneath `docs/`, `doc/`, or `documentation/`; repeated `--include` options are the only expansion mechanism.
- Do not launch/download an embedding model, access private repositories, crawl websites, publish artifacts, or invoke Qemer.
- Emit only a relative artifact filename in manifest `url`; Qemer resolves it beside `manifest.json`.
- The output directory must not exist and is created only by a final atomic rename.
- The tarball contains only `corpus.parquet`; the Parquet schema is `snippet_id`, `kind`, `title`, `source_url`, `text`, and fixed-width float32 `vector`.
- Do not commit or push while executing this plan unless the user explicitly asks in that execution turn.

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | CLI and runtime/test dependencies. |
| `src/qemer_ingest/models.py` | Frozen dataclasses for repository identity, discovery results, parsed units, vectors, manifest entries, and reports. |
| `src/qemer_ingest/github.py` | Validate GitHub URLs, resolve refs, download archives, and safely unpack their root. |
| `src/qemer_ingest/discovery.py` | Apply default paths and `--include` globs. |
| `src/qemer_ingest/parsing.py` | Convert Markdown, RST, and TXT files into deterministic prose/code units. |
| `src/qemer_ingest/embedding.py` | Call the embeddings endpoint and validate vectors. |
| `src/qemer_ingest/artifact.py` | Write Parquet, archive, manifest, report, checksums, and atomic staging output. |
| `src/qemer_ingest/cli.py` | Typer `inspect` and `build` commands. |
| `tests/` | Unit, transport, parser, artifact, and CLI tests using fixtures and no live service. |

### Task 1: Establish the Python package and shared data model

**Files:**

- Modify: `pyproject.toml`
- Modify: `src/qemer_ingest/__init__.py`
- Create: `src/qemer_ingest/__main__.py`
- Create: `src/qemer_ingest/models.py`
- Create: `src/qemer_ingest/cli.py`
- Create: `tests/test_models.py`

**Interfaces:**

- Produces `RepositoryRef(url: str, owner: str, repository: str, requested_ref: str, commit_sha: str)`.
- Produces `DiscoveryReport(selected: tuple[Path, ...], skipped: dict[str, str])` and `BuildReport` with requested ref, resolved commit, selected/skipped files, and prose/code row counts.
- Produces `DocumentUnit(snippet_id: str, kind: Literal["prose", "code"], title: str, source_url: str, text: str)`.
- Produces `EmbeddedUnit(unit: DocumentUnit, vector: list[float])`.
- Produces a Typer application exposed as `qemer-ingest` and `python -m qemer_ingest`.

- [ ] **Step 1: Write failing model tests.**

```python
from qemer_ingest.models import DocumentUnit, RepositoryRef


def test_repository_ref_keeps_the_resolved_commit() -> None:
    source = RepositoryRef("https://github.com/numpy/numpy", "numpy", "numpy", "v2.3.0", "a" * 40)
    assert source.commit_sha == "a" * 40


def test_document_unit_rejects_empty_text() -> None:
    try:
        DocumentUnit("numpy-2.3-readme-0001", "prose", "README", "https://example.test", "")
    except ValueError as error:
        assert "text" in str(error)
    else:
        raise AssertionError("empty text must not become an embedding request")
```

- [ ] **Step 2: Run the test before implementation.**

Run: `uv run pytest tests/test_models.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'qemer_ingest.models'`.

- [ ] **Step 3: Add dependencies and minimal package surface.**

Run: `uv add typer httpx markdown-it-py docutils pyarrow zstandard && uv add --dev pytest pytest-asyncio ruff`

Create frozen, slotted dataclasses in `models.py`. `DocumentUnit.__post_init__` must reject blank snippet IDs, titles, source URLs, texts, and kinds other than `prose`/`code`. Replace the generated `main` with an import from `qemer_ingest.cli`; create `app = typer.Typer(no_args_is_help=True)` and `main()` that calls `app()`; make `__main__.py` call `main()`.

- [ ] **Step 4: Run model and command checks.**

Run: `uv run pytest tests/test_models.py -q && uv run qemer-ingest --help && uv run python -m qemer_ingest --help`

Expected: PASS and both entry points display the same help.

### Task 2: Resolve and download one public GitHub revision

**Files:**

- Create: `src/qemer_ingest/github.py`
- Create: `tests/test_github.py`

**Interfaces:**

- Produces `parse_repository_url(value: str) -> tuple[str, str]`.
- Produces `GitHubClient(transport: httpx.AsyncBaseTransport | None = None)`.
- Produces `await GitHubClient.resolve(url: str, ref: str) -> RepositoryRef`.
- Produces `await GitHubClient.download_archive(source: RepositoryRef, destination: Path) -> Path`.

- [ ] **Step 1: Write URL/ref tests using `httpx.MockTransport`.**

Make a handler return `{"sha": "b" * 40}` for `GET /repos/numpy/numpy/commits/v2.3.0`, then use `@pytest.mark.asyncio` and assert:

```python
client = GitHubClient(transport=httpx.MockTransport(handler))
resolved = await client.resolve("https://github.com/numpy/numpy", "v2.3.0")
assert (resolved.owner, resolved.repository, resolved.commit_sha) == ("numpy", "numpy", "b" * 40)
```

Also assert `git@github.com:numpy/numpy.git`, `https://github.com/numpy/numpy/tree/main`, and `https://gitlab.com/numpy/numpy` raise `ValueError` before a request.

- [ ] **Step 2: Run tests and confirm they fail.**

Run: `uv run pytest tests/test_github.py -q`

Expected: FAIL because `qemer_ingest.github` is absent.

- [ ] **Step 3: Implement validation and ref resolution.**

Use `urllib.parse.urlparse`; accept only scheme `https`, host `github.com`, and two non-empty path segments, with an optional `.git` suffix. Request `https://api.github.com/repos/{owner}/{repository}/commits/{ref}`, call `raise_for_status`, and require a 40-character hexadecimal `sha` before constructing `RepositoryRef`.

- [ ] **Step 4: Add and implement archive extraction.**

Use an in-memory gzipped tar containing `numpy-<sha>/README.md`. Assert `download_archive` returns its sole top-level directory. Reject tar members with absolute paths or a `..` component before extracting `tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")` into the supplied temporary directory.

- [ ] **Step 5: Run focused GitHub tests.**

Run: `uv run pytest tests/test_github.py -q`

Expected: PASS with no live network request.

### Task 3: Discover documentation and implement `inspect`

**Files:**

- Create: `src/qemer_ingest/discovery.py`
- Modify: `src/qemer_ingest/cli.py`
- Create: `tests/test_discovery.py`
- Create: `tests/test_cli_inspect.py`

**Interfaces:**

- Produces `discover(root: Path, includes: tuple[str, ...]) -> DiscoveryReport`.
- Produces `qemer-ingest inspect <repository-url> --ref <ref> [--include <glob> ...]`.

- [ ] **Step 1: Write conservative-discovery tests.**

Create a temporary tree with `README.md`, `docs/guide.md`, `doc/api.rst`, `documentation/intro.txt`, `src/notes.md`, `.github/guide.md`, and `node_modules/readme.md`. Assert only the README and three conventional documentation files are selected. Then call `discover(root, ("src/notes.md",))` and assert `src/notes.md` becomes selected.

- [ ] **Step 2: Run the test and confirm it fails.**

Run: `uv run pytest tests/test_discovery.py -q`

Expected: FAIL because `qemer_ingest.discovery` is absent.

- [ ] **Step 3: Implement deterministic discovery.**

Walk files with `Path.rglob("*")`, reject any path with a hidden segment or a segment in `{node_modules, vendor, .git}`, and operate on repository-relative paths. Default-select case-insensitive root README names and `.md`/`.rst`/`.txt` files below `docs`, `doc`, or `documentation`. Match includes with `PurePosixPath.match`, verify UTF-8 decoding, and sort selections by `as_posix()`.

- [ ] **Step 4: Add inspect CLI tests and implementation.**

Inject a fake `GitHubClient` into `typer.testing.CliRunner`. Assert inspect prints the resolved 40-character SHA, `README.md`, and `docs/guide.md`, does not print `embedding`, and does not create an output directory. Implement inspect with `TemporaryDirectory`, resolution, archive download, discovery, and a line per selected path.

- [ ] **Step 5: Run discovery and inspect tests.**

Run: `uv run pytest tests/test_discovery.py tests/test_cli_inspect.py -q`

Expected: PASS.

### Task 4: Parse Markdown, RST, and plain text into Qemer units

**Files:**

- Create: `src/qemer_ingest/parsing.py`
- Create: `tests/fixtures/guide.md`
- Create: `tests/fixtures/api.rst`
- Create: `tests/test_parsing.py`

**Interfaces:**

- Produces `parse_document(path: Path, source: RepositoryRef, library: str, version: str) -> tuple[DocumentUnit, ...]`.
- Produces source URLs in the form `https://github.com/<owner>/<repository>/blob/<commit-sha>/<repository-relative-path>`.

- [ ] **Step 1: Write fixture-based parsing tests.**

Make `guide.md` contain an H1 introduction with prose and a Python fence, followed by H2 “Arrays” with two fences. Assert parsing yields two prose units and two code units, the “Arrays” code fences become one unit joined by two newlines, and all units use the fixture source's 40-character commit SHA. Make `api.rst` contain ordinary text and a `.. code-block:: python` directive; assert one prose and one code unit. Add a TXT case asserting one prose unit titled with the file stem, and an empty-content case asserting no units.

- [ ] **Step 2: Run parser tests and confirm they fail.**

Run: `uv run pytest tests/test_parsing.py -q`

Expected: FAIL because `qemer_ingest.parsing` is absent.

- [ ] **Step 3: Implement Markdown parsing.**

Use `markdown_it.MarkdownIt("commonmark")` tokens. Start a section at each `heading_open`, consume its immediately following `inline` token as the title rather than prose, collect other inline text outside `fence` tokens as prose, and collect fence content as code. Finalize at the next heading/end of file, dropping empty sides. Use the file stem for pre-heading text. Create IDs with `sha256(f"{library}\0{version}\0{relative_path}\0{ordinal}".encode()).hexdigest()[:16]`, prefixed by `f"{library}-{version}-"`.

- [ ] **Step 4: Implement RST and TXT parsing.**

Use `docutils.core.publish_doctree` for RST. For each `nodes.section`, use the title, normal text for prose, and `nodes.literal_block`/`nodes.doctest_block` for code. TXT strips its UTF-8 contents and emits one prose unit only when non-empty. Route by lowercased suffix and raise `ValueError` for another suffix.

- [ ] **Step 5: Run parser tests.**

Run: `uv run pytest tests/test_parsing.py -q`

Expected: PASS, including deterministic IDs, pinned URLs, multi-fence joining, and empty-section skipping.

### Task 5: Embed units and write a verified local artifact

**Files:**

- Create: `src/qemer_ingest/embedding.py`
- Create: `src/qemer_ingest/artifact.py`
- Create: `tests/test_embedding.py`
- Create: `tests/test_artifact.py`

**Interfaces:**

- Produces `EmbeddingClient(base_url: str, model: str, dimension: int)` and `await embed_all(units: tuple[DocumentUnit, ...]) -> tuple[EmbeddedUnit, ...]`.
- Produces `build_artifact(output: Path, library: str, version: str, model: str, dimension: int, embedded: tuple[EmbeddedUnit, ...], report: BuildReport) -> Path`.

- [ ] **Step 1: Write embedding transport tests.**

Use `httpx.MockTransport` to capture requests. Assert `embed_all` posts one request per unit to `/v1/embeddings` with `{"input": unit.text, "model": "nomic-embed-text-v1.5"}` and preserves input order. Return `[1.0, 2.0, 3.0]` for a dimension-three client and assert success. Return an empty `data` array and a two-element vector and assert each error names the affected source URL.

- [ ] **Step 2: Run embedding tests and confirm they fail.**

Run: `uv run pytest tests/test_embedding.py -q`

Expected: FAIL because `qemer_ingest.embedding` is absent.

- [ ] **Step 3: Implement the endpoint client.**

Strip a trailing slash from `base_url`, require a positive dimension, and use `httpx.AsyncClient`. Post each unit to `{base_url}/v1/embeddings`, call `raise_for_status`, parse `data[0].embedding` as floats, reject any width other than the configured dimension, and return units in their original order.

- [ ] **Step 4: Write artifact tests.**

Build a three-dimension fixture with one prose and one code row. Assert the output contains exactly `manifest.json`, `numpy-2.3.0.tar.zst`, and `build-report.json`; manifest `url` equals `numpy-2.3.0.tar.zst`; the independently computed archive SHA-256 and byte count equal its manifest fields; the zstd tar's sole member is `corpus.parquet`; and PyArrow reads the six expected columns with float32 vectors of width three.

- [ ] **Step 5: Implement atomic artifact building.**

Reject an existing output path before creating a `tempfile.TemporaryDirectory` beside its parent. Build an Arrow table with string columns `snippet_id`, `kind`, `title`, `source_url`, and `text`, plus `pa.list_(pa.float32(), list_size=dimension)` for `vector`. Write `corpus.parquet`, create a zstandard-compressed tar containing only that filename, calculate byte count and SHA-256, then write sorted-key JSON manifest/report. Rename the staging directory to output only after every write succeeds.

- [ ] **Step 6: Run embedding and artifact tests.**

Run: `uv run pytest tests/test_embedding.py tests/test_artifact.py -q`

Expected: PASS with no real embedding server.

### Task 6: Wire `build`, document the workflow, and verify the project

**Files:**

- Modify: `src/qemer_ingest/cli.py`
- Modify: `README.md`
- Create: `tests/test_cli_build.py`

**Interfaces:**

- Produces `qemer-ingest build <repository-url> --ref <ref> --library <library> --version <version> --embedding-url <url> --embedding-model <model> --embedding-dim <positive-int> --output <new-directory> [--include <glob> ...]`.

- [ ] **Step 1: Write an end-to-end build test using injected fakes.**

Inject a fake GitHub client returning a fixture archive root containing `README.md` and a fake embedding client returning dimension-three vectors. Invoke `CliRunner` with all required build arguments. Assert exit code zero, output names `manifest.json`, `numpy-2.3.0.tar.zst`, and `build-report.json`, and report fields `requested_ref`, `resolved_commit`, `selected_files`, `prose_rows`, and `code_rows`.

- [ ] **Step 2: Run the build test and confirm it fails.**

Run: `uv run pytest tests/test_cli_build.py -q`

Expected: FAIL because the build command is absent.

- [ ] **Step 3: Implement command orchestration.**

Validate non-empty `library`/`version`, positive `embedding_dim`, and a nonexistent `output` before contacting the embedding endpoint. Resolve, download, discover, parse, and embed in that order; fail with a non-zero Typer error when no files or no units result. Assemble the report, call `build_artifact`, and print the resolved SHA, selected-file count, emitted-row count, and final manifest path.

- [ ] **Step 4: Replace the generated README.**

Document uv installation, public-GitHub-only input, the required user-managed embeddings endpoint, conservative discovery and `--include`, local output layout, and this handoff:

```sh
qemer-ingest inspect https://github.com/numpy/numpy --ref v2.3.0
qemer-ingest build https://github.com/numpy/numpy --ref v2.3.0 --library numpy --version 2.3.0 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --output ./qemer-corpora/numpy-2.3.0
qemer install numpy@2.3.0 --manifest ./qemer-corpora/numpy-2.3.0/manifest.json
```

- [ ] **Step 5: Run final verification.**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv build`

Expected: all tests, linting, formatting, and package build checks pass with no live GitHub or embedding endpoint.

- [ ] **Step 6: Confirm local-only state.**

Run: `git remote -v && git status --short`

Expected: no remote exists; no commit or push occurs unless the user explicitly requests it.

## Plan Self-Review

**Spec coverage:** Tasks 1–2 establish the Python CLI and pinned public GitHub revision acquisition. Task 3 implements conservative inspect-first selection. Task 4 creates Qemer prose/code rows for each supported format. Task 5 validates embeddings and emits the exact local artifact contract plus a non-contract report. Task 6 wires the build, documents its Qemer handoff, and verifies the package without an external service.

**Placeholder scan:** Source URL validation, archive endpoint, path rules, parser behavior, embedding request/response checks, artifact filenames, JSON fields, command arguments, and verification commands are explicit.

**Type consistency:** `RepositoryRef` flows from `GitHubClient.resolve` into parsing/reports; `DocumentUnit` flows from parsing into `EmbeddingClient.embed_all`; `EmbeddedUnit` flows into `build_artifact`; the CLI commands are exercised with injected clients.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-28-github-repository-ingestion.md`. The repository is local-only; do not add a remote, publish, commit, or push unless the user explicitly requests it.
