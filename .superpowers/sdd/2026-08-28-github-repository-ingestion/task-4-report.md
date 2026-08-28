# Task 4 report: parse documentation into Qemer units

## Implementation

Added `parse_document` for repository-relative Markdown, reStructuredText, and plain-text inputs. Markdown is tokenized with CommonMark, headings define sections, prose excludes heading tokens, and all fenced blocks within a section are combined into one code unit. RST uses docutils doctrees, including document-title and section-title forms, to separate ordinary content from literal and doctest blocks. TXT produces one stripped prose unit only when content remains.

Every emitted row carries a deterministic snippet ID made from the library, version, repository-relative POSIX path, and one-based emitted-section ordinal. The prose and code rows produced from a section intentionally share that ID, title, and a GitHub blob URL built from the resolved repository owner, repository, and full commit SHA. Unsupported suffixes raise `ValueError`.

## TDD RED/GREEN evidence

RED: after adding the two local fixtures and `tests/test_parsing.py`, `QEMER_UV_CACHE=/tmp/qemer-ingest-uv-cache UV_CACHE_DIR="$QEMER_UV_CACHE" uv run pytest tests/test_parsing.py -q` failed during collection with the expected `ModuleNotFoundError: No module named 'qemer_ingest.parsing'`.

GREEN: after adding `src/qemer_ingest/parsing.py` and correcting RST document-title handling exposed by the test, the focused command reported `4 passed in 0.18s`.

## Verification

- Focused parser tests: `4 passed in 0.18s`.
- Full regression suite: `24 passed in 0.13s`.
- Scoped lint: `uv run ruff check src/qemer_ingest/parsing.py tests/test_parsing.py` reported `All checks passed!`.
- Scoped formatting: `uv run ruff format --check src/qemer_ingest/parsing.py tests/test_parsing.py` reported `2 files already formatted`.
- `git diff --check` completed without output.

## Changed files

- `src/qemer_ingest/parsing.py`
- `tests/fixtures/guide.md`
- `tests/fixtures/api.rst`
- `tests/test_parsing.py`
- `.superpowers/sdd/2026-08-28-github-repository-ingestion/task-4-report.md`

## Self-review

Reviewed the output contract section by section: Markdown heading text is never added to prose; adjacent Python fences become exactly one code text joined by two newlines; RST literals do not appear in the prose result; blank TXT and empty Markdown/RST sections produce no invalid `DocumentUnit`; case-folded recognized suffixes work; unsupported extensions fail explicitly; and URLs use the resolved SHA rather than the requested ref. Fixture tests use actual local files and real parser dependencies, with only `monkeypatch.chdir` used to model a repository-relative input path.

## Concerns

The repository-wide `uv run ruff format --check src tests` currently reports pre-existing formatting changes in Task 2 and Task 3 files (`github.py`, `discovery.py`, `test_github.py`, and `test_discovery.py`). The new parser and test files pass scoped formatting, and this task does not modify those unrelated files.

## Review fix round 1

### Root cause and implementation

The original RST collector only classified immediate section children. Docutils represents a code-block directive inside a list as `bullet_list` → `list_item` → `literal_block`, so calling `astext()` on the list child incorrectly copied the literal into prose and never emitted code. `_rst_content` now walks descendants in document order, treats `literal_block` and `doctest_block` as terminal code, omits their descendant text from prose, and continues to exclude nested sections from their parent section.

### Covering test and TDD evidence

Added local fixture `tests/fixtures/nested.rst` and `test_parse_rst_separates_a_nested_literal_block_from_list_prose` in `tests/test_parsing.py`. The fixture contains a Python code-block directive nested beneath a list item, requiring prose `A list item owns a code sample.` and a separate code unit `print("nested")`.

RED: `QEMER_UV_CACHE=/tmp/qemer-ingest-uv-cache UV_CACHE_DIR="$QEMER_UV_CACHE" uv run pytest tests/test_parsing.py::test_parse_rst_separates_a_nested_literal_block_from_list_prose -q` reported `1 failed in 0.11s`. The failure showed the only prose unit was `A list item owns a code sample.\n\nprint("nested")`, while the expected separate code unit was absent.

GREEN: the same targeted command reported `1 passed in 0.10s` after the descendant traversal change.

### Final verification

- `QEMER_UV_CACHE=/tmp/qemer-ingest-uv-cache UV_CACHE_DIR="$QEMER_UV_CACHE" uv run pytest tests/test_parsing.py -q` reported `5 passed in 0.10s`.
- `QEMER_UV_CACHE=/tmp/qemer-ingest-uv-cache UV_CACHE_DIR="$QEMER_UV_CACHE" uv run pytest -q` reported `25 passed in 0.13s`.
- `QEMER_UV_CACHE=/tmp/qemer-ingest-uv-cache UV_CACHE_DIR="$QEMER_UV_CACHE" uv run ruff check src/qemer_ingest/parsing.py tests/test_parsing.py` reported `All checks passed!`.
- `QEMER_UV_CACHE=/tmp/qemer-ingest-uv-cache UV_CACHE_DIR="$QEMER_UV_CACHE" uv run ruff format --check src/qemer_ingest/parsing.py tests/test_parsing.py` reported `2 files already formatted`.
- `git diff --check` completed without output.

### Self-review

Verified that direct and nested literals both flow through the same terminal branch, so nested code cannot be duplicated into prose; text inside ordinary list/quote/container descendants still contributes to prose; code blocks retain traversal order before joining; and the fix leaves deferred empty-section ordinal behavior unchanged.
