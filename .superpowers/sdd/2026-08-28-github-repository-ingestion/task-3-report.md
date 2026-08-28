# Task 3 report: discover documentation and implement inspect

## Implementation

Implemented conservative repository documentation discovery in `qemer_ingest.discovery`. It walks repository files, operates on repository-relative paths, excludes hidden segments and `.git`, `node_modules`, and `vendor`, and selects only root README names or Markdown, RST, and text files below `docs`, `doc`, or `documentation`. Repeated include globs add eligible non-excluded paths via `PurePosixPath.match`. Every selected candidate is read as UTF-8, invalid text is recorded as skipped, and final selections are sorted by POSIX path.

Added `qemer-ingest inspect <repository-url> --ref <ref> [--include <glob> ...]`. The command resolves the requested revision, downloads that immutable archive into `TemporaryDirectory`, discovers selected files, and prints the resolved SHA followed by one relative path per selected file. It has no output-directory option or embedding behavior.

## TDD evidence

RED: after adding `tests/test_discovery.py`, `uv run pytest tests/test_discovery.py -q` failed during collection with `ModuleNotFoundError: No module named 'qemer_ingest.discovery'`.

GREEN: after adding `src/qemer_ingest/discovery.py`, `uv run pytest tests/test_discovery.py -q` reported `1 passed in 0.03s`.

RED: after adding `tests/test_cli_inspect.py`, `uv run pytest tests/test_cli_inspect.py -q` failed because `qemer_ingest.cli` did not expose `GitHubClient`, which the command requires for injected local-fake integration behavior.

GREEN: after adding the inspect command, `uv run pytest tests/test_discovery.py tests/test_cli_inspect.py -q` reported `2 passed in 0.15s`.

## Final verification

`uv run pytest tests/test_discovery.py tests/test_cli_inspect.py -q` reported `2 passed in 0.07s`.

`uv run ruff check src tests` reported `All checks passed!`.

`git diff --check` completed with no output.

## Changed files

- `src/qemer_ingest/discovery.py`
- `src/qemer_ingest/cli.py`
- `tests/test_discovery.py`
- `tests/test_cli_inspect.py`
- `.superpowers/sdd/2026-08-28-github-repository-ingestion/task-3-report.md`

## Self-review

Checked that inclusion cannot bypass excluded hidden or dependency paths, discovery returns deterministic repository-relative selections, and the CLI exercises resolution, extraction, and discovery through a fake backed by actual local files rather than mock-call assertions. The CLI test verifies the SHA and selected paths, absence of embedding output, and no persistent `output` directory.

## Concerns

None identified within this task's scope.

## Review fix round 1

The review identified that `DiscoveryReport.skipped` retained `Path.rglob` traversal order. Added a real-filesystem regression case with undecodable files in `docs/a-first.md`, `docs/z-last/deep.md`, and `documentation/a-intro.md`; this layout exposes the depth-first order differing from sorted repository-relative POSIX order. `discover` now sorts skipped entries by their POSIX-string keys before constructing the report.

Added a real-filesystem exclusion behavior test that explicitly includes `.private/notes.md`, `.git/guide.md`, and `vendor/manual.md`. The test confirms no selection or skipped result is produced, so explicit include globs cannot bypass the hidden, Git, or vendor guards.

RED: `uv run pytest tests/test_discovery.py -q` reported `1 failed, 2 passed in 0.09s`, with `documentation/a-intro.md` occurring before `docs/z-last/deep.md` in skipped iteration order.

GREEN: `uv run pytest tests/test_discovery.py tests/test_cli_inspect.py -q` reported `4 passed in 0.07s`.

Final covering commands: `uv run pytest tests/test_discovery.py tests/test_cli_inspect.py -q`, `uv run ruff check src tests`, and `git diff --check`.

Self-review: verified that the new sort applies only to skipped-map presentation and does not affect selection, UTF-8 validation, or exclusion precedence. The ordering test uses real `Path.rglob` traversal across nested directories, while the include-exclusion test uses actual filesystem paths and no mocks.

Concerns: none identified within this review-fix scope.
