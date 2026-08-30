# Bounded token chunking design

## Status

Proposed. This design supersedes the longest-fitting-slice requirement in `2026-08-30-token-aware-chunking-design.md` while keeping its public CLI, artifact, prefix, overlap, endpoint-preflight, and exact final-payload budget contracts.

## Problem

The current chunker finds the mathematically longest fitting raw-token slice by starting with the entire remaining document and shortening it one token at a time. A measured NumPy unit contained 20,680 raw tokens; its first 2,048-token-budget chunk therefore required roughly 18,640 candidate checks, and later chunks restarted from the document end. Each candidate performs `/detokenize` followed by `/tokenize`, and each token request currently creates a fresh HTTP client. The result is pathological request amplification that prevents large repositories from completing in practical time.

The exhaustive search exists because final token counts after detokenization and prefixing are not guaranteed to be monotonic. This design resolves the conflict in favor of bounded work: every emitted embedding payload must fit, but a chunk need not be the mathematically longest fitting slice.

## Goals

- Preserve the authoritative rule that `/tokenize` with `add_special: true` over `document_prefix + chunk.text` decides whether an embedding payload fits.
- Produce deterministic, near-maximum, source-derived chunks without a local tokenizer or custom llama-server endpoint.
- Bound candidate verification to a small constant per emitted chunk, independent of the remaining document length.
- Reuse one HTTP connection pool across token preflight and chunking.
- Preserve unsplit unit identity, split-child identity and metadata, exact source-token overlap, request-only prefixing, source-oriented artifact text, and failure-before-publication behavior.

## Non-goals

- Do not add or require `POST /chunk`.
- Do not fork or patch llama.cpp.
- Do not guarantee the mathematically longest fitting slice.
- Do not change Markdown, RST, or TXT parsing; Sphinx-aware RST handling is a separate design problem.
- Do not batch embeddings or change artifact publication.

## Backend contract

The backend contract remains stock llama-server-compatible `POST /tokenize`, `POST /detokenize`, and `POST /v1/embeddings`. The request and response schemas remain unchanged. `qemer-ingest` continues to use source token IDs from `/tokenize` with `add_special: false`, source-derived text from `/detokenize`, and final payload counts from `/tokenize` with `add_special: true`.

## Chunk selection algorithm

`chunk_units` validates the configured token size and overlap, tokenizes the prefix once, and rejects a prefix that consumes the complete budget. Each unit is then processed independently.

For a unit whose complete prefixed payload fits, return the original `DocumentUnit` unchanged. For an oversized unit, tokenize its raw source once with `add_special: false` and maintain a raw-token `start` index.

The first candidate length is `min(remaining_source_tokens, max(1, chunk_size_tokens - prefix_token_count))`. This prevents the first probe from containing the complete remaining document when only one budget-sized window can fit.

For each candidate, detokenize the raw-token slice and tokenize `document_prefix + candidate_text` with `add_special: true`. If the result fits, accept it immediately. The successful authoritative tokenization is the final verification; the child is not tokenized again.

If the candidate is oversized, compute a strictly smaller candidate using both the measured overflow and geometric backoff. The proportional proposal is `floor(current_length * chunk_size_tokens / observed_payload_tokens)`. The geometric proposal removes at least one eighth of the current raw-token length. Select the smaller positive proposal, always advancing toward one source token.

Candidate refinement is capped at eight attempts per emitted chunk. If all eight attempts are oversized, test the single-source-token candidate once unless it was already tested. If that candidate fits, emit it; otherwise fail with `ValueError("token budget leaves no room for source tokens")`. This deliberately refuses an exotic non-monotonic case where an untested longer candidate fits but the bounded candidates do not.

After accepting a child, advance to `end - chunk_overlap_tokens`. Preserve the existing non-progress guard and do not emit an overlap-only trailing child.

With the default 2,048-token budget, a normal tokenizer should accept the first candidate or require one proportional correction. The hard ceiling is nine candidate checks per emitted child rather than work proportional to the complete remaining document.

## Token client lifecycle

`TokenizationClient` becomes an async context manager that owns one `httpx.AsyncClient`. All preflight, prefix counting, unit counting, detokenization, and final verification requests in one build share that client and its connection pool.

The synchronous Typer command performs scalar and output-path validation first, then calls one async build coroutine with `asyncio.run`. The async coroutine enters the token client context, preflights both token endpoints, validates prefix capacity, and only then constructs or invokes GitHub operations. Parsing remains synchronous inside this orchestration because parser behavior is out of scope. Chunking and embedding are awaited in the same event loop.

The token client must close its HTTP client on success and failure. The optional mock transport remains injectable for contract tests.

## Error handling

Endpoint failures retain the endpoint path. Errors raised while processing a unit retain its source URL. Prefix-only overflow still fails before GitHub construction. Exhausting bounded refinement produces the existing no-room error with the unit source URL. Any failure before `build_artifact` leaves the requested output path unpublished.

## Performance contract

Let `C` be the number of emitted chunks and `R` the number of refinement attempts, capped at nine including the single-token fallback. Candidate work is at most `C × R` detokenizations and `C × R` authoritative tokenizations, plus constant prefix work and one source tokenization per oversized unit. It no longer depends quadratically on the number of source tokens.

Connection setup is once per build rather than once per token request. The design does not promise a particular wall-clock duration because backend and repository sizes vary, but tests enforce the request-count bound.

## Testing

- Replace the longest-nonmonotonic-slice test with a deterministic-valid-slice test that permits a shorter fitting chunk.
- Add a 20,680-token synthetic unit test whose counting codec proves the first candidate is budget-sized rather than document-sized.
- Add an oversized-boundary codec that forces refinement and assert no child uses more than nine candidate checks.
- Assert every emitted prefixed payload is within budget, overlap and child identities remain stable, and artifact text excludes the prefix.
- Assert an accepted candidate is not redundantly tokenized after authoritative verification.
- Assert `TokenizationClient` reuses one injected async client/transport lifecycle and closes it on both success and endpoint failure.
- Keep CLI ordering tests proving scalar and output-path validation precede clients, token preflight and prefix validation precede GitHub, and failures do not publish.
- Run the complete unit suite, Ruff checks, formatting checks, package build, and a live small-repository build against llama-server.

## Documentation

README token-aware chunking documentation will state that chunks are deterministic and near-maximum rather than mathematically longest. The public options, defaults, backend endpoints, report fields, and source-oriented artifact semantics do not change.
