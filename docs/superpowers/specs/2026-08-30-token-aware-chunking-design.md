# Token-aware chunking design

## Goal

Make every `build` embedding request fit a configured token budget by using a required tokenizer service supplied alongside the embedding service. The default budget is exactly 2048 tokens. This replaces character-based chunking and supports an optional model-specific document prefix without storing that prefix as source text.

## Service contract

`--embedding-url` is the common base URL for all required service endpoints. Every supported embedding backend must provide these endpoints in addition to `POST /v1/embeddings`:

```text
POST /tokenize
{"content": "<text>", "add_special": true}
→ {"tokens": [<integer>, ...]}

POST /detokenize
{"tokens": [<integer>, ...]}
→ {"content": "<text>"}
```

`/tokenize` with `add_special: true` is authoritative for the final embedding-request budget. `/tokenize` with `add_special: false` supplies source-token IDs for splitting. `/detokenize` supplies the source-derived text for each token slice. Responses with a missing or non-integer `tokens` array, or a missing string `content`, are invalid.

`build` preflights both token endpoints with an empty tokenization request and an empty detokenization request before resolving GitHub content. A missing, non-2xx, malformed, or unreachable token endpoint aborts the build before GitHub download or embedding requests. The token service is part of the caller-owned embedding backend contract; `qemer-ingest` does not install, configure, or infer a tokenizer.

## CLI contract

Remove `--chunk-size` and `--chunk-overlap`. Add these options to `build`:

- `--chunk-size-tokens <positive-int>`: maximum token count of each final embedding payload. Default: `2048`.
- `--chunk-overlap-tokens <non-negative-int>`: source-token suffix repeated in the next source-token slice. Default: `0`; it must be smaller than the selected token chunk size.
- `--document-prefix <string>`: exact string prepended to every embedding request. Default: an empty string.

All CLI validation that does not need the token service runs before any external client is constructed. The token-endpoint preflight then runs before GitHub resolution. A nonempty prefix whose tokenized final payload already consumes the full chunk budget produces a nonzero validation error before repository download.

## Token splitting behavior

Markdown, RST, and TXT parsers remain the authority for semantic units, classification, titles, source URLs, parser skips, and parent IDs. Token-aware splitting operates only after parsing a structurally emitted `DocumentUnit`.

For each unit, the final request candidate is `document_prefix + unit.text`. If `/tokenize` with `add_special: true` reports no more than `chunk_size_tokens`, retain the unit unchanged. Its artifact text remains the original source text and its ID remains unchanged.

For an oversized unit, tokenize its source text with `add_special: false`, split that token sequence in order, and detokenize each source-token slice. Choose each slice so tokenizing `document_prefix + detokenized_source_slice` with `add_special: true` is no more than `chunk_size_tokens`. A chunk that becomes too large after prefixing is shortened and rechecked until it fits. If the prefix alone cannot fit, fail with a clear error rather than emit an invalid request.

With nonzero token overlap, each later source-token slice begins with the preceding slice's final `chunk_overlap_tokens` source-token IDs. Its new payload is shortened as necessary so the complete prefixed request remains inside `chunk_size_tokens`. The default zero overlap emits nonoverlapping source-token slices.

Each split child retains its parent kind, title, and pinned source URL. Child IDs remain `<parent-snippet-id>-<kind>-<three-digit-child-ordinal>`, beginning at `001`. Child `DocumentUnit.text` is the detokenized source-derived slice without the document prefix. Embedding requests use the prefix, but `corpus.parquet` remains source-oriented rather than storing a model-specific transform.

## Embedding and reporting

The embedding client accepts `document_prefix` and sends exactly `document_prefix + unit.text` to `/v1/embeddings`. It does not add a prefix when the option is empty. The client embeds one final unit at a time, as it does today.

`build-report.json` replaces `chunk_size` and `chunk_overlap` with `chunk_size_tokens` and `chunk_overlap_tokens`, recorded as JSON integers. It also records `document_prefix` exactly. Prose and code row counts continue to describe post-split units.

## Error handling

Endpoint failures identify the endpoint and source URL when a specific unit is being processed. Failed tokenization, detokenization, or embedding must not publish an artifact: artifact generation continues to use temporary staging followed by an atomic rename. The output path must not already exist.

## Testing

Tests cover token-service preflight before GitHub construction, malformed token-service responses, the exact 2048-token boundary, a 2049-token split, prefix-inclusive budgeting, prefix-only overflow rejection, zero and nonzero token overlap, child metadata and IDs, prefix-free artifact rows, request payload prefixing, report serialization, removed character-option help/validation, and no artifact publication on a token-service failure.

## Non-goals

This change does not add `transformers`, download tokenizer assets, execute remote tokenizer code, infer model limits, batch embedding inputs, replace structural parsers, or provide a fallback character-based chunking mode. A backend without both required token endpoints is unsupported by `build`.
