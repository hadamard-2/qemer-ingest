# LangChain overflow chunking design

## Goal

Add bounded, reproducible overflow chunking to `qemer-ingest` without replacing its format-aware Markdown, RST, and plain-text parsing.

## Decision

The project will depend on the standalone `langchain-text-splitters` package and use `RecursiveCharacterTextSplitter` as the shared overflow splitter. It will not add the full LangChain framework, and it will not use `MarkdownHeaderTextSplitter` as the primary Markdown parser.

Markdown, RST, and TXT parsing remains responsible for semantic structure: section titles, prose-versus-code classification, source URLs, parser-skip reporting, and parent unit identity. The overflow splitter operates only after that stage, on an emitted prose or code unit whose text exceeds the configured character limit.

## CLI contract

The `build` command gains these options:

- `--chunk-size <positive-int>`: maximum characters per emitted embedding unit. Default: `8000`.
- `--chunk-overlap <non-negative-int>`: characters repeated between adjacent chunks. Default: `0`.

`chunk_overlap` must be strictly smaller than `chunk_size`. Validation occurs before GitHub resolution or embedding-client construction. `inspect` remains a discovery-only command and receives neither option.

## Splitting behavior

For every structurally emitted `DocumentUnit`:

- If `len(unit.text) <= chunk_size`, emit the unit unchanged, including its current `snippet_id`.
- If the text exceeds `chunk_size`, create a `RecursiveCharacterTextSplitter` with explicit deterministic separators: paragraph boundaries, line boundaries, spaces, then characters. Apply the configured size and overlap, then emit one child unit per returned text chunk in order.
- Every child retains the parent unit's `kind`, `title`, and pinned `source_url`.
- Child IDs are deterministic and collision-free across a parent prose/code pair: `<parent-snippet-id>-<kind>-<three-digit-child-ordinal>`. Child ordinals start at one.

The same configured overlap applies to prose and code. The default overlap is zero, avoiding duplicate content unless the caller explicitly opts in.

## Reporting and reproducibility

`build-report.json` gains `chunk_size` and `chunk_overlap`. The report therefore records the exact splitting policy used to create an artifact. Existing report fields continue to describe parsed files, skipped files/sections, and prose/code output counts; the counts reflect post-split emitted rows.

Chunking must be deterministic for identical repository content, library, version, and CLI values. It must not change a unit below the limit and must preserve source provenance for every child.

## Error handling

Invalid chunk option values produce nonzero Typer validation errors before any GitHub or embedding request. The splitter's output must be nonempty; an unexpected empty child is ignored rather than becoming an embedding request.

## Testing

Tests cover default values, invalid option combinations, an unchanged below-limit unit, deterministic child IDs, configured overlap, source/title/kind preservation, and report serialization. CLI tests prove preflight option validation runs before external clients. Existing parser tests remain the authority for structural Markdown, RST, and TXT behavior.

## Non-goals

This change does not add model-specific token counting, a token-limit CLI option, semantic/LLM chunking, a new embedding runtime, or a replacement RST parser. A future tokenizer-backed mode may be added as a separate, explicitly model-aware feature.
