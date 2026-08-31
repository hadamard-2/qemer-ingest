# qemer-ingest

`qemer-ingest` turns documentation from a public GitHub repository into a local corpus that Qemer can search. It resolves the requested revision to an immutable commit, selects documentation conservatively, creates embeddings through a service you run, and writes a portable artifact directory.

## What you need

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and choose a public GitHub HTTPS repository, such as `https://github.com/numpy/numpy`.

You also need a running embeddings service. `qemer-ingest` uses it to count and reconstruct tokens while chunking source text, then create the vectors stored in the corpus. The service is not downloaded, started, or configured by `qemer-ingest`.

## Installation

Install the CLI from this checkout:

```sh
uv tool install .
qemer-ingest --help
```

For development, create the locked project environment and run the CLI through uv:

```sh
uv sync
uv run qemer-ingest --help
```

## Usage

### 1. Inspect the source repository

Start with `inspect` to resolve a ref and see the documentation that would be selected. This command does not contact an embeddings service or write a corpus:

```sh
qemer-ingest inspect https://github.com/numpy/numpy --ref v2.3.0
```

Use `--include <glob>` more than once to opt additional files into discovery:

```sh
qemer-ingest inspect https://github.com/example/project --ref v1.0.0 --include 'examples/**/*.md' --include 'CHANGELOG.md'
```

### 2. Build a local corpus

Provide the source revision, corpus identity, embedding contract, and a new output directory:

```sh
qemer-ingest build https://github.com/numpy/numpy --ref v2.3.0 --library numpy --version 2.3.0 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --output ./qemer-corpora/numpy-2.3.0
```

The output directory must not already exist. On success, it contains a `manifest.json`, a compressed corpus archive, and `build-report.json`:

```text
qemer-corpora/numpy-2.3.0/
├── manifest.json
├── numpy-2.3.0.tar.zst
└── build-report.json
```

### 3. Install the corpus in Qemer

`qemer-ingest` only creates local artifacts; it does not invoke Qemer, publish a corpus, or push repository changes. When you are ready, explicitly give Qemer the generated manifest:

```sh
qemer install numpy@2.3.0 --manifest ./qemer-corpora/numpy-2.3.0/manifest.json
```

## Reference

### Repository input and document selection

Repository input is limited to public GitHub HTTPS URLs. Private repositories, credentials, tokens, and authenticated downloads are not supported.

Discovery selects a root `README` plus `.md`, `.rst`, and `.txt` files under top-level `docs`, `doc`, or `documentation` directories. Hidden paths, `.git`, `node_modules`, and `vendor` are excluded, and selected files must be valid UTF-8. Include globs add files but do not override excluded directories.

### Embedding service contract

The service at `--embedding-url` must accept `POST /tokenize`, `POST /detokenize`, and `POST /v1/embeddings`. An embedding request carries an `input` string and `model` name, and the response must provide the vector at `data[0].embedding`. Every vector must have exactly the positive width passed with `--embedding-dim`.

### Chunking

`build` preserves Markdown, reStructuredText, and plain-text structure where it can, then splits prose and code against the final embedding-payload token budget. The default budget is `2048` tokens with no overlap. Set `--chunk-overlap-tokens` to repeat trailing source tokens at the beginning of the next chunk; it must be smaller than `--chunk-size-tokens`.

Use `--document-prefix` to prepend exact caller-selected text to embedding requests. Its default is empty, and corpus text remains source-oriented without the prefix. Chunk selection is deterministic and near-maximum: it starts with a budget-sized raw-token window, performs at most eight refinement checks plus a one-token fallback, and treats `/tokenize` as authoritative for each accepted payload. It does not exhaustively search for the mathematically longest fitting slice.

```sh
qemer-ingest build https://github.com/numpy/numpy --ref v2.5.2 --library numpy --version 2.5.2 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --chunk-size-tokens 2048 --chunk-overlap-tokens 0 --document-prefix 'search_document: ' --output ./qemer-corpora/numpy-2.5.2
```

`build-report.json` records the chunk size, overlap, and document prefix, together with the requested ref, resolved commit, selected and skipped files, and emitted prose and code row counts.

### Artifact layout

`manifest.json` is the local installation contract. It records the archive filename, checksum, byte size, embedding model and dimension, and snippet count. The compressed tar archive contains `corpus.parquet`; `build-report.json` is a build-time record rather than part of the installation contract.
