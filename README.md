# qemer-ingest

`qemer-ingest` builds local Qemer documentation corpora from pinned revisions of public GitHub repositories. It resolves a requested ref to a full commit SHA, downloads that immutable archive, conservatively selects documentation, parses prose and code rows, obtains embeddings from an endpoint you manage, and writes a local artifact directory atomically.

## Install

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then install the CLI from this checkout:

```sh
uv tool install .
qemer-ingest --help
```

For development, create the locked project environment and run the CLI through uv:

```sh
uv sync
uv run qemer-ingest --help
```

## Input and embedding requirements

Repository input is limited to public GitHub HTTPS URLs such as `https://github.com/numpy/numpy`. Private repositories, credentials, tokens, and authenticated downloads are not supported.

The build command requires an already-running embeddings endpoint that you operate. The endpoint must accept `POST /tokenize`, `POST /detokenize`, and `POST /v1/embeddings`; the embedding request carries an `input` string and `model` name, and returns the vector at `data[0].embedding`. Every returned vector must have exactly the positive width passed with `--embedding-dim`; `qemer-ingest` does not download or launch an embedding model.

The output path must not already exist. Builds stay local: the CLI does not invoke Qemer, publish artifacts, or push repository changes.

## Inspect before building

Discovery intentionally selects only a root `README` plus `.md`, `.rst`, and `.txt` files under top-level `docs`, `doc`, or `documentation` directories. Hidden paths, `.git`, `node_modules`, and `vendor` are excluded. Selected files must be valid UTF-8.

Use `inspect` to see the resolved commit and default selection without contacting an embeddings endpoint:

```sh
qemer-ingest inspect https://github.com/numpy/numpy --ref v2.3.0
```

Repeat `--include <glob>` to opt additional files into discovery. Include globs do not override excluded directories.

```sh
qemer-ingest inspect https://github.com/example/project --ref v1.0.0 --include 'examples/**/*.md' --include 'CHANGELOG.md'
```

## Build and local output

Provide the resolved source, corpus identity, embedding contract, and a new output directory:

```sh
qemer-ingest build https://github.com/numpy/numpy --ref v2.3.0 --library numpy --version 2.3.0 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --output ./qemer-corpora/numpy-2.3.0
```

### Token-aware chunking

`build` requires `/tokenize` and `/detokenize` at the embedding URL, preserves Markdown/RST/TXT structure first, then splits prose or code units against the final embedding payload token budget. The default is `2048` tokens with zero token overlap. Set `--chunk-overlap-tokens` to repeat trailing source tokens at the start of the next chunk; it must be smaller than `--chunk-size-tokens`. Use `--document-prefix` to prepend exact caller-selected text only to embedding requests; its default is the empty string. Corpus text remains source-oriented and does not include the prefix.

```sh
qemer-ingest build https://github.com/numpy/numpy --ref v2.5.2 --library numpy --version 2.5.2 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --chunk-size-tokens 2048 --chunk-overlap-tokens 0 --document-prefix 'search_document: ' --output ./qemer-corpora/numpy-2.5.2
```

`build-report.json` records `chunk_size_tokens`, `chunk_overlap_tokens`, and `document_prefix`, while corpus text remains source-oriented.

On success, the output directory contains:

```text
qemer-corpora/numpy-2.3.0/
├── manifest.json
├── numpy-2.3.0.tar.zst
└── build-report.json
```

`manifest.json` is the local Qemer installation contract and records the archive filename, checksum, byte size, embedding model and dimension, and snippet count. The compressed tar archive contains `corpus.parquet`. `build-report.json` is a build-time record of the requested ref, resolved commit, selected and skipped files, and emitted prose and code row counts; it is not part of the installation contract.

## Qemer handoff

Inspect, build locally, then explicitly hand the manifest to Qemer:

```sh
qemer-ingest inspect https://github.com/numpy/numpy --ref v2.3.0
qemer-ingest build https://github.com/numpy/numpy --ref v2.3.0 --library numpy --version 2.3.0 --embedding-url http://127.0.0.1:8080 --embedding-model nomic-embed-text-v1.5 --embedding-dim 768 --output ./qemer-corpora/numpy-2.3.0
qemer install numpy@2.3.0 --manifest ./qemer-corpora/numpy-2.3.0/manifest.json
```
