# GitHub Repository Ingestion Design

## Goal

Build `qemer-ingest` as an independent Python CLI that turns documentation from one public GitHub repository revision into a local Qemer corpus artifact.

## Boundary with Qemer

`qemer-ingest` owns source acquisition, GitHub interaction, document discovery, parsing, embedding, and build reporting. Qemer has no dependency on this project and never receives a repository URL.

The only integration surface is a local output directory containing `manifest.json` and one `.tar.zst` archive per corpus. The archive contains `corpus.parquet` with Qemer's established six-column schema. Manifest artifact URLs are relative filenames, so Qemer resolves them beside the local manifest.

## Supported source and revision

The first release accepts only a public repository URL in the form `https://github.com/<owner>/<repository>` and a required `--ref` value. It resolves the ref through GitHub's commit API, records the resulting full commit SHA, and downloads the GitHub archive for that SHA. It does not clone repositories, use credentials, access private repositories, process documentation websites, or publish artifacts.

The required ref makes the build reproducible: a tag, branch, or short SHA is an input convenience, but the report and all per-file source URLs carry the resolved immutable commit SHA.

## Commands

```text
qemer-ingest inspect <repository-url> --ref <ref> [--include <glob> ...]
qemer-ingest build <repository-url> --ref <ref> --library <library> --version <version> --embedding-url <url> --embedding-model <model> --embedding-dim <positive-int> --output <new-directory> [--include <glob> ...]
```

`inspect` downloads the requested revision to a temporary directory, prints the resolved SHA and every selected documentation file, then removes the temporary archive. It does not call an embedding endpoint and writes no output directory.

`build` uses the same source selection, parses the selected files, embeds the resulting units through an already-running OpenAI-compatible `/v1/embeddings` endpoint, and writes a new output directory. The output path must not exist. Work is staged in a sibling temporary directory and renamed only after every artifact and the manifest are complete.

## Documentation discovery

Discovery is conservative by default:

- include root `README`, `README.md`, `README.rst`, and `README.txt`, case-insensitively;
- include `.md`, `.rst`, and `.txt` files below `docs/`, `doc/`, or `documentation/`, case-insensitively;
- skip hidden paths, `.git`, `node_modules`, `vendor`, generated archives, and files that cannot decode as UTF-8;
- allow repeated `--include <glob>` options to add explicitly requested repository-relative files.

The inspect output and build report state selected, skipped, and explicitly included paths. No file outside the default rules or `--include` patterns is embedded.

## Parsing and rows

Markdown is segmented at headings. A section's prose is the text outside fenced code blocks up to the next heading, and all fenced code blocks in that section are joined into one code unit. RestructuredText is segmented at docutils section nodes; prose comes from ordinary text nodes and code comes from literal/doctest blocks. Plain-text files produce one prose section using the file stem as title.

A source file or section with neither prose nor code is skipped and recorded in the report. Each emitted section receives a deterministic `snippet_id` derived from the library, version, repository-relative path, and section ordinal. Every row uses a GitHub blob URL pinned to the resolved commit SHA as `source_url`. A prose row and optional code row share the same snippet ID, title, and source URL.

The ingester sends one text unit per embeddings request, preserves input order in its output rows, and rejects an empty response or a vector whose dimension does not equal `--embedding-dim`. It does not start, download, or configure the embedding model runtime.

## Output and report

For `--library numpy --version 2.3.0 --output ./out`, the completed directory contains:

```text
out/
├── manifest.json
├── numpy-2.3.0.tar.zst
└── build-report.json
```

The tarball contains only `corpus.parquet`. The manifest entry has the established fields `library`, `version`, `url`, `sha256`, `bytes`, `embedding_model`, `embedding_dim`, and `snippet_count`, with `url` equal to `numpy-2.3.0.tar.zst`. `build-report.json` is not part of Qemer's contract; it records the requested repository/ref, resolved SHA, selected/skipped files, and emitted prose/code row counts.

## Non-goals

This release does not crawl documentation websites, ingest non-GitHub repositories, detect licences, host or publish corpora, invoke Qemer, select a corpus version automatically, support private repositories, launch an embedding runtime, or implement incremental/reused embeddings.
