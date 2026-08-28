from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class RepositoryRef:
    url: str
    owner: str
    repository: str
    requested_ref: str
    commit_sha: str


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    selected: tuple[Path, ...]
    skipped: dict[str, str]
    explicitly_included: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ParserSkip:
    path: Path
    reason: str
    section: str | None = None


@dataclass(frozen=True, slots=True)
class BuildReport:
    repository_url: str
    requested_ref: str
    resolved_commit: str
    selected_files: tuple[Path, ...]
    skipped_files: dict[str, str]
    explicitly_included_files: tuple[Path, ...]
    parser_skips: tuple[ParserSkip, ...]
    prose_rows: int
    code_rows: int


@dataclass(frozen=True, slots=True)
class DocumentUnit:
    snippet_id: str
    kind: Literal["prose", "code"]
    title: str
    source_url: str
    text: str

    def __post_init__(self) -> None:
        for field_name in ("snippet_id", "title", "source_url", "text"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.kind not in ("prose", "code"):
            raise ValueError("kind must be 'prose' or 'code'")


@dataclass(frozen=True, slots=True)
class EmbeddedUnit:
    unit: DocumentUnit
    vector: list[float]


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    units: tuple[DocumentUnit, ...]
    skipped: tuple[ParserSkip, ...]
