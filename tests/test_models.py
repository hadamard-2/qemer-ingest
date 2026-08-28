from qemer_ingest.models import DocumentUnit, RepositoryRef


def test_repository_ref_keeps_the_resolved_commit() -> None:
    source = RepositoryRef(
        "https://github.com/numpy/numpy", "numpy", "numpy", "v2.3.0", "a" * 40
    )
    assert source.commit_sha == "a" * 40


def test_document_unit_rejects_empty_text() -> None:
    try:
        DocumentUnit(
            "numpy-2.3-readme-0001", "prose", "README", "https://example.test", ""
        )
    except ValueError as error:
        assert "text" in str(error)
    else:
        raise AssertionError("empty text must not become an embedding request")
