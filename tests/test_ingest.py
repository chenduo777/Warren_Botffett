import pytest

from src.ingest import load_letter_documents


def test_year_mapping_starts_at_1977(tmp_path):
    f = tmp_path / "letters.txt"
    f.write_text("letter A\nletter B\nletter C\n", encoding="utf-8")

    docs = load_letter_documents(f)

    assert len(docs) == 3
    assert docs[0].page_content == "letter A"
    assert docs[0].metadata == {"year": 1977, "line_number": 1}
    assert docs[1].metadata == {"year": 1978, "line_number": 2}
    assert docs[2].metadata == {"year": 1979, "line_number": 3}


def test_empty_lines_skipped_but_line_number_advances(tmp_path):
    f = tmp_path / "letters.txt"
    f.write_text("line1\n\nline3\n", encoding="utf-8")

    docs = load_letter_documents(f)

    assert len(docs) == 2
    assert docs[0].metadata["year"] == 1977
    assert docs[1].metadata["year"] == 1979


def test_whitespace_only_line_skipped(tmp_path):
    f = tmp_path / "letters.txt"
    f.write_text("real content\n   \nanother\n", encoding="utf-8")

    docs = load_letter_documents(f)

    assert len(docs) == 2
    assert docs[0].page_content == "real content"
    assert docs[1].page_content == "another"


def test_content_is_stripped(tmp_path):
    f = tmp_path / "letters.txt"
    f.write_text("  padded line  \n", encoding="utf-8")

    docs = load_letter_documents(f)

    assert docs[0].page_content == "padded line"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="letters file not found"):
        load_letter_documents("/definitely/does/not/exist.txt")


def test_all_empty_lines_raises(tmp_path):
    f = tmp_path / "letters.txt"
    f.write_text("\n\n   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="No non-empty lines"):
        load_letter_documents(f)


def test_accepts_string_path(tmp_path):
    f = tmp_path / "letters.txt"
    f.write_text("hello\n", encoding="utf-8")

    docs = load_letter_documents(str(f))

    assert len(docs) == 1
