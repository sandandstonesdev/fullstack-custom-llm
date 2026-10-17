"""Corpus cleaning and chunking: pure functions, no network."""

from corpus.build_wolnelektury import MYTHOLOGY_BOOKS, _match_books
from corpus.text import chunk_text, clean_text


def test_clean_text_strips_noise():
    raw = (
        "Pierwsi Piastowie\n\n"
        "Mieszko I byl pierwszym historycznym wladca Polski.\n\n"
        "Przypisy\n\n"
        "1. Kowalski, Historia, Warszawa."
    )
    cleaned = clean_text(raw)
    assert "Mieszko I" in cleaned  # content kept
    assert "Kowalski" not in cleaned  # reference section dropped
    assert "Pierwsi Piastowie" not in cleaned  # heading dropped


def test_chunk_text_window():
    text = "\n\n".join(f"Zdanie numer {i} o historii Polski." for i in range(100))
    samples = chunk_text(text, min_chars=500, max_chars=1000)
    assert samples
    assert all(len(s) <= 1000 for s in samples)
    # Content is preserved (joined with spaces, not lost).
    assert sum(len(s) for s in samples) >= 0.9 * len(text)


def test_mythology_books_curated_list():
    # All entries are unique .txt filenames from ArielUW/wolnelektury.
    assert len(MYTHOLOGY_BOOKS) == 38
    assert len(set(MYTHOLOGY_BOOKS)) == len(MYTHOLOGY_BOOKS)
    assert all(b.endswith(".txt") for b in MYTHOLOGY_BOOKS)
    # One anchor title from every section of the curated collection.
    anchors = {
        "homer-iliada.txt",  # Greek antiquity
        "wergiliusz-eneida.txt",  # Roman literature
        "zipper-opowiadania-z-mitologii-grekow-i-rzymian.txt",  # studies
        "mitologia-slowianska.txt",  # Brückner
        "stryjkowski-kronika-polska-litewska-zmudzka-i-wszystkiej-rusi.txt",  # Baltic
        "slowacki-lilla-weneda.txt",
        "dziejba-lesna-smierc-buddy.txt",  # Eastern accents
        "zmichowska-poganka.txt",
        "faraon-tom-trzeci.txt",  # Egypt
        "z-legend-dawnego-egiptu.txt",
    }
    assert anchors <= set(MYTHOLOGY_BOOKS)


def test_match_books_expands_and_dedupes():
    available = ["faraon-tom-drugi.txt", "faraon-tom-pierwszy.txt", "homer-iliada.txt"]
    # A substring pattern matches every hit, in dataset order, deduped.
    assert _match_books(available, ["faraon", "faraon"]) == [
        "faraon-tom-drugi.txt",
        "faraon-tom-pierwszy.txt",
    ]
    # Exact filenames are just substrings of themselves.
    assert _match_books(available, ["homer-iliada.txt"]) == ["homer-iliada.txt"]
    # A title absent from the dataset (e.g. Ovid's Metamorphoses) must
    # fail loudly, not silently build an empty corpus.
    try:
        _match_books(available, ["metamorfozy"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unmatched pattern")
