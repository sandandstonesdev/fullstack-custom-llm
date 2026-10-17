"""Shared text cleaning and chunking for corpus building.

clean_text strips heading/boilerplate noise and normalizes whitespace;
chunk_text merges paragraphs into ~500-1000-character samples. Both feed
the line-per-sample corpus contract: one sample per line, one document
per line (the dataset preparer appends one EOT token per line).
"""

import re

# Target sample window in characters (per project spec: ~500-1000).
MIN_SAMPLE_CHARS = 500
MAX_SAMPLE_CHARS = 1000

# Section headings that mark reference-only content: everything from the
# first occurrence onwards is dropped.
NOISE_SECTIONS = {
    "przypisy",
    "uwagi",
    "przypisy i uwagi",
    "bibliografia",
    "zobacz też",
    "zobacz takze",
    "linki zewnętrzne",
    "linki zewnetrzne",
    "źródła",
    "zrodla",
    "literatura uzupełniająca",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def clean_text(text):
    """Strip heading/reference noise and normalize whitespace.

    Section headings and header lines (author/title/ISBN) come out as
    short standalone lines without trailing punctuation, so those are
    dropped. Everything from the first reference-section heading
    onwards is removed entirely.
    """
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        low = stripped.lower().rstrip(":")
        if low in NOISE_SECTIONS:
            break
        # Keep only blank lines and actual sentences/paragraphs; drop
        # short punctuation-less lines (section headings).
        if (
            not stripped
            or len(stripped) > 80
            or stripped[-1] in ".!?…)]”\""
        ):
            cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines)
    # Collapse runs of blank lines and internal whitespace.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _split_long_paragraph(paragraph, max_chars):
    """Split a paragraph longer than ``max_chars`` on sentence boundaries."""
    pieces = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        while len(sentence) > max_chars:
            # Hard split for pathological sentences without boundaries.
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if not buf:
            buf = sentence
        elif len(buf) + len(sentence) + 1 <= max_chars:
            buf = f"{buf} {sentence}"
        else:
            pieces.append(buf)
            buf = sentence
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text, min_chars=MIN_SAMPLE_CHARS, max_chars=MAX_SAMPLE_CHARS):
    """Merge paragraphs into samples of roughly ``min_chars``-``max_chars``."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    pieces = []
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            pieces.extend(_split_long_paragraph(paragraph, max_chars))
        else:
            pieces.append(paragraph)

    samples = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > max_chars:
            samples.append(buf)
            buf = piece
        else:
            buf = f"{buf} {piece}".strip()
        if len(buf) >= min_chars:
            samples.append(buf)
            buf = ""
    if buf:
        samples.append(buf)
    return samples
