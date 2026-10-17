"""Build the corpus from Wolne Lektury (Hugging Face: ArielUW/wolnelektury).

The default corpus is the curated mythological-literary collection
(MYTHOLOGY_BOOKS below): Greek and Roman antiquity, mythographic
studies, Baltic/Lithuanian legends, single Eastern accents, and
ancient Egypt - all Polish publications/translations from
wolnelektury.pl. The full dataset (~2400 plain-text books, one .txt
per book) is still reachable with --lektury_all. Books are downloaded
directly over HTTPS (no `datasets` library needed) and pushed through
the shared clean -> chunk -> one-sample-per-line pipeline
(corpus/text.py), so everything downstream (tokenizer, shards,
training) works unchanged.
"""

import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

from corpus.text import chunk_text, clean_text

HF_API = "https://huggingface.co/api/datasets/ArielUW/wolnelektury"
HF_RAW = "https://huggingface.co/datasets/ArielUW/wolnelektury/resolve/main"
USER_AGENT = "polish-llm-corpus-builder/1.0 (educational LLM project)"
DOWNLOAD_DELAY = 0.25  # seconds between book downloads (politeness)
PROGRESS_EVERY = 50  # books between progress lines in "all books" mode
# Measured on the mythology corpus with the BPE 4k tokenizer (Polish
# diacritics cost multiple ByteLevel bytes).
CHARS_PER_TOKEN = 3.1

# Curated mythological-literary corpus. Every filename was verified
# against the dataset (author/title header) and sizes checked; a few
# titles exist there only as fragments (marked "fragment"). Ordering
# follows the project brief: antiquity first, then studies, regional
# legends, single accents, and Egypt.
MYTHOLOGY_BOOKS = (
    # Greek mythology (antiquity): Homer, Hezjod, the tragedians, Aristophanes.
    "homer-iliada.txt",
    "homer-odyseja.txt",
    "hezjod-prace-i-dnie.txt",
    "ajschylos-oresteja.txt",
    "ajschylos-prometeusz-skowany.txt",
    "antygona.txt",  # Sofokles
    "krol-edyp.txt",  # Sofokles
    "elektra.txt",  # Sofokles
    "bachantki.txt",  # Eurypides
    "medea.txt",  # Eurypides
    "eurypides-ifigenia-w-aulidzie.txt",
    "eurypides-hippolytos-uwienczony.txt",
    "arystofanes-chmury.txt",  # fragment
    "arystofanes-zaby.txt",  # fragment
    "arystofanes-gromiwoja.txt",  # fragment
    # Roman literature: Virgil, Ovid, Horace (odes/satires - fragments).
    "wergiliusz-eneida.txt",
    "sztuka-kochania.txt",  # Owidiusz
    "horacy-piesn-iii-1-tlum-siemienski.txt",  # fragment (Odi profanum vulgus)
    "horacy-satyry-i-9.txt",  # fragment
    # Mythographic studies: Zipper's school collection, Brückner on Slavs.
    "zipper-opowiadania-z-mitologii-grekow-i-rzymian.txt",
    "mitologia-slowianska.txt",  # Aleksander Brückner
    # Baltic / Lithuania: Stryjkowski's chronicle, Mickiewicz's
    # Lithuanian motifs, Lithuanian legends (Baranauskas, Asnyk),
    # Słowacki's Venedian/Slavic tales.
    "stryjkowski-kronika-polska-litewska-zmudzka-i-wszystkiej-rusi.txt",
    "ballady-i-romanse-switez.txt",  # Mickiewicz
    "ballady-i-romanse-switezianka.txt",  # Mickiewicz
    "grazyna.txt",  # Mickiewicz
    "konrad-wallenrod.txt",  # Mickiewicz
    "antanas-baranauskas-anyksci-silelis.txt",
    "asnyk-kiejstut.txt",
    "asnykas-kestutis.txt",  # Lithuanian edition (tłum. Kudirka)
    "slowacki-lilla-weneda.txt",
    "slowacki-mindowe.txt",
    # Single Eastern accents: Buddha's death, the Egyptian Mary, Isis mysticism.
    "dziejba-lesna-smierc-buddy.txt",  # Leśmian, fragment
    "hymny-hymn-marii-egipcjanki.txt",  # Kasprowicz
    "zmichowska-poganka.txt",
    # Egypt: Prus - the priests of Amon vs the pharaoh, plus his Egypt tales.
    "faraon-tom-pierwszy.txt",
    "faraon-tom-drugi.txt",
    "faraon-tom-trzeci.txt",
    "z-legend-dawnego-egiptu.txt",
)


def _list_books():
    """Return the .txt book filenames in the dataset repo."""
    req = urllib.request.Request(HF_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [s["rfilename"] for s in data["siblings"] if s["rfilename"].endswith(".txt")]


def list_available_books():
    """Public book listing for the CLI (--lektury_list)."""
    return _list_books()


def _match_books(available, patterns):
    """Expand comma-separated substring patterns into book filenames.

    Each pattern matches every book whose filename contains it
    (case-insensitive), e.g. 'faraon' -> all three Faraon volumes.
    Raises if a pattern matches nothing, so typos don't silently
    produce an empty corpus.
    """
    ordered = []
    seen = set()
    for pattern in patterns:
        matches = [b for b in available if pattern.lower() in b.lower()]
        if not matches:
            raise ValueError(
                f"No Wolne Lektury book matches {pattern!r} "
                f"(use --lektury_list to see available names)"
            )
        for book in matches:
            if book not in seen:
                seen.add(book)
                ordered.append(book)
    return ordered


def _download_book(book):
    """Download one book's text (UTF-8)."""
    url = f"{HF_RAW}/{urllib.parse.quote(book)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def build_corpus_from_lektury(
    output_path,
    target_samples=None,
    books=MYTHOLOGY_BOOKS,
    overwrite=False,
    seed=42,
):
    """Download books, clean and chunk them into the line-per-sample corpus.

    ``books`` is a list of substring patterns (see _match_books); the
    default is the curated MYTHOLOGY_BOOKS collection, and None means
    all books in dataset order. ``target_samples`` is an optional cap;
    None keeps every sample from the selected books.

    Returns the number of samples written to ``output_path``.
    """
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Corpus already exists: {output}")

    available = _list_books()
    ordered = _match_books(available, books) if books else available
    print(f"Wolne Lektury: {len(available)} books available, using {len(ordered)}")

    samples = []
    books_used = 0
    # Per-book progress for small/filtered runs; every 50 books in "all
    # books" mode (~2400 books = 10 minutes of download delay alone).
    progress_every = 1 if len(ordered) <= PROGRESS_EVERY else PROGRESS_EVERY

    for i, book in enumerate(ordered):
        if i:
            time.sleep(DOWNLOAD_DELAY)  # politeness pause BETWEEN requests
        try:
            text = _download_book(book)
        except Exception as exc:
            print(f"  skipping {book}: download failed ({exc})")
            continue
        books_used += 1

        cleaned = clean_text(text)
        samples.extend(chunk_text(cleaned))
        reached_cap = target_samples is not None and len(samples) >= target_samples
        if books_used % progress_every == 0 or reached_cap:
            print(f"  {books_used} books: {len(samples)} samples so far")
        if reached_cap:
            break

    random.Random(seed).shuffle(samples)
    if target_samples is not None:
        samples = samples[:target_samples]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            assert (
                "\n" not in sample
            ), "samples must be single-line (one document per line)"
            f.write(sample + "\n")

    total_chars = sum(len(s) for s in samples)
    print(f"Books used: {books_used}")
    print(
        f"Samples: {len(samples)} ({total_chars:,} chars, "
        f"~{int(total_chars / CHARS_PER_TOKEN):,} tokens estimated)"
    )
    return len(samples)
