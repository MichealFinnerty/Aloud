"""
Aloud - local audiobook backend
--------------------------------
Serves index.html / library.html / readalong.html.

The reading model: opening a book just parses its text (fast, no TTS
involved at all) into chapters and a flat list of sentences. Nothing
gets narrated until a sentence is actually requested, whichever one
you click. That's what makes true jump-anywhere navigation possible,
there's no queue to wait on and no "still narrating" state anywhere.

Accounts + saved libraries: books, collections, and login are now
backed by a database (SQLite locally, Postgres in production - see
the "ACCOUNTS + SAVED LIBRARIES" section below for how that switch
works), created automatically the first time this runs. That's what
lets a library follow you between devices instead of only living in
one browser's localStorage. See the "ACCOUNTS + SAVED LIBRARIES"
section below, everything above and below it is unchanged from before.

Run this file (F5 in Thonny), then open http://localhost:8000 in your browser.
"""

import os
import re
import uuid
import asyncio
import io
import json
import secrets
import sqlite3
import tempfile
import urllib.request
from datetime import datetime, timezone
from functools import wraps
from html import escape as html_escape
from urllib.parse import urljoin, quote

from flask import Flask, request, send_from_directory, send_file, jsonify, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from pypdf import PdfReader
import edge_tts

try:
    from mutagen.mp3 import MP3
except ImportError:
    MP3 = None

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup, NavigableString
except ImportError:
    ebooklib = None
    epub = None
    BeautifulSoup = None
    NavigableString = None

try:
    import fitz  # PyMuPDF, used only to render a PDF's first page as a cover image
except ImportError:
    fitz = None

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "generated_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Drop background ambience tracks (rain, cafe noise, etc) into this
# folder, they'll show up automatically in the library's ambience
# playlist, no code changes needed to add more.
AMBIENCE_DIR = os.path.join(BASE_DIR, "ambience")
os.makedirs(AMBIENCE_DIR, exist_ok=True)
AMBIENCE_MIME_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

VOICE_MAP = {
    "roger": "en-US-RogerNeural",
    "andrew": "en-US-AndrewNeural",
    "aria": "en-US-AriaNeural",
    "ryan": "en-GB-RyanNeural",
    "guy": "en-US-GuyNeural",
}


# ---------------------------------------------------------------------
# ACCOUNTS + SAVED LIBRARIES (SQLite locally, Postgres in production)
# ---------------------------------------------------------------------
#
# Locally (Thonny, F5), this still just uses a SQLite file, aloud.db,
# sitting next to this script - nothing to install or configure, same
# as before. But a hosting platform's free tier (Render, etc.) wipes
# local files like aloud.db every time the app restarts or goes idle,
# so a permanent deployment needs a database that lives somewhere
# else entirely.
#
# The switch is controlled by one environment variable: DATABASE_URL.
#   - Not set (your own computer)  -> uses the local aloud.db file, exactly
#                                      like before.
#   - Set (on Render)               -> connects to that Postgres database
#                                      instead, so your accounts/collections/
#                                      books survive restarts forever.
#
# Every route below still writes plain SQL with "?" placeholders exactly
# as before - the DBConn wrapper is the only thing that knows which
# real database is underneath, translating "?" to Postgres's "%s" and
# handing back dict-like rows either way. Nothing else had to change.
#
# IMPORTANT if this ever goes on GitHub: add "aloud.db" and
# "secret_key.txt" to your .gitignore. The first is everyone's saved
# books (only relevant when running locally without DATABASE_URL), the
# second is what keeps people logged in between visits, neither
# belongs in a public repo.

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

DB_PATH = os.path.join(BASE_DIR, "aloud.db")
SECRET_KEY_PATH = os.path.join(BASE_DIR, "secret_key.txt")

DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL) and psycopg2 is not None

if bool(DATABASE_URL) and psycopg2 is None:
    print("WARNING: DATABASE_URL is set but psycopg2 isn't installed - "
          "run 'pip install psycopg2-binary' (it's in requirements.txt). "
          "Falling back to local SQLite for now.")


class DBConn:
    """Wraps either a sqlite3 or a psycopg2 connection behind the exact
    same interface the rest of this file already uses:
        db.execute(sql, params).fetchone() / .fetchall()
        db.commit()
        db.close()
    Every route in this file was written against sqlite3's ergonomics
    (Connection.execute(...) returning a cursor directly, "?"
    placeholders, dict-like row access). This class makes a Postgres
    connection behave the same way, so none of that code needed to
    change - only get_db()/close_db()/init_db() below know the
    difference.
    """

    def __init__(self, raw_conn, is_postgres):
        self._conn = raw_conn
        self.is_postgres = is_postgres

    def execute(self, sql, params=()):
        if self.is_postgres:
            cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql.replace("?", "%s"), params)
        else:
            cur = self._conn.cursor()
            cur.execute(sql, params)
        return cur

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def get_db():
    """One DB connection per request, reused if something earlier in
    the same request already opened one, closed automatically at the
    end of the request by close_db() below."""
    if "db" not in g:
        if USE_POSTGRES:
            raw = psycopg2.connect(DATABASE_URL)
            g.db = DBConn(raw, is_postgres=True)
        else:
            raw = sqlite3.connect(DB_PATH)
            raw.row_factory = sqlite3.Row
            raw.execute("PRAGMA foreign_keys = ON")
            g.db = DBConn(raw, is_postgres=False)
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Creates the three tables if this is the first time the database
    has been seen. Safe to call every time the app starts, IF NOT
    EXISTS means an existing database with real data is never touched.
    Runs against whichever database DATABASE_URL points at (Postgres
    in production), or the local aloud.db file otherwise."""
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
    else:
        conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            name TEXT NOT NULL,
            icon TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            title TEXT NOT NULL,
            author TEXT,
            chapters_json TEXT NOT NULL,
            sentences_json TEXT NOT NULL,
            voice TEXT,
            narrator_label TEXT,
            collection_ids_json TEXT NOT NULL DEFAULT '[]',
            cover_url TEXT,
            finished INTEGER NOT NULL DEFAULT 0,
            last_position INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def login_required(view_func):
    """Guards a route so it 401s instead of running if nobody's
    logged in, rather than every route re-checking session by hand."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "You need to be logged in for this"}), 401
        return view_func(*args, **kwargs)
    return wrapped


def book_row_to_summary(row):
    """The lightweight shape used for the library grid: no chapters or
    sentences text included, since a full novel's sentence list can be
    genuinely large and the grid never needs it, only chapter count."""
    chapters = json.loads(row["chapters_json"])
    return {
        "id": row["id"],
        "title": row["title"],
        "author": row["author"],
        "chapterCount": len(chapters),
        "voice": row["voice"],
        "narratorLabel": row["narrator_label"],
        "collectionIds": json.loads(row["collection_ids_json"] or "[]"),
        "coverUrl": row["cover_url"],
        "finished": bool(row["finished"]),
    }


def book_row_to_full(row):
    """The full shape used when actually opening a book in the reader,
    matching the book object shape the frontend already works with."""
    return {
        "id": row["id"],
        "title": row["title"],
        "author": row["author"],
        "chapters": json.loads(row["chapters_json"]),
        "sentences": json.loads(row["sentences_json"]),
        "voice": row["voice"],
        "narratorLabel": row["narrator_label"],
        "collectionIds": json.loads(row["collection_ids_json"] or "[]"),
        "coverUrl": row["cover_url"],
        "finished": bool(row["finished"]),
        "lastPosition": row["last_position"],
    }


def load_or_create_secret_key():
    """Flask needs a secret key to sign login session cookies.

    In production (DATABASE_URL set), this reads SECRET_KEY from the
    environment instead of a file - Render's filesystem is wiped on
    every restart, so a file-based key would silently invalidate every
    logged-in session each time the app spins back up. Locally, it
    still falls back to a key file next to this script (kept in its
    own file rather than a random value picked fresh every run) so you
    stay logged in across local restarts too.
    """
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key

    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH, "r") as f:
            existing = f.read().strip()
            if existing:
                return existing
    key = secrets.token_hex(32)
    with open(SECRET_KEY_PATH, "w") as f:
        f.write(key)
    return key


app.secret_key = load_or_create_secret_key()


# ---------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "That doesn't look like a valid email address"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password needs to be at least 6 characters"}), 400

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        return jsonify({"error": "An account with that email already exists"}), 400

    user_id = uuid.uuid4().hex
    db.execute(
        "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (user_id, email, generate_password_hash(password), now_iso()),
    )
    db.commit()

    session["user_id"] = user_id
    return jsonify({"id": user_id, "email": email})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    db = get_db()
    row = db.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Incorrect email or password"}), 401

    session["user_id"] = row["id"]
    return jsonify({"id": row["id"], "email": row["email"]})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return jsonify({"ok": True})


@app.route("/api/me")
def me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"user": None})
    db = get_db()
    row = db.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        session.pop("user_id", None)
        return jsonify({"user": None})
    return jsonify({"user": {"id": row["id"], "email": row["email"]}})


# ---------------------------------------------------------------------
# Book routes
# ---------------------------------------------------------------------

@app.route("/api/books", methods=["GET"])
@login_required
def list_books():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM books WHERE user_id = ? ORDER BY created_at DESC",
        (session["user_id"],),
    ).fetchall()
    return jsonify({"books": [book_row_to_summary(r) for r in rows]})


@app.route("/api/books", methods=["POST"])
@login_required
def create_book():
    """Saves a book that's already been parsed by /api/open_book,
    /api/import_gutenberg, or /api/convert_to_epub. Those endpoints
    are unchanged, they still just return {title, chapters, sentences,
    cover_url}, this is the separate step that persists that result
    against the logged-in user instead of (or alongside) localStorage."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    chapters = data.get("chapters")
    sentences = data.get("sentences")

    if not title or chapters is None or sentences is None:
        return jsonify({"error": "title, chapters, and sentences are required"}), 400

    book_id = "book_" + uuid.uuid4().hex
    db = get_db()
    db.execute(
        """INSERT INTO books
           (id, user_id, title, author, chapters_json, sentences_json,
            voice, narrator_label, collection_ids_json, cover_url, finished, last_position, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)""",
        (
            book_id,
            session["user_id"],
            title,
            data.get("author"),
            json.dumps(chapters),
            json.dumps(sentences),
            data.get("voice", "andrew"),
            data.get("narratorLabel", "Andrew"),
            json.dumps(data.get("collectionIds", [])),
            data.get("coverUrl"),
            now_iso(),
        ),
    )
    db.commit()
    return jsonify({"id": book_id})


@app.route("/api/books/<book_id>", methods=["GET"])
@login_required
def get_book(book_id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM books WHERE id = ? AND user_id = ?",
        (book_id, session["user_id"]),
    ).fetchone()
    if not row:
        return jsonify({"error": "Book not found"}), 404
    return jsonify(book_row_to_full(row))


@app.route("/api/books/<book_id>", methods=["PATCH"])
@login_required
def update_book(book_id):
    """Partial update, only send the fields that changed: title,
    voice, narratorLabel, collectionIds, finished, lastPosition."""
    data = request.get_json(silent=True) or {}
    db = get_db()
    row = db.execute(
        "SELECT id FROM books WHERE id = ? AND user_id = ?",
        (book_id, session["user_id"]),
    ).fetchone()
    if not row:
        return jsonify({"error": "Book not found"}), 404

    fields = []
    values = []
    if "title" in data:
        fields.append("title = ?")
        values.append(data["title"])
    if "voice" in data:
        fields.append("voice = ?")
        values.append(data["voice"])
    if "narratorLabel" in data:
        fields.append("narrator_label = ?")
        values.append(data["narratorLabel"])
    if "collectionIds" in data:
        fields.append("collection_ids_json = ?")
        values.append(json.dumps(data["collectionIds"]))
    if "finished" in data:
        fields.append("finished = ?")
        values.append(1 if data["finished"] else 0)
    if "lastPosition" in data:
        fields.append("last_position = ?")
        values.append(data["lastPosition"])

    if not fields:
        return jsonify({"ok": True})

    values.extend([book_id, session["user_id"]])
    db.execute(f"UPDATE books SET {', '.join(fields)} WHERE id = ? AND user_id = ?", values)
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/books/<book_id>", methods=["DELETE"])
@login_required
def delete_book(book_id):
    db = get_db()
    db.execute("DELETE FROM books WHERE id = ? AND user_id = ?", (book_id, session["user_id"]))
    db.commit()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------
# Collection routes
# ---------------------------------------------------------------------

@app.route("/api/collections", methods=["GET"])
@login_required
def list_collections():
    db = get_db()
    rows = db.execute(
        "SELECT id, name, icon FROM collections WHERE user_id = ? ORDER BY created_at ASC",
        (session["user_id"],),
    ).fetchall()
    return jsonify({"collections": [dict(r) for r in rows]})


@app.route("/api/collections", methods=["POST"])
@login_required
def create_collection():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Collection name is required"}), 400

    collection_id = "col_" + uuid.uuid4().hex
    db = get_db()
    db.execute(
        "INSERT INTO collections (id, user_id, name, icon, created_at) VALUES (?, ?, ?, ?, ?)",
        (collection_id, session["user_id"], name, data.get("icon", "icon1"), now_iso()),
    )
    db.commit()
    return jsonify({"id": collection_id, "name": name, "icon": data.get("icon", "icon1")})


@app.route("/api/collections/<collection_id>", methods=["PATCH"])
@login_required
def update_collection(collection_id):
    data = request.get_json(silent=True) or {}
    db = get_db()
    row = db.execute(
        "SELECT id FROM collections WHERE id = ? AND user_id = ?",
        (collection_id, session["user_id"]),
    ).fetchone()
    if not row:
        return jsonify({"error": "Collection not found"}), 404

    fields = []
    values = []
    if "name" in data:
        fields.append("name = ?")
        values.append(data["name"])
    if "icon" in data:
        fields.append("icon = ?")
        values.append(data["icon"])

    if fields:
        values.extend([collection_id, session["user_id"]])
        db.execute(f"UPDATE collections SET {', '.join(fields)} WHERE id = ? AND user_id = ?", values)
        db.commit()
    return jsonify({"ok": True})


@app.route("/api/collections/<collection_id>", methods=["DELETE"])
@login_required
def delete_collection(collection_id):
    """Deleting a collection also strips its id out of every book that
    referenced it, same behaviour as the localStorage version, just
    done here instead."""
    db = get_db()
    db.execute(
        "DELETE FROM collections WHERE id = ? AND user_id = ?",
        (collection_id, session["user_id"]),
    )

    rows = db.execute(
        "SELECT id, collection_ids_json FROM books WHERE user_id = ?",
        (session["user_id"],),
    ).fetchall()
    for row in rows:
        ids = json.loads(row["collection_ids_json"] or "[]")
        if collection_id in ids:
            ids = [i for i in ids if i != collection_id]
            db.execute(
                "UPDATE books SET collection_ids_json = ? WHERE id = ?",
                (json.dumps(ids), row["id"]),
            )
    db.commit()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------
# Serve the front end
# ---------------------------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "library.html")


@app.route("/library.html")
def library():
    return send_from_directory(BASE_DIR, "library.html")


@app.route("/readalong.html")
def readalong():
    return send_from_directory(BASE_DIR, "readalong.html")


# ---------------------------------------------------------------------
# Text extraction: PDF and EPUB both become the same {title, text}
# chapter structure, so everything downstream treats them identically.
# ---------------------------------------------------------------------

CHAPTER_HEADING_PATTERN = re.compile(r"chapter\s+([ivxlcdm]+|\d+)\.?", re.IGNORECASE)

BOILERPLATE_KEYWORDS = [
    "isbn", "©", "copyright", "all rights reserved", "trademark",
    "this book may be reproduced", "p.o. box", "phonic skills",
    "related phonograms", "vocabulary", "this book belongs to",
    "written and illustrated by",
]


def strip_boilerplate_lines(text):
    """Drop lines that look like front/back-matter (ISBN, copyright,
    standalone page numbers) rather than actual story or document content.
    Blank lines are kept as paragraph-break markers rather than dropped."""
    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        lower = stripped.lower()
        if any(keyword in lower for keyword in BOILERPLATE_KEYWORDS):
            continue
        kept_lines.append(stripped)
    return "\n".join(kept_lines)


def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?<=[.!?])\s+\d{1,4}(?=\s)", " ", text)
    text = re.sub(r"\b[A-Za-z]{1,3}\d{3,6}\b", " ", text)
    text = re.sub(r"\b\d[\d\-]{5,}\d\b", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n\n[ \t]*", "\n\n", text)
    return text.strip()


def convert_pdf_to_epub_chapters(file_bytes):
    """Turns a PDF into the same {title, text} chapter list a real EPUB
    produces, using the page-by-page cleanup and 'Chapter I.' heading
    detection already built for PDFs."""
    reader = PdfReader(io.BytesIO(file_bytes))
    chapters = []
    current_title = None
    current_paragraphs = []

    def flush():
        if current_title or current_paragraphs:
            chapters.append({
                "title": current_title,
                "text": "\n\n".join(current_paragraphs),
            })

    for page in reader.pages:
        raw = page.extract_text() or ""
        raw = strip_boilerplate_lines(raw)
        page_text = clean_text(raw)

        for para in page_text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if CHAPTER_HEADING_PATTERN.fullmatch(para):
                flush()
                current_title = para.rstrip(".")
                current_paragraphs = []
                continue
            current_paragraphs.append(para)

    flush()
    return [c for c in chapters if c["title"] or c["text"].strip()]


def safe_filename(name):
    """Turns a book title into something safe to use as a filename."""
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name or "").strip()
    name = re.sub(r"\s+", "_", name)
    return name[:60] or "book"


def build_epub_file(title, chapters_data):
    """
    Packages a {title, text} chapter list into a real, standalone EPUB
    file, so converting a PDF produces something genuinely useful on
    its own, not just an internal format only Aloud understands.
    """
    if ebooklib is None:
        raise RuntimeError("ebooklib must be installed to build EPUB files")

    book = epub.EpubBook()
    book.set_identifier(uuid.uuid4().hex)
    book.set_title(title or "Untitled")
    book.set_language("en")

    epub_chapters = []
    for i, ch in enumerate(chapters_data):
        chapter_title = ch["title"] or f"Section {i + 1}"
        html_item = epub.EpubHtml(title=chapter_title, file_name=f"chap_{i + 1}.xhtml", lang="en")
        heading = f"<h1>{chapter_title}</h1>" if ch["title"] else ""
        paragraphs = "".join(f"<p>{p}</p>" for p in ch["text"].split("\n\n") if p.strip())
        html_item.content = f"<html><body>{heading}{paragraphs}</body></html>"
        book.add_item(html_item)
        epub_chapters.append(html_item)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
            tmp_path = tmp.name
        epub.write_epub(tmp_path, book)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "li"]
INLINE_ALLOWED_TAGS = ("em", "i", "strong", "b", "u")


def get_runs(tag):
    """Flattens a paragraph-level tag's inner content into a list of
    (text, active_inline_tags) runs in document order. This is the
    first step in keeping italics/bold: rather than throwing formatting
    away with get_text(), each piece of text remembers which emphasis
    tags it was sitting inside."""
    runs = []

    def walk(node, active_tags):
        for child in node.children:
            if isinstance(child, NavigableString):
                if str(child):
                    runs.append((str(child), tuple(active_tags)))
            elif getattr(child, "name", None) in INLINE_ALLOWED_TAGS:
                walk(child, active_tags + [child.name])
            else:
                # Anything else (span, a, sup, footnote markers, etc)
                # gets unwrapped: keep its text, drop its own styling.
                walk(child, active_tags)

    walk(tag, [])
    return runs


def runs_to_sentences(runs):
    """Splits one paragraph's runs into sentences at the same boundary
    rule split_into_sentences() uses, but rebuilds each sentence's HTML
    from its original runs instead of plain text, so a sentence that
    was partly italicised keeps that italic span intact."""
    full_text = "".join(r[0] for r in runs)
    boundaries = [m.start() for m in re.finditer(r"(?<=[.!?])\s+", full_text)]
    ranges = []
    start = 0
    for b in boundaries:
        ranges.append((start, b))
        start = b
    ranges.append((start, len(full_text)))

    run_offsets = []
    pos = 0
    for text, tags in runs:
        run_offsets.append((pos, pos + len(text), tags))
        pos += len(text)

    def html_for_range(s, e):
        html_parts = []
        for (rs, re_, tags) in run_offsets:
            if re_ <= s or rs >= e:
                continue
            seg_start = max(rs, s)
            seg_end = min(re_, e)
            seg_text = full_text[seg_start:seg_end]
            seg_html = html_escape(seg_text)
            for t in reversed(tags):
                seg_html = f"<{t}>{seg_html}</{t}>"
            html_parts.append(seg_html)
        return "".join(html_parts)

    sentences = []
    for (s, e) in ranges:
        piece_text = full_text[s:e].strip()
        if not piece_text:
            continue
        html_piece = html_for_range(s, e).strip()
        sentences.append({"text": clean_text(piece_text), "html": html_piece})
    return sentences


def extract_paragraph_blocks(soup):
    """Walks a chapter's real paragraph-level elements (p, headings,
    blockquotes, list items) in document order, each becoming one
    paragraph made of formatted sentences. This is what lets a book's
    original paragraph breaks survive instead of the whole chapter
    being flattened into one wall of text.

    Also picks up "leaf" <div> and <pre> elements as their own
    paragraph: some books (poetry, epigraphs, typographic shape-poems)
    use a bare <div> or <pre> with line breaks instead of real <p>
    tags, and without this those sections would silently vanish from
    the book entirely rather than just losing their exact layout."""
    LEAF_TAGS = ("div", "pre")

    blocks = []
    for tag in soup.find_all(BLOCK_TAGS):
        if tag.find_parent(BLOCK_TAGS):
            continue  # already covered as part of an ancestor block
        runs = get_runs(tag)
        sentences = runs_to_sentences(runs)
        if not sentences:
            continue
        is_heading = tag.name in ("h1", "h2", "h3", "h4", "h5", "h6")
        blocks.append({"tag": tag, "sentences": sentences, "is_heading": is_heading})

    for leaf in soup.find_all(LEAF_TAGS):
        if leaf.find_parent(BLOCK_TAGS):
            continue  # sits inside a real block tag, already covered by it
        if leaf.find_parent(LEAF_TAGS):
            continue  # only the outermost leaf container in a nested stack counts; its own text walk already covers everything inside
        if leaf.find(BLOCK_TAGS):
            continue  # this just wraps real paragraph/heading tags, which get counted on their own above
        runs = get_runs(leaf)
        sentences = runs_to_sentences(runs)
        if not sentences:
            continue
        blocks.append({"tag": leaf, "sentences": sentences, "is_heading": False})

    if blocks:
        # Paragraphs and leaf-containers were collected in two separate
        # passes, so put them back in real document order.
        blocks.sort(key=lambda b: (b["tag"].sourceline or 0, b["tag"].sourcepos or 0))
        return blocks

    # Fallback for EPUBs that don't use real <p> tags at all: treat the
    # whole file as one paragraph rather than silently dropping its text.
    text = clean_text(soup.get_text(" "))
    if not text.strip():
        return []
    sents = split_into_sentences(text)
    if not sents:
        return []
    fallback_sentences = [{"text": s, "html": html_escape(s)} for s in sents]
    return [{"tag": soup, "sentences": fallback_sentences, "is_heading": False}]


def extract_epub_chapters(file_bytes):
    """Reads an EPUB's real table of contents and paragraph structure
    instead of guessing at it, EPUB is structured HTML under the hood.
    Each chapter comes back as {title, paragraphs}, paragraphs being a
    list of paragraphs, each a list of sentence dicts with both plain
    text (for narration) and lightly-formatted HTML (for display), so
    paragraph breaks and basic emphasis survive into the reader.

    Two things previously broke chapter detection and are fixed here:
    (1) files must be read in spine order, since manifest order isn't
    guaranteed to match actual reading order, and (2) Gutenberg's EPUBs
    often bundle several chapters into one physical file, told apart
    only by an anchor fragment. Naively keying titles by filename alone
    silently drops all but the last chapter title in a bundled file."""
    if ebooklib is None or BeautifulSoup is None:
        raise RuntimeError("ebooklib and beautifulsoup4 must be installed to read EPUB files")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        book = epub.read_epub(tmp_path, options={"ignore_ncx": True})

        # Group TOC entries by the file they point into, keeping each
        # entry's fragment (the part after '#') so chapters that share
        # a single physical file can still be told apart.
        toc_entries_by_file = {}

        def add_toc_entry(href, title):
            if "#" in href:
                filename, fragment = href.split("#", 1)
            else:
                filename, fragment = href, None
            toc_entries_by_file.setdefault(filename, []).append((fragment, title))

        def walk_toc(items):
            for item in items:
                if isinstance(item, tuple):
                    section_or_link, children = item
                    href = getattr(section_or_link, "href", None)
                    title = getattr(section_or_link, "title", None)
                    if href and title:
                        add_toc_entry(href, title.strip())
                    walk_toc(children)
                else:
                    href = getattr(item, "href", None)
                    title = getattr(item, "title", None)
                    if href and title:
                        add_toc_entry(href, title.strip())

        walk_toc(book.toc)

        chapters = []

        for idref, _linear in book.spine:
            item = book.get_item_with_id(idref)
            if item is None:
                continue

            entries = toc_entries_by_file.get(item.get_name(), [])
            soup = BeautifulSoup(item.get_content(), "html.parser")

            if soup.find("nav") is not None:
                continue  # the auto-generated table-of-contents page, not real content

            blocks = extract_paragraph_blocks(soup)
            if not blocks:
                continue

            fragment_entries = [(frag, title) for frag, title in entries if frag]

            if not fragment_entries:
                # Simple case: this whole file is one chapter, or has
                # no TOC entry at all (front/back matter).
                whole_file_title = entries[0][1] if entries else None
                paragraphs = [{"sentences": b["sentences"], "is_heading": b["is_heading"]} for b in blocks if b["sentences"]]
                if paragraphs:
                    chapters.append({"title": whole_file_title, "paragraphs": paragraphs})
                continue

            # Multiple chapters share this physical file. Instead of
            # inserting text markers and splitting get_text() output,
            # find which paragraph block each chapter-start anchor
            # falls at or after (using each element's real source
            # position) and split the block list there directly.
            preceding_title = None
            for frag, title in entries:
                if frag is None:
                    preceding_title = title

            split_points = []
            for frag, title in fragment_entries:
                anchor = soup.find(id=frag) or soup.find(attrs={"name": frag})
                if anchor is None:
                    continue

                # Most commonly the chapter-start anchor sits nested
                # right inside the heading tag itself (<h2><a id=.../>
                # CHAPTER I</h2>), in which case that block IS where the
                # chapter starts. Check that first, before falling back
                # to a position comparison for a standalone anchor that
                # sits as its own sibling between blocks.
                split_index = None
                for i, b in enumerate(blocks):
                    tag = b["tag"]
                    if tag is anchor or tag.find(id=frag) is not None or (
                        anchor.get("name") and tag.find(attrs={"name": anchor.get("name")}) is not None
                    ):
                        split_index = i
                        break

                if split_index is None:
                    anchor_pos = (anchor.sourceline or 0, anchor.sourcepos or 0)
                    split_index = len(blocks)
                    for i, b in enumerate(blocks):
                        tag_pos = (b["tag"].sourceline or 0, b["tag"].sourcepos or 0)
                        if tag_pos >= anchor_pos:
                            split_index = i
                            break

                split_points.append((split_index, title))

            split_points.sort(key=lambda x: x[0])

            cursor = 0
            current_title = preceding_title
            for split_index, title in split_points:
                if split_index > cursor:
                    paragraphs = [{"sentences": b["sentences"], "is_heading": b["is_heading"]} for b in blocks[cursor:split_index] if b["sentences"]]
                    if paragraphs:
                        chapters.append({"title": current_title, "paragraphs": paragraphs})
                current_title = title
                cursor = split_index
            if cursor < len(blocks):
                paragraphs = [{"sentences": b["sentences"], "is_heading": b["is_heading"]} for b in blocks[cursor:] if b["sentences"]]
                if paragraphs:
                    chapters.append({"title": current_title, "paragraphs": paragraphs})

        return chapters
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------
# Turning chapters into a flat, individually-addressable sentence list
# ---------------------------------------------------------------------

def split_into_sentences(text, max_chars=600):
    """Splits chapter text into individual sentences, each one is a
    separately narratable unit. Falls back to a plain length-based
    split for the rare unusually long 'sentence' (missing punctuation,
    etc) so no single narration request gets too large."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            sentences.append(s)
        else:
            words = s.split()
            current = ""
            for w in words:
                if len(current) + len(w) + 1 <= max_chars:
                    current = (current + " " + w).strip()
                else:
                    if current:
                        sentences.append(current)
                    current = w
            if current:
                sentences.append(current)
    return sentences


def extract_epub_cover(file_bytes):
    """
    Looks for a real cover image embedded in the EPUB. Tries the
    properly-declared cover first, then falls back to any image whose
    filename suggests it's the cover, then finally just the first image
    in the book, most EPUBs put the cover first even when it isn't
    explicitly marked.
    """
    if ebooklib is None:
        return None, None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        book = epub.read_epub(tmp_path, options={"ignore_ncx": True})

        cover_item = None
        try:
            covers = list(book.get_items_of_type(ebooklib.ITEM_COVER))
            if covers:
                cover_item = covers[0]
        except Exception:
            pass

        if cover_item is None:
            for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
                if "cover" in item.get_name().lower():
                    cover_item = item
                    break

        if cover_item is None:
            images = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))
            if images:
                cover_item = images[0]

        if cover_item is None:
            return None, None

        content = cover_item.get_content()
        name = cover_item.get_name().lower()
        if name.endswith(".png"):
            mime = "image/png"
        elif name.endswith(".gif"):
            mime = "image/gif"
        else:
            mime = "image/jpeg"
        return content, mime
    except Exception:
        return None, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_pdf_cover(file_bytes):
    """Renders the PDF's first page as an image to use as a cover,
    since PDFs (unlike EPUBs) have no real embedded cover metadata."""
    if fitz is None:
        return None, None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if doc.page_count == 0:
            return None, None
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return pix.tobytes("png"), "image/png"
    except Exception:
        return None, None


def build_document(chapters_data):
    """
    Flattens {title, paragraphs} chapters into one continuous list of
    sentences, with each chapter recording exactly which sentence
    index it starts on. Nothing here involves narration, it's just
    text, which is why opening even a long book is fast: there's no
    TTS work happening yet, only once a sentence is actually requested.

    Alongside the plain sentence list (unchanged, still what gets sent
    for narration), this also returns sentences_html, paragraph_starts
    (the sentence index each new paragraph begins on), and
    paragraph_headings (whether that paragraph is a heading, parallel
    to paragraph_starts) - together, what lets the reader show real
    paragraph breaks and emphasis instead of one wall of plain text.

    sentences_html is deliberately sparse: most sentences in a book
    have no italics or bold at all, so their "formatted" version would
    be character-for-character identical to their plain text, storing
    it separately would double a book's size for nothing. Only
    sentences whose formatting actually differs get an entry here,
    keyed by their index; everything else falls back to the plain
    sentence text (HTML-escaped) on the reading end.
    """
    sentences = []
    sentences_html = {}
    paragraph_starts = []
    paragraph_headings = []
    chapters = []

    def add_sentence(text, html):
        idx = len(sentences)
        sentences.append(text)
        if html and html != html_escape(text):
            sentences_html[str(idx)] = html

    for ch in chapters_data:
        paragraphs = ch.get("paragraphs", [])
        # Most chapters' own content already opens with a real heading
        # tag (<h1>-<h6>) carrying the same title. Only inject a
        # separate title sentence when that's not the case, otherwise
        # the chapter title would show up twice in a row.
        first_is_heading = bool(paragraphs) and paragraphs[0].get("is_heading")

        if ch["title"]:
            chapters.append({"title": ch["title"], "start_sentence_index": len(sentences)})
            if not first_is_heading:
                paragraph_starts.append(len(sentences))
                paragraph_headings.append(True)
                add_sentence(ch["title"], f"<strong>{html_escape(ch['title'])}</strong>")

        for paragraph in paragraphs:
            sents = paragraph.get("sentences", [])
            if not sents:
                continue
            paragraph_starts.append(len(sentences))
            paragraph_headings.append(bool(paragraph.get("is_heading")))
            for sent in sents:
                add_sentence(sent["text"], sent["html"])

    return chapters, sentences, sentences_html, paragraph_starts, paragraph_headings


# ---------------------------------------------------------------------
# Text -> speech, one sentence at a time
# ---------------------------------------------------------------------

async def narrate_text(text, voice, retries=2):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            audio_bytes = bytearray()
            words = []
            duration = 0.0

            async for event in communicate.stream():
                if event["type"] == "audio":
                    audio_bytes.extend(event["data"])
                elif event["type"] == "WordBoundary":
                    start_sec = event["offset"] / 10_000_000
                    dur_sec = event["duration"] / 10_000_000
                    words.append({
                        "text": event["text"],
                        "start": start_sec,
                        "end": start_sec + dur_sec,
                    })
                    duration = max(duration, start_sec + dur_sec)

            return bytes(audio_bytes), words, duration
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(1)
    raise last_exc


def estimate_word_timings(text, duration_seconds):
    """Fallback for the rare case Edge TTS doesn't return real word
    boundaries for a given sentence: spreads words evenly across the
    known audio duration instead of having no highlighting at all."""
    words = text.split()
    if not words or not duration_seconds:
        return []
    total_chars = sum(len(w) for w in words) or 1
    result = []
    t = 0.0
    for w in words:
        share = len(w) / total_chars
        dur = duration_seconds * share
        result.append({"text": w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur
    return result


def get_audio_duration(path):
    if MP3 is None:
        return None
    try:
        return MP3(path).info.length
    except Exception:
        return None


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------

def fetch_url_bytes(url, timeout=20):
    """A plain GET with a real User-Agent, since some sites reject the
    default Python one. Returns (bytes, final_url_after_redirects)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AloudReader/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.geturl()


def find_gutenberg_epub_url(gutenberg_id):
    """
    Project Gutenberg's exact download URL naming has changed over the
    years (epub.images, epub.noimages, epub3.images...), so instead of
    guessing a pattern, this reads the book's real info page and finds
    whatever EPUB link Gutenberg itself is currently offering.
    """
    info_url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}"
    html_bytes, final_url = fetch_url_bytes(info_url)
    soup = BeautifulSoup(html_bytes, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".epub" in href.lower():
            return urljoin(final_url, href)
    return None


# Small cache so the homepage swivel doesn't re-download a full EPUB
# from Gutenberg's servers just to grab its cover every time someone
# loads the page. Filenames are stable per book id, so this also
# survives a server restart, not just this in-memory dict.
GUTENBERG_COVER_CACHE = {}


@app.route("/api/gutenberg_cover/<gutenberg_id>")
def gutenberg_cover(gutenberg_id):
    """Fetches just the real cover image for a Gutenberg book, used to
    show actual cover art in the homepage's book carousel instead of a
    placeholder, without doing the full text-parsing work of opening it."""
    if gutenberg_id in GUTENBERG_COVER_CACHE:
        return jsonify({"cover_url": GUTENBERG_COVER_CACHE[gutenberg_id]})

    # Reuse a previously saved cover file if one already exists on disk
    # from an earlier run, even if the in-memory cache was reset.
    for ext in (".jpg", ".png", ".gif"):
        existing_path = os.path.join(AUDIO_DIR, f"gutenberg_{gutenberg_id}{ext}")
        if os.path.exists(existing_path):
            url = f"/api/cover/gutenberg_{gutenberg_id}{ext}"
            GUTENBERG_COVER_CACHE[gutenberg_id] = url
            return jsonify({"cover_url": url})

    if BeautifulSoup is None or ebooklib is None:
        return jsonify({"cover_url": None})

    try:
        epub_url = find_gutenberg_epub_url(gutenberg_id)
        if not epub_url:
            GUTENBERG_COVER_CACHE[gutenberg_id] = None
            return jsonify({"cover_url": None})
        file_bytes, _ = fetch_url_bytes(epub_url, timeout=30)
        cover_bytes, cover_mime = extract_epub_cover(file_bytes)
    except Exception:
        GUTENBERG_COVER_CACHE[gutenberg_id] = None
        return jsonify({"cover_url": None})

    if not cover_bytes:
        GUTENBERG_COVER_CACHE[gutenberg_id] = None
        return jsonify({"cover_url": None})

    cover_ext = ".png"
    if cover_mime == "image/jpeg":
        cover_ext = ".jpg"
    elif cover_mime == "image/gif":
        cover_ext = ".gif"

    cover_filename = f"gutenberg_{gutenberg_id}{cover_ext}"
    with open(os.path.join(AUDIO_DIR, cover_filename), "wb") as f:
        f.write(cover_bytes)

    cover_url = f"/api/cover/{cover_filename}"
    GUTENBERG_COVER_CACHE[gutenberg_id] = cover_url
    return jsonify({"cover_url": cover_url})


@app.route("/api/import_gutenberg", methods=["POST"])
def import_gutenberg():
    """
    Downloads a real public-domain book directly from Project
    Gutenberg's own servers and runs it through the exact same
    pipeline as a manually uploaded EPUB, so clicking a book on the
    homepage actually opens it, rather than just linking out to it.
    """
    if BeautifulSoup is None or ebooklib is None:
        return jsonify({"error": "ebooklib and beautifulsoup4 must be installed for this"}), 400

    data = request.get_json(silent=True) or {}
    gutenberg_id = str(data.get("gutenberg_id", "")).strip()
    display_title = (data.get("title") or "").strip()

    if not gutenberg_id.isdigit():
        return jsonify({"error": "Invalid Project Gutenberg book id"}), 400

    try:
        epub_url = find_gutenberg_epub_url(gutenberg_id)
        if not epub_url:
            return jsonify({"error": "Could not find an EPUB for that book on Project Gutenberg"}), 400
        file_bytes, _ = fetch_url_bytes(epub_url, timeout=30)
    except Exception as exc:
        return jsonify({"error": f"Could not download that book: {exc}"}), 400

    try:
        chapters_data = extract_epub_chapters(file_bytes)
        cover_bytes, cover_mime = extract_epub_cover(file_bytes)
        chapters, sentences, sentences_html, paragraph_starts, paragraph_headings = build_document(chapters_data)
    except Exception as exc:
        return jsonify({"error": f"Could not read that book: {exc}"}), 400
    if not sentences:
        return jsonify({"error": "No readable text was found in that book"}), 400

    cover_url = None
    if cover_bytes:
        cover_ext = ".png"
        if cover_mime == "image/jpeg":
            cover_ext = ".jpg"
        elif cover_mime == "image/gif":
            cover_ext = ".gif"
        cover_filename = f"{uuid.uuid4().hex}{cover_ext}"
        with open(os.path.join(AUDIO_DIR, cover_filename), "wb") as f:
            f.write(cover_bytes)
        cover_url = f"/api/cover/{cover_filename}"

    return jsonify({
        "title": display_title or f"Project Gutenberg #{gutenberg_id}",
        "chapters": chapters,
        "sentences": sentences,
        "sentences_html": sentences_html,
        "paragraph_starts": paragraph_starts,
        "paragraph_headings": paragraph_headings,
        "cover_url": cover_url,
    })


@app.route("/api/open_book", methods=["POST"])
def open_book():
    """Parses an EPUB into chapters + a flat sentence list. No narration
    happens here at all, just text extraction, which is why this stays
    fast even for a long book. PDFs go through /api/convert_to_epub
    first now, this endpoint only ever handles real EPUB files."""
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded"}), 400

    uploaded_file = request.files["file"]
    ext = os.path.splitext(uploaded_file.filename)[1].lower()
    if ext != ".epub":
        return jsonify({"error": "Please upload an EPUB file. Got a PDF? Convert it to EPUB first using the tool below the drop zone."}), 400

    file_bytes = uploaded_file.read()
    title = os.path.splitext(uploaded_file.filename)[0]

    try:
        chapters_data = extract_epub_chapters(file_bytes)
        cover_bytes, cover_mime = extract_epub_cover(file_bytes)
        chapters, sentences, sentences_html, paragraph_starts, paragraph_headings = build_document(chapters_data)
    except Exception as exc:
        return jsonify({"error": f"Could not read that file: {exc}"}), 400

    if not sentences:
        return jsonify({"error": "No readable text was found in that file"}), 400

    cover_url = None
    if cover_bytes:
        cover_ext = ".png"
        if cover_mime == "image/jpeg":
            cover_ext = ".jpg"
        elif cover_mime == "image/gif":
            cover_ext = ".gif"
        cover_filename = f"{uuid.uuid4().hex}{cover_ext}"
        with open(os.path.join(AUDIO_DIR, cover_filename), "wb") as f:
            f.write(cover_bytes)
        cover_url = f"/api/cover/{cover_filename}"

    return jsonify({
        "title": title,
        "chapters": chapters,
        "sentences": sentences,
        "sentences_html": sentences_html,
        "paragraph_starts": paragraph_starts,
        "paragraph_headings": paragraph_headings,
        "cover_url": cover_url,
    })


@app.route("/api/narrate_sentence", methods=["POST"])
def narrate_sentence():
    """Narrates exactly one sentence, on demand. This is the whole
    trick: since nothing is pre-generated, any sentence can be
    requested at any time, which is what makes clicking anywhere in
    the text actually jump there immediately instead of waiting on
    a queue."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice_key = data.get("voice", "andrew")
    voice = VOICE_MAP.get(voice_key, VOICE_MAP["andrew"])

    if not text:
        return jsonify({"error": "No text was provided"}), 400

    try:
        audio_bytes, words, duration = asyncio.run(narrate_text(text, voice))
    except Exception as exc:
        return jsonify({"error": f"Narration failed: {exc}"}), 500

    filename = f"{uuid.uuid4().hex}.mp3"
    path = os.path.join(AUDIO_DIR, filename)
    with open(path, "wb") as f:
        f.write(audio_bytes)

    if not words:
        real_duration = get_audio_duration(path) or duration
        words = estimate_word_timings(text, real_duration) if real_duration else []
        duration = real_duration

    return jsonify({
        "audio_url": f"/api/audio/{filename}",
        "words": words,
        "duration": duration,
    })


@app.route("/api/audio/<filename>")
def get_audio(filename):
    return send_file(os.path.join(AUDIO_DIR, filename), mimetype="audio/mpeg")


@app.route("/api/cover/<filename>")
def get_cover(filename):
    ext = os.path.splitext(filename)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".gif":
        mime = "image/gif"
    return send_file(os.path.join(AUDIO_DIR, filename), mimetype=mime)


@app.route("/api/convert_to_epub", methods=["POST"])
def convert_to_epub():
    """Converts an uploaded PDF into a real, standalone EPUB file that
    can be downloaded and used anywhere, not just inside Aloud."""
    if ebooklib is None:
        return jsonify({"error": "ebooklib must be installed for this"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    file_bytes = uploaded_file.read()
    title = os.path.splitext(uploaded_file.filename)[0]

    try:
        chapters_data = convert_pdf_to_epub_chapters(file_bytes)
        epub_bytes = build_epub_file(title, chapters_data)
    except Exception as exc:
        return jsonify({"error": f"Could not convert that PDF: {exc}"}), 400

    epub_filename = f"{safe_filename(title)}_{uuid.uuid4().hex[:8]}.epub"
    with open(os.path.join(AUDIO_DIR, epub_filename), "wb") as f:
        f.write(epub_bytes)

    return jsonify({
        "epub_url": f"/api/converted_epub/{epub_filename}",
        "title": title,
    })


@app.route("/api/converted_epub/<filename>")
def get_converted_epub(filename):
    return send_file(
        os.path.join(AUDIO_DIR, filename),
        mimetype="application/epub+zip",
        as_attachment=True,
    )


def format_ambience_track_name(filename):
    """Turns 'Ambient1.mp3' into 'Ambient 1', 'City_Noise3.wav' into
    'City Noise 3', etc, without needing tracks to be named any
    particular way."""
    name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
    name = re.sub(r"(\D)(\d+)$", r"\1 \2", name)
    return name or filename


@app.route("/api/ambience_list")
def ambience_list():
    """Walks the ambience folder's category subfolders (Ambient,
    Nature, Lofi, etc) and lists the tracks inside each one. Add a
    new subfolder or a new track file and it shows up automatically,
    no code changes needed."""
    categories = []
    if os.path.isdir(AMBIENCE_DIR):
        for entry in sorted(os.listdir(AMBIENCE_DIR)):
            entry_path = os.path.join(AMBIENCE_DIR, entry)
            if not os.path.isdir(entry_path):
                continue
            tracks = []
            for fname in sorted(os.listdir(entry_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in AMBIENCE_MIME_TYPES:
                    tracks.append({
                        "name": format_ambience_track_name(fname),
                        "url": f"/api/ambience/{quote(entry)}/{quote(fname)}",
                    })
            if tracks:
                categories.append({"name": entry, "tracks": tracks})
    return jsonify({"categories": categories})


@app.route("/api/ambience/<path:filepath>")
def get_ambience(filepath):
    full_path = os.path.abspath(os.path.join(AMBIENCE_DIR, filepath))
    if not full_path.startswith(os.path.abspath(AMBIENCE_DIR)):
        return jsonify({"error": "Invalid path"}), 400
    ext = os.path.splitext(full_path)[1].lower()
    mime = AMBIENCE_MIME_TYPES.get(ext, "audio/mpeg")
    return send_file(full_path, mimetype=mime)


# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Aloud is running: open http://localhost:8000 in your browser")
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)