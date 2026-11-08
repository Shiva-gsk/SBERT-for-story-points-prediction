"""tawos_dump_to_csv.py — pure-Python MySQL-dump parser + combiner for TAWOS.

Reads a mysqldump .sql file (no MySQL server needed), parses every CREATE TABLE
+ INSERT statement, builds one DataFrame per table, then JOINs
Issue x Project x Repository into a single clean combined CSV with the PS-6
"clean story-point" filters applied. Also writes one CSV per table so every
table in the dump is recoverable.

Usage:
  python tawos_dump_to_csv.py path/to/tawos.sql --out-dir data/processed
  python tawos_dump_to_csv.py --url URL [--cookie-jar cookies.txt] --out-dir data/processed
  cat tawos.sql | python tawos_dump_to_csv.py --stdin --out-dir data/processed

Outputs (in --out-dir, default data/processed):
  tawos_combined.csv          one row per eligible issue, joined columns
  tables/<Table>.csv          one CSV per table in the dump
  tawos_table_index.csv       table -> row count -> column list summary
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError:
    pd = None  # we'll fall back to plain csv if pandas is missing


# ---------------------------------------------------------------------------
# Statement splitter (top-level ';', quote-aware, backslash-aware)
# ---------------------------------------------------------------------------
def iter_statements(fh: Iterable[str]) -> Iterable[str]:
    """Yield complete SQL statements from a line-iterable file handle."""
    buf: list[str] = []
    in_str = False
    for line in fh:
        i = 0
        while i < len(line):
            ch = line[i]
            if in_str:
                if ch == "\\":
                    buf.append(line[i:i + 2]); i += 2; continue
                if ch == "'":
                    # doubled '' inside a string is an escaped quote
                    if i + 1 < len(line) and line[i + 1] == "'":
                        buf.append(line[i:i + 2]); i += 2; continue
                    buf.append(ch); in_str = False; i += 1; continue
                buf.append(ch); i += 1; continue
            else:
                if ch == "'":
                    buf.append(ch); in_str = True; i += 1; continue
                if ch == '"':
                    # MySQL allows double-quoted strings in ANSI_QUOTES off mode;
                    # treat as string delimiter to be safe
                    buf.append(ch); in_str = True; i += 1; continue
                if ch == ";" and not in_str:
                    stmt = "".join(buf).strip()
                    if stmt:
                        yield stmt
                    buf = []
                    i += 1
                    # skip the rest of this line (after ; on the same line)
                    # only if it's just whitespace / a comment
                    rest = line[i:].strip()
                    if rest.startswith("--") or rest.startswith("/*"):
                        buf = []
                        break
                    if rest:
                        buf.append(rest)
                    break
                buf.append(ch); i += 1
    tail = "".join(buf).strip()
    if tail:
        yield tail


# ---------------------------------------------------------------------------
# CREATE TABLE  ->  (table_name, [column_names in declared order])
# ---------------------------------------------------------------------------
_CREATE_RE = re.compile(
    r"^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?([A-Za-z0-9_.$]+)`?\s*\(",
    re.IGNORECASE,
)
_COL_RE = re.compile(
    r"^\s*`([A-Za-z0-9_]+)`\s+([A-Za-z0-9]+(?:\([^)]*\))?(?:\s+\w+)*)",
)


def parse_create_table(stmt: str) -> tuple[str, list[str]] | None:
    m = _CREATE_RE.match(stmt)
    if not m:
        return None
    table = m.group(1).split(".")[-1].strip("`")
    # Take the body inside the outermost ( ) of the CREATE TABLE.
    open_idx = stmt.find("(", m.end() - 1)
    # find matching close paren (top-level)
    depth = 0
    body_end = -1
    in_str = False
    for i in range(open_idx, len(stmt)):
        ch = stmt[i]
        if in_str:
            if ch == "\\":
                continue
            if ch == "'":
                in_str = False
            continue
        if ch == "'":
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                body_end = i
                break
    if body_end < 0:
        return table, []
    body = stmt[open_idx + 1:body_end]
    cols: list[str] = []
    # Split body on top-level commas
    for raw in _split_top_level_commas(body):
        s = raw.strip()
        if not s:
            continue
        # Skip constraint / key / index / primary lines
        if re.match(r"^(PRIMARY\s+KEY|UNIQUE|KEY|INDEX|CONSTRAINT|FOREIGN|FULLTEXT|SPATIAL|CHECK)",
                    s, re.IGNORECASE):
            continue
        cm = _COL_RE.match(s)
        if cm:
            cols.append(cm.group(1))
    return table, cols


def _split_top_level_commas(s: str) -> list[str]:
    out, buf, depth, in_str = [], [], 0, False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == "\\":
                buf.append(s[i:i + 2]); i += 2; continue
            if ch == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    buf.append(s[i:i + 2]); i += 2; continue
                buf.append(ch); in_str = False; i += 1; continue
            buf.append(ch); i += 1; continue
        if ch == "'":
            buf.append(ch); in_str = True; i += 1; continue
        if ch == "(":
            depth += 1; buf.append(ch); i += 1; continue
        if ch == ")":
            depth -= 1; buf.append(ch); i += 1; continue
        if ch == "," and depth == 0:
            out.append("".join(buf)); buf = []; i += 1; continue
        buf.append(ch); i += 1
    if buf:
        out.append("".join(buf))
    return out


# ---------------------------------------------------------------------------
# INSERT  ->  (table_name, [cols or None], [row_values])
# ---------------------------------------------------------------------------
_INSERT_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+`?([A-Za-z0-9_.$]+)`?(?:\s*\(([^)]*)\))?\s+VALUES\s*",
    re.IGNORECASE,
)


def parse_insert(stmt: str) -> tuple[str, list[str] | None, list[list]] | None:
    m = _INSERT_RE.match(stmt)
    if not m:
        return None
    table = m.group(1).split(".")[-1].strip("`")
    cols: list[str] | None = None
    if m.group(2):
        cols = [c.strip().strip("`") for c in m.group(2).split(",") if c.strip()]
    # Rest after 'VALUES' is the tuple list: (v,v,...),(v,v,...),...
    rest = stmt[m.end():].strip().rstrip(";").strip()
    rows = _parse_value_tuples(rest)
    return table, cols, rows


def _parse_value_tuples(s: str) -> list[list]:
    """Parse '(1,'a',NULL),(2,'b',...),...' into a list of value lists."""
    rows = []
    i = 0
    n = len(s)
    while i < n:
        # skip whitespace and commas
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if s[i] != "(":
            break
        i += 1  # consume '('
        vals: list = []
        while i < n:
            # skip whitespace
            while i < n and s[i] in " \t\r\n":
                i += 1
            if i < n and s[i] == ")":
                i += 1
                break
            # parse one value
            val, i = _parse_one_value(s, i)
            vals.append(val)
            # skip whitespace
            while i < n and s[i] in " \t\r\n":
                i += 1
            if i < n and s[i] == ",":
                i += 1
                continue
            if i < n and s[i] == ")":
                i += 1
                break
        rows.append(vals)
    return rows


def _parse_one_value(s: str, i: int) -> tuple[object, int]:
    """Parse a single value at position i; return (value, next_i)."""
    n = len(s)
    # skip whitespace
    while i < n and s[i] in " \t\r\n":
        i += 1
    if i >= n:
        return None, i
    ch = s[i]
    # NULL
    if s[i:i + 4].upper() == "NULL" and (i + 4 >= n or not s[i + 4].isalnum()):
        return None, i + 4
    # Quoted string (with optional introducer like _binary, _utf8, N, X)
    if ch in "'\"":
        return _parse_quoted(s, i, ch)
    # introducer-prefixed strings: _binary'...', _utf8'...', X'...', N'...'
    m = re.match(r"^[NnXx]?'", s[i:i + 10])
    intro = re.match(r"^_[A-Za-z0-9]+'", s[i:i + 32])
    if m or intro:
        # find the opening quote
        qpos = s.find("'", i)
        if qpos == -1:
            qpos = s.find('"', i)
        if qpos != -1:
            return _parse_quoted(s, qpos, s[qpos])
    # Number or identifier
    j = i
    while j < n and s[j] not in ",)\t\r\n ":
        # don't break on '.' for decimals
        j += 1
    token = s[i:j].strip()
    if token == "":
        return None, j
    # try int / float
    try:
        if "." in token or "e" in token.lower():
            return float(token), j
        return int(token), j
    except ValueError:
        return token, j  # bare identifier (e.g., function result)


def _parse_quoted(s: str, i: int, q: str) -> tuple[str, int]:
    """Parse a quoted string starting at s[i]=='q'; return (decoded, next_i)."""
    out = []
    i += 1  # skip opening quote
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            # escape sequence
            if i + 1 < n:
                nxt = s[i + 1]
                mp = {"n": "\n", "t": "\t", "r": "\r", "0": "\x00",
                      "b": "\b", "Z": "\x1a", "\\": "\\", "'": "'", '"': '"',
                      "%": "%", "_": "_"}
                out.append(mp.get(nxt, nxt))
                i += 2
                continue
            i += 1
            continue
        if ch == q:
            # doubled quote = escaped literal
            if i + 1 < n and s[i + 1] == q:
                out.append(q)
                i += 2
                continue
            i += 1
            return "".join(out), i
        out.append(ch)
        i += 1
    return "".join(out), i


# ---------------------------------------------------------------------------
# Driver: parse dump -> {table: DataFrame}
# ---------------------------------------------------------------------------

def parse_dump(
    path: Path | None,
    text: str | None = None,
) -> dict[str, "pd.DataFrame"]:
    """
    Parse a MySQL dump.

    When a file path is supplied, read it incrementally instead
    of loading the entire SQL dump into a single Python string.
    """
    if pd is None:
        raise RuntimeError("pandas is required: pip install pandas")

    tables: dict[str, dict] = {}

    def process_file(fh):
        for stmt in iter_statements(fh):
            ct = parse_create_table(stmt)

            if ct:
                tname, cols = ct
                tables.setdefault(
                    tname, {"cols": cols, "rows": []}
                )
                if cols and not tables[tname]["cols"]:
                    tables[tname]["cols"] = cols
                continue

            ins = parse_insert(stmt)

            if ins:
                tname, cols, rows = ins

                if tname not in tables:
                    tables[tname] = {
                        "cols": cols or [],
                        "rows": [],
                    }

                if cols:
                    tables[tname]["cols"] = cols

                tables[tname]["rows"].extend(rows)

    if text is not None:
        # For callers that already have text in memory.
        process_file(io.StringIO(text))
    elif path is not None:
        # Stream the dump file instead of calling path.read_text().
        with path.open(
            "r", encoding="utf-8", errors="replace"
        ) as fh:
            process_file(fh)
    else:
        raise ValueError("Provide either a file path or SQL text.")

    out = {}

    for tname, table in tables.items():
        rows = table["rows"]
        cols = table["cols"]

        if not cols:
            cols = [
                f"col_{i}"
                for i in range(len(rows[0]) if rows else 0)
            ]

        out[tname] = pd.DataFrame(rows, columns=cols)

    return out



# ---------------------------------------------------------------------------
# Combine Issue x Project x Repository, apply PS-6 filters, write CSVs
# ---------------------------------------------------------------------------
def combine_and_write(tables: dict, out_dir: Path, apply_filters: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    # write every table
    index_rows = []
    for tname, df in tables.items():
        p = out_dir / "tables" / f"{tname}.csv"
        df.to_csv(p, index=False)
        index_rows.append({"table": tname, "n_rows": len(df), "columns": ",".join(df.columns)})
    pd.DataFrame(index_rows).to_csv(out_dir / "tawos_table_index.csv", index=False)

    # combined
    have_issue = "Issue" in tables
    have_project = "Project" in tables
    have_repo = "Repository" in tables
    if not have_issue:
        # nothing to combine
        return out_dir / "tawos_table_index.csv"

    issue = tables["Issue"].copy()
    # project + repository joins
    if have_project and "Project_ID" in issue.columns and "ID" in tables["Project"].columns:
        proj = tables["Project"].rename(columns={"ID": "Project_ID",
                                                  "Name": "project",
                                                  "Project_Key": "project_key"})
        # avoid column collisions
        proj = proj[["Project_ID", "project"] + [c for c in proj.columns
                                                  if c not in ("Project_ID", "project", "Description", "URL", "Start_Date", "Last_Update_Date", "project_key", "Repository_ID")]]
        issue = issue.merge(proj, on="Project_ID", how="left")
    if have_repo and "Repository_ID" in issue.columns and "ID" in tables["Repository"].columns:
        repo = tables["Repository"][["ID", "Name"]].rename(columns={"ID": "Repository_ID",
                                                                     "Name": "repository"})
        issue = issue.merge(repo, on="Repository_ID", how="left")

    # canonical PS-6 columns (rename to friendly names)
    rename = {
        "Issue_Key": "issue_key",
        "Title": "title",
        "Description_Text": "description_text",
        "Description": "description_raw",
        "Description_Code": "description_code",
        "Type": "issue_type",
        "Priority": "priority",
        "Status": "status",
        "Resolution": "resolution",
        "Story_Point": "story_point",
        "Story_Point_Changed_After_Estimation": "sp_changed_after_est",
        "Creation_Date": "created_at",
        "Estimation_Date": "estimation_date",
        "Resolution_Date": "resolution_date",
        "Last_Updated": "last_updated",
    }
    issue = issue.rename(columns={k: v for k, v in rename.items() if k in issue.columns})

    if apply_filters:
        sp = pd.to_numeric(issue["story_point"], errors="coerce") if "story_point" in issue else None
        mask = pd.Series(True, index=issue.index)
        if "resolution" in issue:
            mask &= issue["resolution"].notna()
        if sp is not None:
            mask &= sp.notna() & (sp >= 1) & (sp <= 100)
            issue["story_point"] = sp
        if "sp_changed_after_est" in issue:
            mask &= (issue["sp_changed_after_est"].astype(str).isin(["0", "0.0", "False", "false"]))
        issue = issue[mask]

    # order + sort
    if "project" in issue and "created_at" in issue:
        issue = issue.sort_values(["project", "created_at"]).reset_index(drop=True)

    # prefer description_text over description_raw in the combined CSV
    if "description_text" not in issue and "description_raw" in issue:
        issue = issue.rename(columns={"description_raw": "description_text"})

    out_csv = out_dir / "tawos_combined.csv"
    issue.to_csv(out_csv, index=False)
    return out_csv


# ---------------------------------------------------------------------------
# Download helper (uses cookie jar if supplied; for when run on user's box)
# ---------------------------------------------------------------------------
def download(url: str, dst: Path, cookie_jar: Path | None = None) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    req = urllib.request.Request(url, headers=headers)
    import http.cookiejar
    cj = http.cookiejar.MozillaCookieJar(cookie_jar) if cookie_jar else http.cookiejar.CookieJar()
    if cookie_jar and Path(cookie_jar).exists():
        cj.load(ignore_discard=True, ignore_expires=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    with opener.open(req, timeout=300) as r, dst.open("wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="path to the .sql dump file")
    ap.add_argument("--url", help="download URL (will fetch with optional cookies)")
    ap.add_argument("--cookie-jar", help="Netscape cookie jar for --url (e.g. /tmp/cookies.txt)")
    ap.add_argument("--stdin", action="store_true", help="read dump from stdin")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--no-filters", action="store_true", help="keep all issues (skip PS-6 clean filters)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.url:
        dst = Path("/tmp/tawos_dump.sql")
        print(f"==> Downloading {args.url} -> {dst}")
        download(args.url, dst, Path(args.cookie_jar) if args.cookie_jar else None)
        sql_path = dst
    elif args.stdin:
        text = sys.stdin.read()
        sql_path = None
    elif args.path:
        sql_path = Path(args.path)
        text = None
    else:
        ap.error("provide a path, --url, or --stdin")

    print("==> Parsing dump ...")
    tables = parse_dump(sql_path, text=text) if not args.stdin else parse_dump(None, text=text)
    print(f"    parsed {len(tables)} table(s):")
    for t, df in tables.items():
        print(f"      - {t:20s} {len(df):>8d} rows  cols={list(df.columns)[:6]}{'...' if len(df.columns)>6 else ''}")

    print("==> Writing per-table CSVs + combined CSV ...")
    out = combine_and_write(tables, out_dir, apply_filters=not args.no_filters)
    print(f"\nDONE.")
    print(f"  Combined CSV : {out}")
    print(f"  Per-table    : {out_dir}/tables/")
    print(f"  Table index  : {out_dir}/tawos_table_index.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
