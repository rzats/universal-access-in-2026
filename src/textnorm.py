"""Normalisation and tokenisation shared by ground truth, model outputs and OCR.

Tokens are lower-cased letter runs (apostrophe kept inside words) or digit runs.
Phone-like digit runs (10+ digits separated only by spaces, hyphens or brackets) are
merged into one numeric token, so "+380 (67) 482-19-05" and "+380 67 482 19 05" both
become "380674821905"; dots, colons and commas between digits are dropped, so "14.11.2026" becomes
"14112026", "10:30" == "10.30" == "1030" and "2413,21" == "2413.21"; all other separators split. Leading zeros are stripped from numeric tokens ("08" == "8").
"""
import re
import unicodedata

APOS = "'’ʼ`´‘ʹ"
# Latin look-alikes that sometimes leak into Cyrillic words
HOMO = str.maketrans({"a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р", "x": "х",
                      "y": "у", "k": "к", "m": "м", "t": "т", "h": "н", "b": "в"})
CYR = re.compile(r"[а-яіїєґё]")
TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+")
PHONE = re.compile(r"\+?\(?\d[\d \-() ]{7,}\d")
DOT = re.compile(r"(?<=\d)[.:,](?=\d)")


def _removed(s: str):
    """Indices of separator characters to delete before tokenising."""
    rm = set()
    for m in PHONE.finditer(s):
        if sum(c.isdigit() for c in m.group(0)) >= 10:
            rm.update(i for i in range(m.start(), m.end()) if not s[i].isdigit())
    for m in DOT.finditer(s):
        rm.add(m.start())
    return rm


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    for a in APOS:
        s = s.replace(a, "'")
    s = s.replace("й̈", "ї")
    rm = _removed(s)
    return "".join(c for i, c in enumerate(s) if i not in rm)


def _fix_script(tok: str) -> str:
    """Fold Latin look-alike letters to Cyrillic. Applied to every token on every side (reference,
    model output, OCR), so mixed-script text layers (e.g. 'sніgеllа') match 'shigella'."""
    if re.search(r"[a-z]", tok):
        return tok.translate(HOMO)
    return tok


def tokens(s: str):
    """Return list of (token, start, end) on the normalised string."""
    s = norm_text(s)
    out = []
    for m in TOKEN.finditer(s):
        t = m.group(0)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = _fix_script(t)
        out.append((t, m.start(), m.end()))
    return out


def toks(s: str):
    return [t for t, _, _ in tokens(s)]


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    return lev(hyp, ref) / max(1, len(ref))


def sim(a: str, b: str) -> float:
    return 1.0 - lev(a, b) / max(len(a), len(b), 1)


def tokens_raw(raw: str):
    """Tokenise a ground-truth string, returning (token, [raw char indices]) pairs.

    Same rules as tokens(); assumes the raw string is NFC and that lower-casing is
    length-preserving (true for the Ukrainian/Latin text we render)."""
    s = raw.lower()
    for a in APOS:
        s = s.replace(a, "'")
    assert len(s) == len(raw)
    rm = _removed(s)
    idx = [i for i in range(len(s)) if i not in rm]
    s2 = "".join(s[i] for i in idx)
    out = []
    for m in TOKEN.finditer(s2):
        t = m.group(0)
        t = (t.lstrip("0") or "0") if t.isdigit() else _fix_script(t)
        out.append((t, idx[m.start():m.end()]))
    return out
