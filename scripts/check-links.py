#!/usr/bin/env python3
"""Check the links of the yolo docs: every internal file link and #anchor must resolve.

Files checked: README.md, docs/*.md, steps/*/README.md and the other top-level *.md.
  python scripts/check-links.py              internal links and anchors
  python scripts/check-links.py --external   also GET every http(s) link (retry on 429/5xx)
"""
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urldefrag

from markdown.extensions.toc import slugify

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted({*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md"), *ROOT.glob("steps/*/README.md"),
                *ROOT.glob("steps/README.md"), *ROOT.glob("config/README.md")})
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|(?<!\()<(https?://[^>]+)>|!\[[^\]]*\]\(([^)\s]+)\)")
HERA = "https://novelcore.github.io/hera-pipeline-template/"
FENCE = re.compile(r"^```.*?^```", re.S | re.M)


def anchors(path):
    text = FENCE.sub("", path.read_text())
    out = set()
    for m in re.finditer(r"^#{1,6}\s+(.*?)\s*(?:\{#([\w-]+)\})?$", text, re.M):
        out.add(m.group(2) or slugify(re.sub(r"[`*]", "", m.group(1)), "-"))
    return out


def get(url):
    for attempt in range(4):
        req = urllib.request.Request(url, headers={"User-Agent": "yolo-docs-check"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return e.code
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return str(e)
            time.sleep(2)


def main():
    external = "--external" in sys.argv
    bad, ext = [], {}
    cache = {}
    for f in FILES:
        text = FENCE.sub("", f.read_text())
        for m in LINK.finditer(text):
            link = m.group(1) or m.group(2) or m.group(3)
            if link.startswith(("mailto:", "tel:")):
                continue
            if link.startswith(("http://", "https://")):
                ext.setdefault(urldefrag(link)[0], f)
                continue
            target, frag = urldefrag(unquote(link))
            dest = (f.parent / target).resolve() if target else f
            if not dest.exists():
                bad.append(f"{f.relative_to(ROOT)}: missing file  {link}")
                continue
            if frag and dest.suffix == ".md":
                cache.setdefault(dest, anchors(dest))
                if frag not in cache[dest]:
                    bad.append(f"{f.relative_to(ROOT)}: missing anchor  {link}")
    if external:
        for url, f in sorted(ext.items()):
            code = get(url)
            if code != 200 and url.startswith(HERA):
                print(f"PENDING {f.relative_to(ROOT)}: {code} {url} (hera pages publish from its main only)")
            elif code != 200:
                bad.append(f"{f.relative_to(ROOT)}: external {code}  {url}")
    print(f"checked {len(FILES)} files, {len(ext)} external links{'' if external else ' (not fetched; use --external)'}")
    for b in bad:
        print("BROKEN", b)
    sys.exit(1 if bad else 0)


main()
