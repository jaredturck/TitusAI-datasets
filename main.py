import pathlib, time, sys, re, os, html
from lxml import etree
import mwparserfromhell as mw

INPUT_PATH = "enwiki-20251001-pages-articles-multistream.xml"
OUTPUT_FILE = '/mnt/8TB_HDD/datasets/Wikipedia/wiki_pages'

TABLE_RE = re.compile(r"\{\|[\s\S]*?\|\}", re.DOTALL)
REF_SELF_CLOSE_RE = re.compile(r"(?is)<ref\b[^/>]*/>")
REF_BLOCK_RE      = re.compile(r"(?is)<ref\b[^>]*>.*?</ref>")
GALLERY_RE        = re.compile(r"(?is)<gallery\b[^>]*>.*?</gallery>")
URL_RE            = re.compile(r"https?://\S+")
WS_RE             = re.compile(r"\s+")
HTML_TAG_RE       = re.compile(r"(?is)<[^>]+>")
EXT_LINK_LABELED_RE = re.compile(r"\[(?:(?:https?|ftp)://[^\s\]]+)\s+([^\]]+)\]")
EXT_LINK_BARE_RE    = re.compile(r"\[(?:(?:https?|ftp)://[^\s\]]+)\]")
INT_LINK_LABELED_RE = re.compile(r"\[\[[^|\]]+\|([^\]]+)\]\]")
INT_LINK_SIMPLE_RE  = re.compile(r"\[\[([^\]|#]+)(?:#[^\]]+)?\]\]")
FILE_LINK_RE        = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.IGNORECASE)
SIMPLE_TEMPLATE_RE  = re.compile(r"\{\{[^{}]*\}\}")

OUTPUT_FILE_OBJ = open(os.path.join(OUTPUT_FILE, 'wiki_dump_1.txt'), "ab")

def _looks_hard_wikitext(s: str) -> bool:
    return ("{{" in s and "}}" in s and s.count("{{") > 1) or ("{{#" in s)

def clean_wikitext(s):
    if not s:
        return ""

    s = TABLE_RE.sub(" ", s)
    s = REF_SELF_CLOSE_RE.sub(" ", s)
    s = REF_BLOCK_RE.sub(" ", s)
    s = GALLERY_RE.sub(" ", s)
    s = FILE_LINK_RE.sub(" ", s)
    s = EXT_LINK_LABELED_RE.sub(r"\1", s)
    s = EXT_LINK_BARE_RE.sub(" ", s)
    s = INT_LINK_LABELED_RE.sub(r"\1", s)
    s = INT_LINK_SIMPLE_RE.sub(r"\1", s)

    # 2) Strip simple templates quickly (non-nested) by iterating a few times
    for _ in range(3):  # usually enough to peel nested in layers cheaply
        new_s = SIMPLE_TEMPLATE_RE.sub(" ", s)
        if new_s == s:
            break
        s = new_s

    # 3) If it still looks “hard”, fall back to mwparserfromhell (rare)
    if _looks_hard_wikitext(s):
        code = mw.parse(s)
        # We avoid remove(); rely on strip_code to drop remaining templates/markup
        out = code.strip_code(normalize=True, collapse=True)
    else:
        # Remove any remaining HTML-ish tags
        s = HTML_TAG_RE.sub(" ", s)
        # Unescape entities (e.g., &nbsp;)
        out = html.unescape(s)

    # 4) Final cleanup
    out = URL_RE.sub("", out)         # nuke stray URLs
    out = WS_RE.sub(" ", out).strip() # whitespace collapse
    return out

def slugify(title, maxlen=150):
    title = title.strip()
    title = re.sub(r"\s+", "_", title)
    title = re.sub(r'[\\/:*?"<>|]', "", title)
    title = re.sub(r"_+", "_", title)[:maxlen].strip("._")
    return title or "untitled"

def main(xml_path: str):
    context = etree.iterparse(
        xml_path,
        events=("end",),
        tag="{*}page",
        huge_tree=True,
        recover=True,
        encoding="utf-8",
        remove_comments=True,
        remove_pis=True,
    )

    start = time.time()
    count = 0
    byte_counter = 0
    file_counter = 0

    for _, page in context:
        count += 1
        if time.time() - start > 10:
            print(f"Processed {count:,} pages so far", file=sys.stderr)
            start = time.time()

        text = ""
        rev = page.find("{*}revision")
        if rev is not None:
            text_el = rev.find("{*}text")
            if text_el is not None and text_el.text:
                text = text_el.text

        if len(text) >= 1024:
            current_row = clean_wikitext(text).encode("utf-8") + b"\n[EOS]\n"
            byte_counter += len(current_row)
            OUTPUT_FILE_OBJ.write(current_row)

            if byte_counter >= 1024 * 1024 * 1024:
                OUTPUT_FILE_OBJ.close()
                global OUTPUT_FILE_OBJ
                OUTPUT_FILE_OBJ = open(os.path.join(OUTPUT_FILE, f'wiki_dump_{file_counter}.txt'), "ab")
                file_counter += 1

        # Drop local references BEFORE clearing the page node
        text = None
        text_el = None
        rev = None

        # Now free the subtree for real
        page.clear()
        parent = page.getparent()
        while parent is not None and page.getprevious() is not None:
            del parent[0]

    del context
    print(f"Done. Wrote {count:,} pages to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    try:
        main(INPUT_PATH)
    except KeyboardInterrupt:
        OUTPUT_FILE_OBJ.close()
    finally:
        OUTPUT_FILE_OBJ.close()
