import time, sys, os
from lxml import etree
import mwparserfromhell as mw

INPUT_PATH = "enwiki-20251001-pages-articles-multistream.xml"
OUTPUT_FILE = '/mnt/8TB_HDD/datasets/TitusAI-datasets/wiki_pages'

OUTPUT_FILE_OBJ = open(os.path.join(OUTPUT_FILE, 'wiki_dump_1.txt'), "ab+")

def clean_wikitext(s):
    out = mw.parse(s).strip_code(normalize=True, collapse=True)
    return out.encode('utf-8', errors='ignore')

def main(xml_path):
    global OUTPUT_FILE_OBJ
    context = etree.iterparse(xml_path, events=("end",), tag="{*}page", huge_tree=True, recover=True, encoding="utf-8", remove_comments=True, remove_pis=True)

    start = time.time()
    count = 0
    byte_counter = 0
    file_counter = 1

    for _, page in context:
        count += 1
        if time.time() - start > 10:
            print(f"Processed {count:,} pages so far", file=sys.stderr)
            start = time.time()

        text = page.findtext("{*}revision/{*}text") or ''

        if len(text) >= 1024:
            current_row = clean_wikitext(text) + b"\n[EOS]\n"
            byte_counter += len(current_row)
            OUTPUT_FILE_OBJ.write(current_row)

            if byte_counter >= 1024 * 1024 * 1024:
                OUTPUT_FILE_OBJ.close()
                file_counter += 1
                OUTPUT_FILE_OBJ = open(os.path.join(OUTPUT_FILE, f'wiki_dump_{file_counter}.txt'), "ab+")
                byte_counter = 0
                
        # Now free the subtree for real
        text = text_el = rev = None
        page.clear()
        while page.getprevious() is not None:
            del page.getparent()[0]

    del context
    print(f"Done. Wrote {count:,} pages to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    try:
        main(INPUT_PATH)
    except KeyboardInterrupt:
        OUTPUT_FILE_OBJ.close()
    finally:
        OUTPUT_FILE_OBJ.close()
