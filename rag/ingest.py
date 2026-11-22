"""PDF extraction, NFC normalization, heading-aware Hindi chunking and citations."""
import json, re, unicodedata, hashlib
from pathlib import Path
import fitz
from config import PDF_PATH, CHUNKS_PATH, DATA_DIR, CHUNK_WORDS, CHUNK_OVERLAP

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\u00ad', '').replace('\ufeff', '')
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _is_heading(line: str) -> bool:
    s = line.strip()
    # Explicit numbered section headings used throughout this source PDF.
    if re.match(r"^\d{1,2}\s*·\s*\S", s) and len(s) < 120:
        return True
    return False

def extract_pages(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []
    table_rows = 0
    for page_no, page in enumerate(doc, start=1):
        raw_text = page.get_text('text', sort=True)
        text = normalize_text(raw_text)
        tables = []
        # PyMuPDF's geometric table detector can fail on PDFs with embedded/custom fonts.
        # These known source pages expose table cells as sequential text lines, so pair
        # cells according to their visible column headers as a deterministic fallback.
        raw_lines = [x.strip() for x in raw_text.splitlines() if x.strip()]
        page_table_rows = []
        heading_line = raw_lines[0] if raw_lines else ''
        if 'पुरस्काार एवं सम्माान' in heading_line:
            for line in raw_lines:
                cells = [normalize_text(c) for c in re.split(r'\s{2,}', line) if normalize_text(c)]
                if len(cells) >= 3 and re.match(r'^(?:19|20)\d{2}|विभिन्न', cells[0]):
                    page_table_rows.append(cells[:3])
        elif 'जीवन-रेखा' in heading_line:
            for line in raw_lines:
                cells = [normalize_text(c) for c in re.split(r'\s{2,}', line) if normalize_text(c)]
                if len(cells) >= 2 and re.match(r'^(?:19|20)\d{2}|\d{4} का दशक', cells[0]):
                    page_table_rows.append(cells[:2])
        elif 'त्वरित तथ्य-सारांश' in heading_line:
            for line in raw_lines:
                cells = [normalize_text(c) for c in re.split(r'\s{2,}', line) if normalize_text(c)]
                if len(cells) >= 2 and cells[0] not in {'क्षेत्र', '20 · त्वरित तथ्य-सारांश'} and not cells[0].startswith('मौलिक अध्ययन सहायिका'):
                    page_table_rows.append(cells[:2])
        if page_table_rows:
            tables.append({'table_index': 1, 'rows': page_table_rows})
            table_rows += len(page_table_rows)
        if text or tables:
            pages.append({'page': page_no, 'text': text, 'tables': tables})
    return pages, table_rows

def parse_paragraphs(page_text: str, initial_section='दस्तावेज़'):
    lines = [re.sub(r'\s+', ' ', x).strip() for x in page_text.splitlines()]
    lines = [x for x in lines if x]
    paras, current, section = [], [], initial_section
    for line in lines:
        if _is_heading(line):
            if current:
                paras.append((section, ' '.join(current))); current = []
            section = re.sub(r'^\d+\s*·\s*', '', line).strip()
            continue
        starts_new = bool(re.match(r'^(?:\d+\s*[·.]|[-•✺]|[१२३४५६७८९०]+\.)', line))
        if current and (starts_new or re.search(r'[।!?]$|[:：]$', current[-1])):
            paras.append((section, ' '.join(current))); current = []
        current.append(line)
    if current:
        paras.append((section, ' '.join(current)))
    return [(s, p) for s, p in paras if len(p) >= 25 and not re.match(r'^(विषय-सूची|मेगामाइंड्ज़ एआई)', p)]

def _make_chunk(text, page_no, section, para_idx, start, end, source_type='paragraph'):
    uid = hashlib.sha1(f'{page_no}|{section}|{para_idx}|{start}|{text}'.encode('utf-8')).hexdigest()[:16]
    return {'chunk_id': uid, 'text': text, 'page': int(page_no), 'section': section or 'दस्तावेज़',
            'paragraph_id': int(para_idx), 'start_word': int(start), 'end_word': int(end),
            'language': 'hi', 'script': 'Devanagari', 'source_type': source_type}

def chunk_paragraphs(paragraphs, page_no, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    chunks = []
    for para_idx, (section, para) in enumerate(paragraphs):
        words = para.split()
        if len(words) <= chunk_words:
            windows = [(0, len(words))]
        else:
            step = max(1, chunk_words - overlap)
            windows = [(i, min(i + chunk_words, len(words))) for i in range(0, len(words), step)]
        for start, end in windows:
            content = ' '.join(words[start:end]).strip()
            if len(content) >= 25:
                kind = 'practice_question' if any(k in section for k in ('विचार एवं बोध-प्रश्न', 'प्रश्न')) else 'paragraph'
                chunks.append(_make_chunk(content, page_no, section, para_idx, start, end, kind))
    return chunks

def table_to_chunks(tables, page_no, section):
    chunks = []
    for table in tables:
        for row_idx, row in enumerate(table['rows']):
            cells = [c for c in row if c]
            if len(cells) < 2:
                continue
            # Keep row context together so timeline/award labels remain attached to values.
            text = ' | '.join(cells)
            if len(text) >= 12:
                chunks.append(_make_chunk(text, page_no, section, 10000 + table['table_index'] * 100 + row_idx,
                                          0, len(cells), 'table_row'))
    return chunks

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f'PDF not found: {PDF_PATH}')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pages, table_rows = extract_pages(PDF_PATH)
    all_chunks, paragraph_count = [], 0
    current_section = 'दस्तावेज़'
    for item in pages:
        paras = parse_paragraphs(item['text'], current_section)
        paragraph_count += len(paras)
        if paras:
            current_section = paras[-1][0]
        all_chunks.extend(chunk_paragraphs(paras, item['page']))
        all_chunks.extend(table_to_chunks(item['tables'], item['page'], current_section))
    # Stable unique IDs are mandatory because Chroma uses them as primary keys.
    seen, unique = set(), []
    for chunk in all_chunks:
        if chunk['chunk_id'] not in seen:
            seen.add(chunk['chunk_id']); unique.append(chunk)
    with CHUNKS_PATH.open('w', encoding='utf-8') as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    stats = {'source': PDF_PATH.name, 'pages_extracted': len(pages), 'paragraphs_detected': paragraph_count,
             'table_rows_extracted': table_rows, 'chunks_created': len(unique),
             'characters': sum(len(p['text']) for p in pages), 'chunks_path': str(CHUNKS_PATH),
             'note': 'PDF text and table extraction are best-effort; inspect table_rows_extracted and extracted chunks.',
             'citation_pages': '1-based physical PDF page numbers', 'unicode_normalization': 'NFC'}
    (DATA_DIR / 'ingestion_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
