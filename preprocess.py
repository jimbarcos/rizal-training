import re
import unicodedata

def clean_gutenberg_text(file_path):
    """
    Reads a raw text file, removes Gutenberg boilerplate, 
    and fixes hard line breaks to form proper paragraphs.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    start_match = re.search(r'\*\*\* START OF [^\n]*\*\*\*', text)
    end_match = re.search(r'\*\*\* END OF [^\n]*\*\*\*', text)
    
    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]

    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for p in paragraphs:
        clean_p = re.sub(r'\s+', ' ', p).strip()
        
        if len(clean_p) > 15 and not re.match(r'^(CHAPTER|KABANATA|Kabanata)\s+[IVXLCDM]+', clean_p, re.IGNORECASE):
            clean_p = re.sub(r'\[\d+\]', '', clean_p)
            cleaned_paragraphs.append(clean_p)

    return ' '.join(cleaned_paragraphs)

def normalize_historical_tagalog(text):
    """
    Converts 19th-century Tagalog orthography to modern standards,
    strips Spanish punctuation, and removes heavy diacritics.
    """
    text_lower = text.lower()
    
    # 1. Remove Spanish inverted punctuation
    text_lower = text_lower.replace('¡', '').replace('¿', '')
    
    # 2. Fix the archaic "ng" and "mga" representations 
    text_lower = re.sub(r'n~g', 'ng', text_lower)
    text_lower = re.sub(r'm~ga', 'mga', text_lower)
    text_lower = text_lower.replace('ng̃', 'ng').replace('mg̃a', 'mga')
    
    # 3. Conservative Archaic → Modern Map (from v3)
    # Using word boundaries (\b) prevents accidentally changing modern words
    ORTHO_MAP = [
        (r'\baco\b',    'ako'),
        (r'\bica\b',    'ika'),
        (r'\bcayo\b',   'kayo'),
        (r'\bca\b',     'ka'),
        (r'\bninyong\b','ninyong'),
        (r'\banak\b',   'anak'),
        (r'\bmalaking\b','malaking'),
        (r'\bcamag\b',  'kamag'),
        (r'\bcarun\b',  'karun'),
        (r'\bcasama\b', 'kasama'),
        (r'\bcata\b',   'kata'),
        (r'\bcapit\b',  'kapit'),
        (r'\bcailangan\b','kailangan'),
        (r'\bcawawa\b', 'kawawa'),
        (r'\bcupkop\b', 'kupkop'),
        (r'\bcustura\b','kustura'),
        (r'\bquita\b',  'kita'),
        (r'\bquilos\b', 'kilos'),
        (r'\bguinoo\b', 'ginoo'),
        (r'\bguitna\b', 'gitna'),
        (r"(\w)'ng\b",  r'\1ng'),
        (r'\bboong\b',  'buong'),
        (r'\bnoong\b',  'noong'),
        (r'\bnang\b',   'nang'),
        (r'\bsalarin\b','salarin'),
        (r'\byao\b',    'yao'),
        (r'\bmarahil\b','marahil'),
        # Broad fallbacks
        (r'ic\b', 'ik'),         
        (r'll', 'y'),            
        (r'\b([bcdfghjklmnpqrstvwxyz])\1', r'\1') # Remove double consonants
    ]
    
    for pattern, replacement in ORTHO_MAP:
        text_lower = re.sub(pattern, replacement, text_lower)
        
    # 4. Strip Diacritics (Accents)
    text_lower = ''.join(c for c in unicodedata.normalize('NFD', text_lower) if unicodedata.category(c) != 'Mn')
    
    return text_lower.capitalize()

if __name__ == "__main__":
    print("Cleaning Noli Me Tangere...")
    noli_en_clean = clean_gutenberg_text('noli_english_raw.txt')
    noli_tl_clean = clean_gutenberg_text('noli_tagalog_raw.txt')

    print("Cleaning El Filibusterismo...")
    fili_en_clean = clean_gutenberg_text('elfili_english_raw.txt')
    fili_tl_clean = clean_gutenberg_text('elfili_tagalog_raw.txt')

    with open('noli_english_clean.txt', 'w', encoding='utf-8') as f:
        f.write(noli_en_clean)
    with open('noli_tagalog_clean.txt', 'w', encoding='utf-8') as f:
        f.write(noli_tl_clean)
        
    with open('elfili_english_clean.txt', 'w', encoding='utf-8') as f:
        f.write(fili_en_clean)
    with open('elfili_tagalog_clean.txt', 'w', encoding='utf-8') as f:
        f.write(fili_tl_clean)
        
    print("All files cleaned and ready for sentence alignment!")