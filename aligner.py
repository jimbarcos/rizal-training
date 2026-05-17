import spacy
import torch
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from preprocess import normalize_historical_tagalog

nlp = spacy.load("xx_sent_ud_sm")
nlp.max_length = 3000000  # Increase the character limit to 3 million

def clean_text(text):
    """Normalizes special characters, punctuation, whitespace, and comprehensive mojibake."""
    
    # 1. Comprehensive Mojibake Dictionary (UTF-8 decoded as Windows-1252)
    mojibake_fixes = {
        'Â¿': '¿', 'Â¡': '¡', 'â€™': "'", 'â€œ': '"', 'â€': '"',
        'â€”': ' - ', 'â€“': ' - ', 'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í',
        'Ã-': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã¢': 'â', 'Ãª': 'ê',
        'Ã®': 'î', 'Ã´': 'ô', 'Ã»': 'û', 'Ã ': 'à', 'Ã¨': 'è',
        'Ã¬': 'ì', 'Ã²': 'ò', 'Ã¹': 'ù', 'Ã±': 'ñ', 'Ã‘': 'Ñ',
        'ngÃ®f': 'ng̃', 'm~ga': 'mga',
    }
    
    for broken, fixed in mojibake_fixes.items():
        text = text.replace(broken, fixed)
        
    # 2. Normalize standard smart quotes to straight quotes
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    
    # 3. Pad remaining standard em-dashes and en-dashes with spaces
    text = re.sub(r'[—–]', ' - ', text)
    
    # 4. Replace multiple spaces, tabs, or newlines with a single standard space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def segment_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    cleaned_text = clean_text(raw_text)
    doc = nlp(cleaned_text)
    
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

def calculate_difficulty(tl_text):
    words = re.findall(r'\b\w+\b', tl_text)
    word_count = len(words)
    
    if word_count <= 8:
        return "Easy"
    elif word_count <= 18:
        return "Medium"
    else:
        return "Hard"

def process_and_align(tl_file, en_file, model, source_label):
    """Processes a pair of cleaned files and returns a list of aligned dictionaries."""
    print(f"\n--- Processing {source_label} ---")
    
    tagalog_sentences = segment_text(tl_file)
    english_sentences = segment_text(en_file)
    
    tagalog_normalized = [normalize_historical_tagalog(sent) for sent in tagalog_sentences]
    
    print(f"Encoding {len(tagalog_normalized)} Tagalog sentences...")
    embeddings_tl = model.encode(tagalog_normalized, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    
    print(f"Encoding {len(english_sentences)} English sentences...")
    embeddings_en = model.encode(english_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    
    print("Computing alignments...")
    cosine_scores = util.cos_sim(embeddings_tl, embeddings_en).cpu().numpy()
    
    aligned_data = []
    SIMILARITY_THRESHOLD = 0.65
    WINDOW_SIZE = 2000
    
    ratio = len(english_sentences) / max(1, len(tagalog_normalized))
    
    potential_matches = []
    for i in range(len(tagalog_normalized)):
        expected_j = int(i * ratio)
        start_j = max(0, expected_j - WINDOW_SIZE)
        end_j = min(len(english_sentences), expected_j + WINDOW_SIZE)
        
        for j in range(start_j, end_j):
            score = cosine_scores[i, j]
            if score >= SIMILARITY_THRESHOLD:
                potential_matches.append((score, i, j))
                
    potential_matches.sort(key=lambda x: x[0], reverse=True)
    
    used_tl_indices = set()
    used_en_indices = set()
    
    for score, i, j in potential_matches:
        if i not in used_tl_indices and j not in used_en_indices:
            used_tl_indices.add(i)
            used_en_indices.add(j)
            
            # THE FIX: lstrip("-=+@ ") removes leading dashes and equals signs 
            # that cause Excel to break, without ruining the rest of the sentence!
            aligned_data.append({
                "original_tagalog": tagalog_sentences[i].lstrip("-=+@ "), 
                "normalized_tagalog": tagalog_normalized[i].lstrip("-=+@ "),
                "english": english_sentences[j].lstrip("-=+@ "),
                "difficulty": calculate_difficulty(tagalog_sentences[i]),
                "similarity_score": round(float(score), 4),
                "source": source_label
            })
            
    print(f"Found {len(aligned_data)} matches for {source_label}.")
    return aligned_data

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    noli_aligned = process_and_align('noli_tagalog_raw.txt', 'noli_english_raw.txt', model, 'Noli Me Tangere')
    fili_aligned = process_and_align('elfili_tagalog_raw.txt', 'elfili_english_raw.txt', model, 'El Filibusterismo')

    all_data = noli_aligned + fili_aligned
    df = pd.DataFrame(all_data)
    
    df = df.sort_values(by="similarity_score", ascending=False) 
    df.to_csv("rizal_aligned_dataset.csv", index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully aligned and combined {len(df)} total sentence pairs into 'rizal_aligned_dataset.csv'!")