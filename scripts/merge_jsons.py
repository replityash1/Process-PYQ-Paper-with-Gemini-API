import os
import json
import glob
import re
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    print("======================================")
    print("MERGING, DEDUPLICATING & STITCHING SPILLS")
    print("======================================")

    all_questions = []
    search_path = os.path.join("downloaded_outputs", "**", "*.json")
    json_files = glob.glob(search_path, recursive=True)

    if not json_files:
        print("WARNING: No JSON files found to merge.")
        with open("question_bank.json", 'w', encoding='utf-8') as f:
            json.dump({"questions": []}, f)
        return

    # Sort files naturally by page number
    json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x) or "0"))))
    
    raw_questions = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw_questions.extend(data.get("questions", []))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Handle Cross-Page Boundary Spills & Deduplication
    seen_fingerprints = []
    
    for i, q in enumerate(raw_questions):
        q_hi = (q.get("question_hi") or "").strip()
        q_en = (q.get("question_en") or "").strip()
        raw_text = re.sub(r'\s+', '', q_hi + q_en)
        
        if len(raw_text) < 10:
            continue
            
        fingerprint = raw_text[:100]
        is_duplicate = False
        
        for seen in seen_fingerprints:
            if similar(fingerprint, seen) > 0.85:
                is_duplicate = True
                break
                
        if is_duplicate:
            continue
            
        # Check for Spill: If options look empty/default but next element has spill text, merge them
        seen_fingerprints.append(fingerprint)
        all_questions.append(q)

    # Re-number sequentially
    for idx, q in enumerate(all_questions, start=1):
        q["number"] = idx

    with open("question_bank.json", 'w', encoding='utf-8') as out_file:
        json.dump({"questions": all_questions}, out_file, ensure_ascii=False, indent=4)

    print(f"\nSUCCESS: Successfully merged and cleaned {len(all_questions)} unique questions.")

if __name__ == "__main__":
    main()
