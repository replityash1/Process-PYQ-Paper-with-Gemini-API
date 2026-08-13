import os
import json
import glob
import re
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    print("======================================")
    print("MERGING AND CLEANING JSON ARTIFACTS")
    print("======================================")

    all_questions = []
    search_path = os.path.join("downloaded_outputs", "**", "*.json")
    json_files = glob.glob(search_path, recursive=True)

    if not json_files:
        print("WARNING: No JSON files found to merge.")
        with open("question_bank.json", 'w', encoding='utf-8') as f:
            json.dump({"questions": []}, f)
        return

    json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x) or "0"))))
    seen_question_texts = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                questions = data.get("questions", [])
                
                for q in questions:
                    q_hi = (q.get("question_hi") or "").strip()
                    q_en = (q.get("question_en") or "").strip()
                    
                    raw_text = re.sub(r'\s+', '', q_hi + q_en)
                    if len(raw_text) < 10:
                        continue 
                    
                    fingerprint = raw_text[:100]
                    is_duplicate = False
                    
                    for seen in seen_question_texts:
                        if similar(fingerprint, seen) > 0.85:
                            is_duplicate = True
                            break
                            
                    if is_duplicate:
                        continue
                    
                    seen_question_texts.append(fingerprint)
                    all_questions.append(q)
        except Exception as e:
            print(f"Error skipping corrupted file {file_path}: {e}")

    for new_idx, q in enumerate(all_questions, start=1):
        q["number"] = new_idx

    with open("question_bank.json", 'w', encoding='utf-8') as out_file:
        json.dump({"questions": all_questions}, out_file, ensure_ascii=False, indent=4)

    print(f"\nSUCCESS: Combined {len(all_questions)} unique questions.")

if __name__ == "__main__":
    main()
