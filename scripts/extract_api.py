import os
import sys
import json
import time
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

class Option(BaseModel):
    label: str = Field(description="Option label: 1, 2, 3, 4, or 5")
    hi: str = Field(description="Hindi text of option or empty string")
    en: str = Field(description="English text of option or empty string")

class Question(BaseModel):
    number: int = Field(description="Original question number on page")
    question_hi: str = Field(description="Full Hindi text including statements/tables. Wrap math in $...$.")
    question_en: str = Field(description="Full English text including statements/tables. Wrap math in $...$.")
    diagram_box: list[int] | None = Field(
        default=None, 
        description="Bounding box coordinates [ymin, xmin, ymax, xmax] scaled 0 to 1000 if a diagram/chemical structure is present, else null"
    )
    options: list[Option] = Field(description="List of options")

class QuestionBank(BaseModel):
    questions: list[Question]

@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=4, min=10, max=60))
def call_gemini_with_retry(client, image_bytes, prompt):
    return client.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QuestionBank,
            temperature=0.0
        ),
    )

def validate_and_fix_questions(data):
    """Post-processing validation & schema assertions to guarantee strict 5-option compliance."""
    for q in data.get("questions", []):
        options = q.get("options", [])
        
        cleaned_options = []
        for i, opt in enumerate(options[:4], start=1):
            cleaned_options.append({
                "label": str(i),
                "hi": str(opt.get("hi", "")),
                "en": str(opt.get("en", ""))
            })
            
        # Enforce exact 5 options rule (5th is always Question not attempted)
        cleaned_options.append({
            "label": "5",
            "hi": "अनुत्तरित प्रश्न",
            "en": "Question not attempted"
        })
        q["options"] = cleaned_options
    return data

def crop_diagrams(page_num, data, output_json_path):
    page_img_path = f"pages/page_{page_num}.jpg"
    if not os.path.exists(page_img_path):
        return data

    img = Image.open(page_img_path)
    img_w, img_h = img.size
    os.makedirs("images", exist_ok=True)

    for q in data.get("questions", []):
        box = q.get("diagram_box")
        if box and len(box) == 4:
            ymin, xmin, ymax, xmax = box
            
            left = max(0, (xmin / 1000.0) * img_w - 8)
            top = max(0, (ymin / 1000.0) * img_h - 8)
            right = min(img_w, (xmax / 1000.0) * img_w + 8)
            bottom = min(img_h, (ymax / 1000.0) * img_h + 8)

            cropped = img.crop((left, top, right, bottom))
            diag_path = f"images/page{page_num}_q{q['number']}_diagram.jpg"
            cropped.save(diag_path)
            
            q["diagram_path"] = diag_path
        else:
            q["diagram_path"] = None

    return data

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
        
    page_num = sys.argv[1]
    image_path = f"pages/page_{page_num}.jpg"
    output_json_path = f"output/page_{page_num}.json"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    prompt = """
    Extract all multiple-choice questions from this RPSC exam paper page into a structured format.
    1. Each question MUST have exactly 5 options. Option 5 is ALWAYS "Question not attempted" / "अनुत्तरित प्रश्न".
    2. Do NOT use placeholder text like "Option 1". Extract the actual text from the image.
    3. Put all premises, match-columns, and statements inside question_hi and question_en using line breaks (\n).
    4. Wrap ALL math, chemical formulas, and fractions in inline LaTeX dollar signs (e.g., $\\frac{w_1 + w_2}{2}$).
    5. If a question contains a diagram or chemical structure, return its bounding box coordinates [ymin, xmin, ymax, xmax] scaled 0-1000 in diagram_box.
    """

    print(f"Sending Page {page_num} to Gemini 2.5 Flash API...")
    response = call_gemini_with_retry(client, image_bytes, prompt)

    data = json.loads(response.text)
    data = validate_and_fix_questions(data)
    data = crop_diagrams(page_num, data, output_json_path)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"SUCCESS: Page {page_num} processed successfully!")
    time.sleep(10)  # Pacing control to prevent rate limit spikes

if __name__ == "__main__":
    main()
