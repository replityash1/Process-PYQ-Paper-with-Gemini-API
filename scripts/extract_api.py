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
    hi: str = Field(description="Hindi text of option")
    en: str = Field(description="English text of option")

class Question(BaseModel):
    number: int = Field(description="Original question number on page")
    question_hi: str = Field(description="Full Hindi question text. Wrap math in $...$.")
    question_en: str = Field(description="Full English question text. Wrap math in $...$.")
    has_visuals: bool = Field(description="True ONLY IF the question contains diagrams, physics circuits, chemical structures, or complex tables.")
    box_hi: list[int] | None = Field(
        default=None, 
        description="Bounding box [ymin, xmin, ymax, xmax] (0-1000) wrapping the ENTIRE Hindi question AND its options. Null if has_visuals is False."
    )
    box_en: list[int] | None = Field(
        default=None, 
        description="Bounding box [ymin, xmin, ymax, xmax] (0-1000) wrapping the ENTIRE English question AND its options. Null if has_visuals is False."
    )
    options: list[Option] = Field(description="List of exactly 5 options")

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
    """Creates dummy options if visuals are present and enforces the 5-option rule."""
    for q in data.get("questions", []):
        options = q.get("options", [])
        has_visuals = q.get("has_visuals", False)
        
        cleaned_options = []
        for i, opt in enumerate(options[:4], start=1):
            # If the question is visual, we wipe the text to create dummy blank options
            hi_text = "" if has_visuals else str(opt.get("hi", ""))
            en_text = "" if has_visuals else str(opt.get("en", ""))
            
            cleaned_options.append({
                "label": str(i),
                "hi": hi_text,
                "en": en_text
            })
            
        # Option 5 is always strictly enforced
        cleaned_options.append({
            "label": "5",
            "hi": "अनुत्तरित प्रश्न",
            "en": "Question not attempted"
        })
        q["options"] = cleaned_options
    return data

def crop_box(img, box, output_filepath):
    """Safely crops a bounding box with strict coordinate clamping."""
    if not box or len(box) != 4:
        return False

    img_w, img_h = img.size
    
    ymin = max(0, min(1000, box[0]))
    xmin = max(0, min(1000, box[1]))
    ymax = max(0, min(1000, box[2]))
    xmax = max(0, min(1000, box[3]))
    
    box_w_px = ((xmax - xmin) / 1000.0) * img_w
    box_h_px = ((ymax - ymin) / 1000.0) * img_h

    # Reject false crops (too small or capturing the whole page)
    if ymax <= ymin or xmax <= xmin or box_w_px < 25 or box_h_px < 25 or (box_h_px > 0.90 * img_h):
        return False

    left = max(0, (xmin / 1000.0) * img_w - 10)
    top = max(0, (ymin / 1000.0) * img_h - 10)
    right = min(img_w, (xmax / 1000.0) * img_w + 10)
    bottom = min(img_h, (ymax / 1000.0) * img_h + 10)

    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_filepath)
    return True

def process_diagram_crops(page_num, data):
    page_img_path = f"pages/page_{page_num}.jpg"
    if not os.path.exists(page_img_path):
        return data

    img = Image.open(page_img_path)
    os.makedirs("output", exist_ok=True)

    for q in data.get("questions", []):
        if q.get("has_visuals"):
            # Crop Hindi Block
            if q.get("box_hi"):
                hi_filename = f"page{page_num}_q{q['number']}_hi.jpg"
                if crop_box(img, q.get("box_hi"), os.path.join("output", hi_filename)):
                    q["image_hi"] = f"images/{hi_filename}"
            
            # Crop English Block
            if q.get("box_en"):
                en_filename = f"page{page_num}_q{q['number']}_en.jpg"
                if crop_box(img, q.get("box_en"), os.path.join("output", en_filename)):
                    q["image_en"] = f"images/{en_filename}"

        # Clean up coordinates from JSON output to save space
        q.pop("box_hi", None)
        q.pop("box_en", None)

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
    Extract ALL multiple-choice questions from this RPSC exam paper page into structured JSON.
    
    STRICT EXTRACTION RULES:
    1. EXHAUSTIVE: Extract EVERY single question on this page. Do NOT skip bottom questions.
    2. VISUAL QUESTIONS: If a question contains ANY diagram, circuit, or chemical structure, set `has_visuals` to true.
    3. THE FULL BLOCK CROP: If `has_visuals` is true, draw a bounding box around the ENTIRE question block (The question text AND all its options) for Hindi in `box_hi` and for English in `box_en`.
    4. NEVER attempt to draw benzene rings, bonds, or circuits using ASCII symbols (no Ŧ or geometric shapes). 
    5. Options must always be 5.
    """

    print(f"Sending Page {page_num} to Gemini 3.5 Flash-Lite API...")
    response = call_gemini_with_retry(client, image_bytes, prompt)

    data = json.loads(response.text)
    data = validate_and_fix_questions(data)
    data = process_diagram_crops(page_num, data)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"SUCCESS: Page {page_num} processed.")
    time.sleep(6)  # Pacing for 15 RPM limit with 3 parallel workers

if __name__ == "__main__":
    main()
