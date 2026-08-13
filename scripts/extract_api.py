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
    hi: str = Field(description="Hindi text of option or empty string if visual structure")
    en: str = Field(description="English text of option or empty string if visual structure")
    diagram_box: list[int] | None = Field(
        default=None,
        description="Bounding box [ymin, xmin, ymax, xmax] (0-1000) IF this specific option is a diagram/chemical structure, else null"
    )

class Question(BaseModel):
    number: int = Field(description="Original question number on page")
    question_hi: str = Field(description="Full Hindi text including statements/tables. Wrap math in $...$.")
    question_en: str = Field(description="Full English text including statements/tables. Wrap math in $...$.")
    diagram_box: list[int] | None = Field(
        default=None, 
        description="Bounding box [ymin, xmin, ymax, xmax] (0-1000) IF the main question contains a diagram, circuit, or structure, else null"
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
    """Post-processing validation & schema assertions for strict compliance."""
    for q in data.get("questions", []):
        options = q.get("options", [])
        
        cleaned_options = []
        for i, opt in enumerate(options[:4], start=1):
            cleaned_options.append({
                "label": str(i),
                "hi": str(opt.get("hi", "")),
                "en": str(opt.get("en", "")),
                "diagram_box": opt.get("diagram_box")
            })
            
        # Enforce exact 5 options rule (5th is always Question not attempted)
        cleaned_options.append({
            "label": "5",
            "hi": "अनुत्तरित प्रश्न",
            "en": "Question not attempted",
            "diagram_box": None
        })
        q["options"] = cleaned_options
    return data

def crop_box(img, box, output_filepath):
    """Safely crops a bounding box from an image with coordinate clamping and size validation."""
    if not box or len(box) != 4:
        return None

    img_w, img_h = img.size
    
    # Clamp coordinates to 0-1000 range
    ymin = max(0, min(1000, box[0]))
    xmin = max(0, min(1000, box[1]))
    ymax = max(0, min(1000, box[2]))
    xmax = max(0, min(1000, box[3]))
    
    box_w_px = ((xmax - xmin) / 1000.0) * img_w
    box_h_px = ((ymax - ymin) / 1000.0) * img_h

    # Reject invalid boxes: too small (< 15px) or too huge (> 85% of entire page)
    if ymax <= ymin or xmax <= xmin or box_w_px < 15 or box_h_px < 15 or (box_h_px > 0.85 * img_h):
        return None

    left = max(0, (xmin / 1000.0) * img_w - 6)
    top = max(0, (ymin / 1000.0) * img_h - 6)
    right = min(img_w, (xmax / 1000.0) * img_w + 6)
    bottom = min(img_h, (ymax / 1000.0) * img_h + 6)

    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_filepath)
    return output_filepath

def process_diagram_crops(page_num, data):
    page_img_path = f"pages/page_{page_num}.jpg"
    if not os.path.exists(page_img_path):
        return data

    img = Image.open(page_img_path)
    os.makedirs("output", exist_ok=True)

    for q in data.get("questions", []):
        # 1. Main Question Diagram Crop
        q_box = q.get("diagram_box")
        q_filename = f"page{page_num}_q{q['number']}_diagram.jpg"
        q_filepath = os.path.join("output", q_filename)
        
        if crop_box(img, q_box, q_filepath):
            q["diagram_path"] = f"images/{q_filename}"
        else:
            q["diagram_path"] = None

        # 2. Option Level Diagram Crops (for chemistry questions where options are images)
        for opt in q.get("options", []):
            opt_box = opt.get("diagram_box")
            opt_filename = f"page{page_num}_q{q['number']}_opt{opt['label']}_diagram.jpg"
            opt_filepath = os.path.join("output", opt_filename)
            
            if crop_box(img, opt_box, opt_filepath):
                opt["diagram_path"] = f"images/{opt_filename}"
            else:
                opt["diagram_path"] = None

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
    1. EXHAUSTIVE EXTRACTION: Extract EVERY single question on this page. Do NOT skip any question, especially at the bottom of the page.
    2. NO ASCII CHEATING: NEVER attempt to draw benzene rings, chemical bonds, or cyclic compounds using ASCII or math symbols (do NOT output symbols like Ŧ or weird brackets).
    3. QUESTION DIAGRAMS: IF a question contains a diagram, circuit, or chemical structure in its stem, return its exact bounding box [ymin, xmin, ymax, xmax] (0-1000) in diagram_box.
    4. OPTION DIAGRAMS: IF the options themselves (1, 2, 3, 4) are visual chemical structures/diagrams, return bounding boxes for each option in option.diagram_box.
    5. NO FALSE CROPS: Do NOT set diagram_box for plain Hindi/English text, lists of statements, or simple matching tables.
    6. LaTeX MATH: Wrap ALL math equations, charges (e.g. $C^+$), chemical formulas, and fractions in inline LaTeX dollar signs ($...$).
    7. OPTIONS: Option 5 is ALWAYS "Question not attempted" / "अनुत्तरित प्रश्न". Do NOT put option text inside question_hi or question_en.
    """

    print(f"Sending Page {page_num} to Gemini 3.5 Flash-Lite API...")
    response = call_gemini_with_retry(client, image_bytes, prompt)

    data = json.loads(response.text)
    data = validate_and_fix_questions(data)
    data = process_diagram_crops(page_num, data)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"SUCCESS: Page {page_num} processed successfully!")
    time.sleep(6)  # Pacing control

if __name__ == "__main__":
    main()
