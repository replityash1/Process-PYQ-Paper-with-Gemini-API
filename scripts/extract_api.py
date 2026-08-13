import os
import sys
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# 1. Enforce Structured JSON Output with Pydantic
class Option(BaseModel):
    label: str = Field(description="Option label like 1, 2, 3, 4, or 5")
    hi: str = Field(description="Actual Hindi text of option or empty string if absent")
    en: str = Field(description="Actual English text of option or empty string if absent")

class Question(BaseModel):
    number: int = Field(description="Original question number on page")
    question_hi: str = Field(description="Full Hindi question text including statements/tables. Wrap ALL math in $...$.")
    question_en: str = Field(description="Full English question text including statements/tables. Wrap ALL math in $...$.")
    options: list[Option] = Field(description="List of exactly 5 options where option 5 is Question not attempted")

class QuestionBank(BaseModel):
    questions: list[Question]

# 2. Add Automatic Retry Logic for 429 Rate Limits
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=30))
def call_gemini_with_retry(client, image_bytes, prompt):
    return client.models.generate_content(
        model='gemini-3.6-flash',
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

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
        
    page_num = sys.argv[1]
    image_path = f"pages/page_{page_num}.jpg"
    output_json_path = f"output/page_{page_num}.json"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY secret is missing.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    prompt = """
    Extract all multiple-choice questions from this RPSC exam paper page into a structured format.
    
    EXTRACTION RULES:
    1. Each question MUST have exactly 5 options. Option 5 is ALWAYS "Question not attempted" / "अनुत्तरित प्रश्न".
    2. Do NOT use placeholder text like "Option 1" or "विकल्प 1". You MUST extract the actual text of each option from the image.
    3. Put all premises, match-columns, and statements (I, II, III or A, B, C) inside the main question_hi and question_en text fields using line breaks (\n).
    4. Wrap ALL math, equations, variables, physics values, and fractions in inline LaTeX dollar signs (e.g., $\\frac{w_1 + w_2}{2}$ or $x^2$).
    5. Do NOT translate single-language questions; leave the absent language field as "".
    6. If a page contains no numbered questions, return an empty list under "questions".
    """

    print(f"Sending Page {page_num} to Gemini API...")
    response = call_gemini_with_retry(client, image_bytes, prompt)

    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    print(f"SUCCESS: Page {page_num} extracted and saved!")

if __name__ == "__main__":
    main()
