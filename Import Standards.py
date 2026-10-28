import os
import time
import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2
from google import genai
from google.genai import types, errors

# -------------------------
# CONFIG
# -------------------------
# Set these paths to your actual files
EXAMPLES_JSON_PATH = r"C:\Users\ljcau\Downloads\DA 401\Math Project\algebra_1_standards_most.json"
STANDARDS_PDF_PATH = r"C:\Users\ljcau\Downloads\DA 401\Math Project\Algebra1 Standards.pdf"
OUTPUT_JSON_PATH = r"C:\Users\ljcau\Downloads\DA 401\Math Project\algebra_1_standards_complete.json"

# Use environment variable for API key: set GOOGLE_GENAI_API_KEY in your system/user env
## API_KEY = insert api key here

MODEL_NAME = "gemini-3.8-flash"
MAX_RETRIES = 5
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 8192

# -------------------------
# IO HELPERS
# -------------------------
def read_json(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"⚠️ JSON not found at {path}. Returning empty list.")
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: Any, path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(data) if isinstance(data, list) else 'data'} entries to {path}")

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            parts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                parts.append(f"\n--- Page {i+1} ---\n{page_text}")
            return "".join(parts).strip()
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

# -------------------------
# DEDUP / SCHEMA HELPERS
# -------------------------
POSSIBLE_ID_KEYS = [
    "code", "standard_code", "standard", "id", "identifier", "number", "slug"
]

def best_key(d: Dict[str, Any]) -> Optional[str]:
    # Try preferred keys first
    for k in POSSIBLE_ID_KEYS:
        if k in d and isinstance(d[k], str) and d[k].strip():
            return k
    # Fallback: combine some text fields to form a hashable identity
    for alt in ["title", "name", "description", "statement"]:
        if alt in d and isinstance(d[alt], str) and d[alt].strip():
            return alt
    return None

def item_identity(d: Dict[str, Any]) -> str:
    # Build a robust identity string for dedupe
    k = best_key(d)
    if k:
        return f"{k}::{d.get(k, '')}".strip().lower()
    # fallback: hash subset of fields
    fields = []
    for f in ["domain", "cluster", "strand", "category", "title", "description", "statement"]:
        v = d.get(f, "")
        if isinstance(v, str) and v.strip():
            fields.append(v.strip().lower())
    return "|".join(fields) or json.dumps(d, sort_keys=True)

def merge_dedupe(originals: List[Dict[str, Any]], additions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {}
    for item in originals:
        seen[item_identity(item)] = item
    for item in additions:
        seen[item_identity(item)] = item
    return list(seen.values())

# -------------------------
# PROMPT
# -------------------------
def build_prompt(standards_text: str, examples: List[Dict[str, Any]]) -> str:
    examples_formatted = json.dumps(examples, ensure_ascii=False, indent=2)
    prompt = f"""
You are an expert curriculum specialist and instructional designer for South Carolina Algebra 1.

GOAL:
From the SC Algebra 1 standards text below, extract all Algebra 1 standards and produce ONLY the standards that are NOT already present in the examples. Follow the EXACT JSON schema used by the examples (same keys, casing, and nesting).

STRICT REQUIREMENTS:
- OUTPUT: A single JSON array only. No prose, no markdown, no comments, no trailing text.
- Schema: Exactly match the keys and structure in the examples you were given.
- Content: Return only genuinely missing standards (avoid duplicates of any example entries).
- Granularity: Keep each standard at the same granularity as examples (no combining or splitting).
- Values: Accurately copy official codes/labels/numbers if present in the PDF. Do not invent IDs.
- If NOTHING is missing, return [].

EXAMPLES (SCHEMA TO FOLLOW):
{examples_formatted}

SC ALGEBRA 1 STANDARDS (SOURCE TEXT):
\"\"\"{standards_text[:120000]}\"\"\"  # truncated if extremely long
""".strip()
    return prompt

# -------------------------
# GENAI CALL
# -------------------------
def call_gemini_for_missing(examples: List[Dict[str, Any]], standards_text: str) -> List[Dict[str, Any]]:
    if not API_KEY:
        raise RuntimeError("Missing GOOGLE_GENAI_API_KEY environment variable. Please set it before running.")

    client = genai.Client(api_key=API_KEY)
    prompt = build_prompt(standards_text, examples)

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )
            text = getattr(resp, "text", None) or ""
            # Extract JSON array from response safely
            arr = extract_json_array(text)
            if not isinstance(arr, list):
                raise ValueError("Model did not return a JSON array.")
            # Ensure each item is dict
            clean = [x for x in arr if isinstance(x, dict)]
            return clean
        except errors.ClientError as e:
            last_err = e
            s = str(e)
            if "503" in s or "UNAVAILABLE" in s:
                if attempt < MAX_RETRIES:
                    wait = 2 ** (attempt - 1)
                    print(f"⚠️ Service unavailable. Retry {attempt}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)
                    continue
            raise
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                print(f"⚠️ Error parsing/response. Retry {attempt}/{MAX_RETRIES} in {wait}s... ({e})")
                time.sleep(wait)
                continue
            raise
    if last_err:
        raise last_err
    return []

def extract_json_array(s: str) -> Any:
    # Find the first top-level JSON array in the string
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try to clean trailing commas
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)
    # If the whole string is JSON
    try:
        return json.loads(s)
    except Exception:
        raise ValueError("No valid JSON array found in model response.")

# -------------------------
# MAIN
# -------------------------
def main():
    # 1) Load examples
    examples = read_json(EXAMPLES_JSON_PATH)
    print(f"Loaded {len(examples)} example entries.")

    # 2) Extract standards text from PDF
    print("Extracting text from standards PDF...")
    standards_text = extract_text_from_pdf(STANDARDS_PDF_PATH)
    if not standards_text:
        raise RuntimeError("No text extracted from the PDF. Check the file or consider OCR if it's scanned.")

    # 3) Get missing standards from Gemini
    print(f"Querying {MODEL_NAME} for missing standards...")
    new_items = call_gemini_for_missing(examples, standards_text)
    print(f"Model returned {len(new_items)} new items.")

    # 4) Merge and dedupe
    final = merge_dedupe(examples, new_items)
    print(f"Final merged total: {len(final)} items.")

    # 5) Save
    write_json(final, OUTPUT_JSON_PATH)

if __name__ == "__main__":
    main()