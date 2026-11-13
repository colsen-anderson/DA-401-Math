import os
import time
import json
import re
from pathlib import Path
from google import genai
from google.genai import types, errors

# ============================================
# API SETUP
# ============================================
try:
    import gemini_api_key

    API_KEY = gemini_api_key.get_api_key()
except (ImportError, AttributeError):
    API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in ~/.zshrc or gemini_api_key.py.")

client = genai.Client(api_key=API_KEY)

# ============================================
# CONFIGURATION
# ============================================
MODEL_NAME = "gemini-3.6-flash"
MAX_RETRIES = 5
TEMPERATURE = 0.2


# ============================================
# HELPER FUNCTIONS
# ============================================
def load_json(filepath: str):
    """Load JSON file safely."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, filepath: str) -> None:
    """Save formatted JSON output."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 Successfully saved {len(data)} generated problems to: {filepath}")


def clean_json_response(raw_text: str) -> list:
    """Safely extract and parse a JSON array from raw model text."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start:end + 1].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)

    return json.loads(raw_text)


# ============================================
# CORE GENERATION FUNCTION
# ============================================
def generate_similar_practice_problems(
        input_problems: list,
        standards_data: list,
        model_name: str = MODEL_NAME,
        max_retries: int = MAX_RETRIES
) -> list:
    """
    Takes textbook problems + official standards list, generates 1-to-1 parallel
    problems, and maps each problem to the correct SC standard and indicator.
    """
    total_problems = len(input_problems)
    input_formatted = json.dumps(input_problems, indent=2, ensure_ascii=False)
    standards_formatted = json.dumps(standards_data, indent=2, ensure_ascii=False)

    prompt = f"""You are an expert mathematics curriculum specialist and assessment designer for South Carolina High School Algebra 1.

### MISSION:
You are provided with:
1. An official JSON list of South Carolina Algebra 1 standards and indicators.
2. A JSON list containing exactly {total_problems} original algebra practice problems.

Generate an EXACT 1-to-1 parallel JSON list containing {total_problems} BRAND NEW, original practice problems mapped accurately against the provided SC standards.

### OFFICIAL SC STANDARDS LIST:
{standards_formatted}

### SOURCE PROBLEMS TO REPLICATE (JSON):
{input_formatted}

### STRICT GENERATION & MAPPING RULES:
1. **EXACT 1-TO-1 COUNT**: Produce exactly {total_problems} problems numbered 1 through {total_problems}.
2. **PARALLEL DIFFICULTY & CONTEXT**:
   - Each new problem must test the same mathematical concept with new coefficients/variables.
   - For grouped problems sharing context (e.g., functions $f(x)$ and $g(x)$ for questions 6–10), define fresh functions in `problem_context` and evaluate the questions against those functions.
3. **ACCURATE SOLUTIONS & LATEX**:
   - Compute exact simplified answers for `solution_text`.
   - Format all math expressions with standard LaTeX notation (e.g., `8x^{{2}} + 5x - 9`).
4. **STANDARDS ALIGNMENT**:
   - Map each generated problem to its most precise indicator from the provided standards JSON.
   - Set `"mapped_standard"` to the exact indicator code (e.g., `"A1.PAFR.1.4"` or `"A1.PAFR.3.1"`).
   - Set `"extraction_metadata"."indicator_alignment"` to the exact `"indicator_description"` from the standards JSON matching that code.
   - Set `"extraction_metadata"."source_url"` to `"Generated from SC <mapped_standard>"` (e.g., `"Generated from SC A1.PAFR.1.4"`).
5. **SCHEMA FORMAT**:
   - `"book"`: `"SC Standards-Aligned Practice"`
   - `"unit"`: Keep from source item
   - `"lesson"`: Keep from source item
   - `"block_code"`: Append `"-generated"` (e.g., `"6.1.6-generated"`)
   - `"section_title"`: Keep from source item
   - `"problem_number"`: Keep from source item
   - `"problem_context"`: Parallel context/instructions
   - `"problem_text"`: New problem text
   - `"solution_text"`: Exact answer

### OUTPUT FORMAT:
Return ONLY a valid JSON list of {total_problems} dictionaries adhering to the schema.
"""

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        max_output_tokens=8192,
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🤖 Sending request to {model_name} (Attempt {attempt}/{max_retries})...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )

            raw_text = getattr(response, "text", "") or ""
            parsed_data = clean_json_response(raw_text)

            if isinstance(parsed_data, list) and len(parsed_data) == total_problems:
                print(f"✓ Successfully generated and validated {len(parsed_data)} parallel problems.")
                return parsed_data
            elif isinstance(parsed_data, list):
                print(f"⚠️ Model returned {len(parsed_data)} items instead of {total_problems}. Retrying...")
            else:
                print(f"⚠️ Response was not a JSON list. Retrying...")

        except errors.ClientError as e:
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait_time = 2 ** attempt
                print(f"⚠️ Service 503. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            print(f"⚠️ Error during processing: {e}")
            if attempt < max_retries:
                time.sleep(2)

    raise RuntimeError("❌ Failed to generate the expected problem set after maximum retry attempts.")


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # File paths
    standards_file = "algebra_1_standards_most.json"
    input_file = "section_6_1_6_problems.json"
    output_file = "generated_section_6_1_6_problems.json"

    print("=" * 60)
    print("🎯 STARTING 1-TO-1 SIMILAR PROBLEM GENERATION WITH STANDARDS")
    print("=" * 60)

    # 1. Load Standards and Input Problems
    try:
        standards_data = load_json(standards_file)
        print(f"📋 Loaded SC standards from '{standards_file}'")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{standards_file}'. Please place it in the same directory.")
        return

    try:
        source_problems = load_json(input_file)
        print(f"📂 Loaded {len(source_problems)} source problems from '{input_file}'")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{input_file}'. Please place it in the same directory.")
        return

    # 2. Generate new problems mapped to standards
    generated_problems = generate_similar_practice_problems(
        input_problems=source_problems,
        standards_data=standards_data,
        model_name=MODEL_NAME
    )

    # 3. Save output
    save_json(generated_problems, output_file)

    print("=" * 60)
    print("✅ PROCESS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()