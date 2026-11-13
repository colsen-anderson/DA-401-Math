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
TEMPERATURE = 0.3


# ============================================
# HELPER FUNCTIONS
# ============================================
def load_json(filepath: str) -> list:
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
    # Strip markdown code blocks if present
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    # Locate array bounds
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start:end + 1].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Clean trailing commas
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)

    return json.loads(raw_text)


# ============================================
# CORE GENERATION FUNCTION
# ============================================
def generate_similar_practice_problems(
        input_problems: list,
        model_name: str = MODEL_NAME,
        max_retries: int = MAX_RETRIES
) -> list:
    """
    Takes a list of textbook problems and generates a 1-to-1 parallel list
    of new problems matching the SC Standards-Aligned Practice schema.
    """
    total_problems = len(input_problems)
    input_formatted = json.dumps(input_problems, indent=2, ensure_ascii=False)

    prompt = f"""You are an expert mathematics curriculum specialist and assessment designer for South Carolina High School Algebra 1.

### MISSION:
You are provided with a JSON list containing exactly {total_problems} original algebra practice problems.
Generate an EXACT 1-to-1 parallel JSON list containing {total_problems} BRAND NEW, original practice problems.

### STRICT GENERATION RULES:
1. **EXACT 1-TO-1 COUNT**: You must produce exactly {total_problems} problems matching the original problem numbers (1 through {total_problems}).
2. **PARALLEL STRUCTURE & DIFFICULTY**:
   - Each new problem must test the exact same mathematical skill, concept, and structure as its source problem, but with different coefficients, variables, constants, or algebraic forms.
   - If a group of problems shares a context (e.g., questions 6–10 share two functions $f(x)$ and $g(x)$), create a new consistent set of shared functions in `problem_context` and evaluate the new problems against those new functions.
3. **ACCURATE SOLUTIONS**: Solve each generated problem step-by-step internally to ensure `solution_text` is mathematically correct and in simplest form.
4. **LATEX NOTATION**: Format all mathematical expressions using valid LaTeX syntax (e.g., `5m^{{2}} - 7m + 22`, `\\frac{{1}}{{2}}`, etc.).
5. **METADATA & SCHEMA CONVENTIONS**:
   - `"book"`: Must be set to `"SC Standards-Aligned Practice"`
   - `"unit"`: Keep identical to source problem.
   - `"lesson"`: Keep identical to source problem.
   - `"block_code"`: Append `"-generated"` (e.g., `"6.1.6"` -> `"6.1.6-generated"`).
   - `"section_title"`: Keep identical to source problem (e.g., `"Practice"`).
   - `"problem_number"`: Keep identical to source problem (e.g., `"1"`, `"2"`, etc.).
   - `"problem_context"`: Parallel to source instructions/context.
   - `"problem_text"`: The new problem statement.
   - `"solution_text"`: The exact, simplified answer to the new problem.
   - `"mapped_standard"`: Keep the exact standard code from the source (e.g., `"A1.PAFR.1.4"`).
   - `"extraction_metadata"`:
       - `"source_url"`: Set to `"Generated from SC <mapped_standard>"` (e.g., `"Generated from SC A1.PAFR.1.4"`).
       - `"indicator_alignment"`: Keep the exact indicator alignment text from the source item.

### SOURCE PROBLEMS TO REPLICATE (JSON):
{input_formatted}

### OUTPUT REQUIREMENTS:
Return ONLY a valid JSON list of {total_problems} dictionaries adhering strictly to the required schema. No conversational prose or explanations.
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
                print(f"✓ Successfully generated and verified {len(parsed_data)} parallel problems.")
                return parsed_data
            elif isinstance(parsed_data, list):
                print(f"⚠️ Warning: Model returned {len(parsed_data)} items instead of {total_problems}. Retrying...")
            else:
                print(f"⚠️ Warning: Response was not a JSON list. Retrying...")

        except errors.ClientError as e:
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait_time = 2 ** attempt
                print(f"⚠️ Service temporarily unavailable (503). Retrying in {wait_time}s...")
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
    # File paths (adjust as needed)
    input_file = "section_6_1_6_problems.json"
    output_file = "generated_section_6_1_6_problems.json"

    print("=" * 60)
    print("🎯 STARTING 1-TO-1 SIMILAR PROBLEM GENERATION")
    print("=" * 60)

    # 1. Load input problems
    try:
        source_problems = load_json(input_file)
        print(f"📂 Loaded {len(source_problems)} source problems from '{input_file}'")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{input_file}'.")
        print("   Please make sure the source JSON is located in the working directory.")
        return

    # 2. Generate new problems
    generated_problems = generate_similar_practice_problems(
        input_problems=source_problems,
        model_name=MODEL_NAME
    )

    # 3. Save output
    save_json(generated_problems, output_file)

    print("=" * 60)
    print("✅ PROCESS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()