import os
from google import genai
from google.genai import types, errors
from pathlib import Path
import time
import json

#API Set up
try:
    import gemini_api_key
    API_KEY = gemini_api_key.get_api_key()
except (ImportError, AttributeError):
    API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in ~/.zshrc or gemini_api_key.py.")

client = genai.Client(api_key=API_KEY)

#Template for the AI to follow

textbook_problem_data = [
    {
        "book": "OpenStax Algebra 1",
        "unit": "5",
        "lesson": "5.2",
        "block_code": "5.2.5",
        "section_title": "Practice",
        "problem_number":"1",
        "problem_context": " -- ",
        "problem_text": "Simplify the expression: 81^{\\frac{1}{4}}",
        "solution_text": "3",
        "mapped_standard": "A1.NR.1.1",
        "extraction_metadata": {
            "source_url": (
                "https://openstax.org/books/"
                "algebra-1/pages/5-2-5-practice"
            ),
            "indicator_alignment": (
                "Rewrite numerical and algebraic expressions "
                "involving rational and irrational numbers "
                "with radicals/roots"
            )
        }
    },

    {
        "book": "OpenStax Algebra 1",
        "unit": "1",
        "lesson": "1.1",
        "block_code": "1.1.5",
        "section_title": "Practice",
        "problem_number":"2",
        "problem_context": "For questions 1 – 5, use the following scenario to identify the expression, equation, or inequality described."
                            "To support a local senior citizens center, a student club sent a flier home to the students in the school. The flier said, “Please bring in money to support the senior citizens center. Paper money and coins accepted!” Their goal is to raise dollars."
                            "Choose the expression, equation, or inequality, from the list, that describes the given quantity.",
        "problem_text": "The dollar amount the club would have if every student at the school donated 50 cents to the cause. Assume there are n students",
        "solution_text": "0.5n",
        "mapped_standard": "MPS.RC.1",
        "extraction_metadata": {
            "source_url": (
                "https://openstax.org/books/"
                "algebra-1/pages/1-1-5-practice"
            ),
            "indicator_alignment": (
                "Understand and use defintions and verbal/written information to construct arguments and prove conjectures"
            )

        }
    },

    {
        "book": "OpenStax Algebra 1",
        "unit": "2",
        "lesson": "2.1",
        "block_code": "2.1.5",
        "section_title": "Practice",
        "problem_number":"1a",
        "problem_context": " Determine if the point is a solution to the system of equations: 2x-6y=0 and 3x-4y=5 ",
        "problem_text": "(1, \\frac{3}{4})",
        "solution_text": "not a solution to the system",
        "mapped_standard": "A1.PAFR.2.1",
        "extraction_metadata": {
            "source_url": (
                "https://openstax.org/books/"
                "algebra-1/pages/2-1-5-practice"
            ),
            "indicator_alignment": (
                " Determine which form of a function is used to identify the information "
            )
        }
    },

    {
        "book": "OpenStax Algebra 1",
        "unit": "6",
        "lesson": "6.1",
        "block_code": "6.1.6",
        "section_title": "Practice",
        "problem_number":"2",
        "problem_context": "Simplify each expression completely",
        "problem_text": "(8k^2 - 3k + 14) + (8k +17)",
        "solution_text": "8k^2 +5k + 31",
        "mapped_standard": "A1.PAFR.1.4",
        "extraction_metadata": {
            "source_url": (
                "https://openstax.org/books/"
                "algebra-1/pages/6-1-6-practice"
            ),
            "indicator_alignment": (
                "Add, subtract, and multiply polynomials with initial terms up to a degree 2 "
            )
        }
    },

    {
        "book": "OpenStax Algebra 1",
        "unit": "4",
        "lesson": "4.7",
        "block_code": "4.7.5",
        "section_title": "Practice",
        "problem_number":"2",
        "problem_context": "--",
        "problem_text": "Find the slope of the line that goes through the points (0,3) and (2,4) .",
        "solution_text": "7",
        "mapped_standard": "A1.DPSR.2.1",
        "extraction_metadata": {
            "source_url": (
                "https://openstax.org/books/"
                "algebra-1/pages/4-7-5-practice"
            ),
            "indicator_alignment": (
                "Interpret the slope and the intercept of a linear model in the context of the data"
            )
        }
    },

    {
        "book": "OpenStax Algebra 1",
        "unit": "9",
        "lesson": "9.2",
        "block_code": "9.2.5",
        "section_title": "Practice",
        "problem_number":"15",
        "problem_context": "For questions 15 - 20, solve choose the equivalent"
                           " equation with a perfect square on one side.",
        "problem_text":"x^2 + 8x = 2",
        "solution_text": "(x+4)^2 = 18",
        "mapped_standard": "A1.PAFR.2.2",
        "extraction_metadata": {
            "source_url": (
                "https://openstax.org/books/"
                "algebra-1/pages/9-2-5-practice"
            ),
            "indicator_alignment": (
                " Solve quadratic equations by completing the square,"
                " factoring, and the quadratic formula, explaining the "
                "connection between the zeros of the function"
                " derived from the equation, its linear factors "
                "(if it factors), the x-intercepts of its graph"
                " (if they exist), and the solutions (if any)"
                " to the corresponding quadratic equation."
            )
        }
    }
]

#Mapping function to use the template
def map_problems_to_standards(
        Unit6Textbook_pdf,
        algebra1Standards_pdf,
        target_section,
        target_section_url,
        model_name="gemini-3.6-flash",
        max_retries=5):

    print("📖 Loading Unit 6 Textbook PDF...")
    with open(Unit6Textbook_pdf, "rb") as f:
        textbook_doc = types.Part.from_bytes(
            data=f.read(),
            mime_type="application/pdf"
        )

    print("📋 Loading Standards PDF...")
    with open(algebra1Standards_pdf, "rb") as f:
        standards_doc = types.Part.from_bytes(
            data=f.read(),
            mime_type="application/pdf"
        )

    # Convert your examples to a formatted string for the prompt
    examples_formatted = json.dumps(textbook_problem_data, indent=4)

    # Build the prompt with few-shot examples
    # Build prompt with strict instructions and references to the attached PDFs
    prompt = f"""You are an expert curriculum specialist and instructional designer specializing in South Carolina Algebra 1 standards alignment.

    ### MISSION:
    Locate the section titled "{target_section}" (Block Code: 6.6.5) in the attached Unit 6 Textbook PDF.
    Extract EVERY SINGLE practice problem in this section from beginning to end without skipping any.

    For each problem in section {target_section}:
    1. Extract the exact problem number, context/instructions (e.g., "Simplify each expression completely"), and the exact problem text.
    2. Provide the solution text if given or derive the exact solution.
    3. Match it to the most specific SC Algebra 1 standard code from the attached Standards PDF.
    4. Copy or closely paraphrase the indicator alignment from the Standards PDF.
    5. Set "source_url" to "https://openstax.org/books/algebra-1/pages/{target_section_url}".

    ### SCHEMA & PATTERN TO FOLLOW:
    {examples_formatted}

    ### STRICT EXTRACTION RULES:
    1. EXHAUSTIVE EXTRACTION: Do NOT stop after 6 problems. Extract EVERY problem listed under section {target_section} (Problem 1, 2, 3, etc., up to the last problem in that section).
    2. VERBATIM: Problem text and instructions must match the textbook PDF character-for-character.
    3. LATEX NOTATION: Format all math expressions in proper LaTeX (e.g., (8k^2 - 3k + 14) + (8k + 17) -> (8k^{{2}} - 3k + 14) + (8k + 17)).
    4. OUTPUT FORMAT: Return a valid JSON list of dictionaries adhering strictly to the schema keys.
    """

    # Configure generation settings
    config = types.GenerateContentConfig(
        temperature=0.0,  # Lower temperature for more consistent structured output
        response_mime_type="application/json",
        max_output_tokens=9000,
    )
    # Contents list combines instructions and both PDF documents
    contents = [
        prompt,
        textbook_doc,
        standards_doc
    ]
    # Retry logic for 503 errors
    for attempt in range(max_retries):
        try:
            print(f"\n🤖 Sending request to {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            return response.text

        except errors.ClientError as e:
            error_str = str(e)

            if "503" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Service temporarily unavailable")
                    print(f"   Attempt {attempt + 1}/{max_retries} - Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return "❌ Error: Gemini service is currently unavailable. Please try again later."
            else:
                return f"❌ Error: {error_str}"

    return "❌ Error: Maximum retry attempts exceeded"


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    target_section = "6.4.6 Practice"
    target_section_url = "6-4-6-practice"
    textbook_pdf = "Unit6Textbook.pdf"
    standards_pdf = "algebra1Standards.pdf"

    # Check if files exist
    if not Path(textbook_pdf).exists():
        print(f"❌ Error: Textbook PDF not found at {textbook_pdf}")
        print(f"   Make sure the file is in the same directory as this script")
    elif not Path(standards_pdf).exists():
        print(f"❌ Error: Standards PDF not found at {standards_pdf}")
        print(f"   Make sure the file is in the same directory as this script")
    else:
        print("=" * 60)
        print(f"🎯 EXTRACTING ALL PROBLEMS: SECTION {target_section}")
        print("=" * 60)

        # Run extraction
        result = map_problems_to_standards(
            textbook_pdf,
            standards_pdf,
            target_section,
            target_section_url
        )
        print("\n" + "=" * 60)
        print("📊 RAW OUTPUT:")
        print("=" * 60)
        print(result)

        # Validate and parse the output
        print("\n" + "=" * 60)
        print("🔍 VALIDATION:")
        print("=" * 60)
        try:
            # Parse the JSON response
            section_problems = json.loads(result)

            if isinstance(section_problems, list) and len(section_problems) > 0:
                print(f"✓ Successfully parsed {len(section_problems)} new problem mappings")

                # Validate structure of first item
                required_keys = ["book", "unit", "lesson", "block_code",
                                 "section_title", "problem_number", "problem_context",
                                 "problem_text", "solution_text", "mapped_standard",
                                 "extraction_metadata"]

                first_problem = section_problems[0]
                missing_keys = [key for key in required_keys if key not in first_problem]

                if missing_keys:
                    print(f"⚠️  Warning: Missing keys in output: {missing_keys}")
                else:
                    print("✓ All required fields present")



                output_filename = f"{target_section}_problems.json"
                with open(output_filename, 'w', encoding="utf-8") as f:
                    json.dump(section_problems, f, indent=4)
                print(f"💾 Saved {len(section_problems)} problems to {output_filename}")
            else:
                print("⚠️ Output parsed, but is not a non-empty list.")


        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print("   Response text was not valid JSON.")
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")

        print("\n" + "=" * 60)
        print("✅ PROCESS COMPLETE")
        print("=" * 60)