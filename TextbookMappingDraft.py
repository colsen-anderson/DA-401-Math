import os
from google import genai
from google.genai import types, errors
import PyPDF2
from pathlib import Path
import time

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
        "problem_text": "(1, \frac{3}{4})",
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

#PDF Extraction Function (so the AI can read the pdf)
def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()

            return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

#Mapping function to use the template
def map_problems_to_standards(Unit6Textbook_pdf, algebra1Standards_pdf, model_name="gemini-3.8-flash", max_retries=5):
    """
    Use Gemini to map 6 NEW textbook problems to SC standards,
    following the pattern of your existing 6 examples.

    Args:
        textbook_pdf (str): Path to Unit 6 textbook PDF
        standards_pdf (str): Path to SC Algebra 1 standards PDF
        model_name (str): Gemini model to use
        max_retries (int): Number of retry attempts for 503 errors

    Returns:
        str: Gemini's response with 6 new problem mappings
    """
    # Extract text from both PDFs
    print("📖 Extracting textbook content...")
    textbook_text = extract_text_from_pdf(Unit6Textbook_pdf)

    print("📋 Extracting standards content...")
    standards_text = extract_text_from_pdf(algebra1Standards_pdf)

    # Convert your examples to a formatted string for the prompt
    import json
    examples_formatted = json.dumps(textbook_problem_data, indent=4)

    # Build the prompt with few-shot examples
    prompt = f"""You are an expert curriculum specialist and instructional designer specializing in South Carolina Algebra 1 standards alignment.
    ### TASK:
    Select 6 DISTINCT, UNMAPPED problems from the Unit 6 textbook content provided below. For each problem:
    1. Identify the mathematical skill being assessed
    2. Match it to the most specific SC Algebra 1 standard code
    3. Create a complete dictionary entry following the EXACT schema shown in the examples

    ### EXAMPLE MAPPINGS (YOUR PATTERN TO FOLLOW):
    ```json
    {examples_formatted}
    ###INSTRUCTIONS:
    Analyze the structure, field names, and level of detail in the 6 examples above
    Your output MUST follow this IDENTICAL schema
    Select 6 NEW problems from Unit 6 that are NOT in the examples above
    Use proper LaTeX formatting for math expressions (e.g., x^{{2}}, \frac{{a}}{{b}})
    Match each problem to the most appropriate SC standard code from the standards document
    Copy or closely paraphrase the indicator text from the standards document
    ###UNIT 6 TEXTBOOK CONTENT:
    {textbook_text}
    ###SC ALGEBRA 1 STANDARDS REFERENCE:
    {standards_text}
    ###OUTPUT FORMAT:
    Return ONLY a valid Python list of 6 dictionaries. Do not include any conversational text, markdown formatting, or code blocks. Output should start with [ and end with ].
    Begin your response now:"""

    # Configure generation settings
    config = types.GenerateContentConfig(
        temperature=0.3,  # Lower temperature for more consistent structured output
        top_p=0.95,
        max_output_tokens=8192,
    )

    # Retry logic for 503 errors
    for attempt in range(max_retries):
        try:
            print(f"\n🤖 Sending request to {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
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
        print("🎯 TEXTBOOK PROBLEM TO STANDARDS MAPPING")
        print("=" * 60)
        print(f"📖 Textbook: {textbook_pdf}")
        print(f"📋 Standards: {standards_pdf}")
        print(f"📝 Using {len(textbook_problem_data)} examples as pattern")
        print("=" * 60)

        # Run the mapping function
        result = map_problems_to_standards(textbook_pdf, standards_pdf)

        print("\n" + "=" * 60)
        print("📊 GEMINI'S OUTPUT:")
        print("=" * 60)
        print(result)

        # Validate and parse the output
        print("\n" + "=" * 60)
        print("🔍 VALIDATION:")
        print("=" * 60)
        try:
            import ast

            # Try to parse the response as a Python list
            new_problems = ast.literal_eval(result)

            if isinstance(new_problems, list) and len(new_problems) > 0:
                print(f"✓ Successfully parsed {len(new_problems)} new problem mappings")

                # Validate structure of first item
                if len(new_problems) > 0:
                    required_keys = ["book", "unit", "lesson", "block_code",
                                     "section_title", "problem_number", "problem_context",
                                     "problem_text", "solution_text", "mapped_standard",
                                     "extraction_metadata"]

                    first_problem = new_problems[0]
                    missing_keys = [key for key in required_keys if key not in first_problem]

                    if missing_keys:
                        print(f"⚠️  Warning: Missing keys in output: {missing_keys}")
                    else:
                        print("✓ All required fields present")

                # Merge with your original 6 examples
                all_problems = textbook_problem_data + new_problems
                print(f"✓ Total problems in dataset: {len(all_problems)}")

                # Optional: Save to a file
                print("\n" + "=" * 60)
                print("💾 SAVING RESULTS:")
                print("=" * 60)

                import json

                output_filename = "textbook_problems_complete.json"
                with open(output_filename, 'w') as f:
                    json.dump(all_problems, f, indent=4)
                print(f"✓ Saved complete dataset to {output_filename}")

                # Also save just the new problems
                new_output_filename = "textbook_problems_new.json"
                with open(new_output_filename, 'w') as f:
                    json.dump(new_problems, f, indent=4)
                print(f"✓ Saved new problems to {new_output_filename}")

            else:
                print("⚠️  Output is not a valid list or is empty")

        except SyntaxError as e:
            print(f"⚠️  Syntax error parsing output: {e}")
            print("   The AI may have included extra text. Try extracting just the list portion.")
        except Exception as e:
            print(f"⚠️  Could not parse output: {e}")
            print("   You may need to manually clean the output")

        print("\n" + "=" * 60)
        print("✅ PROCESS COMPLETE")
        print("=" * 60)

#convert the pdf to json
