import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIRECTORY = PROJECT_ROOT / "evaluation"

QUESTIONS_FILE = EVALUATION_DIRECTORY / "test_questions.csv"
RESULTS_FILE = EVALUATION_DIRECTORY / "results.csv"

EVALUATION_DOCUMENT = (
    PROJECT_ROOT
    / "data"
    / "sample_documents"
    / "Week4_AI_Development.pdf"
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.document_processor import load_pdf, split_documents  # noqa: E402
from src.rag_chain import answer_question  # noqa: E402
from src.vector_store import create_vector_store  # noqa: E402


def parse_boolean(value: str) -> bool:
    """Convert a CSV boolean value into a Python boolean."""

    return value.strip().lower() == "true"


def normalize_text(value: str) -> str:
    """Normalize text for keyword comparison."""

    return " ".join(value.lower().split())


def evaluate_keywords(
    answer: str,
    expected_keywords: str,
) -> tuple[int, int, float]:
    """Check how many expected keywords appear in an answer."""

    keywords = [
        keyword.strip()
        for keyword in expected_keywords.split("|")
        if keyword.strip()
    ]

    if not keywords:
        return 0, 0, 1.0

    normalized_answer = normalize_text(answer)

    matched_keywords = sum(
        1
        for keyword in keywords
        if normalize_text(keyword) in normalized_answer
    )

    score = matched_keywords / len(keywords)

    return matched_keywords, len(keywords), score


def extract_source_pages(
    sources: list[dict[str, Any]],
) -> list[str]:
    """Extract unique source-page numbers from a response."""

    pages = {
        str(source.get("page_number", "Unknown"))
        for source in sources
    }

    return sorted(pages)


def check_expected_page(
    actual_pages: list[str],
    expected_pages: str,
) -> bool:
    """Check whether an expected page was retrieved."""

    expected = [
        page.strip()
        for page in expected_pages.split("|")
        if page.strip() and page.strip() != "0"
    ]

    if not expected:
        return True

    return any(page in actual_pages for page in expected)


def prepare_evaluation_knowledge_base() -> None:
    """
    Build a fresh evaluation knowledge base.

    This makes the test independent from the Streamlit session
    and prevents Clear All from causing an empty evaluation database.
    """

    if not EVALUATION_DOCUMENT.exists():
        raise FileNotFoundError(
            "Evaluation document was not found:\n"
            f"{EVALUATION_DOCUMENT}"
        )

    pages = load_pdf(EVALUATION_DOCUMENT)

    chunks = split_documents(
        documents=pages,
        chunk_size=500,
        chunk_overlap=100,
    )

    if not chunks:
        raise ValueError(
            "No text chunks were generated from the evaluation document."
        )

    create_vector_store(
        documents=chunks,
        reset_database=True,
    )

    print(
        f"Indexed 1 sample document, "
        f"{len(pages)} page(s), and "
        f"{len(chunks)} chunk(s).\n"
    )


def run_evaluation() -> None:
    """Run all test questions and save the evaluation results."""

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions not found: {QUESTIONS_FILE}"
        )

    print("\nPreparing evaluation knowledge base...")

    prepare_evaluation_knowledge_base()

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        questions = list(csv.DictReader(file))

    if not questions:
        raise ValueError("The evaluation question file is empty.")

    results: list[dict[str, Any]] = []

    total_questions = len(questions)
    passed_questions = 0

    print(f"Running {total_questions} evaluation questions...\n")

    for row in questions:
        question_id = row["id"]
        question = row["question"]

        expected_fallback = parse_boolean(
            row["expect_fallback"]
        )

        print(
            f"[{question_id}/{total_questions}] "
            f"{question}"
        )

        try:
            response = answer_question(
                question=question,
                number_of_results=3,
            )

            answer = response["answer"]
            sources = response["sources"]
            actual_fallback = response["is_fallback"]

            matched, total_keywords, keyword_score = (
                evaluate_keywords(
                    answer=answer,
                    expected_keywords=row["expected_keywords"],
                )
            )

            source_pages = extract_source_pages(sources)

            source_page_correct = check_expected_page(
                actual_pages=source_pages,
                expected_pages=row["expected_pages"],
            )

            normalized_answer = answer.strip()

            answer_is_complete = (
                actual_fallback
                or (
                    len(normalized_answer) >= 15
                    and normalized_answer.endswith(
                        (".", "!", "?", '"', "'", "”")
                    )
                )
            )

            if expected_fallback:
                passed = (
                    actual_fallback
                    and len(sources) == 0
                    and answer_is_complete
                )
            else:
                passed = (
                    not actual_fallback
                    and keyword_score >= 0.50
                    and source_page_correct
                    and answer_is_complete
                )

            error = ""

        except Exception as exception:
            answer = ""
            source_pages = []
            actual_fallback = False
            matched = 0
            total_keywords = 0
            keyword_score = 0.0
            source_page_correct = False
            answer_is_complete = False
            passed = False
            error = str(exception)

            print(f"Error: {exception}")

        if passed:
            passed_questions += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"Result: {status}")
        print(f"Answer: {answer}")
        print(
            "Source pages: "
            f"{', '.join(source_pages) or 'None'}"
        )
        print("-" * 70)

        results.append(
            {
                "id": question_id,
                "category": row["category"],
                "question": question,
                "expected_fallback": expected_fallback,
                "actual_fallback": actual_fallback,
                "matched_keywords": matched,
                "total_keywords": total_keywords,
                "keyword_score": f"{keyword_score:.2f}",
                "expected_pages": row["expected_pages"],
                "retrieved_pages": "|".join(source_pages),
                "source_page_correct": source_page_correct,
                "answer_is_complete": answer_is_complete,
                "status": status,
                "actual_answer": answer,
                "error": error,
                "manual_grounded_review": "",
                "manual_notes": "",
            }
        )

    fieldnames = list(results[0].keys())

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    pass_rate = (
        passed_questions / total_questions * 100
    )

    print("\nEvaluation completed.")
    print(
        f"Passed: {passed_questions}/{total_questions}"
    )
    print(f"Pass rate: {pass_rate:.1f}%")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_evaluation()