import json
from pathlib import Path
from collections import defaultdict

from app.detectors.jailbreak_detector import detect as detect_jailbreak
from app.detectors.context_reasoning_analyzer import analyze as analyze_context


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "evaluation_dataset.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "evaluation"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "ablation_results.json"
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Rule detector
# ---------------------------------------------------------

def run_rule_detector(prompt):

    result = detect_jailbreak(
        prompt,
        prompt
    )

    score = result.jailbreak_score

    return score


# ---------------------------------------------------------
# Context detector
# ---------------------------------------------------------

def run_context_detector(prompt):

    result = analyze_context(
        prompt,
        prompt,
        None,
    )

    return result.context_reasoning_score


# ---------------------------------------------------------
# Convert score to binary detection
# ---------------------------------------------------------

def detected(score):

    return score > 0


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(results):

    tp = tn = fp = fn = 0

    for result in results:

        expected = (
            result["expected_label"] == "jailbreak"
        )

        predicted = result["predicted"]

        if expected and predicted:
            tp += 1

        elif not expected and not predicted:
            tn += 1

        elif not expected and predicted:
            fp += 1

        elif expected and not predicted:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if fn + tp
        else 0.0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }


# ---------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------

def evaluate_configuration(
    dataset,
    detector_name,
    detector
):

    results = []

    for case in dataset:

        prompt = case["prompt"]

        score = detector(prompt)

        prediction = detected(score)

        results.append({
            "id": case["id"],
            "language": case["language"],
            "category": case["category"],
            "expected_label": case["expected_label"],
            "score": score,
            "predicted": prediction,
        })

    metrics = calculate_metrics(results)

    return {
        "detector": detector_name,
        "metrics": metrics,
        "results": results,
    }


# ---------------------------------------------------------
# Combined rule + context
# ---------------------------------------------------------

def combined_rule_context(prompt):

    rule_score = run_rule_detector(prompt)

    context_score = run_context_detector(prompt)

    combined_score = max(
        rule_score,
        context_score,
    )

    return combined_score


# ---------------------------------------------------------
# Language analysis
# ---------------------------------------------------------

def language_metrics(results):

    groups = defaultdict(list)

    for result in results:

        groups[
            result["language"]
        ].append(result)

    output = {}

    for language, cases in groups.items():

        output[language] = calculate_metrics(
            cases
        )

    return output


# ---------------------------------------------------------
# Category analysis
# ---------------------------------------------------------

def category_metrics(results):

    groups = defaultdict(list)

    for result in results:

        groups[
            result["category"]
        ].append(result)

    output = {}

    for category, cases in groups.items():

        output[category] = calculate_metrics(
            cases
        )

    return output


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    dataset = load_dataset()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("CrossShield Ablation Study")
    print("=" * 70)

    print(
        f"Dataset : {DATASET_PATH}"
    )

    print(
        f"Cases   : {len(dataset)}"
    )


    # -----------------------------------------------------
    # Configuration 1: Rule only
    # -----------------------------------------------------

    print()

    print(
        "Running Rule-only detector..."
    )

    rule_result = evaluate_configuration(
        dataset,
        "rule_only",
        run_rule_detector,
    )


    # -----------------------------------------------------
    # Configuration 2: Context only
    # -----------------------------------------------------

    print(
        "Running Context-only detector..."
    )

    context_result = evaluate_configuration(
        dataset,
        "context_only",
        run_context_detector,
    )


    # -----------------------------------------------------
    # Configuration 3: Rule + Context
    # -----------------------------------------------------

    print(
        "Running Rule + Context..."
    )

    combined_result = evaluate_configuration(
        dataset,
        "rule_plus_context",
        combined_rule_context,
    )


    # -----------------------------------------------------
    # Print comparison
    # -----------------------------------------------------

    configurations = [
        rule_result,
        context_result,
        combined_result,
    ]

    print()

    print("=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)

    print(
        f"{'Detector':<25}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 70)

    for result in configurations:

        metrics = result["metrics"]

        print(
            f"{result['detector']:<25}"
            f"{metrics['accuracy']:<12.2%}"
            f"{metrics['precision']:<12.2%}"
            f"{metrics['recall']:<12.2%}"
            f"{metrics['f1']:<12.2%}"
        )


    # -----------------------------------------------------
    # Language-wise results
    # -----------------------------------------------------

    print()

    print("=" * 70)
    print("LANGUAGE-WISE RESULTS")
    print("=" * 70)

    for result in configurations:

        print()

        print(
            result["detector"]
        )

        language_results = language_metrics(
            result["results"]
        )

        for language, metrics in sorted(
            language_results.items()
        ):

            print(
                f"  {language}: "
                f"accuracy={metrics['accuracy']:.2%}, "
                f"recall={metrics['recall']:.2%}, "
                f"F1={metrics['f1']:.2%}"
            )


    # -----------------------------------------------------
    # Category-wise results
    # -----------------------------------------------------

    print()

    print("=" * 70)
    print("CATEGORY-WISE RESULTS")
    print("=" * 70)

    for result in configurations:

        print()

        print(
            result["detector"]
        )

        category_results = category_metrics(
            result["results"]
        )

        for category, metrics in sorted(
            category_results.items()
        ):

            print(
                f"  {category}: "
                f"accuracy={metrics['accuracy']:.2%}, "
                f"recall={metrics['recall']:.2%}, "
                f"F1={metrics['f1']:.2%}"
            )


    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output = {

        "dataset": str(
            DATASET_PATH
        ),

        "total_cases": len(
            dataset
        ),

        "configurations": configurations,

        "note": (
            "Semantic and cross-lingual model-dependent "
            "detectors are excluded because the real "
            "LaBSE model is currently unavailable."
        ),
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


    print()

    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Saved to: {RESULTS_PATH}"
    )


# ---------------------------------------------------------
# Run program
# ---------------------------------------------------------

if __name__ == "__main__":
    main()