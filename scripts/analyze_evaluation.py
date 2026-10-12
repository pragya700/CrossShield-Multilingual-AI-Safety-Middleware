import json
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "evaluation_results.json"
)


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["results"]


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(results):
    tp = tn = fp = fn = 0

    for result in results:
        expected = result["expected_label"] == "jailbreak"
        predicted = result["predicted_detection"] is True

        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1
        elif expected and not predicted:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------
# Language-wise analysis
# ---------------------------------------------------------

def language_analysis(results):
    groups = defaultdict(list)

    for result in results:
        groups[result["language"]].append(result)

    print()
    print("=" * 70)
    print("LANGUAGE-WISE ANALYSIS")
    print("=" * 70)

    for language, cases in sorted(groups.items()):
        metrics = calculate_metrics(cases)

        print()
        print(f"Language: {language}")
        print(f"Cases: {len(cases)}")
        print(
            f"TP={metrics['TP']} | "
            f"TN={metrics['TN']} | "
            f"FP={metrics['FP']} | "
            f"FN={metrics['FN']}"
        )
        print(f"Accuracy : {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall   : {metrics['recall']:.2%}")
        print(f"F1       : {metrics['f1']:.2%}")


# ---------------------------------------------------------
# Category-wise analysis
# ---------------------------------------------------------

def category_analysis(results):
    groups = defaultdict(list)

    for result in results:
        groups[result["category"]].append(result)

    print()
    print("=" * 70)
    print("CATEGORY-WISE ANALYSIS")
    print("=" * 70)

    for category, cases in sorted(groups.items()):
        metrics = calculate_metrics(cases)

        print()
        print(f"Category: {category}")
        print(f"Cases: {len(cases)}")
        print(
            f"TP={metrics['TP']} | "
            f"TN={metrics['TN']} | "
            f"FP={metrics['FP']} | "
            f"FN={metrics['FN']}"
        )
        print(f"Accuracy : {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall   : {metrics['recall']:.2%}")
        print(f"F1       : {metrics['f1']:.2%}")


# ---------------------------------------------------------
# Missed attacks
# ---------------------------------------------------------

def show_missed_attacks(results):
    missed = [
        result
        for result in results
        if result["expected_label"] == "jailbreak"
        and result["predicted_detection"] is False
    ]

    print()
    print("=" * 70)
    print("MISSED ATTACKS")
    print("=" * 70)

    print(f"Total missed attacks: {len(missed)}")

    for result in missed:
        print()
        print(f"ID       : {result['id']}")
        print(f"Language : {result['language']}")
        print(f"Category : {result['category']}")
        print(f"Decision : {result['decision']}")
        print(f"Prompt   : {result['prompt']}")


# ---------------------------------------------------------
# Detected but not blocked
# ---------------------------------------------------------

def show_detected_not_blocked(results):
    cases = [
        result
        for result in results
        if result["expected_label"] == "jailbreak"
        and result["predicted_detection"] is True
        and result["predicted_block"] is False
    ]

    print()
    print("=" * 70)
    print("DETECTED BUT NOT BLOCKED")
    print("=" * 70)

    print(f"Total: {len(cases)}")

    for result in cases:
        print()
        print(f"ID       : {result['id']}")
        print(f"Language : {result['language']}")
        print(f"Category : {result['category']}")
        print(f"Decision : {result['decision']}")
        print(f"Risk     : {result['overall_risk']}")
        print(f"Jailbreak Score: {result['jailbreak_score']}")
        print(f"Prompt   : {result['prompt']}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    results = load_results()

    print("=" * 70)
    print("CrossShield Evaluation Analysis")
    print("=" * 70)

    print(f"Results file: {RESULTS_PATH}")
    print(f"Total cases : {len(results)}")

    overall = calculate_metrics(results)

    print()
    print("=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)

    print(f"TP        : {overall['TP']}")
    print(f"TN        : {overall['TN']}")
    print(f"FP        : {overall['FP']}")
    print(f"FN        : {overall['FN']}")
    print(f"Accuracy  : {overall['accuracy']:.2%}")
    print(f"Precision : {overall['precision']:.2%}")
    print(f"Recall    : {overall['recall']:.2%}")
    print(f"F1        : {overall['f1']:.2%}")

    language_analysis(results)

    category_analysis(results)

    show_missed_attacks(results)

    show_detected_not_blocked(results)


if __name__ == "__main__":
    main()