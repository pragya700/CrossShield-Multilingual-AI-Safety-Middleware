import csv
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "datasets" / "evaluation_dataset.json"
RESULTS_DIR = BASE_DIR / "data" / "evaluation"

RESULTS_JSON = RESULTS_DIR / "evaluation_results.json"
RESULTS_CSV = RESULTS_DIR / "evaluation_results.csv"


# ---------------------------------------------------------
# Load evaluation dataset
# ---------------------------------------------------------

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(results):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for result in results:
        expected_jailbreak = result["expected_label"] == "jailbreak"
        predicted_jailbreak = result["predicted_detection"] is True

        if expected_jailbreak and predicted_jailbreak:
            tp += 1

        elif not expected_jailbreak and not predicted_jailbreak:
            tn += 1

        elif not expected_jailbreak and predicted_jailbreak:
            fp += 1

        elif expected_jailbreak and not predicted_jailbreak:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
    }


# ---------------------------------------------------------
# Blocking metrics
# ---------------------------------------------------------

def calculate_blocking_metrics(results):
    blocked_attacks = 0
    missed_attacks = 0
    blocked_benign = 0
    allowed_benign = 0

    for result in results:
        expected_jailbreak = result["expected_label"] == "jailbreak"
        blocked = result["predicted_block"] is True

        if expected_jailbreak and blocked:
            blocked_attacks += 1

        elif expected_jailbreak and not blocked:
            missed_attacks += 1

        elif not expected_jailbreak and blocked:
            blocked_benign += 1

        elif not expected_jailbreak and not blocked:
            allowed_benign += 1

    total = (
        blocked_attacks
        + missed_attacks
        + blocked_benign
        + allowed_benign
    )

    blocking_rate = (
        blocked_attacks / (blocked_attacks + missed_attacks)
        if (blocked_attacks + missed_attacks)
        else 0.0
    )

    benign_block_rate = (
        blocked_benign / (blocked_benign + allowed_benign)
        if (blocked_benign + allowed_benign)
        else 0.0
    )

    return {
        "attacks_blocked": blocked_attacks,
        "attacks_not_blocked": missed_attacks,
        "benign_prompts_blocked": blocked_benign,
        "benign_prompts_allowed": allowed_benign,
        "attack_blocking_rate": round(blocking_rate, 4),
        "benign_block_rate": round(benign_block_rate, 4),
        "total_cases": total,
    }


# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------

def run_evaluation():
    dataset = load_dataset()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)

    results = []

    print("=" * 60)
    print("CrossShield Evaluation")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total cases: {len(dataset)}")
    print()

    for index, case in enumerate(dataset, start=1):

        prompt = case["prompt"]

        response = client.post(
            "/analyze",
            json={
                "prompt": prompt
            }
        )

        status_code = response.status_code

        if status_code != 200:
            print(
                f"[{index}/{len(dataset)}] "
                f"{case['id']} -> ERROR {status_code}"
            )

            results.append({
                **case,
                "predicted_detection": False,
                "predicted_block": False,
                "decision": None,
                "risk_level": None,
                "overall_risk": None,
                "jailbreak_score": None,
                "status_code": status_code,
                "error": response.text,
            })

            continue

        data = response.json()

        decision = data.get("decision")
        risk_level = data.get("risk_level")

        overall_risk = data.get("overall_risk")
        jailbreak_score = data.get("jailbreak_score")

        # -------------------------------------------------
        # Detection definition:
        #
        # Any decision other than ALLOW means CrossShield
        # detected meaningful risk.
        # -------------------------------------------------

        predicted_detection = decision != "ALLOW"

        # -------------------------------------------------
        # Blocking definition:
        #
        # Only BLOCK counts as an actual block.
        # -------------------------------------------------

        predicted_block = decision == "BLOCK"

        result = {
            **case,
            "predicted_detection": predicted_detection,
            "predicted_block": predicted_block,
            "decision": decision,
            "risk_level": risk_level,
            "overall_risk": overall_risk,
            "jailbreak_score": jailbreak_score,
            "status_code": status_code,
        }

        results.append(result)

        expected = case["expected_label"]

        prediction = (
            "jailbreak"
            if predicted_detection
            else "benign"
        )

        status = "✓" if expected == prediction else "✗"

        print(
            f"[{index:02d}/{len(dataset)}] "
            f"{status} "
            f"{case['id']} | "
            f"expected={expected} | "
            f"predicted={prediction} | "
            f"decision={decision}"
        )

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    detection_metrics = calculate_metrics(results)
    blocking_metrics = calculate_blocking_metrics(results)

    evaluation_output = {
        "dataset": str(DATASET_PATH),
        "total_cases": len(dataset),
        "detection_metrics": detection_metrics,
        "blocking_metrics": blocking_metrics,
        "results": results,
    }

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    with open(RESULTS_JSON, "w", encoding="utf-8") as file:
        json.dump(
            evaluation_output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    with open(
        RESULTS_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fieldnames = [
            "id",
            "language",
            "category",
            "expected_label",
            "predicted_detection",
            "predicted_block",
            "decision",
            "risk_level",
            "overall_risk",
            "jailbreak_score",
            "status_code",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result.get(field)
                for field in fieldnames
            })

    # -----------------------------------------------------
    # Print final report
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("DETECTION METRICS")
    print("=" * 60)

    for key, value in detection_metrics.items():
        print(f"{key}: {value}")

    print()
    print("=" * 60)
    print("BLOCKING METRICS")
    print("=" * 60)

    for key, value in blocking_metrics.items():
        print(f"{key}: {value}")

    print()
    print("=" * 60)
    print("RESULT FILES")
    print("=" * 60)
    print(f"JSON: {RESULTS_JSON}")
    print(f"CSV : {RESULTS_CSV}")


if __name__ == "__main__":
    run_evaluation()