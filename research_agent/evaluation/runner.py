"""对已保存的真实运行结果计算固定指标，不生成虚构预测。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_agent.evaluation.metrics import evaluate_case


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation cases must be a JSON array.")
    return data


def evaluate_predictions(
    cases_path: str | Path,
    predictions_path: str | Path,
) -> list[dict[str, Any]]:
    cases = {item["id"]: item for item in load_cases(cases_path)}
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON array.")
    rows = []
    for prediction in predictions:
        case_id = prediction.get("case_id")
        if case_id not in cases:
            raise ValueError(f"Unknown evaluation case: {case_id}")
        result = prediction.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"Prediction {case_id} has no result object.")
        rows.append(evaluate_case(cases[case_id], result))
    return rows
