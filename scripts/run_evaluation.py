import argparse
import json

from research_agent.evaluation.runner import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved, real ReasoningAgent run results."
    )
    parser.add_argument("--cases", default="evals/cases.json")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    rows = evaluate_predictions(args.cases, args.predictions)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
