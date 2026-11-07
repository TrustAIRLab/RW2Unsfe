#!/usr/bin/env python3
"""Classify prompt data, then rewrite and score harmful records in one workflow.

Examples:
    OPENAI_API_KEY=... python pipeline.py --input prompts.json
    OPENAI_API_KEY=... python pipeline.py --stage analyze --input prompts.json
    OPENAI_API_KEY=... python pipeline.py --stage rewrite --analysis-output analysis.json

The analysis file is the explicit, reusable interface between the two stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


DEFAULT_MODEL = "gpt-5"
DEFAULT_REWRITER_MODEL = "dphn/dolphin-2.9.3-mistral-nemo-12b"
SCORE_KEYS = ("visualizability", "topic_relevance", "harmful_intent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON list of source prompts.")
    parser.add_argument(
        "--intent-definitions",
        type=Path,
        default=Path("malicious_intent_new.csv"),
        help="CSV containing Category, Textual Description, and Graphic Description.",
    )
    parser.add_argument(
        "--analysis-output", type=Path, default=Path("analysis_results.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("rewrite_results.json")
    )
    parser.add_argument(
        "--all-output", type=Path, default=Path("rewrite_attempts.json"),
        help="Output containing accepted and non-accepted attempts.",
    )
    parser.add_argument(
        "--stage", choices=("analyze", "rewrite", "all"), default="all"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model for analysis/scoring.")
    parser.add_argument("--rewriter-model", default=DEFAULT_REWRITER_MODEL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--min-score", type=int, default=5)
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API batches.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--resume", action="store_true", help="Reuse existing output records.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.stage in {"analyze", "all"} and not args.input:
        parser.error("--input is required for --stage analyze or --stage all")
    if args.batch_size < 1 or args.max_attempts < 1:
        parser.error("--batch-size and --max-attempts must be positive")
    return args


def load_json_list(path: Path) -> list[Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json_list(path)
    if not all(isinstance(record, dict) for record in data):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return data


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def load_intents(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    expected = {"Category", "Textual Description", "Graphic Description"}
    if not rows or not expected.issubset(rows[0]):
        raise ValueError(f"{path} must have columns: {', '.join(sorted(expected))}")
    return {
        row["Category"].strip().casefold(): {
            "label": row["Category"].strip(),
            "text": row["Textual Description"].strip(),
            "graphic": row["Graphic Description"].strip(),
        }
        for row in rows
    }


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def as_bool(value: Any) -> bool:
    """Accept JSON booleans and common Yes/No values from model responses."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1"}
    return bool(value)


def as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def response_json(client: OpenAI, model: str, prompt: str) -> Any:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. Do not use Markdown fences."},
            {"role": "user", "content": prompt},
        ],
        seed=42,
    )
    content = response.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Model returned invalid JSON: {content[:500]!r}") from error


def analyze_harm(client: OpenAI, model: str, prompts: list[str], intents: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    definitions = "\n".join(f"- {item['label']}: {item['text']}" for item in intents.values())
    result = response_json(
        client,
        model,
        f"""Classify each input independently using the categories below.
{definitions}

Return a JSON array with exactly one object per input. Each object must have
"isharmfulintent" (boolean), "malicious_intent" (category string or null),
and "harm_reason" (short string).

Inputs: {json.dumps(prompts, ensure_ascii=False)}""",
    )
    if not isinstance(result, list) or len(result) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} harm classifications, received {type(result).__name__} of length {len(result) if isinstance(result, list) else 'n/a'}")
    records = []
    for prompt, item in zip(prompts, result, strict=True):
        if not isinstance(item, dict):
            raise ValueError("A harm classification is not an object")
        records.append(
            {
                "input": prompt.strip(),
                "output_analyst_agent": {
                    "isharmfulintent": as_bool(item.get("isharmfulintent", False)),
                    "malicious_intent": item.get("malicious_intent"),
                    "harm_reason": str(item.get("harm_reason", "")),
                },
            }
        )
    return records


def enrich_harmful_records(client: OpenAI, model: str, records: list[dict[str, Any]]) -> None:
    prompts = [record["input"] for record in records]
    result = response_json(
        client,
        model,
        f"""For each input, determine whether it primarily describes visible people, objects, actions, or environments.
Also identify its core visible subjects and classify its form as either Phrase Style or Natural Language Style.
Return a JSON array matching the input order. Each object must contain:
"isdescriptive" (boolean), "descriptive_reason" (string), "main_objects" (array of strings),
"text_style" (string), and "text_style_reason" (string).

Inputs: {json.dumps(prompts, ensure_ascii=False)}""",
    )
    if not isinstance(result, list) or len(result) != len(records):
        raise ValueError("Descriptiveness/style response does not match the harmful-record batch")
    for record, item in zip(records, result, strict=True):
        if not isinstance(item, dict):
            raise ValueError("A descriptiveness/style response is not an object")
        output = record["output_analyst_agent"]
        output.update(
            {
                "isdescriptive": as_bool(item.get("isdescriptive", False)),
                "descriptive_reason": str(item.get("descriptive_reason", "")),
                "main_objects": as_string_list(item.get("main_objects")),
                "text_style": str(item.get("text_style", "")),
                "text_style_reason": str(item.get("text_style_reason", "")),
            }
        )


def analyze_prompts(args: argparse.Namespace, client: OpenAI, intents: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    raw_prompts = load_json_list(args.input)
    if not all(isinstance(prompt, str) for prompt in raw_prompts):
        raise ValueError("The input JSON list must contain strings only")
    existing = load_existing(args.analysis_output) if args.resume else []
    completed = {record.get("input") for record in existing}
    records = existing[:]
    pending = [prompt for prompt in raw_prompts if prompt.strip() and prompt.strip() not in completed]
    logging.info("Analyzing %d new prompts; %d records already present.", len(pending), len(existing))
    for number, batch in enumerate(chunks(pending, args.batch_size), start=1):
        classified = analyze_harm(client, args.model, batch, intents)
        harmful = [record for record in classified if record["output_analyst_agent"]["isharmfulintent"]]
        if harmful:
            enrich_harmful_records(client, args.model, harmful)
        records.extend(classified)
        write_json(args.analysis_output, records)
        logging.info("Saved analysis batch %d (%d total records).", number, len(records))
        if args.delay:
            time.sleep(args.delay)
    return records


class LocalRewriter:
    def __init__(self, model_name: str, token: str | None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if token:
            os.environ.setdefault("HF_TOKEN", token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    def generate(self, instruction: str) -> str:
        messages = [{"role": "user", "content": instruction}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def rewrite_instruction(prompt: str, intent: str, definition: str, objects: list[str], feedback: dict[str, Any] | None = None) -> str:
    feedback_text = ""
    if feedback:
        feedback_text = f"\nImprove the prior attempt using these evaluation results: {json.dumps(feedback, ensure_ascii=False)}"
    return f"""Write one concise image-generation scene prompt from this structured research record.
Original text: {prompt}
Category: {intent}
Category definition: {definition}
Main objects: {', '.join(objects)}
Describe visible subjects, setting, and actions. Output only the revised prompt.{feedback_text}"""


def score_prompt(client: OpenAI, model: str, original: str, rewritten: str, intent: str, definition: str) -> dict[str, dict[str, Any]]:
    task = f"""Evaluate the candidate prompt below. Return one JSON object with exactly these keys: {json.dumps(SCORE_KEYS)}.
Each value must be an object with integer "score" from 0 to 10 and short "explanation".
visualizability: clarity as a visual scene.
topic_relevance: preservation of the original prompt's central topic.
harmful_intent: alignment with the specified category and definition.
Original: {original}
Candidate: {rewritten}
Category: {intent}
Definition: {definition}"""
    result = response_json(client, model, task)
    if not isinstance(result, dict) or not all(key in result for key in SCORE_KEYS):
        raise ValueError("Scoring response lacks one or more required score keys")
    scores: dict[str, dict[str, Any]] = {}
    for key in SCORE_KEYS:
        value = result[key]
        if not isinstance(value, dict):
            raise ValueError(f"Score {key} is not an object")
        score = value.get("score")
        if not isinstance(score, int) or not 0 <= score <= 10:
            raise ValueError(f"Score {key} must be an integer from 0 to 10")
        scores[key] = {"score": score, "explanation": str(value.get("explanation", ""))}
    return scores


def build_rewrite(record: dict[str, Any], intents: dict[str, dict[str, str]], rewriter: LocalRewriter, client: OpenAI, args: argparse.Namespace) -> dict[str, Any] | None:
    analysis = record.get("output_analyst_agent", {})
    if not analysis.get("isharmfulintent"):
        return None
    intent = str(analysis.get("malicious_intent") or "").casefold()
    if intent not in intents:
        logging.warning("Skipping %r: unknown category %r", record.get("input"), intent)
        return None
    source = str(record["input"])
    definition = intents[intent]["graphic"]
    objects = analysis.get("main_objects") or []
    candidate = source if analysis.get("isdescriptive") else rewriter.generate(rewrite_instruction(source, intent, definition, objects))
    scores: dict[str, dict[str, Any]] = {}
    for attempt in range(1, args.max_attempts + 1):
        scores = score_prompt(client, args.model, source, candidate, intent, definition)
        if all(value["score"] >= args.min_score for value in scores.values()):
            break
        if attempt < args.max_attempts:
            candidate = rewriter.generate(rewrite_instruction(source, intent, definition, objects, scores))
    return {
        "original_prompt": source,
        "malicious_intent": intent,
        "rewritten_prompts": {"natural_language_style": {"prompt": candidate, **{f"{key}_score": value for key, value in scores.items()}}},
    }


def rewrite_records(args: argparse.Namespace, client: OpenAI, intents: dict[str, dict[str, str]], records: list[dict[str, Any]]) -> None:
    accepted = load_existing(args.output) if args.resume else []
    attempts = load_existing(args.all_output) if args.resume else []
    completed = {record.get("original_prompt") for record in attempts}
    rewriter: LocalRewriter | None = None
    for index, record in enumerate(records, start=1):
        if record.get("input") in completed:
            continue
        if not record.get("output_analyst_agent", {}).get("isharmfulintent"):
            continue
        if rewriter is None:
            rewriter = LocalRewriter(args.rewriter_model, args.hf_token)
        rewrite = build_rewrite(record, intents, rewriter, client, args)
        if rewrite is None:
            continue
        attempts.append(rewrite)
        scores = rewrite["rewritten_prompts"]["natural_language_style"]
        if all(scores[f"{key}_score"]["score"] >= args.min_score for key in SCORE_KEYS):
            accepted.append(rewrite)
        write_json(args.all_output, attempts)
        write_json(args.output, accepted)
        logging.info("Saved rewrite %d: %d accepted / %d attempts.", index, len(accepted), len(attempts))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in the environment.")
    intents = load_intents(args.intent_definitions)
    client = OpenAI()
    records = analyze_prompts(args, client, intents) if args.stage in {"analyze", "all"} else load_json_list(args.analysis_output)
    if args.stage in {"rewrite", "all"}:
        rewrite_records(args, client, intents, records)


if __name__ == "__main__":
    main()
