# `pipeline.py` Guide

`pipeline.py` combines the former two scripts into one resumable workflow: it first analyses each input prompt, then generates (or retains) an image description and scores only records classified as harmful. Analysis results are saved to disk and passed directly to the next stage.

## Environment and Dependencies

The pipeline requires Python 3.10+, the OpenAI SDK, and `torch` plus `transformers` for the rewrite stage. The default rewrite model is large, so run it in a GPU environment with sufficient VRAM.

```sh
pip install openai transformers torch
export OPENAI_API_KEY="your OpenAI API key"
export HF_TOKEN="your Hugging Face token"  # needed only if the rewrite model requires access
```

Do not put secrets in scripts, JSON files, or commits. The default category-definition file is `malicious_intent_new.csv` in the repository root.

## Input

`--input` must point to a UTF-8 JSON file whose root value is an array of strings. Each string is one source prompt to analyse.

```json
[
  "a short source prompt",
  "another source prompt"
]
```

The category CSV must contain the columns `Category`, `Textual Description`, and `Graphic Description`. To use another file, pass `--intent-definitions path/to/categories.csv`.

### Customising Categories

You can edit `malicious_intent_new.csv` to define the categories used by the pipeline. Add, remove, or revise rows to customise categories and definitions for different research purposes. Keep the three required column headers unchanged:

| Column | Used for |
| --- | --- |
| `Category` | The category name returned by analysis and used to match a definition in the rewrite stage. Category names must be unique. |
| `Textual Description` | The definition supplied to the analysis model when classifying source prompts. |
| `Graphic Description` | The definition supplied to the rewrite and harmful-intent scoring stages. |

For a category to reach the rewrite stage, the model's `malicious_intent` result must match its `Category` value (matching is case-insensitive). After editing the CSV, run the analysis stage again so that existing records use the updated taxonomy.

## Running the Pipeline

Run the complete workflow in one command:

```sh
python pipeline.py --input prompts.json
```

Run analysis only (the local rewrite model is not loaded):

```sh
python pipeline.py --stage analyze --input prompts.json \
  --analysis-output outputs/analysis.json
```

Resume rewriting and scoring from an existing analysis result:

```sh
python pipeline.py --stage rewrite --analysis-output outputs/analysis.json \
  --output outputs/accepted.json --all-output outputs/attempts.json
```

After an interruption, add `--resume` to skip records already written to output. Use `--batch-size 10` to control the analysis batch size, `--max-attempts 5` to limit scoring attempts per rewrite, and `--min-score 5` to set the required threshold for all three scores.

## Output

By default, the pipeline creates three UTF-8 JSON arrays:

| File | Contents |
| --- | --- |
| `analysis_results.json` | A classification record for every input. Harmful records also include `isdescriptive`, `main_objects`, text-style fields, and related explanations. |
| `rewrite_attempts.json` | The final attempt for every record that enters the rewrite stage, whether or not it meets the score threshold. |
| `rewrite_results.json` | Only records whose three scores are all at least `--min-score`. |

The analysis record uses the following interface, which the rewrite stage reads directly:

```json
{
  "input": "source prompt",
  "output_analyst_agent": {
    "isharmfulintent": true,
    "malicious_intent": "category name",
    "harm_reason": "short reason",
    "isdescriptive": false,
    "main_objects": ["object A"]
  }
}
```

The `natural_language_style` object in a rewrite record contains `prompt`, `visualizability_score`, `topic_relevance_score`, and `harmful_intent_score`. Every score contains a 0–10 `score` and a short `explanation`. Non-harmful records remain only in the analysis output and do not enter either rewrite output file.

## Common Parameters and Troubleshooting

Run `python pipeline.py --help` to view all parameters. `--model` selects the OpenAI model used for analysis and scoring, `--rewriter-model` selects the local Hugging Face rewrite model, and `--delay` adjusts the interval between API batches. JSON output is written through a temporary file and then replaced to reduce corruption risk during interruptions. If a model returns invalid JSON, the program exits with an error while preserving batches already written.
