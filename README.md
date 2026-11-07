# RW2Unsafe

> **Content warning:** This repository concerns the safety evaluation of text-to-image (T2I) systems and includes a small sample of unsafe textual prompts. The material is provided solely for responsible safety research.

**RW2Unsafe: Towards More Accurate Real-World Unsafe Generation Evaluation for T2I Models** was accepted to **Findings of EMNLP 2026**. RW2Unsafe is an automated, three-module framework that reformulates publicly available real-world posts into visually descriptive T2I prompts while preserving their harmful intent. It is designed to measure T2I safety more faithfully, not to facilitate harmful-content generation or bypass safety controls.

The accompanying paper is available in [RW2Unsafe.pdf](RW2Unsafe.pdf).

## Repository Contents

| File | Description |
| --- | --- |
| [pipeline.py](pipeline.py) | End-to-end implementation of the Analyst, Prompt Rewrite, and Quality Evaluation workflow. |
| [PIPELINE.md](PIPELINE.md) | Detailed installation, configuration, input/output, and execution guide. |
| [malicious_intent_new.csv](malicious_intent_new.csv) | Configurable unsafe-policy database used for category definitions. |
| [rewritten_prompts_sample.json](rewritten_prompts_sample.json) | Public format sample containing 10 rewritten prompts. |
| [RW2Unsafe.pdf](RW2Unsafe.pdf) | Accepted paper. |

The sample file is a UTF-8 JSON array of strings:

```json
[
  "a rewritten prompt",
  "another rewritten prompt"
]
```

## Framework Overview

RW2Unsafe follows a closed-loop, three-stage pipeline:

1. **Analyst Module** classifies harmful intent, filters benign inputs, identifies visible subjects, and assesses whether a source text is already visually descriptive.
2. **Prompt Rewrite Module** reformulates underspecified harmful source texts into concrete visual prompts, subject to intent-level constraints.
3. **Quality Evaluation Module** scores each candidate for visualizability, topic relevance, and harmful-intent alignment; scores guide iterative refinement.

The policy database is configurable. Edit `malicious_intent_new.csv` to add, remove, or revise categories and their textual/graphic definitions for a different research objective. Keep the required headers (`Category`, `Textual Description`, and `Graphic Description`) unchanged. See [PIPELINE.md](PIPELINE.md#customising-categories) for details.

## Quick Start

Install the dependencies and configure credentials:

```sh
pip install openai transformers torch
export OPENAI_API_KEY="your OpenAI API key"
export HF_TOKEN="your Hugging Face token"  # if required by the rewrite model
```

Run the complete workflow on a UTF-8 JSON array of source prompts:

```sh
python pipeline.py --input prompts.json
```

To run the stages separately:

```sh
python pipeline.py --stage analyze --input prompts.json \
  --analysis-output outputs/analysis.json

python pipeline.py --stage rewrite --analysis-output outputs/analysis.json \
  --output outputs/accepted.json --all-output outputs/attempts.json
```

The default rewrite model is large; use a GPU environment with sufficient VRAM. Consult [PIPELINE.md](PIPELINE.md) for the exact input schema, output schema, resuming interrupted runs, and all configuration options.

## Dataset Access and Responsible Use

The full **rw_t2i_4chan** dataset contains 2,000 high-quality rewritten prompts derived from real-world unsafe content. It is **not distributed in this public repository**. Public release is limited to the 10-record sample above because the full dataset includes offensive, violent, sexually explicit, hateful, and otherwise disturbing material. Broad redistribution could enable misuse and unnecessarily amplify harmful content.

Researchers who need the full dataset for a legitimate safety-evaluation purpose should contact the corresponding author listed in [RW2Unsafe.pdf](RW2Unsafe.pdf) from an institutional email address. Requests are manually reviewed for intended use before access is granted.

By using this repository or requesting the dataset, you agree to use the materials only for legitimate research, evaluation, and safety-improvement purposes; to comply with applicable laws, institutional review requirements, and platform policies; and not to use the materials to create, promote, or distribute harmful content.

## Citation

If you use RW2Unsafe, please cite the paper:

```text
Ye Leng, Junjie Chu, Mingjie Li, Michael Backes, Yun Shen, and Yang Zhang.
2026. RW2Unsafe: Towards More Accurate Real-World Unsafe Generation
Evaluation for T2I Models. Findings of EMNLP 2026.
```

## Contact

For questions about the framework or controlled dataset access, please contact the authors listed in the paper.
