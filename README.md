# SciNex at NTCIR-19 SciClaimEval

Code for **"SciNex at the NTCIR-19 SciClaimEval Task: Structured Adversarial
Reasoning for Multimodal Scientific Claim Verification"**

The systems are training-free and use open-weight Qwen3-VL models:

* **One-stage:** a four-step structured adversarial reasoning prompt, with
  tolerance and burden-of-proof rules for hedged scientific language.
* **Two-stage:** a claim-blind captioner (CapRL-Qwen3VL-4B) de-renders each
  evidence image before the reasoner (Qwen3-VL-30B-A3B-Thinking) sees the
  claim.

| Run | Config | Dev acc. | Test acc. | Test F1 | Test pair acc. |
|---|---|---|---|---|---|
| ST1-one-stage | `configs/st1/v6.yaml` | 0.8099 | **0.7710** | 0.7710 | 0.5688 |
| ST1-two-stages | `configs/st1/two_stage.yaml` | 0.8059 | 0.7525 | 0.7513 | 0.5550 |
| ST2-one-stage | `configs/st2/v10.yaml` | 0.8523 | 0.8349 | — | — |
| ST2-two-stage | `configs/st2/two_stage.yaml` | 0.9034 | **0.8807** | — | — |

Development numbers use exact table cells; official test runs read tables
from PNG images only.

---

## 1. Installation

Requirements: Python ≥ 3.10 and an NVIDIA GPU (tested on one RTX PRO
6000 Blackwell, 96 GB). The Qwen3-VL-30B-A3B weights alone take about 60 GB
in bfloat16, and the two-stage systems also load the 4B captioner.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins exact versions, but the original environment was
never frozen, so some pins are reconstructed (see the comments in the file).
Greedy decoding is bit-reproducible only on a fixed hardware and software
stack.

## 2. Data

The task data is on the Hugging Face Hub. Use the **v1.0 revision**: the
v1.1 update of 2026-08-19 changed the Subtask 2 dev set from 352 to 338
samples, so downloading `main` gives different numbers.

```bash
# development split (dataset v1.0)
hf download --repo-type dataset alabnii/sciclaimeval-shared-task \
  --revision efb3807399acec43854fdf7741c1bcfe605a72b9 --local-dir sciclaimeval-shared-task
# official test split (no gold labels; access may require accepting the task terms)
hf download --repo-type dataset alabnii/sciclaimeval-shared-task-test \
  --local-dir sciclaimeval-shared-task-test
```

`--data_root` must point at the `data/` directory, which contains
`<split>_task1_release.json`, `<split>_task2_release.json`, `figures/`,
`tables_png/`, `tables_json/` and `papers/`.

## 3. Models

Configs name the checkpoints by Hub id. They are downloaded into the Hugging
Face cache on first use, or you can pass a local directory with `--model` /
`--captioner`.

| Role | Checkpoint | Used by |
|---|---|---|
| Reasoner, one-stage | `Qwen/Qwen3-VL-30B-A3B-Instruct` | all one-stage configs |
| Reasoner, two-stage | `Qwen/Qwen3-VL-30B-A3B-Thinking` | `*/two_stage.yaml` |
| Stage-1 captioner | `internlm/CapRL-Qwen3VL-4B` | `*/two_stage.yaml` |

Model revisions are not pinned, so later snapshots may give slightly different outputs.

## 4. Running

```bash
export DATA_ROOT=sciclaimeval-shared-task/data TEST_ROOT=sciclaimeval-shared-task-test/data

# one configuration on the development split
python -m scinex.run --config configs/st1/v6.yaml --data_root $DATA_ROOT \
  --split dev --output outputs/dev/st1_v6.json

# the four official test runs: PNG input only, as the task requires
for cfg in st2/two_stage st1/v6 st1/two_stage st2/v10; do
  python -m scinex.run --config configs/$cfg.yaml --data_root $TEST_ROOT \
    --split test --png_only --output outputs/test/${cfg//\//_}.json
done
```

Every run writes the following, next to each other:

* `<output>.json`: predictions with full reasoning traces and Stage-1 captions.
* `<output>_submission.json`: the official submission format. Test labels are
  not public, so test scores come from the task evaluator.
* `<output>_metrics.json`: metrics, when gold labels exist.
* `<output>_run.json`: the configuration and wall-clock timing
  (`seconds_per_sample`).
* `<output>.log`: the run log.

Re-running with the same `--output` resumes from the saved predictions.

Options for `python -m scinex.run`:

* `--config` (required): one of `configs/st{1,2}/*.yaml`.
* `--data_root` (required) and `--split dev|test` (default `dev`).
* `--output` (required): path of the prediction file.
* `--png_only`: never inject structured table values. Required for the test
  split.
* `--device` (default `cuda:0`) and `--captioner_device`: place the two
  models on different GPUs.
* `--model` / `--captioner`: local checkpoints or other Hub ids.
* `--n_samples N`: self-consistency (default 1); N>1 enables voting
  and the mid-confidence adjudication.
* `--limit N`: the first N samples only.
* `--data_path`: a release JSON other than `<data_root>/<split>_taskN_release.json`.
* `--workers` (image-loading threads, default 6) and `--save_every`
  (checkpoint interval in samples, default 5).

## 5. Layout

```
configs/      one YAML per system/prompt version (all hyper-parameters)
prompts/      verbatim prompt texts, one directory per version
scinex/       package: config, prompts, images, data, engine, claim_label, claim_evidence, metrics, run
scripts/      compare_runs.py (agreement between two runs), evaluate.py (recompute metrics)
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{scinex2026,
  author    = {Anak Ponggun and Patipon Wiangnak and Natthawut Kertkeidkachorn and
               Areerat Trongratsameethong and Hideaki Takeda},
  title     = {{SciNex} at the {NTCIR}-19 {SciClaimEval} Task: Structured Adversarial
               Reasoning for Multimodal Scientific Claim Verification},
  booktitle = {Proceedings of the 19th {NTCIR} Conference on Evaluation of
               Information Access Technologies},
  year      = {2026}
}
```

If you use the SciClaimEval data, please also cite the task overview:

```bibtex
@inproceedings{sciclaimevaloverview,
  author    = {Xanh Ho and Andr{\'e} Greiner-Petter and Florian Boudin and Yun-Ang Wu and
               Sunisth Kumar and Tian Cheng Xia and Hiba Akbi and Tim Schopf and Akiko Aizawa},
  title     = {Overview of the {NTCIR}-19 {SciClaimEval} Task: Cross-modal Claim
               Verification in Scientific Papers},
  booktitle = {Proceedings of the 19th {NTCIR} Conference on Evaluation of
               Information Access Technologies},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The SciClaimEval data
and the model checkpoints are not part of this repository and have their own
licenses.
