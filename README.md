# retina-cascade

Normal-gated, grouped-head, knowledge-distilled multi-disease fundus classifier trained on ODIR-5K
(all 8 labels, patient-level splits, seed 42) with zero-shot external validation (RFMiD).
Implements the pipeline in the literature-review document (Part III).

## Setup
```
pip install -r requirements.txt
```
Pretrained backbone weights are downloaded by timm on first use (internet needed on the GPU machine).
RETFound weights must be downloaded separately (gated, CC BY-NC) and placed at
`weights/RETFound_cfp_weights.pth`; sweep entries with a missing checkpoint are skipped.

## Get the data
Official ODIR-5K: https://odir2019.grand-challenge.org/ (register, download the training set and the
annotation spreadsheet). Use the training images only (3,500 patients, 7,000 images).

## Run order
```
# 0. data audit and splits (see data_prep/README.md; send back audit_summary.json if unsure)
python data_prep/01_audit_odir.py --ann <annotations.xlsx> --img_dir <training images> --out audit
python data_prep/02_make_splits.py --eye_table audit/eye_table.csv --out splits --label_mode eye \
       --seed 42 --drop_missing --drop_cross_patient_dups
python data_prep/03_cache_images.py --splits splits --out cache --res 512

# 1. Phase A: backbone sweep (flat 8-label, plain BCE, validation selection only)
python scripts/run_sweep.py --set data.splits_dir=splits output_dir=runs/sweep data.num_workers=8
#    -> runs/sweep/selection.json (teacher = best, student = best within budget)
#    parallel GPUs: --shard 0/4 ... --shard 3/4, then --select_only

# 2. Phase B/C: ablation grid. Edit configs/ablation.yaml with the selected teacher and student.
python scripts/run_ablation.py --set data.splits_dir=splits output_dir=runs/abl
python scripts/run_ablation.py --only A5 A6 A8 --seeds 42 49 202 --eval_test --set data.splits_dir=splits output_dir=runs/abl

# 3. test evaluation (once per frozen config), tables, paired comparisons
python scripts/final_eval.py --runs runs/abl/abl_A8_s42
python scripts/make_tables.py --runs runs/abl --out tables --pair abl_A8_s42 abl_A5_s42 --pair abl_A6_s42 abl_A1_s42

# 4. external validation (verify configs/rfmid_mapping.yaml against the RFMiD CSV first)
python scripts/run_external.py --run runs/abl/abl_A8_s42 --csv <RFMiD csvs> --img_root <RFMiD images>

# 5. explainability (CNN backbones)
python scripts/explain.py --run runs/abl/abl_A8_s42 --split test --n 6 --out explain
```
Any config key can be overridden with `--set key=value` (for example `optim.lr=1e-4 data.res=384`).

## Smoke test (CPU, synthetic data)
```
python tests/smoke_test.py --work /tmp/retina_smoke
```

## Layout
```
configs/   base.yaml, sweep.yaml, ablation.yaml, rfmid_mapping.yaml
data_prep/ audit, patient-level iterative-stratified splits, image cache
src/retina datasets, models (CBAM, grouped heads, cross-head attention), losses (BCE, ASL, DB, KD),
           metrics (AUC/AP/F1/MCC, ECE, thresholds, patient bootstrap, DeLong), engine (train/eval)
scripts/   train, run_sweep, run_ablation, final_eval, run_external, make_tables, explain
tests/     smoke_test.py
```

## Design notes
- Heads: `flat` (8 sigmoid outputs) or `cascade` (gate for abnormal + 7 disease outputs, grouped as
  vascular D,H / structural G,A,M / media_other C,O, optional cross-head attention). Disease losses are
  masked to abnormal images. End-to-end scores: P(N) = 1 - P(abn), P(d) = P(abn) * P(d | abn).
- Cascade decision rule: below the gate threshold -> Normal; otherwise every disease above its
  validation-tuned threshold; if none -> Others.
- KD: per-label binary KL at temperature T, `L = alpha * L_task + (1 - alpha) * T^2 * L_KD`. The teacher
  must use the same head type as the student. Teacher and student may use different resolutions.
- Thresholds (gate sensitivity 0.95, per-label F1) are tuned on validation and frozen for test and
  external data. The test split is only used when `eval.eval_test=true` or `final_eval.py` is run.
- Efficiency (`efficiency.json`): params, counted FLOPs (2 x MACs), GMACs, batch-1 latency.
