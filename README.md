# MVQAN-CORAL: baseline and branch-ablation experiments

This package contains the experiment implementation (`run_experiments.py`), its dependencies (`requirements.txt`), and these instructions.

## Run

Use Python 3.11 and a CUDA-enabled PyTorch installation. Install the dependencies in that environment:

```bash
python -m pip install -r requirements.txt
```

Run all experiments on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --data-root ./data --output-root ./outputs
```

The script downloads public datasets and pretrained checkpoints when needed. Existing Hugging Face credentials are used when available. Dataset files remain in `data`; experiment artifacts are written to `outputs`. Keep the output directory when resuming an interrupted run. Completed jobs and training states are reused only when their experiment identities match.

Print the plan without loading the training libraries or downloading data:

```bash
python run_experiments.py --plan
```

## Experiment specification

There are **12 configurations × 3 datasets × 3 seeds = 108 evaluations**. Seeds are **11, 22, 33**. Each configuration uses the same fixed fold for a dataset.

| Table entry | Configuration |
|---|---|
| LaBraM, fine-tuned | `LaBraM` |
| CBraMod, fine-tuned | `CBraMod` |
| REVE, fine-tuned | `REVE` |
| MAtt | `MAtt` |
| SPDNet | `SPDNet` |
| TSMNet, SPDDSMBN | `TSMNet_SPDDSMBN` |
| SPDIM-style, IM-adapted SPD bias | `SPDIM` |
| EA + EEGNet | `EA_EEGNet` |
| EA + tangent-space logistic regression | `EA_TS_LR` |
| Residual only | `Residual_only` |
| Residual only, parameter-matched | `Residual_only_matched` |
| Quantum only | `Quantum_only` |

BNCI2014_009 uses target/non-target P300 classification, holding out S01 for testing and S05 for source validation. UCL uses left/right motor imagery, holding out the first sorted participant; source validation is selected with registry seed 42. EEGET uses ALS/control classification, holding out one ALS participant and a paired control, with a disjoint ALS/control pair for source validation. For the complete EEGET cohort, the test pair is ALS_01/id110 and the validation pair is ALS_04/id012. Exact membership is recorded in the fold files.

Macro-F1 is reported as a percentage, with mean and sample standard deviation (`ddof=1`) over the three seeds. These are single-fold evaluations; the SD describes seed variability on that fold. The EA table entry reports EEGNet and tangent-space logistic regression separately, in that order.

## Preprocessing and training

UCL uses 4–40 Hz filtering, resampling from the recorded acquisition rate to 128 Hz, and a fixed two-second analysis window. The default acquisition rate is 256 Hz. A trial must supply the full window after resampling; shorter trials are excluded before filtering. Eligibility is determined by sample length. All eight participants must remain eligible, with at least two trials per class each. The acquisition metadata, retained/excluded trial identities, and participant/class counts are saved under `audit/`.

LaBraM, CBraMod, and REVE load published pretrained backbones and fine-tune both backbone and task classifier. Backbone weight coverage must reach 99%. Foundation-model training uses an effective batch size of 32 with microbatches of eight. Checkpoint and decision-threshold selection use source validation labels.

MAtt, SPDNet, and TSMNet are dataset-adapted reimplementations using the common trainer. MAtt uses squared log-Euclidean attention distances and SPD covariance representations. SPDNet uses QR-reparameterized BiMaps with ReEig/LogEig. TSMNet uses affine-invariant domain-specific momentum normalization. These implementations and their settings are defined in the source.

EA uses per-participant unlabeled alignment statistics. TSMNet estimates evaluation-domain normalization statistics from unlabeled data. SPDIM-style freezes the source model and adapts only its tangent-space bias with 200 fixed information-maximization steps. Held-out labels are used for final scoring. Checkpoint and threshold selection use source validation only. These methods are marked as transductive in the tables and result records.

EA + TS-LR uses OAS covariances, Riemannian tangent projection, standardization, and balanced logistic regression. Its numerical policy applies a covariance eigenvalue floor when the unregularized computation is invalid. The floor is `1e-6 × max(trial maximum eigenvalue, median source-training average variance)`. The scale is fitted on source training covariances; C and the threshold are selected on source validation. Activation of this policy is recorded per evaluation.

The three branch ablations share the source-fitted feature views, fold membership, and training settings. The parameter-matched residual network uses the full architecture's trainable parameter count as its reference, with a 1% tolerance. All numerical hyperparameters are declared in `Config` and `RunConfig`.

## Reproducibility inputs

Use the package versions recorded in the accompanying experiment's `audit/environment.json` for numerical replication. The requirements file fixes Braindecode and PennyLane versions and declares the remaining dependencies. Fold membership, dataset provenance, preprocessing decisions, configuration, and pretrained checkpoint identities are recorded under `audit/`.

To select the exact pretrained revisions and checkpoint hashes recorded with a reference experiment, provide its audit directory:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --data-root ./data --output-root ./outputs --pretrained-metadata ./reference/audit
```

That directory must contain `pretrained_LaBraM.json`, `pretrained_CBraMod.json`, and `pretrained_REVE.json`. Their checkpoint hashes are checked against the downloaded files. An existing UCL acquisition manifest in `data/ucl_manifest.csv` is used when present. Use the same recordings, acquisition metadata, package versions, and checkpoint identities as the accompanying results.

## Outputs

- `results/per_seed_results.csv`: all 108 evaluations, including Macro-F1, accuracy, balanced accuracy, ROC AUC, and average precision.
- `results/macro_f1_mean_sd.csv`: means, sample SDs, and seed counts for 36 dataset/configuration combinations.
- `results/baseline_table.md` and `.csv`: eight baseline rows; the EA row contains two configurations.
- `results/ablation_table.md` and `.csv`: three ablation rows.
- `results/completion.json`: completion status and any failed evaluations.

Each evaluation saves held-out and validation predictions, labels, trial IDs, participant IDs, decision thresholds, fitted model artifacts, and training metadata. A table entry is numeric only after all three seeds succeed. Completion requires 108/108 successful evaluations.
