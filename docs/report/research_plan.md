# Research plan: data-driven executive function modes (multi-dataset design)

This document defines the current research plan and the repository-level design assumptions for the EF (executive function) project. It is written to be consistent with the modular project structure defined in `PROJECT_STRUCTURE.md`.

Status: EFNY is the current working dataset. ABCD support is treated as a planned extension and is not assumed to be implemented unless explicitly documented elsewhere.

## 1) Scientific motivation and scope

Executive function (EF) is commonly operationalized using composite scores and latent factors derived from multiple tasks. A recurring limitation of this approach is that (i) aggregation can obscure task- and condition-specific variance that may be neurobiologically informative, and (ii) psychologically defined factors do not necessarily map cleanly onto a single neural system.

This project adopts a data-driven, multivariate approach that relates fine-grained behavioral features to whole-brain functional connectivity (FC) features. The aim is to identify reproducible covariance modes linking neuroimaging and behavior, while enforcing strict out-of-sample evaluation to reduce overfitting and information leakage.

This plan specifies hypotheses and analysis logic; it does not contain empirical results.

## 2) Project architecture (modular, multi-dataset)

### 2.1 Dataset-agnostic workflow layers

The repository is organized such that general logic is reusable across datasets, while dataset-specific differences are isolated (see `PROJECT_STRUCTURE.md`):

1) **Behavioral feature construction**: read raw behavioral records, apply task-specific preprocessing/QC, and export a subject-by-feature table.
2) **Neuroimaging preprocessing/QC**: preprocess resting-state fMRI, screen data quality, and export subject-level QC summaries.
3) **FC feature construction**: build connectome-derived features (e.g., vectorized FC, atlas-specific representations).
4) **Modeling and evaluation**: estimate multivariate brain–behavior associations using nested cross-validation; compute permutation-based significance where applicable.
5) **Result organization and summaries**: write fold-level artifacts and dataset/model summaries to runtime output directories (not version-controlled).

### 2.2 Dataset-specific layer (current: EFNY)

Dataset-specific assumptions (file naming, subject ID conventions, table schemas, QC thresholds) are documented under `docs/datasets/`. For EFNY, use:

- `docs/datasets/EFNY.md`

## 3) Current dataset focus: EFNY

### 3.1 EFNY overview (inputs and derived tables)

EFNY analyses are built from:

- **Behavioral tasks** recorded as per-subject Excel workbooks, converted to a wide behavioral metrics table.
- **Resting-state fMRI** processed with an established pipeline, accompanied by subject- and run-level motion/QC summaries.
- **Subject lists and alignment tables** used to ensure consistent inclusion across modalities.

The exact input/output paths and QC logic are dataset-specific and are therefore maintained in `docs/datasets/EFNY.md` (with script references treated as authoritative).

### 3.2 Key validity requirements (EFNY)

All EFNY modeling runs are considered valid only if the following conditions are met:

- **Cross-modality alignment**: FC features, behavioral features, and subject lists refer to the same subjects and are aligned consistently (preferably by explicit ID-based joins rather than row-order assumptions).
- **Leakage control**: any preprocessing that uses information from data distributions (imputation, standardization, confound regression, PCA) is fit on training splits only and applied to held-out splits.
- **Traceability**: fold-level artifacts required for downstream summaries are saved deterministically and can be regenerated from raw inputs and configs.

## 4) Modeling strategy (dataset-agnostic principles)

### 4.1 Target estimands

The primary estimand is the out-of-sample association between a multivariate brain feature representation (X) and a multivariate behavioral representation (Y), quantified using test-set canonical correlations and related summary statistics derived from fitted multivariate models.

### 4.2 Models

The repository supports adaptive multivariate models (e.g., adaptive PLS / sparse CCA / regularized CCA) with hyperparameter selection performed strictly inside inner cross-validation. Model family selection and tuning are treated as part of the analysis plan and must be fully nested.

### 4.3 Evaluation and inference

- **Nested cross-validation** provides an unbiased estimate of generalization performance under a fixed modeling protocol.
- **Permutation testing** (when used) must preserve the evaluation protocol and avoid reusing information across permutations in a way that changes the effective model selection procedure.

Any reported association strength must be tied to an explicit evaluation definition (e.g., outer-fold test-set correlation) and should include uncertainty summaries across folds.

## 5) Planned analyses (EFNY)

### 5.1 Primary analysis

1) Construct EFNY behavioral metrics (Y) and FC-derived features (X), harmonized to a shared subject set.
2) Fit multivariate association models under nested cross-validation.
3) Save fold-level artifacts required for interpretability (e.g., loadings/weights) and for summary scripts.

### 5.2 Sensitivity analyses (pre-registered logic)

These analyses are intended to assess robustness rather than to optimize outcomes:

- Alternative confound sets (e.g., demographic covariates and motion/QC summaries).
- Alternative atlas/feature representations (if supported by the existing pipeline and configuration).
- Alternative missing-data handling strategies, provided they are implemented without leakage (fit on training splits only).

### 5.3 Developmental and subgroup analyses (optional, hypothesis-driven)

If scientifically justified and sample size permits, EFNY analyses may be extended to:

- Age-stratified or sliding-window evaluations to test whether brain–behavior covariance structure changes across development.
- Group comparisons, provided group definitions are pre-specified and confounding is handled explicitly.

These extensions require careful control of multiplicity and should be framed as confirmatory only when justified by pre-defined hypotheses.

## 6) Future extension: ABCD (planned, not assumed)

ABCD integration is planned as a dataset adapter layer that conforms to the same dataset-agnostic workflow:

- ABCD-specific ingestion, QC, and feature definitions must be documented under `docs/datasets/ABCD.md` (to be created when work begins).
- ABCD-specific configs should live under `configs/datasets/` as described in `PROJECT_STRUCTURE.md`.
- Cross-dataset comparisons should be performed only after within-dataset pipelines are validated and harmonization decisions are explicitly documented.

Until ABCD documentation and configs exist, ABCD should be treated as “out of scope” for execution and as “planned” for design discussions only.

## 7) Reproducibility and reporting

- All runtime outputs are written to `data/`, `outputs/`, or `logs/` and are not committed.
- Documentation changes must not claim outcomes; observed values belong in versioned run logs or external result archives, referenced with dates and run identifiers.
- Each AI-assisted documentation session should create or update a brief note under `docs/notes/` describing the change rationale and affected files.
