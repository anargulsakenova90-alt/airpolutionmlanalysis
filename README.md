# Almaty air-quality analysis

This repository contains the data-processing, imputation, statistical-analysis, modelling, and figure-generation scripts used for the Almaty air-quality study (January–June 2026).

## Data access

The monitoring data are not included in this repository. They may be requested from the corresponding author through an official request, subject to approval by the data owner and applicable access conditions.

## Analysis workflow

1. Put the authorised raw manual and automatic monitoring workbooks in the project directory.
2. Run `combine_datasets.py` to parse, harmonise, and quality-control the source workbooks.
3. Run `impute_datasets.py` to produce station-specific mean, median, linear-interpolation, and KNN-imputed datasets.
4. Run `analyze_dataset.py`, `model_all_imputations.py`, and `name_clusters.py` to generate descriptive statistics, model outputs, and station typology.
5. Run `add_evidence_outputs.py`, `create_full_imputation_heatmaps.py`, and `create_english_publication_figures.py` to prepare the evidence workbooks and figures.

The scripts write generated workbooks, models, and figures locally; those derived outputs are intentionally not versioned here.

## Reproducibility note

Use Python 3.11+ and install the packages in `requirements.txt`. Paths and source workbook names reflect the original study workspace and may need adjustment after receiving authorised data.
