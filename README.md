# Modeling Multimodal Regression with Mixture Density Networks

**MVA course — Probabilistic Graphical Models**

When one input has several plausible outputs, predicting a single mean can miss the structure of the data. This joint project by **Ying JIN and Gloire LINVANI** models the full conditional distribution with Mixture Density Networks (MDNs) and compares them with a gated two-expert model.

**My contribution.** I designed and ran experiments 2–4, implemented GatedDual, and contributed to the theoretical framing and analysis. Start with [experiment 4](exp4.py) for the gated-expert comparison and the [report](report.pdf) for results and figures.

## Research questions

- How does an MDN represent multimodal inverse relationships that a point-prediction model cannot capture?
- How do the number of mixture components, data noise and model capacity affect the experiments?
- How does the `GatedDual` implementation compare with a full MDN in the synthetic two-dimensional experiment, including parameter counts and the FLOP estimate used in the code?

The [course report](report.pdf) contains the experimental setup, figures, discussion and limitations. It is a student project report, not a claim of peer-reviewed publication. The exploratory fifth experiment discussed there was **not fully validated**; this source package contains experiments 1–4.

## Authorship and individual contribution

The report's Contributions section (page 7) records that **Ying JIN designed and ran experiments 2, 3 and 4, implemented GatedDual, and worked on the theoretical framing, result analysis and discussion**. Gloire LINVANI worked on experiments 1 and 5 and the baseline MLP/standard MDN. The report was written jointly. This repository preserves the complete submitted code needed to understand those experiments; it does not assign every file exclusively to Ying.

## Contents

| File | Purpose |
|---|---|
| `exp1.py`–`exp4.py` | Standalone experiment entry points |
| `exp4.py` | Full MDN / GatedDual comparison and FLOP profiling |
| `Models_training.py` | Model and training helpers |
| `data.py` | Synthetic data generators |
| `data_plot.py`, `utils.py`, `config.py` | Plotting, evaluation utilities and seeding |
| `report.pdf` | Joint course report |

## Running the experiments

Create an isolated Python environment, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python exp2.py
python exp3.py
python exp4.py
```

`python exp1.py` runs the baseline experiment. Experiments generate their own synthetic data and may save figures and checkpoints or open Matplotlib windows. Training settings are defined inside the scripts; several use up to 2,000 epochs with early stopping, so these commands are training runs, not quick checks. `thop` is required by the FLOP profiling in experiment 4. Dependencies are listed from actual imports and are not version-locked.

## Validation and limits

For this portfolio package, all Python files were successfully parsed for syntax on 22 September 2026. Training, numerical results and dependency compatibility were **not rerun** during packaging. Consult the report for the original experiments; no new accuracy or efficiency result is claimed here. The FLOP counts are the profiler estimates for the specific comparison, not a measured end-to-end speedup.

## Reuse

The original final submission did not include an explicit software license. No new license is asserted by this packaging step. Preserve both authors' credit and consult them before redistributing or relicensing their work.


[Back to the MVA project portfolio](https://github.com/PineappleBlowsnow/mva-portfolio)
