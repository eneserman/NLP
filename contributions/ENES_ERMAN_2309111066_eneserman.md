# Enes Erman (2309111066, eneserman)

## What I did

* I chose the language (Croatian), the domain (parliamentary speech) and the source corpus
  (ParlaSpeech-HR 2.0).
* I set up the repository and its structure.
* I worked on the data pipeline in `scripts/` and `hrasr/`: selecting speakers and sessions,
  downloading the audio, cleaning the clips, and making the speaker- and session-disjoint split.
* I worked on the evaluation code: WER and CER with confidence intervals, the scoring
  normalization, the error taxonomy and `scripts/score.py`.
* I worked on the tests in `tests/` and checked that they pass.
* I wrote the README and the config files in `configs/`.
* I prepared the Kaggle notebook `notebooks/00_build_dataset.ipynb`.
* I set up the Kaggle account used for all runs and defined the rules for the experiments (one test
  set, choices on dev only, pinned packages, a compute budget).
* With an AI coding assistant, I added the model code and ran it on Kaggle: the published
  checkpoints, the four course baselines we trained ourselves, and the fine-tuning of Whisper-small
  and XLS-R. I reviewed the training logs and the results.
* I prepared the presentation.

## Use of AI tools

I used an AI coding assistant (Claude Code). It wrote most of the code for inference, training and
fine-tuning and their tests, ran the Kaggle notebooks with my account, and drafted the
documentation and the slides. I set the goals and the rules, approved the steps and checked the
results.
