# Enes Erman (2309111066, eneserman)

## What I did so far

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

## Use of AI tools

I used an AI coding assistant to help write and review the code, the tests and the
documentation. I checked the results, ran the tests and made the decisions myself.

## Not done yet

* The baseline runs and our own fine-tuning have not been run yet. There are no model results
  in `results/` yet.
