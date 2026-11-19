# Croatian Parliament ASR: a speaker- and session-disjoint benchmark on ParlaSpeech-HR

COE025 Natural Language Processing, Project 3 (Automatic Speech Recognition).

We build a cleaned Croatian speech dataset from parliamentary recordings (domain: legislative
speech), with a train/dev/test split that is disjoint by speaker and by session. We run every
course baseline that supports Croatian on one fixed test set with one scoring policy, and compare
them with our own fine-tuned models.

* Language: **Croatian (hr)**. Not English and not the mother tongue of any group member.
* Domain: **parliamentary / legislative speech** (Croatian Sabor).
* Source: **ParlaSpeech-HR 2.0** (Ljubešić, Koržinek, Rupnik; CLARIN.SI,
  http://hdl.handle.net/11356/1914, CC BY-SA 4.0). We read the Parquet copy of this release on the
  Hugging Face Hub, `classla/ParlaSpeech-HR`, because it lets us download single clips instead of
  157 GB of archives. The exact commit is recorded in `data/raw/SOURCE.json`.
* The source has no train/dev/test split. We make our own (see "Split").

> Status: the data pipeline, the scoring code and the tests are finished. Baseline and training
> results are produced by the notebooks in `notebooks/`. Every number in this repository comes
> from a committed file: `data/clean/*.json` for the data, `results/RESULTS.test.md` for the models.

## Repository layout

| Path | What it is |
|---|---|
| `README.md` | This file. |
| `LICENSE` | MIT license for our code. |
| `.gitignore` | Ignores Python caches and `work/` (see below). |
| `configs/data.yaml` | Every setting of the data pipeline: seed, target hours, speaker caps, cleaning thresholds, audio format. |
| `configs/baselines.yaml` | Every course baseline and every published Croatian checkpoint, with the reason when one is not run. |
| `configs/train_scratch.yaml` | Settings of the neural baselines we train from scratch (model sizes, training, the decoding settings searched on dev). |
| `configs/train_gmm_hmm.yaml` | Settings of the MFCC + GMM-HMM baseline (mixture schedule, the decoding settings searched on dev). |
| `configs/finetune.yaml` | Settings of our contribution: fine-tuning `openai/whisper-small` and `facebook/wav2vec2-xls-r-300m` (pinned revisions, training, the decoding settings searched on dev). |
| `hrasr/` | Python package used by all scripts. |
| `hrasr/config.py` | Paths and config loading. |
| `hrasr/source.py` | Reads the source Parquet files (Hugging Face Hub or a local folder). |
| `hrasr/selection.py` | Chooses speakers, sessions and clips to download. |
| `hrasr/cleaning.py` | Cleaning rules and reason codes; session IDs; word-alignment features. |
| `hrasr/textnorm.py` | Transcript fixes and the scoring normalization. |
| `hrasr/split.py` | Applies and checks the speaker- and session-disjoint split; statistics. |
| `hrasr/audio.py` | ffmpeg decoding and encoding. |
| `hrasr/metrics.py` | Corpus WER and CER with bootstrap confidence intervals. |
| `hrasr/errors.py` | Error taxonomy: recall of names and negation words. |
| `hrasr/subgroups.py` | Subgroups of the reference clips (speaking rate, speaker gender, clip length) and WER/CER per group. |
| `hrasr/manifest.py` | JSON Lines read/write. |
| `hrasr/inference.py` | Inference with published checkpoints (NeMo, wav2vec 2.0 CTC, Whisper): same decoded audio for every system, pinned revisions, timing. |
| `hrasr/features.py` | Log-mel and MFCC features, CMVN and SpecAugment in plain torch. |
| `hrasr/vocab.py` | Character vocabulary of the models we train (targets: the scoring-normalized text). |
| `hrasr/conformer.py` | Conformer encoder (convolution subsampling, self-attention, convolution module). |
| `hrasr/transducer.py` | RNN-T loss and greedy transducer decoding in plain torch. |
| `hrasr/scratch_models.py` | CTC, hybrid CTC/attention encoder-decoder and RNN-T models; their decoding. |
| `hrasr/scratch_data.py` | Feature cache per split and length-bucketed batches. |
| `hrasr/gmm_hmm.py` | MFCC + GMM-HMM: character HMMs, Viterbi training, decoding with a character bigram LM. |
| `hrasr/finetune.py` | Fine-tuning helpers: audio in memory, Whisper labels with the Croatian prompt, the XLS-R character vocabulary, learning-rate schedule. |
| `scripts/01_fetch_metadata.py` ... `scripts/08_pack_audio.py` | The data pipeline, one script per step (see "Pipeline"). |
| `scripts/build_dataset.sh` | Runs steps 1-8. |
| `scripts/unpack_audio.py` | Extracts the audio shards into `work/audio/`. |
| `scripts/infer_pretrained.py` | Runs one published checkpoint on dev, test and the no-speech clips; writes predictions and `.meta.json`. |
| `scripts/train_scratch.py` | Trains `ctc_scratch`, `aed_hybrid_scratch` or `rnnt_scratch`, chooses the checkpoint and decoding settings on dev, decodes test once. |
| `scripts/train_gmm_hmm.py` | Trains `gmm_hmm`, chooses the decoding settings on dev, decodes test once. |
| `scripts/finetune.py` | Fine-tunes `whisper_small_ft` or `xlsr_300m_ft`, chooses the checkpoint and decoding settings on dev, decodes test once. |
| `scripts/score.py` | Scores every system's predictions and writes the result tables. |
| `data/LICENSE.md` | License and citation of the data. |
| `data/raw/` | Raw data: `manifest.jsonl.gz` (every downloaded clip exactly as in the source, with transcripts, word alignment and speaker metadata), `SOURCE.json`, `v1_speakers.json`, `selection_report.json`. |
| `data/clean/` | Cleaned data: `train.jsonl`, `dev.jsonl`, `test.jsonl`, `silence_probe.jsonl`; logs and reports: `cleaning_log.csv`, `text_fixes.csv`, `cleaning_report.json`, `split_log.csv`, `split_stats.json`. |
| `data/audio/` | Audio of every raw clip and every no-speech clip, as Opus files in tar shards (`shard_*.tar`, each < 90 MB), and `INDEX.csv`. The raw and the clean manifests point into these shards. |
| `notebooks/` | Kaggle notebooks: `00_build_dataset.ipynb` runs the data pipeline (CPU); `01_baselines_pretrained.ipynb` runs every published checkpoint (GPU); `02_train_scratch.ipynb` trains one neural baseline from scratch (GPU); `03_train_gmm_hmm.ipynb` trains the GMM-HMM (CPU); `04_finetune.ipynb` runs our fine-tuning (GPU). |
| `results/` | `predictions/` (raw per-clip output of every system and a `.meta.json` per system), `train_logs/` (training curve and every choice made on dev, for each model we trained), `scores.*.json`, `RESULTS.*.md`. |
| `tests/` | Tests of the data pipeline and the evaluation (see "Tests"); `conftest.py` puts the repository on the Python path. |
| `requirements/data.txt` | Pinned package versions for the pipeline, scoring and tests. |
| `requirements/baselines.txt` | Pinned package versions for inference with published checkpoints (own virtual environment on the Kaggle GPU image). |
| `requirements/train.txt` | Pinned package versions for training our models on Kaggle (own virtual environment: the data requirements plus torch). |
| `requirements/finetune.txt` | Pinned package versions for fine-tuning on Kaggle (own virtual environment: torch, transformers and the scoring packages). |
| `contributions/` | One file per group member: `<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md`. |

`work/` is not in the repository. It only holds files that a script rebuilds: the metadata table
of the full corpus (step 1), the selection table (step 3), the kept-clip table (step 5) and the
unpacked audio. The dataset itself, raw and clean, is committed in `data/`.

## Reproduce

Requirements: Python 3.11-3.13 and ffmpeg with libopus.

```bash
pip install -r requirements/data.txt
python -m pytest -q                 # tests
bash scripts/build_dataset.sh       # source corpus -> data/raw, data/clean, data/audio
```

To work from the committed data only (no download):

```bash
python scripts/unpack_audio.py      # data/audio -> work/audio
python scripts/05_clean.py          # data/raw -> work/clean_kept.parquet + cleaning logs
python scripts/06_split.py          # -> data/clean/{train,dev,test}.jsonl
```

`05_clean.py` unpacks the audio by itself if `work/audio/` is empty. On Kaggle,
`notebooks/00_build_dataset.ipynb` runs steps 1-8. The seed is `20261119` (`configs/data.yaml`);
the same config gives the same selection, cleaning and split.

Models (Kaggle, `notebooks/01`-`04`): each notebook clones the repository and makes a virtual
environment with only the pinned packages of its requirements file, so the preinstalled packages
of the Kaggle image cannot change the result (NeMo needs other versions of protobuf and jiwer than
the image and than our scoring). Each requirements file is the complete `pip freeze` of that
environment on Kaggle. Notebooks 02 and 04 train up to two systems at the same time, one per GPU of a
"T4 x2" session; every training run has a compute budget (`max_hours` in its config, set so that all
runs fit Kaggle's weekly GPU quota). The notebooks write predictions; all systems are then scored
together, with `requirements/data.txt`:

```bash
python scripts/score.py --split dev
python scripts/score.py --split test
```

## Dataset

### Why this source

* **Trusted references.** The transcripts are the official records of the Croatian parliament
  (ParlaMint), aligned to the parliament's YouTube recordings by the ParlaSpeech authors. We never
  use a model's output as a reference.
* **Speaker and session identity.** Every clip has the speaker's name and gender, and its ID names
  the sitting it comes from, so the split can be disjoint by both.
* **Size.** The source has about 3,061 hours, so our subset can be large and still have many
  speakers.

What counts as the transcript (W9 slide 3): the references are **edited** parliamentary records,
not verbatim transcripts. Hesitations, false starts and repetitions may be missing from them. Every system is
scored against the same edited text.

### Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_fetch_metadata.py` | Reads the metadata of every clip (about 870,000 rows) without audio. Reduces each word alignment to five numbers. Records the source commit SHA. |
| 2 | `02_fetch_v1_speakers.py` | Lists every speaker of ParlaSpeech-HR v1.0 (see "Leakage"). |
| 3 | `03_select_subset.py` | Chooses speakers, sessions and clips (see "Selection"). |
| 4 | `04_fetch_audio.py` | Downloads only the chosen rows, copies them unchanged into `data/raw/manifest.jsonl.gz`, hashes the original audio bytes and stores the audio as 16 kHz mono Opus. |
| 5 | `05_clean.py` | Fixes transcript formatting and removes broken clips; logs every change. |
| 6 | `06_split.py` | Applies and checks the speaker- and session-disjoint split; writes the statistics. |
| 7 | `07_make_silence_probe.py` | Makes 60 no-speech clips with empty references (W9 slide 44). |
| 8 | `08_pack_audio.py` | Packs the audio into shards under GitHub's 100 MiB file limit. |

### Selection

* Targets after cleaning: 50 h train, 5 h dev, 5 h test. Targets and speaker caps count audio
  that is expected to pass cleaning: the metadata-only part of the cleaning rules is run on the
  whole corpus first. We still download problem clips from the same speakers and row groups (at
  most 1.5 times the speaker's expected-clean audio), so the raw data keeps its real problems and
  step 5 removes them with a logged reason. We download 25% more than the targets, for problems
  that only the audio shows. `data/clean/cleaning_report.json` compares the prediction with the
  real cleaning result.
* Result: the cleaning removed exactly the clips that the metadata rules predicted
  (`selection_prediction_check` in `data/clean/cleaning_report.json`: 0 and 0), so the 25% margin
  was not used up, and the splits are larger than the targets: 62.502 h train, 6.319 h dev,
  6.617 h test (`data/clean/split_stats.json`). We keep them as built; the config is unchanged.
* Speaker gender is "-" (unknown) in the source for some speakers (`speakers_by_gender` in
  `data/clean/split_stats.json`); the subgroup table reports them as "unknown".
* Speaker caps: at most 0.75 h of clean audio per training speaker and 0.5 h per dev/test
  speaker, so the data has many voices instead of a few frequent speakers.
* Dev/test speakers come from an eval pool: speakers who are **not in ParlaSpeech-HR v1.0** and
  who first speak on or after **2020-07-22**. Test is filled first, then dev, alternating female
  and male speakers. A dev or test speaker never appears in train.

### Cleaning

Reformatting is not cleaning. Each rule below finds a real problem. A removed clip is written to
`data/clean/cleaning_log.csv` with its first reason and all its reasons; the counts are in
`data/clean/cleaning_report.json`.

| Reason code | Problem |
|---|---|
| `missing_speaker` | No speaker name: the clip cannot be placed in a speaker-disjoint split. |
| `duplicate_id` | The same clip ID appears twice. |
| `duplicate_audio` | Identical original audio bytes (SHA-1) as an earlier clip. |
| `decode_error` | The audio cannot be decoded. |
| `duration_mismatch` | Decoded length differs from the metadata length by more than 0.25 s. |
| `duration_out_of_range` | Shorter than 1 s or longer than 30 s. |
| `empty_text` | Nothing left after cleaning. |
| `annotation` | Brackets or symbols that mark editorial notes (e.g. interjections), which the audio may not match. |
| `digits` | Digits in the transcript. "5" can be spoken as *pet*, *peti*, *petog*, ... (Croatian case endings), so the reference is ambiguous. |
| `non_latin_script` | Letters from another script. |
| `speaking_rate` | Fewer than 6 or more than 25 characters per second: text and audio probably do not match. |
| `no_word_alignment` | No word-level timing. |
| `alignment_gap` | More than 2.5 s before the first or after the last aligned word, or more than 3 s between two words: probably untranscribed speech. |
| `silence` | RMS level below -50 dBFS. |

Fixes that keep the clip (logged in `data/clean/text_fixes.csv`): Unicode NFC, typographic quotes
and dashes to ASCII, invisible characters removed, repeated spaces collapsed.

### Split

Course rule (W9, "Assign related recordings together before splitting"): group by speaker and by
recording session, then assign whole groups.

* **Speakers:** a speaker is in exactly one split.
* **Sessions:** a session is one sitting of the parliament (the part of the clip ID before `.u`).
  The sessions of eval-pool speakers are split at random (fixed seed) into test sessions and dev
  sessions; test clips come only from test sessions and dev clips only from dev sessions. A train
  clip never comes from a session that has a dev or test clip.
* **Recordings:** a recording in this corpus is one speech by one speaker in one session, so the
  clips cut from it stay together by both rules.
* `hrasr/split.py` enforces both rules on the cleaned clips. A clip that breaks them is removed and
  logged in `data/clean/split_log.csv`; if a speaker or session is still in two splits, the script
  stops with an error.
* `data/clean/split_stats.json` reports, per split, the number of speakers, sessions, utterances
  and hours, the speakers by gender, and how many dev/test sentences also appear word for word
  (after normalization) in train.
* Dev is used to choose settings and checkpoints. Test is used once, at the end, after everything
  is frozen.

### Leakage

Several published Croatian baselines were trained on **ParlaSpeech-HR v1.0**:
`classla/wav2vec2-xls-r-parlaspeech-hr`, `nvidia/stt_hr_conformer_ctc_large`,
`nvidia/stt_hr_conformer_transducer_large` and `nvidia/stt_hr_fastconformer_hybrid_large_pc`.
v1.0 is built from ParlaMint 2.1, whose debates mostly end in mid-2020. If our test speakers or
sessions were in that data, these baselines would look better than they are. Therefore:

1. Step 2 downloads the v1.0 metadata (http://hdl.handle.net/11356/1494) and lists every speaker
   in it (`data/raw/v1_speakers.json`). None of them can be a dev or test speaker.
2. Dev/test speakers must also first speak on or after 2020-07-22, the first sitting of the 10th
   Sabor, so their sessions are after the end of ParlaMint 2.1.

`5roop/whisper-large-v3-ParlaSpeech-HR` was fine-tuned on ParlaSpeech-HR 2.0, the corpus our test
set comes from, so it may have seen our test speakers. We do not report it.

### Audio format

The source audio (FLAC) is too large for a Git repository. We store 16 kHz mono Opus at 32 kbit/s.
Every system reads the same Opus files through the same ffmpeg decoder (`hrasr/audio.py`), so the
comparison is fair. All models in this benchmark take 16 kHz input. Clips are at most 30 s long,
so no system needs long-audio chunking (W9 slide 39).

## Evaluation

* One test set: the same clip IDs and references for every system. `scripts/score.py` stops if a
  system has no prediction for a clip.
* One normalization policy for every system (`hrasr/textnorm.normalize_for_scoring`): Unicode NFKC,
  lowercase, every character that is not a letter or a digit becomes a space, repeated spaces
  collapsed. The Croatian letters č, ć, đ, š, ž are kept, because they distinguish words. We do not
  use Whisper's normalizer: it is built for English. References contain no digits (see "Cleaning");
  a system that writes a number as digits is counted as wrong, because the task is to write the
  spoken words.
* Corpus-level scores: edit counts are added over all clips and divided by the total number of
  reference words or characters (W9 slide 12). A 95% bootstrap interval (1,000 resamples of clips)
  shows whether two systems really differ.
* Raw predictions are saved before scoring in `results/predictions/`, with checkpoint, revision,
  language setting, decoding settings, hardware, precision and batch size in `.meta.json`
  (W9 slide 45).
* Error taxonomy (W9 slide 13): recall of **names** (capitalized words that do not start a
  sentence: people, parties, places, laws) and of **negation** words (*ne, nije, nisu, nema, neće,
  ...*), next to recall of all words, and the most often missed names.
* Subgroups (`hrasr/subgroups.py`): WER and CER per group of reference clips, with the same
  normalization and corpus-level counts. The groups come from the references only, so every
  system is scored on the same clips in each group. **Speaking rate**: words per second of the
  normalized reference; the borders between slow, medium and fast are the 1/3 and 2/3 quantiles
  of the scored split and are printed in the table. **Speaker gender**: from the source metadata.
  **Clip length**: 0-5 s, 5-10 s, 10-20 s, >= 20 s.
* No-speech clips (W9 slide 44): 60 clips of digital silence, quiet white noise and low-frequency
  noise with empty references. WER is not defined here (zero reference words), so we report how
  many clips get any output and how many words are emitted.

### Why these metrics

Following the Hugging Face Audio Course, chapter 5 "Evaluation"
(https://huggingface.co/learn/audio-course/en/chapter5/evaluation):

* **WER** is the standard metric for speech recognition. It counts word substitutions, deletions
  and insertions against the number of reference words, and a word counts only if it is fully right.
* **CER** uses the same counts on characters, so a word with one wrong letter is only partly wrong.
  This matters for Croatian: it is highly inflected, and many errors are a wrong case ending
  (*zakon, zakona, zakonu, zakonom*). WER counts such a word as fully wrong; CER shows that most of
  the word was right. We report both.
* **Normalized scoring.** The course recommends training on orthographic text (with case and
  punctuation) and normalizing predictions and references before computing WER. We do the same,
  with a normalizer that keeps Croatian letters.
* **RTFx** (audio seconds / processing seconds) is reported next to WER, because a slightly better
  model can be much slower. Hardware and batch size are in each system's `.meta.json`.

## Baselines

The full list is in `configs/baselines.yaml`. `results/RESULTS.test.md` shows a row for every
entry: a score, or the reason it was not run. A published checkpoint that someone else fine-tuned
is a baseline. A course baseline that we implement and train ourselves is still a baseline and is
reported in its own table.

| Family | Checkpoint / setup | Status |
|---|---|---|
| MFCC + GMM-HMM | trained by us | planned |
| CTC encoder | trained by us | planned |
| Attention encoder-decoder, hybrid CTC/attention | trained by us | planned |
| Transducer (RNN-T) | trained by us | planned |
| CTC encoder (Conformer-CTC, Croatian) | `nvidia/stt_hr_conformer_ctc_large` | planned |
| Transducer (Conformer-Transducer, Croatian) | `nvidia/stt_hr_conformer_transducer_large` | planned |
| Hybrid transducer/CTC (FastConformer, Croatian) | `nvidia/stt_hr_fastconformer_hybrid_large_pc` | planned |
| wav2vec 2.0 + CTC (Croatian fine-tune) | `classla/wav2vec2-xls-r-parlaspeech-hr` | planned |
| HuBERT + CTC | none: no HuBERT checkpoint with a Croatian ASR head found (searched 2026-11-19); a bare encoder cannot transcribe | not available |
| Whisper (multilingual) | `openai/whisper-small` | planned |
| Canary | `nvidia/canary-1b-v2` | planned |
| Parakeet TDT | `nvidia/parakeet-tdt-0.6b-v3` | planned |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | not run: Croatian is not a supported language ([1.7B card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [0.6B card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)) |

## Our contribution

Our own fine-tuning on the train split, with settings and checkpoints chosen on dev and one final
run on test. Reported in a separate table in `results/RESULTS.test.md`.

## Tests

`python -m pytest -q`

* `tests/test_textnorm.py`: scoring normalization (Croatian letters kept, punctuation removed,
  Unicode forms, idempotence) and the text detectors.
* `tests/test_cleaning.py`: each cleaning rule on good and broken clips; metadata-only mode;
  word-time units; session IDs.
* `tests/test_split_and_selection.py`: v1.0 speakers excluded; dev/test speakers are new speakers
  and never in train; test, dev and train sessions do not overlap; caps and targets in
  expected-clean audio; determinism; session-overlap removal; the leak check.
* `tests/test_metrics.py`: the W9 slide 33-34 example (WER 0.3333, CER 0.2273), corpus vs clip
  averaging, empty hypothesis, empty reference, confidence interval.
* `tests/test_errors_and_scoring.py`: name and negation tagging, category recall, and
  `scripts/score.py` end to end (tables, RTFx, no-speech results, subgroup table, "not run" rows).
* `tests/test_inference.py`: exact 16-bit WAV round trip, the shared ffmpeg decoding, batching,
  NeMo output formats, greedy CTC decoding with a small random wav2vec 2.0 model, the baseline
  config, and that a trial run cannot write into `results/predictions/`. Tests that need torch,
  transformers or ffmpeg are skipped where these are missing; the notebooks run them on Kaggle.
* `tests/test_scratch.py`: features, SpecAugment, vocabulary, batching; the RNN-T loss against a sum
  over all alignments and its gradient; greedy CTC and transducer decoding; that the encoder output of
  an utterance does not depend on the padding of its batch; that each of the three models learns to
  transcribe a small training set; beam search with CTC rescoring; `scripts/train_scratch.py` end to end.
* `tests/test_gmm_hmm.py`: Gaussian mixture likelihood, normalized bigram LM, Viterbi alignment with
  optional pauses, flat start, mixture splitting; training from a flat start recovers the true
  alignments and decodes the training utterances; `scripts/train_gmm_hmm.py` end to end.
* `tests/test_finetune.py`: learning-rate schedule, the XLS-R vocabulary and labels, Whisper labels
  with the Croatian prompt; small random Whisper and wav2vec 2.0 models trained with the real code
  learn the training transcripts; batched CTC decoding equals decoding each clip alone;
  `scripts/finetune.py` end to end.
* `tests/test_subgroups.py`: speaking-rate borders and groups, gender and length groups, and that
  the WER of each group equals scoring that group's clips directly.
* `tests/test_pipeline_e2e.py`: runs steps 1-8 and the unpack script on a synthetic corpus with the
  same Parquet schema, checks that planted problems are removed for the right reason, and that
  step 5 runs from the committed data alone.

## Limitations

* References are edited, not verbatim (see "Dataset").
* No group member speaks Croatian, so we did not listen to and correct clips by hand. Quality
  control is automatic (the rules above), and every decision is logged.
* Parliamentary speech is formal and often prepared; results do not transfer directly to
  conversational Croatian.
