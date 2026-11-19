# Data license

All audio and transcripts in `data/` are a cleaned subset of **ParlaSpeech-HR 2.0**
(Ljubešić, Koržinek, Rupnik; Jožef Stefan Institute, 2024), http://hdl.handle.net/11356/1914,
licensed under **Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)**:
https://creativecommons.org/licenses/by-sa/4.0/

Our changes: we selected a subset, re-encoded the audio to 16 kHz mono Opus, fixed
transcript formatting, removed clips (see `data/clean/cleaning_log.csv`), and added a
train/dev/test split. Under the ShareAlike term, this derived data is also shared under
CC BY-SA 4.0.

Please cite the original dataset and its paper:

- Dataset: Ljubešić, N., Koržinek, D., Rupnik, P. (2024). *ParlaSpeech-HR 2.0*. Slovenian language
  resource repository CLARIN.SI. http://hdl.handle.net/11356/1914
- Paper: "ParlaSpeech-HR - a Freely Available ASR Dataset for Croatian Bootstrapped from the
  ParlaMint Corpus", ParlaCLARIN III workshop, LREC 2022.
  http://www.lrec-conf.org/proceedings/lrec2022/workshops/ParlaCLARINIII/pdf/2022.parlaclariniii-1.16.pdf
