# Speech-to-Text

`perception.speech` is a thin, replaceable adapter over the project's own speech
factories (`core.speech.load_recognizer`), adding the perception guarantees.

## Requirements

* **Windows-compatible and CPU-first**: it uses whatever engine the project's own
  discovery picks — **Qwen3-ASR** (`qwen3-asr`, 52 languages incl. Arabic), then
  the `whisper` program, then a transcript stand-in. No GPU is assumed.
* **Memory**: a Qwen3-ASR model folder is a one-time, human-approved download
  (`python -m cli qwen-asr download --approve`) under `data/qwen-speech`
  (git-ignored). On the target machine (16 GB RAM, integrated graphics) the ASR
  models are small; the loader reports the real next step when the extra or the
  weights are absent.
* **Model selection**: `load_recognizer(name="auto", **options)` loads the engine
  discovery would pick, or a named one.

## Loading and unloading

The project's engines hold no global state, so `load_recognizer()` returns an
object a caller **drops to release it**; `transcribe_audio()` loads, uses, and
releases in one call. Nothing stays resident between calls.

## Transcription

`transcribe_audio(path, recognizer=None, language="", limits=...)` returns
`(text, metadata)`:

* the file is converted to a WAV the recognizer can read (reusing the project's
  converter), then read with `core.speech.AudioClip`;
* the metadata carries the engine name, a bounded confidence (0.6), whether
  timestamps were reported, the audio reference, the language, and the engine
  version;
* a `recognizer` may be injected (a test's, or a caller's own engine) so the path
  is exercised with **no model and no audio hardware**;
* it **raises `SpeechPerceptionError`** — never a bare import error — when no
  engine is installed, the clip is empty, or the engine fails.

## Never verified knowledge

Speech recognition output is **never** treated as verified factual knowledge:

* `content_type` is `ASR`;
* the confidence is a bounded quality signal (`< 1.0`), never a truth claim;
* the caller is invited to **correct** the transcript.

## Transcript correction

`correct_transcript(original, corrected, *, result=None, audio_reference="",
corrected_by="user")` returns a `TranscriptCorrection` — a **new** record that
points back at the original (its `result_id` and `audio_reference`), so:

* the original machine reading is **preserved** (nothing perceived is overwritten);
* the corrected text is honestly labelled as user-corrected, not machine-recognised;
* a blank correction is refused (`SpeechPerceptionError`).

## Offline

After the engine and its weights are installed, transcription runs entirely on
this machine — no online transcription service is used or required.

## On this machine

No STT engine is currently installed (`speech-to-text: not available`); the
coordinator reports the real next step (`python -m cli qwen-asr download
--approve`, or install the `whisper` program). The tests exercise the path with an
injected recognizer, so the pipeline is verified without an engine.
