# Text-to-Speech

`perception.speech` is a thin, replaceable adapter over the project's own speech
factories (`core.speech.load_synthesizer`). TTS is **optional** and never touches
reasoning or knowledge storage.

## Requirements

* **Fully offline**: it uses a local system engine (`espeak-ng`/`espeak`/`flite`/
  Windows OneCore/SAPI/`say`), a local **Piper** ONNX voice, **Qwen3-TTS**, or the
  local **audio.cpp** VoiceTut voice. Never a cloud service.
* **Voice and language** are configurable through the project's own voice catalog
  (`core.voices`): a catalog key, a real installed `system:<engine>:<name>` voice,
  or a local `model:<engine>:<name>` voice.
* **Output**: a WAV file (`synthesize_to_file`) or a playable
  `core.speech.AudioClip` (`speak_clip`) that a window can hand to the project's
  `Speaker`.

## Generating speech

`synthesize_to_file(text, *, output=None, voice="", language="",
synthesizer=None)`:

* the engine is chosen by discovery unless `voice` names one (a `system:`/
  `model:` key routes to its engine through the project's reader);
* the language is detected from the text when not pinned;
* `output` defaults to a temporary file under the data folder; the caller owns the
  returned path;
* returns a `SpeechOutput` (`path`, `engine`, `voice`, `language`,
  `duration_seconds`, `silent`).

`synthesize_to_file("   ")` raises `SpeechPerceptionError` (empty text).

## Graceful fallback

When **no** engine is installed the project's `silent` synthesizer is used and
`SpeechOutput.silent` is set — so a caller gets an honest empty clip and a status
line ("produced a silent clip (no speech engine is installed)"), never an
exception. Playback via `speak_clip` likewise degrades to an empty clip.

## Resource limits

* One engine is loaded per call; the project's rule is that TTS never loads every
  model at once.
* `PerceptionLimits` bounds the operation (a timeout and a temp workspace that is
  always cleaned up).
* The voice is exclusive to the caller's studio/window; choosing one never changes
  another surface's.

## Isolation from reasoning

TTS produces audio and nothing else. It writes no knowledge, records no evidence,
and cannot affect the Cognitive Core, Memory, Learning, or Reasoning — pinned by
the architecture's own rule that TTS "must remain optional and must not affect core
reasoning or knowledge storage".

## On this machine

No speech engine is currently installed (`text-to-speech: not available`); the
status reports the real next step (`python -m cli deps --install speech`, or
install Piper). The tests exercise the path through the silent stand-in and assert
the honest outcome rather than a false success.
