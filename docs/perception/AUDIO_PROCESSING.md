# Audio Processing

`perception.audio` is the local audio interface: it validates and measures an
audio file and (optionally) transcribes it with a local engine. Nothing reaches an
online service.

## Supported formats

| Format | Read directly | Notes |
| --- | --- | --- |
| **WAV** | yes (stdlib `wave`) | duration, sample rate, channels from the header |
| **MP3, M4A, AAC, FLAC, OGG, OPUS, WMA** | via a converter | read only when `ffmpeg` or `sox` is installed |

A compressed format is reported as *readable* only if a converter exists; its
duration is left `None` with an honest note, because guessing it would be a lie.
A missing converter is reported as the real next step, never a decode error.

## The pipeline

`perceive_audio(path, transcribe=True, language="", recognizer=None)`:

1. **Validate** — the file exists, is a readable audio file, and is within the
   limits (`validate_audio`).
2. **Measure** — `audio_info(path)` returns an `AudioInfo`
   (`readable`, `duration_seconds`, `sample_rate`, `channels`, `format`, `note`).
3. **Transcribe** (unless `transcribe=False`) — local speech-to-text through
   `perception.speech.transcribe_audio`, stored with `content_type=ASR` and a
   confidence below 1.0.
4. **Segment** — the transcript becomes a `Segment` with `start_seconds` /
   `end_seconds` and an `mm:ss` label.

`transcribe=False` reads only the metadata (a cheap probe) and returns
`content_type=METADATA`.

## What is reported honestly

* **Speaker separation** is *not* performed — no compatible local diarisation is
  bundled — and the result says so in a `PerceptionWarning`. The project never
  claims speaker-identity recognition it does not support.
* **Timestamps**: the bundled engines report no word-level timing; the result says
  so rather than inventing one.
* **Silence** is refused for transcription (`transcribe_audio` raises
  `SpeechPerceptionError("the recording is silent...")`), so silence is never
  turned into invented speech.
* **Optional audio classification / analysis**: not bundled; a caller may inject
  its own `recognizer` (any object with `transcribe(clip)` and a `name`) to extend
  the pipeline without changing it.

## Audio references and provenance

The original audio reference (`source_reference`), a content digest, the engine
name and version, and whether timestamps were reported all travel on the result's
provenance — so a transcript is always traceable to the audio it came from.

## Errors

Every failure is a result: a missing file, an unreadable WAV, a missing converter,
or a missing STT engine yields a `FAILED` result naming the reason (or, for a
missing engine, the install command), never a bare exception.
