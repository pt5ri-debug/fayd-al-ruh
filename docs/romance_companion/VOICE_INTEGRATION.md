# Romance & Companion Studio — Voice Adapter

Phase 8 gives each character a **voice** (`core/romance/voice.py`). This is a
**thin adapter** over the project's own speech stack — it does not add a second
TTS or STT engine. It reads a character's settings from its own
`CharacterVoiceProfile`, offers only the voices this machine **really** has
(through the project's own `core.voices.voice_sections`), and speaks or transcribes
through the project's own `core.reading.TextReader` and
`core.speech.load_recognizer`.

## Character voice settings

Each character has a `CharacterVoiceProfile`:

| Field | Meaning |
|---|---|
| `engine` | `auto` / `espeak` / `piper` / `qwen3-tts` / `audio.cpp` |
| `voice_key` | the project's own voice key (`woman-1`, `system:…`, `model:…`) |
| `language` / `dialect` | the language the character speaks |
| `speed` / `pitch` / `volume` | the reading's prosody (clamped) |
| `mood` | an emotional state from the project's own mood catalog |
| `enabled` | whether the character speaks at all |

`set_voice` validates (an unknown engine, field, or mood is refused) and clamps
(`speed` 0.25–4.0, `pitch` −1.0–1.0, `volume` 0.0–1.0).

## The three honesty rules

**Only real voices are offered.** `available_voices` / `voices_for_language` are
built from the project's own `voice_sections`, which drops a source that
contributed no voice here — so a listed voice is always one that can actually
speak. A saved voice that has since been uninstalled is reported as unavailable
(`view().voice_available is False`) rather than silently swapped.

**A missing engine never crashes a scene.** `read_aloud` / `speak` / `transcribe`
are best-effort: a disabled voice, an empty text, or an engine that fails returns
`None`, never an exception into the conversation.

**Nothing is sent off the machine.** The adapter only calls the local reader and
recognizer the project already uses.

## The API

| Concern | Method |
|---|---|
| Voices | `available_voices`, `voices_for_language` |
| Profile | `profile`, `set_voice`, `view`, `engine_available` |
| Speaking | `read_aloud`, `speak`, `stop` |
| Listening | `transcribe`, `recognizer_available` |

`read_aloud` applies the character's `voice_key` and `mood` to the reader before
rendering, so a character's voice and mood reach the engine. `stop` stops the
playback the adapter last started. `transcribe` accepts the microphone's
`AudioClip` or a WAV path.

## What this phase does **not** do

- It does not add a speech engine, a voice catalog, or a model downloader — all of
  those are the project's own (`core/speech.py`, `core/voices.py`,
  `core/reading.py`, `core/piper_speech.py`, `core/qwen_speech.py`,
  `core/audiocpp_speech.py`), reused unchanged.
- It does not decide *when* a character speaks — the orchestrator (Phase 10)
  drives that from the conversation.
