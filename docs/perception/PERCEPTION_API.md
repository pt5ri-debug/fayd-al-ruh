# The Perception API

Two surfaces: the Python API (the coordinator and the bridge) and the CLI.

## Python API

### The coordinator (the one entry point)

```python
from perception.coordinator import PerceptionCoordinator

coordinator = PerceptionCoordinator(limits=None, provenance=None, token=None)
result = coordinator.perceive(
    "report.pdf",
    ocr=None,               # None = auto when installed; True forces; False refuses
    languages=("eng", "ara"),
    normalize=False,        # the deterministic text pass
    describe_image=False,   # ask the local vision model to describe an image
    transcribe=True,        # run local STT on audio
    file_into_ledger=True,  # file a provenance record
)

result = coordinator.perceive_text("waterflood increases recovery")  # a bare string
```

The coordinator routes the input, runs the right pipeline, validates the result,
attaches provenance, and returns a `PerceptionResult`. A convenience function
`perception.perceive(path, ...)` does the same with a default coordinator.

### The pipelines (each independently usable)

```python
from perception.documents import perceive_document     # text/structured/Word/PDF
from perception.pdf import classify_pdf, perceive_pdf  # text/scanned/mixed
from perception.ocr import perceive_image_ocr          # OCR a plain image
from perception.images import perceive_image, describe_image_with_model
from perception.audio import audio_info, perceive_audio
from perception.speech import transcribe_audio, synthesize_to_file, correct_transcript
```

### The bridge (perception → evidence)

```python
from core.perception_bridge import PerceptionBridge, to_documents, to_evidence

bridge = PerceptionBridge(
    controller=controller,          # validates claims, provides long-term memory
    retriever=retriever,            # makes a perceived document citable
    graph=controller.graph,
    evidence_manager=controller.evidence_manager,
    ontology=controller.ontology,
)
outcome = bridge.ingest(result, index=True, record_evidence=True,
                        submit_claims=True, remember=False)
print(outcome.describe())
```

Or, from the default assistant:

```python
from core.perception_bridge import default_bridge, ingest_result
outcome = default_bridge().ingest(result)
```

### Detection and status

```python
from perception.detector import detect_file, detect_kind, validate_integrity
from perception.validator import perception_status, validate_result
from perception.models import perception_model_status, can_load_vision_model

print(perception_status().describe())        # what this build can perceive
```

## CLI (`python -m cli perception …`)

| Command | What it does |
| --- | --- |
| `status` | what this build can perceive right now, engine by engine |
| `detect <path>` | file type detection and integrity (no content read) |
| `read <path>` | perceive a document/PDF/Word/text file (`--ocr`/`--no-ocr`, `--normalize`, `--languages`) |
| `classify <pdf>` | whether a PDF is text, scanned, or mixed |
| `ocr <image>` | OCR a plain image (`--preprocess`, `--region`, `--languages`) |
| `describe <image>` | ask the local vision model to describe an image |
| `audio <file>` | audio metadata, and a transcript unless `--no-transcribe` |
| `speak <text>` | render text to a WAV with a local engine (`--voice`, `--output`) |
| `ingest <path>` | perceive a file **and** submit it to the Cognitive Core (`--remember`, `--no-claims`) |
| `ingest-text <text>` | ingest a bare string through the same bridge |

Every command accepts `--json` where it produces structured output.

**Exit codes**: `0` success, `2` invalid arguments, `3` an engine/model is not
available, `4` a gate refused the operation, `5` an unexpected failure.

Registered in `cli/__main__.py` as `perception` (alias `perceive`), and mirrored in
`cli/menu.py` as five menu entries.

## Example

```bash
python -m cli perception status
python -m cli perception read docs/architecture/IMPLEMENTATION_ROADMAP.md
python -m cli perception classify report.pdf
python -m cli perception ingest scan.pdf --remember
python -m cli perception audio interview.wav
```
