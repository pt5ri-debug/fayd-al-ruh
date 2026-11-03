# Image Perception

An image can be read two ways, and the project keeps them strictly apart.

## 1. Deterministic perception (facts about the picture)

`perception.images.perceive_image(path, ocr=None)`:

* the image's own **measurable features** via `core.vision`'s dependency-free
  `ImageBuffer`/`extract_features` (width, height, aspect ratio, brightness,
  saturation, edge density, landscape/portrait) — for a PPM, the one format the
  project's own decoder reads;
* the image's **dimensions** for PNG/JPEG/GIF read from the header by
  `perception.metadata` (no full decode);
* the image's **text** via OCR when it is installed (`content_type=OCR`).

`content_type` is `OCR` when text was read and `NONE` when the image carries only
features, so a result is never ambiguous about whether text was read.

## 2. Model interpretation (a reading, not truth)

`perception.images.describe_image_with_model(path, question=...)` asks the local
vision model — Doc's Studio's **Qwen3-VL** GGUF — to describe an image. This is a
*model's reading*, not ground truth, so:

* `content_type` is `MODEL`;
* `confidence` is `MODEL_DESCRIPTION_CONFIDENCE` (0.4), deliberately low;
* `evidence_strength()` ranks it **below** OCR, ASR, and native text
  (`native 0.8 > ocr 0.6 > asr 0.5 > model 0.3`);
* `describe_source()` labels it `image (model), model-derived`.

When no vision model is installed the result is `FAILED` with the real next step,
**never** a fabricated description:

```
python -m pip install -r requirements-docs.txt
python -m cli docs download --approve
```

The `runner` parameter is the injection seam: a test (or the gate) passes its own
runner, so the whole path is exercised with no model and no weights.

## The four things kept apart

| Kind | `content_type` | Treated as |
| --- | --- | --- |
| Text directly in a document | `NATIVE` | the source's own words |
| OCR-derived content | `OCR` | a machine's reading of a picture |
| Model-generated description | `MODEL` | an interpretation |
| A verified fact | — | *never produced by perception* |

A generated image description is **not** ground truth and is never stored as a
verified fact: the bridge stores it as perceived evidence that says so.

## No cloud vision

Every image path is local: the features come from the project's own decoder, OCR
from Tesseract, and the description from a GGUF loaded in this process. There is
no cloud vision API anywhere in the pipeline.
