# PDF and OCR

A PDF is not one thing, and the project keeps the three kinds strictly apart so a
scanned page's OCR reading is never mistaken for a native text layer.

## The three kinds

| Kind | What it is | How it is read |
| --- | --- | --- |
| **Text** | every page has a real text layer | `pypdf` (`core.pdf_extract`) |
| **Scanned** | no page has a text layer | OCR of rendered page images (`core.pdf_ocr`) |
| **Mixed** | some pages of each | text where present, OCR only for the low-text pages |

`perception.pdf.classify_pdf(path)` returns `"text"`, `"scanned"`, `"mixed"`, or
`"empty"` **before** any expensive rasterisation runs, so a caller knows whether
OCR is needed at all. `python -m cli perception classify <pdf>` prints it.

## Reading a PDF

`perceive_pdf(path, ocr=None, languages=("eng","ara"), dpi=200)`:

* `ocr=True` forces the OCR path on; `ocr=False` refuses it (native text only);
  `ocr=None` (the default) enables it **automatically when the OCR engine is
  installed** — the same auto-when-available rule the rest of the project uses.
* **OCR only where needed.** A page with real text is never sent to OCR and never
  overwritten; only a low-text page (fewer than `MIN_CHARS_PER_PAGE` = 16
  characters) is. This is `core.pdf_ocr.extract_pdf_pages`'s own rule, reused.
* Each page becomes a `Segment` labelled `"page N"`, with
  `content_type=OCR` for an OCR'd page and `content_type=NATIVE` for a text-layer
  page, so the result records which method read each page.
* A page that failed is named in the warnings; a page skipped by a cap or the time
  budget is named too. **Nothing is silently omitted.**
* The confidence is a quality signal: `1.0 - 0.3 × (OCR share)`, so a pure text
  layer is near-certain and an OCR-heavy read is honestly lower.

A scanned PDF with **no** OCR installed is reported as `EMPTY` with the next step
("Enable OCR (install the [ocr] extra) to read it"), not as a crash.

## OCR for images

`perception.ocr.perceive_image_ocr(path, languages=("eng","ara"), preprocess=False,
region="")` reads a **plain image** file — the one thing the PDF OCR path does not
do. It reuses `core.pdf_ocr`'s availability and language resolvers and the
project's raster guards.

* **Supported image formats**: whatever Pillow opens (`.png`, `.jpg`, `.jpeg`,
  `.webp`, `.bmp`, `.tif`, `.tiff`).
* **Confidence where supported**: Tesseract's own per-word confidence is averaged
  when reported; a `-1` sentinel falls back to the honest engine default (0.75),
  never invented.
* **Preprocessing** is minimal and off by default: an optional greyscale
  conversion (sometimes helps a photographed page, sometimes hurts a clean
  screenshot). A `region="x,y,w,h"` crops to a sub-rectangle.
* **Unreadable images** return a `FAILED` result naming the reason, never raise.
* **Arabic + English** work only when the language data is installed
  (`tesseract-ocr-ara` / `tesseract-ocr-eng`); a missing Arabic model never stops
  an English scan, and `perception.ocr.language_report()` reports what is present.

## OCR text is never ground truth

The result's `content_type` is `OCR` and its confidence is bounded at or below
`MACHINE_QUALITY_CEILING` (0.95), so a downstream layer never treats it as a
native text layer or a verified fact. On this machine: Tesseract 5.5.0, `eng` and
`ara` both installed.

## Optional extras

OCR needs the `[ocr]` extra and the Tesseract **system** package (plus its
language data). Both are installable from the app:

```
python -m cli deps --install ocr --ocr-languages eng+ara
```

`perception.ocr.ocr_install_hint()` is the one next-step string a window or the
CLI shows, and `perception.validator.perception_status()` reports whether OCR,
PDF text, vision, and speech are usable here.
