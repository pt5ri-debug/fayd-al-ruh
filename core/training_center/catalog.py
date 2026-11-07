"""One bilingual description of every Training Center control.

The Training Center grew to a dozen sections and dozens of controls, and a reader
must be able to tell what a control *does* and what it *returns* before pressing
it -- "Build a dataset" and "Import a folder" must not look equally weighted,
because one writes a durable artifact and the other only reads files.

This module is the single source of truth, exactly like
:mod:`core.cognition_catalog` is for the autonomous mind: one
:class:`Capability` per control, carrying its title, a one-sentence summary, what
it returns, whether it is **governed** (asks for confirmation and/or writes
something durable), the exact headless CLI command, and the same text in Arabic.
The window draws each button's label and a hover tooltip from here, and the CLI
prints the table, so no surface can describe a control differently.

It is dependency-free and Tk-free: it imports nothing but the standard library, so
a test, the CLI, and the window all reach it the same way. Nothing here executes
anything -- it only *describes* what already exists.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The sections, in the order the window draws them.
SECTION_DASHBOARD = "dashboard"
SECTION_SOURCES = "sources"
SECTION_KNOWLEDGE = "knowledge"
SECTION_QA = "qa"
SECTION_DATASETS = "datasets"
SECTION_JOBS = "jobs"
SECTION_FINETUNE = "finetune"
SECTION_PRETRAIN = "pretrain"
SECTION_EVALUATION = "evaluation"
SECTION_KB = "kb"
SECTION_HISTORY = "history"
SECTION_RESOURCES = "resources"
SECTION_SETTINGS = "settings"
SECTION_PIPELINES = "pipelines"

SECTIONS: tuple[str, ...] = (
    SECTION_DASHBOARD,
    SECTION_SOURCES,
    SECTION_KNOWLEDGE,
    SECTION_QA,
    SECTION_DATASETS,
    SECTION_JOBS,
    SECTION_FINETUNE,
    SECTION_PRETRAIN,
    SECTION_EVALUATION,
    SECTION_KB,
    SECTION_PIPELINES,
    SECTION_HISTORY,
    SECTION_RESOURCES,
    SECTION_SETTINGS,
)

SECTION_TITLES: dict[str, str] = {
    SECTION_DASHBOARD: "Dashboard",
    SECTION_SOURCES: "Learning Sources",
    SECTION_KNOWLEDGE: "Knowledge Acquisition",
    SECTION_QA: "Question-and-Answer Learning",
    SECTION_DATASETS: "Dataset Studio",
    SECTION_JOBS: "Training Jobs",
    SECTION_FINETUNE: "Model Fine-Tuning",
    SECTION_PRETRAIN: "Pretraining & Continued Training",
    SECTION_EVALUATION: "Evaluation & Benchmarking",
    SECTION_KB: "Knowledge Base Management",
    SECTION_PIPELINES: "Learning Pipelines",
    SECTION_HISTORY: "Training History",
    SECTION_RESOURCES: "Resource Monitor",
    SECTION_SETTINGS: "Settings",
}

SECTION_TITLES_AR: dict[str, str] = {
    SECTION_DASHBOARD: "لوحة المعلومات",
    SECTION_SOURCES: "مصادر التعلّم",
    SECTION_KNOWLEDGE: "اكتساب المعرفة",
    SECTION_QA: "التعلّم بالأسئلة والأجوبة",
    SECTION_DATASETS: "استوديو مجموعات البيانات",
    SECTION_JOBS: "مهام التدريب",
    SECTION_FINETUNE: "الضبط الدقيق للنموذج",
    SECTION_PRETRAIN: "التدريب المسبق والمستمر",
    SECTION_EVALUATION: "التقييم والمقارنة",
    SECTION_KB: "إدارة قاعدة المعرفة",
    SECTION_PIPELINES: "خطوط التعلّم",
    SECTION_HISTORY: "سجلّ التدريب",
    SECTION_RESOURCES: "مراقبة الموارد",
    SECTION_SETTINGS: "الإعدادات",
}

#: One short sentence describing what a whole section is for (English + Arabic).
SECTION_SUMMARIES: dict[str, str] = {
    SECTION_DASHBOARD: "Everything the project has learned and trained, read from the "
    "real stores: sources, documents, chunks, datasets, jobs, models, and resources.",
    SECTION_SOURCES: "Choose where learning comes from: local databases, the internet, "
    "Gemini or another cloud model, a local model, uploaded documents, and books.",
    SECTION_KNOWLEDGE: "Fold documents into the project's own knowledge base and "
    "retrieval, then search, inspect, reindex, back up, and export it.",
    SECTION_QA: "Ask models questions on a topic, compare their answers, review them, "
    "and approve only what is good -- nothing is trained on unreviewed data.",
    SECTION_DATASETS: "Create, validate, split, merge, inspect, and version governed "
    "datasets for every training method, with provenance and lineage.",
    SECTION_JOBS: "Start, pause, resume, cancel, and inspect training jobs, with live "
    "progress, logs, and metrics -- every long run off the interface thread.",
    SECTION_FINETUNE: "Fine-tune a compatible model with the method it really supports "
    "(LoRA over the in-project transformer, or the optional transformers path).",
    SECTION_PRETRAIN: "Train a small language model from scratch, or continue training "
    "an existing checkpoint, with a resource estimate before anything starts.",
    SECTION_EVALUATION: "Measure a model on held-out data -- loss, perplexity, bits per "
    "byte, and instruction-following -- and compare before and after.",
    SECTION_KB: "The bulk knowledge base: build it, index it, search it, and report "
    "exactly how much text and how many chunks it holds.",
    SECTION_PIPELINES: "Combine stages into a saved pipeline (search, download, "
    "extract, clean, ask, review, dataset, train, evaluate) and run it bounded.",
    SECTION_HISTORY: "Every learning session, import, dataset, run, and evaluation, "
    "searchable and filterable, with the lineage of every model.",
    SECTION_RESOURCES: "This machine's own CPU, memory, disk, and GPU figures, and an "
    "honest estimate of what a training run would need.",
    SECTION_SETTINGS: "The Training Center's own preferences: default language, chunk "
    "size, review policy, and which approval gates are in force.",
}


@dataclass(frozen=True)
class Capability:
    """One control the Training Center offers, described once."""

    key: str
    title: str
    section: str
    summary: str
    outcome: str
    cli: str
    governed: bool = False
    #: True for a control that writes durable data (a dataset, a run, an import).
    writes: bool = False
    title_ar: str = ""
    summary_ar: str = ""

    def describe(self) -> str:
        governed = " (asks first)" if self.governed else ""
        return f"{self.title}{governed}\n  {self.summary}\n  Returns: {self.outcome}\n  {self.cli}"


def _cap(
    key: str,
    title: str,
    section: str,
    summary: str,
    outcome: str,
    cli: str,
    *,
    governed: bool = False,
    writes: bool = False,
    title_ar: str = "",
    summary_ar: str = "",
) -> Capability:
    return Capability(
        key=key,
        title=title,
        section=section,
        summary=summary,
        outcome=outcome,
        cli=cli,
        governed=governed,
        writes=writes,
        title_ar=title_ar,
        summary_ar=summary_ar,
    )


CAPABILITIES: tuple[Capability, ...] = (
    # -- dashboard --------------------------------------------------------
    _cap(
        "dashboard.refresh", "Refresh the dashboard", SECTION_DASHBOARD,
        "Read every figure again from the real stores and redraw the dashboard.",
        "A dashboard snapshot: sources, documents, chunks, datasets, jobs, models, resources.",
        "python -m cli training-center dashboard",
        title_ar="تحديث اللوحة", summary_ar="اقرأ كل الأرقام من المخازن الحقيقية وأعد رسم اللوحة.",
    ),
    _cap(
        "dashboard.history", "Recent activity", SECTION_DASHBOARD,
        "Show the most recent learning and training activity.",
        "A bounded list of the newest history entries.",
        "python -m cli training-center history --limit 20",
        title_ar="النشاط الأخير", summary_ar="اعرض أحدث نشاط للتعلّم والتدريب.",
    ),
    # -- learning sources -------------------------------------------------
    _cap(
        "sources.database", "Import from a database (SQLite)", SECTION_SOURCES,
        "Inspect a SQLite database's schema, choose tables and columns, and import it "
        "read-only through the project's resumable SQL importer.",
        "A per-table import report with row and chunk counts.",
        "python -m cli ingest sql-inspect <db>",
        writes=True,
        title_ar="استيراد من قاعدة بيانات", summary_ar="افحص مخطط قاعدة SQLite واختر الجداول والأعمدة واستوردها للقراءة فقط.",
    ),
    _cap(
        "sources.internet", "Learn from the internet", SECTION_SOURCES,
        "Fetch user-supplied URLs (or crawl a site within bounded depth and page "
        "limits) and import the extracted text, honouring robots and rate limits.",
        "A crawl/import report with per-page outcomes.",
        "python -m cli training-center sources --url <url> --approve",
        governed=True, writes=True,
        title_ar="التعلّم من الإنترنت", summary_ar="اجلب روابط يحدّدها المستخدم واستورد النصّ المستخرج ضمن حدود آمنة.",
    ),
    _cap(
        "sources.gemini", "Learn from Gemini (or another cloud model)", SECTION_SOURCES,
        "Send learning questions to Gemini (or any configured OpenAI-compatible "
        "provider) and collect structured answers, Q&A pairs, and synthetic examples.",
        "Generated examples saved as pending learning examples.",
        "python -m cli training-center ask --provider gemini --topic <t>",
        governed=True, writes=True,
        title_ar="التعلّم من Gemini", summary_ar="أرسل أسئلة تعلّم إلى Gemini واجمع أجوبة منظّمة وأزواج أسئلة وأجوبة.",
    ),
    _cap(
        "sources.local_model", "Learn from a local model", SECTION_SOURCES,
        "Ask an installed local model (Ultron, Vision, Banner, the companion, the "
        "project's own checkpoint) and collect its answers for review.",
        "Generated examples saved as pending learning examples.",
        "python -m cli training-center ask --local <mouth> --topic <t>",
        writes=True,
        title_ar="التعلّم من نموذج محلي", summary_ar="اسأل نموذجًا محليًا مثبّتًا واجمع أجوبته للمراجعة.",
    ),
    _cap(
        "sources.documents", "Upload documents", SECTION_SOURCES,
        "Import TXT, Markdown, PDF, DOCX, DOCM, CSV, JSON, JSONL, XML, HTML, and ZIP "
        "archives through the project's own readers (with optional OCR for scans).",
        "A per-file import report with chunk counts and any error.",
        "python -m cli training-center import <path>",
        writes=True,
        title_ar="رفع مستندات", summary_ar="استورد الملفات المدعومة عبر قارئات المشروع نفسها.",
    ),
    _cap(
        "sources.books", "Import books and collections", SECTION_SOURCES,
        "Import EPUB/PDF/TXT/Markdown books and whole folders as collections, keeping "
        "chapter structure and book-level metadata where the format allows.",
        "A collection report with the books and chapters found.",
        "python -m cli training-center import <folder>",
        writes=True,
        title_ar="استيراد كتب ومجموعات", summary_ar="استورد الكتب والمجلدات كمقتنيات مع بنية الفصول.",
    ),
    _cap(
        "sources.characters", "Learn with a character", SECTION_SOURCES,
        "Choose one of the project's characters as a learning participant, using its "
        "own knowledge scope and language, without changing its personality.",
        "Examples attributed to the chosen character.",
        "python -m cli training-center ask --character <key> --topic <t>",
        title_ar="التعلّم مع شخصية", summary_ar="اختر شخصية كمشارك في التعلّم دون تغيير شخصيتها.",
    ),
    # -- knowledge acquisition -------------------------------------------
    _cap(
        "knowledge.ingest", "Fold documents into the knowledge base", SECTION_KNOWLEDGE,
        "Chunk, normalise, dedupe, and index imported text into the project's own "
        "knowledge base so retrieval can cite it.",
        "An ingestion report: sources, chunks, and bytes written.",
        "python -m cli training-center import <folder>",
        writes=True,
        title_ar="إدخال المستندات إلى قاعدة المعرفة", summary_ar="قسّم النصّ وطبّعه وأزل التكرار وأدخله إلى قاعدة المعرفة.",
    ),
    _cap(
        "knowledge.search", "Search the knowledge base", SECTION_KNOWLEDGE,
        "Run the project's own retrieval (lexical, semantic, and ontology-guided) over "
        "the knowledge base and preview the matching chunks with their sources.",
        "Ranked passages with citations.",
        "python -m cli training-center knowledge --query <text>",
        title_ar="ابحث في قاعدة المعرفة", summary_ar="شغّل استرجاع المشروع واعرض المقاطع المطابقة مع مصادرها.",
    ),
    _cap(
        "knowledge.reindex", "Rebuild the search index", SECTION_KNOWLEDGE,
        "Rebuild the SQLite FTS index over the knowledge base's shards.",
        "An index summary: chunk count and size.",
        "python -m cli training-center knowledge --reindex",
        writes=True,
        title_ar="أعد بناء الفهرس", summary_ar="أعد بناء فهرس FTS على مقاطع قاعدة المعرفة.",
    ),
    _cap(
        "knowledge.export", "Back up / export the knowledge base", SECTION_KNOWLEDGE,
        "Copy the knowledge base's shards and manifest to a chosen folder as a backup.",
        "The destination path and the number of files copied.",
        "python -m cli training-center knowledge --export <dir>",
        writes=True,
        title_ar="نسخ احتياطي / تصدير", summary_ar="انسخ مقاطع قاعدة المعرفة إلى مجلد مختار.",
    ),
    # -- question & answer ------------------------------------------------
    _cap(
        "qa.ask", "Ask a question", SECTION_QA,
        "Put a question to the selected provider(s) and record the answer as a "
        "pending learning example for review.",
        "A pending learning example (never trained on until approved).",
        "python -m cli training-center ask --question <q> --provider <p>",
        writes=True,
        title_ar="اطرح سؤالًا", summary_ar="اطرح سؤالًا على المزوّد المختار وسجّل الجواب كمثال قيد المراجعة.",
    ),
    _cap(
        "qa.generate", "Generate questions automatically", SECTION_QA,
        "Generate a set of questions from a topic, a document, or the project's own "
        "knowledge, then ask them to the selected provider(s).",
        "A batch of pending learning examples.",
        "python -m cli training-center ask --topic <t> --count 5",
        writes=True,
        title_ar="توليد أسئلة تلقائيًا", summary_ar="ولّد مجموعة أسئلة من موضوع أو مستند ثم اطرحها على المزوّد.",
    ),
    _cap(
        "qa.review", "Review and approve answers", SECTION_QA,
        "Approve, reject, correct, or mark an answer for review; only approved "
        "answers can be exported to a dataset.",
        "The example's new status.",
        "python -m cli training-center review --approve <id>",
        writes=True,
        title_ar="راجع واعتمد الأجوبة", summary_ar="اعتمد أو ارفض أو صحّح الجواب؛ لا يُصدَّر إلا المعتمد.",
    ),
    _cap(
        "qa.export", "Export approved answers to a dataset", SECTION_QA,
        "Build a governed dataset from the approved learning examples, with their "
        "provenance and review status recorded.",
        "A built dataset version.",
        "python -m cli training-center export --dataset <id>",
        governed=True, writes=True,
        title_ar="صدّر الأجوبة المعتمدة", summary_ar="ابنِ مجموعة بيانات محوكمة من الأمثلة المعتمدة.",
    ),
    # -- datasets ---------------------------------------------------------
    _cap(
        "datasets.list", "List datasets", SECTION_DATASETS,
        "List every governed dataset version with its size, splits, quality, and "
        "eligibility.",
        "A table of dataset versions.",
        "python -m cli train datasets",
        title_ar="اسرد مجموعات البيانات", summary_ar="اسرد كل نسخ مجموعات البيانات مع حجمها وجودتها.",
    ),
    _cap(
        "datasets.build", "Build a dataset", SECTION_DATASETS,
        "Build a validated, versioned dataset from chosen sources (approved learning "
        "examples, a document, a paste), with train/validation/test splits.",
        "A built, eligible dataset version.",
        "python -m cli train build-dataset <id> --text <text>",
        writes=True,
        title_ar="ابنِ مجموعة بيانات", summary_ar="ابنِ مجموعة بيانات مُتحقَّقة ومُرقّمة من مصادر مختارة.",
    ),
    _cap(
        "datasets.validate", "Validate a dataset", SECTION_DATASETS,
        "Re-run the governed validator over a stored dataset and report every problem "
        "(duplicates, leakage, malformed, empty, unverified).",
        "A validation report.",
        "python -m cli train validate <id>",
        title_ar="تحقّق من مجموعة بيانات", summary_ar="أعد تشغيل المُتحقِّق على مجموعة مخزّنة وأبلغ بكل مشكلة.",
    ),
    _cap(
        "datasets.export", "Export a dataset", SECTION_DATASETS,
        "Write a stored dataset version to JSONL in a chosen folder.",
        "The destination path and the number of examples written.",
        "python -m cli training-center datasets --export <id> <dir>",
        writes=True,
        title_ar="صدّر مجموعة بيانات", summary_ar="اكتب نسخة مخزّنة بصيغة JSONL في مجلد مختار.",
    ),
    # -- training jobs ----------------------------------------------------
    _cap(
        "jobs.list", "List training jobs", SECTION_JOBS,
        "List every training run with its status, progress, losses, and resources.",
        "A table of training runs.",
        "python -m cli train runs",
        title_ar="اسرد مهام التدريب", summary_ar="اسرد كل تشغيل تدريب مع حالته وتقدّمه.",
    ),
    _cap(
        "jobs.feasibility", "Check feasibility", SECTION_JOBS,
        "Assess whether a run can start -- compatibility, dataset eligibility, memory, "
        "and disk -- without starting it.",
        "A feasibility report with blockers, warnings, and unknowns.",
        "python -m cli train feasibility --model <m> --dataset <id>",
        title_ar="تحقّق من الجدوى", summary_ar="قيّم إمكانية بدء التشغيل دون بدئه فعليًا.",
    ),
    _cap(
        "jobs.start", "Start a training job", SECTION_JOBS,
        "Start a governed training run on a compatible model and dataset, off the "
        "interface thread, with live progress.",
        "A running job; its result is recorded in the run log.",
        "python -m cli train run --dataset <id> --approve",
        governed=True, writes=True,
        title_ar="ابدأ مهمة تدريب", summary_ar="ابدأ تشغيل تدريب محوكم على نموذج ومجموعة متوافقين.",
    ),
    _cap(
        "jobs.cancel", "Cancel a running job", SECTION_JOBS,
        "Trip the cooperative cancel flag of a running job; it stops at the next step "
        "and records what it did.",
        "The job's new status (cancelled).",
        "python -m cli training-center jobs --cancel <run_id>",
        governed=True, writes=True,
        title_ar="ألغِ مهمة جارية", summary_ar="أوقف المهمة عند الخطوة التالية وسجّل ما أنجزته.",
    ),
    _cap(
        "jobs.resume", "Resume from a checkpoint", SECTION_JOBS,
        "Continue a training run from the adapter an earlier run saved, instead of "
        "starting a fresh one.",
        "A continued run.",
        "python -m cli train run --dataset <id> --resume-from <adapter> --approve",
        governed=True, writes=True,
        title_ar="استأنف من نقطة حفظ", summary_ar="تابع التشغيل من مُهايئ حفظه تشغيل سابق.",
    ),
    _cap(
        "jobs.retry", "Retry a failed job", SECTION_JOBS,
        "Re-run a job that failed, using the same configuration it recorded.",
        "A new run attempt.",
        "python -m cli training-center jobs --retry <run_id>",
        governed=True, writes=True,
        title_ar="أعد محاولة مهمة فاشلة", summary_ar="أعد تشغيل مهمة فشلت بالإعداد نفسه.",
    ),
    # -- fine-tuning ------------------------------------------------------
    _cap(
        "finetune.compat", "Check method compatibility", SECTION_FINETUNE,
        "Report which fine-tuning methods the selected model and runtime really "
        "support on this machine -- a GGUF is never a training base and QLoRA needs a "
        "CUDA GPU.",
        "A compatibility verdict, method by method.",
        "python -m cli train compat --model <m>",
        title_ar="تحقّق من توافق الطريقة", summary_ar="أبلغ أي طرق ضبط دقيق يدعمها النموذج فعلًا على هذا الجهاز.",
    ),
    _cap(
        "finetune.presets", "Show the presets", SECTION_FINETUNE,
        "Show the conservative hyper-parameter presets for a CPU-first machine.",
        "A table of presets.",
        "python -m cli train presets",
        title_ar="اعرض الإعدادات الجاهزة", summary_ar="اعرض الإعدادات المسبقة المتحفّظة لجهاز يعتمد المعالج.",
    ),
    _cap(
        "finetune.adapters", "Adapters and promotion", SECTION_FINETUNE,
        "List the trained adapters, promote an accepted one, or roll back to the base "
        "-- never automatic, always approved.",
        "The adapter registry and its active entry.",
        "python -m cli train adapters",
        title_ar="المُهايئات والترقية", summary_ar="اسرد المُهايئات، ورقِّ مقبولًا، أو ارجع للأساس.",
    ),
    # -- pretraining ------------------------------------------------------
    _cap(
        "pretrain.status", "Pretraining status", SECTION_PRETRAIN,
        "Report the in-project language model's checkpoint, its training history, and "
        "whether continued training is possible.",
        "The checkpoint path, step count, and evaluation.",
        "python -m cli training-center dashboard",
        title_ar="حالة التدريب المسبق", summary_ar="أبلغ عن نقطة حفظ نموذج المشروع وتاريخ تدريبه.",
    ),
    _cap(
        "pretrain.estimate", "Estimate the resources", SECTION_PRETRAIN,
        "Estimate the memory and time a from-scratch or continued run would need, "
        "labelled as an estimate.",
        "A resource estimate with its unknowns named.",
        "python -m cli training-center resources",
        title_ar="قدّر الموارد", summary_ar="قدّر الذاكرة والوقت اللازمين لتدريب من الصفر أو مستمر.",
    ),
    _cap(
        "pretrain.continue", "Continue training the checkpoint", SECTION_PRETRAIN,
        "Continue the project's own language model from its checkpoint -- off the "
        "interface thread, with live progress and a resumable schedule.",
        "A continued checkpoint and a training report.",
        "python -m cli train run --dataset <id> --resume-from <adapter> --approve",
        governed=True, writes=True,
        title_ar="تابع تدريب نقطة الحفظ", summary_ar="تابع نموذج المشروع من نقطة حفظه مع تقدّم مباشر.",
    ),
    # -- evaluation -------------------------------------------------------
    _cap(
        "evaluation.run", "Evaluate a model", SECTION_EVALUATION,
        "Measure a model on a dataset's held-out test split -- next-token loss, "
        "accuracy, perplexity, and a rough instruction-following score.",
        "An evaluation report comparing the base and the candidate.",
        "python -m cli train evaluate <run_id>",
        title_ar="قيّم نموذجًا", summary_ar="قِس النموذج على انقسام اختبار محجوز.",
    ),
    _cap(
        "evaluation.compare", "Compare before and after", SECTION_EVALUATION,
        "Compare two evaluation reports metric by metric and flag a regression.",
        "A regression report.",
        "python -m cli train evaluate <run_id>",
        title_ar="قارن قبل وبعد", summary_ar="قارن تقريري تقييم معيارًا بمعيار.",
    ),
    _cap(
        "evaluation.gate", "Run the training gate", SECTION_EVALUATION,
        "Run the held-out Phase-6 training gate and report its verdict.",
        "The gate's accuracy and its per-judgement detail.",
        "python -m cli train gate",
        title_ar="شغّل بوابة التدريب", summary_ar="شغّل بوابة التدريب المحجوزة وأبلغ بنتيجتها.",
    ),
    # -- knowledge base management ---------------------------------------
    _cap(
        "kb.build", "Build the bulk knowledge base", SECTION_KB,
        "Build the bulk knowledge base from public sources (Wikipedia, Gutenberg, "
        "Wikisource) up to a target size, resumably.",
        "A build report with the sources and chunks stored.",
        "python -m cli knowledge build --target-mb 100",
        governed=True, writes=True,
        title_ar="ابنِ قاعدة المعرفة الكبرى", summary_ar="ابنِ قاعدة المعرفة من مصادر عامة حتى حجم مستهدف.",
    ),
    _cap(
        "kb.stats", "Knowledge base status", SECTION_KB,
        "Report the knowledge base's real size, chunk count, and sources.",
        "A status report.",
        "python -m cli knowledge stats",
        title_ar="حالة قاعدة المعرفة", summary_ar="أبلغ عن الحجم الحقيقي وعدد المقاطع والمصادر.",
    ),
    _cap(
        "kb.index", "Build the search index", SECTION_KB,
        "Build the SQLite FTS index over the knowledge base for millisecond queries.",
        "An index summary.",
        "python -m cli knowledge build-index",
        writes=True,
        title_ar="ابنِ فهرس البحث", summary_ar="ابنِ فهرس FTS على قاعدة المعرفة لاستعلامات سريعة.",
    ),
    # -- pipelines --------------------------------------------------------
    _cap(
        "pipelines.list", "List saved pipelines", SECTION_PIPELINES,
        "List every saved learning pipeline and its stages.",
        "A table of pipelines.",
        "python -m cli training-center pipeline --list",
        title_ar="اسرد الخطوط المحفوظة", summary_ar="اسرد كل خط تعلّم محفوظ ومراحله.",
    ),
    _cap(
        "pipelines.run", "Run a pipeline", SECTION_PIPELINES,
        "Run a saved pipeline's enabled stages in order, with bounded, resumable "
        "execution and a live report.",
        "A per-stage outcome report.",
        "python -m cli training-center pipeline --run <id> --approve",
        governed=True, writes=True,
        title_ar="شغّل خطًا", summary_ar="شغّل مراحل خط محفوظ بالترتيب مع تقرير مباشر.",
    ),
    # -- history ----------------------------------------------------------
    _cap(
        "history.show", "Show the history", SECTION_HISTORY,
        "Show the durable history of learning sessions, imports, datasets, runs, and "
        "evaluations, newest first.",
        "A bounded list of history entries.",
        "python -m cli training-center history",
        title_ar="اعرض السجل", summary_ar="اعرض سجل جلسات التعلّم والاستيراد والمجموعات والتشغيل.",
    ),
    _cap(
        "history.search", "Search the history", SECTION_HISTORY,
        "Filter the history by kind or free text.",
        "The matching history entries.",
        "python -m cli training-center history --kind import",
        title_ar="ابحث في السجل", summary_ar="رشّح السجل بالنوع أو بنصّ حرّ.",
    ),
    # -- resources --------------------------------------------------------
    _cap(
        "resources.read", "Read this machine's resources", SECTION_RESOURCES,
        "Read the machine's own CPU, memory, disk, and GPU figures, each with its "
        "provenance and an honest unknown where the platform cannot say.",
        "A resource snapshot.",
        "python -m cli training-center resources",
        title_ar="اقرأ موارد الجهاز", summary_ar="اقرأ أرقام المعالج والذاكرة والقرص وكرت الرسوم.",
    ),
    # -- settings ---------------------------------------------------------
    _cap(
        "settings.show", "Show the settings", SECTION_SETTINGS,
        "Show the Training Center's own preferences and the approval gates in force.",
        "A settings summary.",
        "python -m cli training-center settings",
        title_ar="اعرض الإعدادات", summary_ar="اعرض تفضيلات مركز التدريب وبوابات الموافقة السارية.",
    ),
    _cap(
        "settings.set", "Change a setting", SECTION_SETTINGS,
        "Change one Training Center preference (default language, chunk size, review "
        "policy) and persist it.",
        "The saved settings.",
        "python -m cli training-center settings --set chunk_chars=1000",
        writes=True,
        title_ar="غيّر إعدادًا", summary_ar="غيّر تفضيلًا واحدًا في مركز التدريب واحفظه.",
    ),
    _cap(
        "settings.catalog", "Show every control", SECTION_SETTINGS,
        "Print this catalogue: what every control does, what it returns, and the "
        "headless command that runs it.",
        "The capability table.",
        "python -m cli training-center catalog",
        title_ar="اعرض كل الأدوات", summary_ar="اطبع هذا الدليل: ما تفعله كل أداة وما تُرجعه وأمرها.",
    ),
)

_BY_KEY: dict[str, Capability] = {cap.key: cap for cap in CAPABILITIES}


def capability(key: str) -> Capability | None:
    """The capability with this key, or ``None`` for an unknown key."""
    return _BY_KEY.get(key)


def capabilities_for(section: str) -> tuple[Capability, ...]:
    """Every capability in one section, in declaration order."""
    return tuple(cap for cap in CAPABILITIES if cap.section == section)


def section_title(section: str) -> str:
    return SECTION_TITLES.get(section, section)


def section_summary(section: str) -> str:
    return SECTION_SUMMARIES.get(section, "")


def catalog_text() -> str:
    """The whole catalogue as one readable block, grouped by section."""
    lines: list[str] = ["Training Center — what every control does", ""]
    for section in SECTIONS:
        caps = capabilities_for(section)
        if not caps:
            continue
        lines.append(f"{SECTION_TITLES[section]}")
        if SECTION_SUMMARIES.get(section):
            lines.append(f"  {SECTION_SUMMARIES[section]}")
        for cap in caps:
            governed = " (asks first)" if cap.governed else ""
            lines.append(f"  • {cap.title}{governed}")
            lines.append(f"      {cap.summary}")
            lines.append(f"      Returns: {cap.outcome}")
            lines.append(f"      CLI: {cap.cli}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "CAPABILITIES",
    "SECTIONS",
    "SECTION_DASHBOARD",
    "SECTION_DATASETS",
    "SECTION_EVALUATION",
    "SECTION_FINETUNE",
    "SECTION_HISTORY",
    "SECTION_JOBS",
    "SECTION_KB",
    "SECTION_KNOWLEDGE",
    "SECTION_PIPELINES",
    "SECTION_PRETRAIN",
    "SECTION_QA",
    "SECTION_RESOURCES",
    "SECTION_SETTINGS",
    "SECTION_SOURCES",
    "SECTION_SUMMARIES",
    "SECTION_TITLES",
    "SECTION_TITLES_AR",
    "Capability",
    "capabilities_for",
    "capability",
    "catalog_text",
    "section_summary",
    "section_title",
]
