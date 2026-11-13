"""The Training Center's own explanation, in Arabic and English.

The window grew to fifteen sections and sixty-four controls, and a reader has to
be able to learn what the whole place is *for*, how the pieces fit together, and
what pressing any one control will actually do -- before pressing it. This module
is that explanation, written once.

It is **generated from the catalogue** (:mod:`core.training_center.catalog`), never
hand-copied: every control's title, summary, what it returns, whether it is
governed, and its exact command come from the one source of truth, so the help can
never describe a control differently from the window that draws it. A test
compares the two.

Dependency-free and Tk-free, so the window, the CLI (``python -m cli
training-center help``), and a test all read the same text.
"""

from __future__ import annotations

from core.training_center.catalog import (
    CAPABILITIES,
    SECTION_SUMMARIES,
    SECTION_TITLES,
    SECTION_TITLES_AR,
    SECTIONS,
    capabilities_for,
)

#: The language keys the help understands. ``ar`` renders the Arabic text with the
#: technical fields (a command, a return value) left as they are -- those are code,
#: not prose, and inventing an Arabic command would be a lie.
LANGUAGES: tuple[str, ...] = ("ar", "en")

_LABELS: dict[str, dict[str, str]] = {
    "ar": {
        "title": "مركز التدريب — الدليل الكامل",
        "intro_heading": "ما هو مركز التدريب؟",
        "flow_heading": "كيف يعمل مركز التدريب؟ (الخطوات بالترتيب)",
        "sections_heading": "الأقسام وكل زرّ فيها",
        "rules_heading": "قواعد الأمان التي تحمي المشروع",
        "returns": "ما يُرجعه",
        "command": "الأمر المكافئ",
        "governed": "يطلب تأكيدك أوّلاً",
        "writes": "يكتب بيانات دائمة",
        "reads": "للقراءة فقط",
        "steps_label": "الخطوات",
        "button_label": "الزرّ",
        "hint_label": "ماذا يحدث عند الضغط",
        "note": "ملاحظة",
    },
    "en": {
        "title": "Training Center — the complete guide",
        "intro_heading": "What is the Training Center?",
        "flow_heading": "How the Training Center works (the steps in order)",
        "sections_heading": "Every section and every control",
        "rules_heading": "The safety rules that protect the project",
        "returns": "Returns",
        "command": "Equivalent command",
        "governed": "asks for your confirmation first",
        "writes": "writes durable data",
        "reads": "read-only",
        "steps_label": "Steps",
        "button_label": "Button",
        "hint_label": "What happens when you press it",
        "note": "Note",
    },
}

_INTRO: dict[str, tuple[str, ...]] = {
    "ar": (
        "مركز التدريب هو المكان الوحيد في المشروع الذي يعلّم فيض الروح: يجمع "
        "المعرفة، ويراجعها، ويبني منها مجموعات بيانات، ثم يدرّب النموذج اللغوي "
        "ويقيسه. لا يوجد مكان آخر يُدرَّب فيه الموديل — كل خيارات التدريب القديمة "
        "نُقلت إلى هنا وبقيت تعمل من داخله.",
        "كل زرّ في النافذة يأتي من «كتالوج» واحد: له عنوان، وشرح في سطر، وما "
        "يُرجعه، وهل يطلب تأكيدك قبل أن يكتب، وأمر سطر أوامر مكافئ. هذا الدليل "
        "مبنيٌّ من الكتالوج نفسه، فلا يمكن أن يصف زرًّا بشكل يخالف النافذة.",
    ),
    "en": (
        "The Training Center is the one place in the project that teaches Fayd "
        "Al-Ruh: it gathers knowledge, reviews it, builds datasets from it, then "
        "trains the language model and measures it. No other place trains the "
        "model -- every older training entry point was moved in here and still "
        "works from inside it.",
        "Every button in the window comes from one catalogue: a title, a one-line "
        "summary, what it returns, whether it asks for your confirmation before it "
        "writes, and an equivalent command-line command. This guide is generated "
        "from that catalogue, so it can never describe a button differently from "
        "the window that draws it.",
    ),
}

_FLOW: dict[str, tuple[tuple[str, str], ...]] = {
    "ar": (
        ("١. اختر مصدر التعلّم", "من قسم «مصادر التعلّم»: قاعدة بيانات، أو الإنترنت، أو Gemini، أو نموذج محلّي، أو مستنداتك، أو كتبك. كل مصدر يُنتج «أمثلة تعلّم» بحالة انتظار."),
        ("٢. راجع الأمثلة", "من قسم «التعلّم بالأسئلة والأجوبة»: اقرأ كل مثال، ثم اقبله أو ارفضه أو صحّحه. لا يُدرَّب النموذج أبدًا على بيانات غير مراجَعة."),
        ("٣. ابنِ مجموعة البيانات", "من قسم «استوديو مجموعات البيانات»: حوّل الأمثلة المقبولة إلى مجموعة بيانات مُصدَّرة بإصدار وسلسلة نسب، وصدّق عليها."),
        ("٤. درّب", "من قسم «الموديل اللغوي (فيض الروح)»: اضبط المعمارية والكوربوس والجدول والمحسّن، ثم درّب أو استكمل. أو من أقسام الضبط الدقيق والتدريب المسبق لطرق أخرى."),
        ("٥. قِس", "من قسم «التقييم والمقارنة» ومن «الموديل اللغوي»: قِس على بيانات محجوزة (خسارة، حيرة، بتات لكل بايت) وقارن قبل وبعد."),
        ("٦. تابِع وسجّل", "من قسمَي «مهام التدريب» و«سجلّ التدريب»: راقب التقدّم الحيّ، وأوقف أو استأنف، ثم راجع كل ما جرى."),
    ),
    "en": (
        ("1. Choose a learning source", "From Learning Sources: a database, the internet, Gemini, a local model, your documents, or your books. Each source produces learning examples in the pending state."),
        ("2. Review the examples", "From Question-and-Answer Learning: read each example, then approve it, reject it, or correct it. The model is never trained on unreviewed data."),
        ("3. Build a dataset", "From the Dataset Studio: turn the approved examples into a versioned, provenance-carrying dataset and validate it."),
        ("4. Train", "From Language Model (Fayd Al-Ruh): set the architecture, the corpus, the schedule, and the optimiser, then train or continue. Or use Fine-Tuning / Pretraining for the other methods."),
        ("5. Measure", "From Evaluation & Benchmarking and the Language Model section: measure on held-out data (loss, perplexity, bits per byte) and compare before and after."),
        ("6. Follow and record", "From Training Jobs and Training History: watch live progress, pause or resume, then review everything that ran."),
    ),
}

_RULES: dict[str, tuple[str, ...]] = {
    "ar": (
        "لا تدريب على بيانات غير مراجَعة: المثال يبقى «منتظرًا» حتى تقبله بنفسك.",
        "الإنترنت مغلق افتراضيًا: أي مصدر خارجي (إنترنت، سحابة، جلب أوزان) يحتاج تشغيلك للمفتاح وموافقتك الصريحة.",
        "الأزرار التي «تطلب تأكيدك» مكتوب عليها ذلك في التلميح، ولن تنفّذ شيئًا قبل أن توافق.",
        "لا يُلمس أي وزن للنموذج إلا عبر مسار التدريب المتحكَّم فيه (LoRA/محسّن)، ولا يُستبدل الأساس.",
        "شخصية «ناتاشا» خاصة: سجلّها لا يُقرأ ولا يُشارك ولا يدخل أي مجموعة تدريب، بحكم البنية لا بحكم الإعداد.",
        "كل رقم في اللوحة مقروء من المخازن الحقيقية، وإن تعذّر القياس على هذه الآلة يُقال ذلك صراحةً بدل تخمينه.",
    ),
    "en": (
        "No training on unreviewed data: an example stays pending until you approve it yourself.",
        "The internet is off by default: any external source (internet, cloud, fetching weights) needs you to switch it on and approve it explicitly.",
        "A control that asks for confirmation says so in its tooltip, and will not do anything before you agree.",
        "No model weight is touched except through the controlled training path (LoRA/optimiser), and the base is never replaced.",
        "Natasha is private: her records are never read, shared, or included in any training set, enforced structurally rather than by a setting.",
        "Every figure on the dashboard is read from the real stores; where this machine cannot measure something, that is said plainly instead of guessed.",
    ),
}

#: A one-line note per section, explaining its *place* in the flow, on top of the
#: catalogue's own summary. Optional: a section without one shows only its summary.
_SECTION_NOTES: dict[str, dict[str, str]] = {
    "ar": {
        "dashboard": "ابدأ من هنا: كل الأرقام الحقيقية في مكان واحد.",
        "sources": "المدخل: من أين تأتي المعرفة.",
        "knowledge": "يحوّل المستندات إلى قاعدة معرفة يمكن الاستشهاد بها.",
        "qa": "بوّابة الجودة: لا شيء يُدرَّب قبل مراجعتك.",
        "datasets": "يحوّل الأمثلة المقبولة إلى بيانات تدريب مُصدَّرة.",
        "jobs": "كل تدريب طويل يعمل في الخلفية ويمكن إيقافه واستئنافه.",
        "finetune": "ضبط دقيق لنموذج متوافق بالطريقة التي يدعمها فعلًا.",
        "pretrain": "تدريب نموذج صغير من الصفر أو استكمال نقطة محفوظة.",
        "language_model": "الموطن الوحيد لكل أوجه تدريب الموديل اللغوي لفيض الروح.",
        "evaluation": "قياس أمين على بيانات محجوزة، ومقارنة قبل/بعد.",
        "kb": "قاعدة المعرفة الكبيرة: بناؤها وفهرستها والبحث فيها.",
        "pipelines": "تجميع الخطوات في خطّ محفوظ وتشغيله بحدود.",
        "history": "سجلّ كل جلسة واستيراد ومجموعة وتدريب وتقييم.",
        "resources": "أرقام هذه الآلة وتقدير أمين لما يحتاجه التدريب.",
        "settings": "تفضيلات مركز التدريب نفسه.",
    },
    "en": {
        "dashboard": "Start here: every real figure in one place.",
        "sources": "The entrance: where the knowledge comes from.",
        "knowledge": "Turns documents into a knowledge base answers can cite.",
        "qa": "The quality gate: nothing is trained before you review it.",
        "datasets": "Turns approved examples into a versioned training set.",
        "jobs": "Every long training run works in the background and can be stopped and resumed.",
        "finetune": "Fine-tune a compatible model with the method it really supports.",
        "pretrain": "Train a small model from scratch, or continue a saved checkpoint.",
        "language_model": "The one home of every way the project trains Fayd Al-Ruh's own language model.",
        "evaluation": "Honest measurement on held-out data, and a before/after comparison.",
        "kb": "The bulk knowledge base: build it, index it, search it.",
        "pipelines": "Combine the steps into a saved pipeline and run it bounded.",
        "history": "Every session, import, dataset, run, and evaluation.",
        "resources": "This machine's own figures and an honest estimate of what a run needs.",
        "settings": "The Training Center's own preferences.",
    },
}


def resolve_language(language: str) -> str:
    """The supported language nearest ``language`` (an unknown value is English)."""
    key = (language or "").strip().lower()[:2]
    return key if key in LANGUAGES else "en"


def help_title(language: str = "ar") -> str:
    return _LABELS[resolve_language(language)]["title"]


def help_text(language: str = "ar") -> str:
    """The whole guide as one readable block in the chosen language."""
    lang = resolve_language(language)
    label = _LABELS[lang]
    lines: list[str] = [label["title"], "=" * len(label["title"]), ""]

    lines.append(label["intro_heading"])
    lines.append("-" * len(label["intro_heading"]))
    for paragraph in _INTRO[lang]:
        lines.append(paragraph)
        lines.append("")

    lines.append(label["flow_heading"])
    lines.append("-" * len(label["flow_heading"]))
    for step, detail in _FLOW[lang]:
        lines.append(f"  {step}")
        lines.append(f"      {detail}")
    lines.append("")

    lines.append(label["rules_heading"])
    lines.append("-" * len(label["rules_heading"]))
    for rule in _RULES[lang]:
        lines.append(f"  • {rule}")
    lines.append("")

    lines.append(label["sections_heading"])
    lines.append("-" * len(label["sections_heading"]))
    for section in SECTIONS:
        caps = capabilities_for(section)
        if not caps:
            continue
        lines.append("")
        lines.append(_section_heading(section, lang))
        summary = SECTION_SUMMARIES.get(section, "")
        note = _SECTION_NOTES.get(lang, {}).get(section, "")
        if summary:
            lines.append(f"  {summary}")
        if note:
            lines.append(f"  ({note})")
        for cap in caps:
            lines.extend(_capability_lines(cap, lang))
    return "\n".join(lines).rstrip()


def _section_heading(section: str, lang: str) -> str:
    title = SECTION_TITLES_AR.get(section, "") if lang == "ar" else SECTION_TITLES.get(section, "")
    english = SECTION_TITLES.get(section, section)
    if lang == "ar" and title:
        return f"■ {title}  —  {english}"
    return f"■ {english}"


def _capability_lines(cap, lang: str) -> list[str]:
    label = _LABELS[lang]
    title = cap.title_ar if (lang == "ar" and cap.title_ar) else cap.title
    summary = cap.summary_ar if (lang == "ar" and cap.summary_ar) else cap.summary
    tags: list[str] = []
    tags.append(label["governed"] if cap.governed else "")
    tags.append(label["writes"] if cap.writes else label["reads"])
    tag_line = " · ".join(t for t in tags if t)
    lines = [f"  • {title}   [{tag_line}]"]
    lines.append(f"      {summary}")
    lines.append(f"      {label['returns']}: {cap.outcome}")
    lines.append(f"      {label['command']}: {cap.cli}")
    return lines


def help_sections() -> tuple[str, ...]:
    """The section keys the guide documents, in the window's own order."""
    return tuple(section for section in SECTIONS if capabilities_for(section))


def help_capability_count() -> int:
    """How many controls the guide documents (it must equal the catalogue's own)."""
    return len(CAPABILITIES)


__all__ = [
    "LANGUAGES",
    "help_capability_count",
    "help_sections",
    "help_text",
    "help_title",
    "resolve_language",
]
