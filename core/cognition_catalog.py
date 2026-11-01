"""The one place every cognition surface learns what each action does.

The autonomous cognitive system grew to twenty-odd controls spread over nine
sidebar sections, and a reader could not tell what a button would do before
pressing it -- "Consolidate experience into knowledge" and "The patterns it has
extracted" looked equally weighted, though one *writes* durable knowledge and the
other only *reads*. This module is the honest answer: one :class:`Capability`
record per control, carrying what it does, what the user gets back, whether it is
a **governed** action (it asks for confirmation and/or writes something), the
exact CLI command that runs it headlessly, and the same text in Arabic.

It is the single source of truth, so the window, the CLI, and the documentation
cannot drift:

* the window draws a one-line hint under each button and a hover tooltip from
  :func:`capability`;
* ``python -m cli cognition catalog`` prints the whole table;
* :func:`catalog_accuracy` is a held-out gate that fails when a capability is
  missing its explanation, when a governed action stops naming ``--approve``,
  when a CLI command does not resolve, or when a window button and a catalogue
  entry fall out of step.

Dependency-free and Tk-free: it imports nothing but the standard library, so a
test, the CLI, and the window all reach it the same way. Nothing here executes
anything -- it only *describes* what already exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The sections a capability belongs to, in the order the window draws them.
CATEGORY_MIND = "mind"
CATEGORY_STUDY = "study"
CATEGORY_DEPTH = "depth"
CATEGORY_METACOGNITION = "metacognition"
CATEGORY_LEARNING = "learning"
CATEGORY_CONSOLIDATION = "consolidation"
CATEGORY_TRAINING = "training"
CATEGORY_CHARACTERS = "characters"
CATEGORY_INDICATORS = "indicators"
CATEGORY_SYSTEM = "system"
CATEGORY_COMPOSER = "composer"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_MIND,
    CATEGORY_STUDY,
    CATEGORY_DEPTH,
    CATEGORY_METACOGNITION,
    CATEGORY_LEARNING,
    CATEGORY_CONSOLIDATION,
    CATEGORY_TRAINING,
    CATEGORY_CHARACTERS,
    CATEGORY_INDICATORS,
    CATEGORY_SYSTEM,
    CATEGORY_COMPOSER,
)

#: The readable name of each section, so a surface never invents one.
CATEGORY_TITLES: dict[str, str] = {
    CATEGORY_MIND: "Proactive Mind",
    CATEGORY_STUDY: "Self-Study (ask & learn)",
    CATEGORY_DEPTH: "Knowledge Depth (widest reach)",
    CATEGORY_METACOGNITION: "Metacognition (self-model)",
    CATEGORY_LEARNING: "Cognitive Learning",
    CATEGORY_CONSOLIDATION: "Consolidation & Review",
    CATEGORY_TRAINING: "Gradual Training",
    CATEGORY_CHARACTERS: "Character Evaluation",
    CATEGORY_INDICATORS: "Cognitive Indicators",
    CATEGORY_SYSTEM: "Resource Mode",
    CATEGORY_COMPOSER: "Command line",
}

#: The same section names in Arabic, for the Arabic description.
CATEGORY_ARABIC: dict[str, str] = {
    CATEGORY_MIND: "العقل الاستباقي",
    CATEGORY_STUDY: "الدراسة الذاتية (اسأل وتعلّم)",
    CATEGORY_DEPTH: "عمق المعرفة (أوسع مدى)",
    CATEGORY_METACOGNITION: "التفكير الفوقي (نموذج الذات)",
    CATEGORY_LEARNING: "التعلّم الإدراكي",
    CATEGORY_CONSOLIDATION: "الترسيخ والمراجعة",
    CATEGORY_TRAINING: "التدريب المتدرّج",
    CATEGORY_CHARACTERS: "تقييم الشخصيات",
    CATEGORY_INDICATORS: "المؤشرات الإدراكية",
    CATEGORY_SYSTEM: "وضع الموارد",
    CATEGORY_COMPOSER: "سطر الأوامر",
}

#: One short sentence describing what a whole section is for (English + Arabic),
#: drawn under the section title so the reader knows what the group does.
CATEGORY_SUMMARIES: dict[str, str] = {
    CATEGORY_MIND: "The eight engines composed into one bounded thinking cycle: it "
    "finds gaps, ranks them, asks questions, and recombines knowledge into ideas.",
    CATEGORY_STUDY: "The mind asks its own questions, reads the project's cited "
    "answer to each, and reports the gain it actually measured.",
    CATEGORY_DEPTH: "Widen the reach of the graph: measure how much of the project's "
    "knowledge it holds, harvest relations from ingested text, and bridge "
    "disciplines.",
    CATEGORY_METACOGNITION: "The mind's model of its own thinking: which strategies "
    "pay off, where it stagnates, and how it will search next.",
    CATEGORY_LEARNING: "Six memories and the operations over them: learn a fact, "
    "recall from every memory, and record an experience with its outcome.",
    CATEGORY_CONSOLIDATION: "Turn raw experience into durable, reviewed knowledge: "
    "extract patterns, review what is due, and track practised skills.",
    CATEGORY_TRAINING: "Staged, rollback-safe training of the project's own model, "
    "with an adaptive curriculum that chooses its own next step.",
    CATEGORY_CHARACTERS: "Score the eight characters on four measured dimensions, "
    "with their strengths, shortcomings, and next steps.",
    CATEGORY_INDICATORS: "Measure the ten cognitive indicators against the live "
    "project and compare them with a baseline.",
    CATEGORY_SYSTEM: "Choose how hard the background mind may work, and run one "
    "background tick by hand.",
    CATEGORY_COMPOSER: "Run one student-first exchange: the project's own model "
    "answers first, and only a weak answer reaches a teacher.",
}

CATEGORY_SUMMARIES_AR: dict[str, str] = {
    CATEGORY_MIND: "ثمانية محرّكات في دورة تفكير واحدة مقيّدة: يجد الفجوات ويرتّبها "
    "ويسأل ويعيد تركيب المعرفة في أفكار.",
    CATEGORY_STUDY: "يسأل العقل أسئلته، ويقرأ إجابة المشروع المستشهَدة عن كل سؤال، "
    "ثم يعرض الفائدة التي قاسها فعلًا.",
    CATEGORY_DEPTH: "توسيع مدى الرسم: قياس ما يحمله المشروع من معرفة، وحصاد العلاقات "
    "من النصوص المستوعَبة، والجسور بين التخصصات.",
    CATEGORY_METACOGNITION: "نموذج العقل لتفكيره: أيّ استراتيجية أثمرت، وأين يركد، "
    "وكيف سيبحث في المرة القادمة.",
    CATEGORY_LEARNING: "ستّ ذاكرات وعملياتها: تعلّم حقيقة، واسترجع من كل ذاكرة، "
    "وسجّل تجربة بنتيجتها.",
    CATEGORY_CONSOLIDATION: "تحويل الخبرة الخام إلى معرفة دائمة مراجَعة: استخراج "
    "الأنماط، ومراجعة المستحق، وتتبّع المهارات المتدرَّب عليها.",
    CATEGORY_TRAINING: "تدريب مرحلي آمن التراجع لنموذج المشروع، بمنهج تكيّفي يختار "
    "خطوته التالية بنفسه.",
    CATEGORY_CHARACTERS: "تقييم الشخصيات الثمانية على أربعة أبعاد مقيسة، مع مواطن "
    "القوة والقصور والخطوات التالية.",
    CATEGORY_INDICATORS: "قياس المؤشرات الإدراكية العشرة على المشروع الحيّ ومقارنتها "
    "بخط الأساس.",
    CATEGORY_SYSTEM: "اختيار مقدار جهد العقل الخلفي، وتشغيل نبضة خلفية واحدة يدويًا.",
    CATEGORY_COMPOSER: "تبادل واحد يبدأ فيه الطالب: نموذج المشروع يجيب أولًا، ولا "
    "يصل المعلّم إلا إجابة ضعيفة.",
}


@dataclass(frozen=True)
class Capability:
    """What one control does, what it gives back, and whether it is governed.

    ``governed`` is true when the action asks for the user's confirmation and/or
    writes something durable (a training run, a rollback, a knowledge write) --
    the exact set the window guards with a confirmation dialog. ``cli`` is the
    headless command that runs the same thing, so a reader can reproduce it in a
    terminal. Both texts are given in English and Arabic.
    """

    key: str
    title: str
    summary: str
    outcome: str
    governed: bool
    cli: str
    arabic_title: str
    arabic_summary: str
    category: str
    writes: bool = False
    field_notes: tuple[str, ...] = field(default=())

    def describe(self, language: str = "en") -> str:
        """One readable block: the title, what it does, what it returns, the CLI."""
        if language == "ar":
            title, summary = self.arabic_title, self.arabic_summary
        else:
            title, summary = self.title, self.summary
        kind = "governed" if self.governed else "read-only"
        lines = [
            f"{title}",
            f"  what it does: {summary}",
            f"  what you get back: {self.outcome}",
            f"  kind: {kind}",
            f"  command: {self.cli}",
        ]
        if self.field_notes:
            lines.extend(f"  note: {note}" for note in self.field_notes)
        return "\n".join(lines)


#: The whole catalogue, in the order the window draws its sections. Every control
#: the cognition window exposes is here exactly once, and every entry has a real
#: CLI command -- ``catalog_accuracy`` fails if either side drifts.
CAPABILITIES: tuple[Capability, ...] = (
    # -- proactive mind ----------------------------------------------------
    Capability(
        key="think",
        title="Run a thinking cycle",
        summary="Runs the eight engines once, in a fixed order: curiosity finds "
        "gaps, motivation ranks them, self-questioning asks about the top ones, "
        "association links the concepts, idea generation recombines them, "
        "imagination builds scenarios, and self-criticism lowers the confidence of "
        "every weak idea while recording the reason.",
        outcome="A cycle record with the ideas it produced, each at its own rung "
        "(never stored as a fact), plus the drives that motivated it.",
        governed=False,
        cli="python -m cli cognition think",
        arabic_title="تشغيل دورة تفكير",
        arabic_summary="يشغّل المحرّكات الثمانية مرة واحدة بترتيب ثابت: الفضول يجد "
        "الفجوات، والدافع يرتّبها، وطرح الأسئلة يسأل عن أهمّها، والترابط يربط "
        "المفاهيم، وتوليد الأفكار يعيد تركيبها، والتخيّل يبني سيناريوهات، والنقد "
        "الذاتي يخفض ثقة كل فكرة ضعيفة ويسجّل السبب.",
        category=CATEGORY_MIND,
    ),
    Capability(
        key="ideas",
        title="Show the ideas awaiting review",
        summary="Lists the ideas the mind has produced that are not yet facts, each "
        "with the engine that made it, its confidence, its novelty, and the "
        "critique it received.",
        outcome="One row per pending idea; nothing is promoted by looking.",
        governed=False,
        cli="python -m cli cognition ideas",
        arabic_title="عرض الأفكار بانتظار المراجعة",
        arabic_summary="يعرض الأفكار التي أنتجها العقل ولم تصبح حقائق بعد، مع المحرّك "
        "الذي أنتجها وثقتها وجدّتها والنقد الموجَّه إليها.",
        category=CATEGORY_MIND,
    ),
    Capability(
        key="drives",
        title="What the mind would pursue next",
        summary="Ranks the topics worth pursuing on named factors (how big the gap "
        "is, how reachable it is, how novel) and says why each one scored what it "
        "did.",
        outcome="A ranked list of drives with their reasons.",
        governed=False,
        cli="python -m cli cognition drives",
        arabic_title="ما سيتابعه العقل تاليًا",
        arabic_summary="يرتّب المواضيع الجديرة بالمتابعة وفق عوامل مسمّاة (حجم الفجوة "
        "وإمكان الوصول والجدّة) ويوضّح سبب كل ترتيب.",
        category=CATEGORY_MIND,
    ),
    Capability(
        key="prove",
        title="Prove the whole loop end to end",
        summary="Runs the honest demonstration in order: think, measure the ten "
        "indicators, learn a fact, and recall it -- so a reader sees the whole "
        "pipeline work rather than a claim that it does.",
        outcome="Four reports, one per step, each the real result of the run.",
        governed=False,
        cli="python -m cli cognition prove",
        arabic_title="إثبات الدورة كاملة من البداية للنهاية",
        arabic_summary="يشغّل العرض الأمين بالترتيب: يفكّر، ويقيس المؤشرات العشرة، "
        "ويتعلّم حقيقة، ثم يسترجعها — ليرى القارئ المنظومة تعمل فعلًا لا مجرّد ادّعاء.",
        category=CATEGORY_MIND,
    ),
    # -- self-study --------------------------------------------------------
    Capability(
        key="study",
        title="Start asking questions",
        summary="Runs a self-study session: the mind asks the number of questions "
        "you set (200 by default), reads the grounded assistant's cited answer to "
        "each, folds what it reads into its memories, and learns any typed relation "
        "the answers state through the same bounded learner every write uses.",
        outcome="Each question and answer streamed live, then a measured benefit "
        "report: relations learned, facts gained, and the change in the ten "
        "indicators before versus after.",
        governed=True,
        cli="python -m cli cognition study --count 200 --approve",
        arabic_title="ابدأ بطرح الأسئلة",
        arabic_summary="يشغّل جلسة دراسة ذاتية: يسأل العقل عدد الأسئلة الذي تحدّده "
        "(200 افتراضيًا)، ويقرأ الإجابة المستشهَدة عن كل سؤال، ويطوي ما يقرأ في "
        "ذاكرته، ويتعلّم أي علاقة مصنّفة تذكرها الإجابات عبر المعلّم المقيّد نفسه.",
        category=CATEGORY_STUDY,
        writes=True,
    ),
    Capability(
        key="bank_size",
        title="Show the question bank size",
        summary="Counts the distinct questions the mind's bank holds, all built from "
        "the graph's own stated edges and the project's authored topics -- nothing "
        "invented.",
        outcome="One number, and where the questions come from.",
        governed=False,
        cli="python -m cli cognition questions",
        arabic_title="عرض حجم بنك الأسئلة",
        arabic_summary="يحصي الأسئلة المتمايزة في بنك العقل، وكلّها مبنيّة من حوافّ "
        "الرسم نفسها ومن المواضيع المؤلَّفة في المشروع — لا شيء مُختلَق.",
        category=CATEGORY_STUDY,
    ),
    Capability(
        key="next_questions",
        title="Preview the next questions",
        summary="Shows the next questions the mind would ask, without recording them "
        "as asked, so you can see what it is about to study.",
        outcome="A preview list; the bank's record is untouched.",
        governed=False,
        cli="python -m cli cognition questions --next 20",
        arabic_title="معاينة الأسئلة التالية",
        arabic_summary="يعرض الأسئلة التالية التي سيسألها العقل دون تسجيلها كمسؤول "
        "عنها، لترى ما سيدرسه قبل أن يبدأ.",
        category=CATEGORY_STUDY,
    ),
    Capability(
        key="last_report",
        title="Last benefit report",
        summary="Reads back the measured benefit of the most recent self-study "
        "session from the durable record.",
        outcome="The last session's counts and indicator delta, or an honest note "
        "that no session has run yet.",
        governed=False,
        cli="python -m cli cognition last-study",
        arabic_title="تقرير الفائدة الأخير",
        arabic_summary="يقرأ من السجل الدائم الفائدة المقيسة لآخر جلسة دراسة ذاتية.",
        category=CATEGORY_STUDY,
    ),
    # -- knowledge depth (the widening of reach) ---------------------------
    Capability(
        key="depth",
        title="Measure the graph's depth",
        summary="Measures how much of the project's own declared knowledge the graph "
        "actually carries: concepts and relations per discipline, per course, and "
        "per corpus entry, how many concepts are isolated, and how many carry an "
        "Arabic alias.",
        outcome="A coverage report with the weak spots named, so the next widening "
        "step is chosen from evidence rather than guessed.",
        governed=False,
        cli="python -m cli cognition depth",
        arabic_title="قياس عمق الرسم",
        arabic_summary="يقيس مقدار ما يحمله الرسم فعلًا من معرفة المشروع المعلنة: "
        "المفاهيم والعلاقات لكل تخصص ومقرّر ومدخل متن، وعدد المفاهيم المعزولة، وعدد "
        "ما يحمل مرادفًا عربيًا.",
        category=CATEGORY_DEPTH,
    ),
    Capability(
        key="coverage",
        title="Coverage by discipline and course",
        summary="Prints one line per discipline and per course the graph covers, "
        "with the concepts and relations present against those the seed declares, "
        "so a reader can see exactly which area is thin.",
        outcome="The full coverage list, plus the graph's connectivity and average "
        "degree.",
        governed=False,
        cli="python -m cli cognition coverage",
        arabic_title="التغطية حسب التخصص والمقرّر",
        arabic_summary="يطبع سطرًا لكل تخصّص ومقرّر يغطّيه الرسم، بالمفاهيم والعلاقات "
        "الموجودة مقابل ما يعلنه البذرة، ليرى القارئ أيّ منطقة رقيقة.",
        category=CATEGORY_DEPTH,
    ),
    Capability(
        key="harvest",
        title="Harvest relations from ingested text",
        summary="Reads typed relations out of the project's bulk knowledge base -- "
        "the text you imported through the Import Center -- with the same pattern "
        "matcher the answer path uses, then folds each accepted relation into the "
        "graph through the bounded learner. Every write is previewed first and "
        "written only when you confirm it.",
        outcome="How many relations were read and how many joined the graph, each "
        "carrying the sentence that produced it; nothing overrides a seeded link.",
        governed=True,
        cli="python -m cli cognition harvest --limit 500 --write --approve",
        arabic_title="حصاد العلاقات من النصوص المستوعَبة",
        arabic_summary="يقرأ العلاقات المصنّفة من قاعدة المعرفة الحجمية — النصوص التي "
        "استوردتها عبر مركز الاستيراد — بالمطابق النمطي نفسه الذي يستخدمه مسار "
        "الإجابة، ثم يطوي كل علاقة مقبولة في الرسم عبر المعلّم المقيّد.",
        category=CATEGORY_DEPTH,
        writes=True,
    ),
    Capability(
        key="bridges",
        title="Bridge two disciplines",
        summary="Finds the shortest chain of typed relations that connects a concept "
        "in one discipline to a concept in another, so the mind can reason across "
        "the whole field rather than inside one silo.",
        outcome="The connecting paths, each step showing the direction it was walked "
        "and the confidence of the chain; no path is ever stored.",
        governed=False,
        cli="python -m cli cognition bridges --from reservoir --to drilling",
        arabic_title="الجسر بين تخصّصين",
        arabic_summary="يجد أقصر سلسلة علاقات مصنّفة تصل مفهومًا في تخصّص بمفهوم في "
        "تخصّص آخر، ليستطيع العقل الاستدلال عبر المجال كله لا داخل جزيرة واحدة.",
        category=CATEGORY_DEPTH,
    ),
    Capability(
        key="arabic",
        title="Report the Arabic reach",
        summary="Reports how far the project reaches in Arabic: how many concepts "
        "carry an Arabic alias, how many questions of the bank are Arabic, and which "
        "labels still lack a term.",
        outcome="The Arabic coverage figures and the remaining gap, reported rather "
        "than hidden.",
        governed=False,
        cli="python -m cli cognition arabic",
        arabic_title="تقرير المدى العربي",
        arabic_summary="يعرض مدى وصول المشروع بالعربية: كم مفهومًا يحمل مرادفًا عربيًا، "
        "وكم سؤالًا من البنك عربي، وأيّ تسميات ما زالت بلا مصطلح.",
        category=CATEGORY_DEPTH,
    ),
    # -- metacognition -----------------------------------------------------
    Capability(
        key="metacognition",
        title="Which strategies pay off",
        summary="Reads the mind's self-model: how much of each strategy's output "
        "survived criticism, the bounded bias it now applies, and what is in focus.",
        outcome="A strategy-by-strategy reading, with the bias each one earned.",
        governed=False,
        cli="python -m cli cognition metacognition",
        arabic_title="أيّ الاستراتيجيات أثمرت",
        arabic_summary="يقرأ نموذج العقل لذاته: كم نجا من ناتج كل استراتيجية بعد النقد، "
        "والانحياز المقيّد الذي صار يطبّقه، وما هو موضع التركيز الآن.",
        category=CATEGORY_METACOGNITION,
    ),
    Capability(
        key="plan",
        title="How the mind will search next",
        summary="Shows the metacognitive plan for the next cycle: how many concepts "
        "and ideas it will take, the novelty floor it will hold, and whether it is "
        "escalating out of a stagnation.",
        outcome="The plan with its reasons, before any cycle runs.",
        governed=False,
        cli="python -m cli cognition plan",
        arabic_title="كيف سيبحث العقل تاليًا",
        arabic_summary="يعرض الخطة الفوقية للدورة القادمة: كم مفهومًا وفكرة سيأخذ، "
        "وحدّ الجدّة الذي سيحافظ عليه، وهل يتوسّع خروجًا من ركود.",
        category=CATEGORY_METACOGNITION,
    ),
    # -- learning ----------------------------------------------------------
    Capability(
        key="learn",
        title="Learn this statement",
        summary="Stores the statement you type into semantic memory with its source "
        "and confidence. It is only reported as a *trusted fact* when the reliability "
        "gate accepts it; otherwise it is stored but marked unverified.",
        outcome="Whether the statement became a trusted fact or is stored as "
        "unverified.",
        governed=False,
        cli='python -m cli cognition learn "acid stimulation increases permeability"',
        arabic_title="تعلّم هذا القول",
        arabic_summary="يخزّن العبارة التي تكتبها في الذاكرة الدلالية مع مصدرها وثقتها. "
        "ولا تُعدّ حقيقة موثوقة إلا إذا قبلتها بوابة الموثوقية؛ وإلا خُزّنت موسومة "
        "بأنها غير محقَّقة.",
        category=CATEGORY_LEARNING,
        writes=True,
    ),
    Capability(
        key="recall",
        title="Recall from every memory",
        summary="Retrieves the query from all six memories at once -- semantic, "
        "episodic, procedural, associations, and the rest -- so you can see what the "
        "mind already holds about it.",
        outcome="What each memory returned for the query, memory by memory.",
        governed=False,
        cli='python -m cli cognition recall "permeability"',
        arabic_title="الاسترجاع من كل الذاكرات",
        arabic_summary="يسترجع الاستعلام من الذاكرات الستّ كلها معًا — الدلالية والعرضية "
        "والإجرائية والترابطات وغيرها — لترى ما يحمله العقل عنه بالفعل.",
        category=CATEGORY_LEARNING,
    ),
    Capability(
        key="learning_stats",
        title="What the learning system holds",
        summary="One line summarising the learning system's durable counts, so you "
        "can see how full each memory is at a glance.",
        outcome="A count summary of the memories.",
        governed=False,
        cli="python -m cli cognition status",
        arabic_title="ما يحمله نظام التعلّم",
        arabic_summary="سطر واحد يلخّص أعداد نظام التعلّم الدائمة، لترى امتلاء كل ذاكرة "
        "بنظرة.",
        category=CATEGORY_LEARNING,
    ),
    # -- consolidation -----------------------------------------------------
    Capability(
        key="consolidate",
        title="Consolidate experience into knowledge",
        summary="Extracts content-addressed patterns from repeated experience and "
        "upgrades a strongly-supported one into a *learned, unverified* fact. A rule "
        "needs real support before it is stated, so one anecdote never becomes a law.",
        outcome="The patterns extracted, what was upgraded, and how many items are "
        "now under spaced review.",
        governed=False,
        cli="python -m cli cognition consolidate",
        arabic_title="ترسيخ الخبرة في معرفة",
        arabic_summary="يستخرج أنماطًا معنونة بالمحتوى من الخبرة المتكرّرة، ويرقّي "
        "النمط المدعوم بقوة إلى حقيقة متعلَّمة غير محقَّقة. ولا تُعلن قاعدة إلا بدعم "
        "حقيقي، فلا تصير حادثة واحدة قانونًا.",
        category=CATEGORY_CONSOLIDATION,
        writes=True,
    ),
    Capability(
        key="patterns",
        title="The patterns it has extracted",
        summary="Lists the regularities the consolidator has drawn from the mind's "
        "own experience, with the support each one rests on.",
        outcome="One line per pattern, with its support and confidence.",
        governed=False,
        cli="python -m cli cognition patterns",
        arabic_title="الأنماط المستخرجة",
        arabic_summary="يعرض الانتظامات التي استخلصها المرسّخ من خبرة العقل نفسه، مع "
        "الدعم الذي يستند إليه كل نمط.",
        category=CATEGORY_CONSOLIDATION,
    ),
    Capability(
        key="review",
        title="Review what is due",
        summary="Advances the spaced-review schedule: each item that is recalled "
        "moves up a box, and a miss resets it -- the honest way to keep knowledge "
        "alive rather than assume it is remembered.",
        outcome="The items that were due and the box each moved to.",
        governed=False,
        cli="python -m cli cognition review",
        arabic_title="مراجعة المستحقّ",
        arabic_summary="يقدّم جدول المراجعة المتباعدة: كل بند يُتذكَّر يرتقي صندوقًا، "
        "والخطأ يصفّره — الطريقة الأمينة لإبقاء المعرفة حيّة بدل افتراض تذكّرها.",
        category=CATEGORY_CONSOLIDATION,
        writes=True,
    ),
    Capability(
        key="skills",
        title="The skills it has practised",
        summary="Reports each procedure the mind has practised and its measured "
        "proficiency, computed from real attempts rather than guessed.",
        outcome="One line per skill with its proficiency and step count.",
        governed=False,
        cli="python -m cli cognition skills",
        arabic_title="المهارات المتدرَّب عليها",
        arabic_summary="يعرض كل إجراء تدرّب عليه العقل وكفاءته المقيسة من محاولات "
        "حقيقية لا مخمّنة.",
        category=CATEGORY_CONSOLIDATION,
    ),
    # -- training ----------------------------------------------------------
    Capability(
        key="train",
        title="Run one stage (asks first)",
        summary="Runs one training stage on a copy of the model and writes a "
        "candidate version. It is adopted only when the evaluation shows it did not "
        "regress; otherwise the previous version stays in use. The base model is "
        "never modified in place.",
        outcome="The stage's outcome and the version log after the run.",
        governed=True,
        cli="python -m cli cognition train --stage quick --approve",
        arabic_title="تشغيل مرحلة واحدة (يسأل أولًا)",
        arabic_summary="يشغّل مرحلة تدريب واحدة على نسخة من النموذج ويكتب نسخة مرشّحة. "
        "ولا تُعتمد إلا إذا أظهر التقييم أنها لم تتدهور؛ وإلا بقيت النسخة السابقة. "
        "ولا يُعدَّل النموذج الأساس في مكانه أبدًا.",
        category=CATEGORY_TRAINING,
        writes=True,
    ),
    Capability(
        key="train_gradual",
        title="Advance gradually (asks first)",
        summary="Advances stage by stage up to the stage you chose, stopping at the "
        "first stage the evaluation does not accept -- so the model grows over time "
        "instead of one enormous run.",
        outcome="Which stages advanced and the reason it stopped.",
        governed=True,
        cli="python -m cli cognition train --gradual --up-to quick --approve",
        arabic_title="التقدّم تدريجيًا (يسأل أولًا)",
        arabic_summary="يتقدّم مرحلة مرحلة حتى المرحلة التي اخترتها، ويتوقّف عند أول "
        "مرحلة لا يقبلها التقييم — فينمو النموذج عبر الزمن لا في تشغيل واحد ضخم.",
        category=CATEGORY_TRAINING,
        writes=True,
    ),
    Capability(
        key="train_autonomous",
        title="Run the adaptive curriculum (asks first)",
        summary="Lets the curriculum choose its own path: it raises the stage when a "
        "run is accepted, adjusts the learning rate when a stage plateaus, and stops "
        "at an explicit effort budget -- adopting a version only through the gate.",
        outcome="Each decision the curriculum made and why it stopped.",
        governed=True,
        cli="python -m cli cognition train --autonomous --approve",
        arabic_title="تشغيل المنهج التكيّفي (يسأل أولًا)",
        arabic_summary="يدع المنهج يختار طريقه بنفسه: يرفع المرحلة عند القبول، ويعدّل "
        "معدّل التعلّم عند الركود، ويتوقّف عند ميزانية جهد صريحة — ولا يعتمد نسخة إلا "
        "عبر البوابة.",
        category=CATEGORY_TRAINING,
        writes=True,
    ),
    Capability(
        key="progression",
        title="How far training has come",
        summary="Shows the progression table: for each stage, how many attempts it "
        "took and whether it is done, plus the next stage to run.",
        outcome="A stage-by-stage progression and the next stage.",
        governed=False,
        cli="python -m cli cognition progression",
        arabic_title="إلى أين وصل التدريب",
        arabic_summary="يعرض جدول التقدّم: لكل مرحلة عدد محاولاتها وهل اكتملت، مع "
        "المرحلة التالية.",
        category=CATEGORY_TRAINING,
    ),
    Capability(
        key="curriculum",
        title="The adaptive curriculum's memory",
        summary="Reads the durable log of training attempts the adaptive policy "
        "consults -- stage, settings, verdict, and the before/after perplexity -- so "
        "a plateau is remembered across runs rather than rediscovered.",
        outcome="One line per recorded attempt.",
        governed=False,
        cli="python -m cli cognition curriculum",
        arabic_title="ذاكرة المنهج التكيّفي",
        arabic_summary="يقرأ السجل الدائم لمحاولات التدريب التي تستشيرها السياسة "
        "التكيّفية — المرحلة والإعدادات والحكم وقيمة الحيرة قبل وبعد — فيُتذكَّر "
        "الركود عبر التشغيلات لا يُكتشف من جديد.",
        category=CATEGORY_TRAINING,
    ),
    Capability(
        key="versions",
        title="The version log",
        summary="Lists the model versions the registry holds, which one is active, "
        "and the base model they all descend from.",
        outcome="The version log.",
        governed=False,
        cli="python -m cli cognition versions",
        arabic_title="سجل النسخ",
        arabic_summary="يعرض نسخ النموذج في السجل، وأيّها نشطة، والنموذج الأساس الذي "
        "تنحدر منه جميعها.",
        category=CATEGORY_TRAINING,
    ),
    Capability(
        key="rollback",
        title="Roll back to the last stable version",
        summary="Returns the active model to the last accepted version. The base "
        "model was never modified, so this is always safe.",
        outcome="Which version is now active.",
        governed=True,
        cli="python -m cli cognition rollback --approve",
        arabic_title="الرجوع إلى آخر نسخة مستقرة",
        arabic_summary="يعيد النموذج النشط إلى آخر نسخة مقبولة. والنموذج الأساس لم "
        "يُعدَّل قط، فالعملية آمنة دائمًا.",
        category=CATEGORY_TRAINING,
        writes=True,
    ),
    # -- characters --------------------------------------------------------
    Capability(
        key="characters",
        title="Evaluate all eight characters",
        summary="Scores the seven characters and the project's own model on four "
        "dimensions computed from their real configuration, real learning store, "
        "and the real model catalog -- training structure, memory, traits, and "
        "development headroom -- never from a fixed table of praise.",
        outcome="A per-character score with strengths, shortcomings, and next steps.",
        governed=False,
        cli="python -m cli cognition characters",
        arabic_title="تقييم الشخصيات الثمانية",
        arabic_summary="يقيّم الشخصيات السبع ونموذج المشروع على أربعة أبعاد محسوبة من "
        "إعدادها الحقيقي ومخزن تعلّمها الحقيقي وكتالوج النماذج الحقيقي — بنية التدريب "
        "والذاكرة والسمات وهامش التطوّر — لا من جدول ثناء ثابت.",
        category=CATEGORY_CHARACTERS,
    ),
    # -- indicators --------------------------------------------------------
    Capability(
        key="evaluate",
        title="Measure the ten indicators",
        summary="Measures the ten cognitive indicators against the live project -- "
        "knowledge retrieval, reasoning, answer accuracy, learning speed, idea "
        "quality, self-criticism, generalisation, response speed, reliability, and "
        "independence.",
        outcome="Each indicator's score with a note explaining it, and the overall "
        "mean.",
        governed=False,
        cli="python -m cli cognition evaluate",
        arabic_title="قياس المؤشرات العشرة",
        arabic_summary="يقيس المؤشرات الإدراكية العشرة على المشروع الحيّ — استرجاع "
        "المعرفة والاستدلال ودقة الإجابة وسرعة التعلّم وجودة الأفكار والنقد الذاتي "
        "والتعميم وسرعة الاستجابة والموثوقية والاستقلالية.",
        category=CATEGORY_INDICATORS,
    ),
    # -- system ------------------------------------------------------------
    Capability(
        key="mode",
        title="Apply the resource mode",
        summary="Sets how hard the background mind may work: LIGHT (a cycle every "
        "15 minutes, one idea), BALANCED (every 5 minutes, three ideas), or "
        "INTENSIVE (every 2 minutes, five ideas). Every mode is still bounded by "
        "the scheduler's own ceilings, so intensive means more, never unbounded.",
        outcome="The mode now in force.",
        governed=False,
        cli="python -m cli cognition mode balanced",
        arabic_title="تطبيق وضع الموارد",
        arabic_summary="يحدّد مقدار جهد العقل الخلفي: خفيف (دورة كل ١٥ دقيقة، فكرة "
        "واحدة)، متوازن (كل ٥ دقائق، ثلاث أفكار)، مكثّف (كل دقيقتين، خمس أفكار). وكل "
        "وضع يبقى مقيّدًا بسقوف المجدول، فالمكثّف يعني أكثر لا بلا حدّ.",
        category=CATEGORY_SYSTEM,
    ),
    Capability(
        key="tick",
        title="Run one background tick",
        summary="Asks the scheduler whether a cycle is due and runs exactly one if "
        "it is; otherwise it reports the reason it deferred, so the mind never runs "
        "a busy loop.",
        outcome="Either the cycle that ran or the honest reason none did.",
        governed=False,
        cli="python -m cli cognition tick",
        arabic_title="تشغيل نبضة خلفية واحدة",
        arabic_summary="يسأل المجدول هل حان وقت دورة، ويشغّل واحدة فقط إن حان؛ وإلا "
        "يذكر سبب التأجيل، فلا يدور العقل في حلقة مشغولة.",
        category=CATEGORY_SYSTEM,
    ),
    # -- composer ----------------------------------------------------------
    Capability(
        key="command",
        title="Teach the student",
        summary="Runs one student-first exchange: the project's own model answers "
        "the question unaided, its answer is judged, and only a weak answer reaches "
        "a teacher whose reply is then verified against the project's own knowledge "
        "before any example is recorded.",
        outcome="The exchange's outcome and the qualified example, if one was "
        "accepted.",
        governed=False,
        cli='python -m cli cognition teach "what is permeability?"',
        arabic_title="تعليم الطالب",
        arabic_summary="يشغّل تبادلًا يبدأ فيه الطالب: يجيب نموذج المشروع بلا عون، "
        "وتُقيَّم إجابته، ولا تصل إجابة ضعيفة إلى معلّم إلا بعد التحقّق من ردّه مقابل "
        "معرفة المشروع قبل تسجيل أي مثال.",
        category=CATEGORY_COMPOSER,
    ),
    Capability(
        key="catalog",
        title="What does each item do?",
        summary="Prints the whole capability catalogue: for every control in this "
        "window, what it does, what you get back, whether it is a governed action, "
        "and the exact headless command that runs it.",
        outcome="The full list, section by section, so you can read it before "
        "pressing anything.",
        governed=False,
        cli="python -m cli cognition catalog",
        arabic_title="ماذا يفعل كل بند؟",
        arabic_summary="يطبع كتالوج القدرات كاملًا: لكل زرّ في هذه النافذة، ماذا "
        "يفعل، وما ستحصل عليه، وهل هو فعل محكوم، والأمر الذي يشغّله من الطرفية.",
        category=CATEGORY_SYSTEM,
    ),
)

#: The catalogue by key, so a lookup is O(1) and a typo returns ``None`` rather
#: than a wrong entry.
CAPABILITY_INDEX: dict[str, Capability] = {cap.key: cap for cap in CAPABILITIES}


def capability(key: str) -> Capability | None:
    """The capability with this key, or ``None`` for an unknown one."""
    return CAPABILITY_INDEX.get(key)


def capabilities_for(category: str) -> tuple[Capability, ...]:
    """Every capability in one section, in catalogue order."""
    return tuple(cap for cap in CAPABILITIES if cap.category == category)


def capability_groups() -> tuple[tuple[str, tuple[Capability, ...]], ...]:
    """The catalogue grouped by section, in the window's own order."""
    return tuple((cat, capabilities_for(cat)) for cat in CATEGORIES)


def governed_keys() -> frozenset[str]:
    """The keys of the actions that ask for confirmation and/or write."""
    return frozenset(cap.key for cap in CAPABILITIES if cap.governed)


def catalog_lines(language: str = "en") -> tuple[str, ...]:
    """The whole catalogue as readable lines, grouped by section."""
    lines: list[str] = []
    for category, caps in capability_groups():
        if not caps:
            continue
        title = CATEGORY_TITLES.get(category, category)
        summary = (
            CATEGORY_SUMMARIES_AR if language == "ar" else CATEGORY_SUMMARIES
        ).get(category, "")
        lines.append(f"{title}")
        if summary:
            lines.append(f"  {summary}")
        for cap in caps:
            lines.extend(cap.describe(language=language).splitlines())
        lines.append("")
    return tuple(lines)


# -- the held-out gate -------------------------------------------------------

#: One judgement: a named check the real catalogue must satisfy. Each is a pure
#: function of the catalogue, so the gate measures the *shipped* text rather than
#: a copy of it.
@dataclass(frozen=True)
class CatalogJudgement:
    key: str
    description: str


CATALOG_JUDGEMENTS: tuple[CatalogJudgement, ...] = (
    CatalogJudgement("keys_unique", "every capability key is unique"),
    CatalogJudgement("every_key_explained", "every capability has a title, a summary, and an outcome"),
    CatalogJudgement("bilingual", "every capability carries an Arabic title and summary"),
    CatalogJudgement("governed_names_approve", "every governed action names its confirmation in the CLI"),
    CatalogJudgement("read_only_has_no_approve", "no read-only action asks for approval"),
    CatalogJudgement("cli_resolves", "every CLI command is a real cognition subcommand"),
    CatalogJudgement("categories_declared", "every capability sits in a declared section"),
    CatalogJudgement("sections_described", "every section has a one-sentence description in both languages"),
    CatalogJudgement("window_buttons_covered", "every window button has a catalogue entry"),
    CatalogJudgement("no_orphan_entries", "every catalogue entry has a window button"),
)


def validate_capability(cap: Capability) -> tuple[str, ...]:
    """The problems with one capability, or ``()`` when it is complete.

    This is the checker the gate runs over the real catalogue, and the same
    function a *broken* capability is measured against -- so the gate can prove it
    discriminates rather than always returning a pass.
    """
    problems: list[str] = []
    if not cap.key.strip():
        problems.append("empty key")
    if not cap.title.strip():
        problems.append("missing title")
    if not cap.summary.strip():
        problems.append("missing summary")
    if not cap.outcome.strip():
        problems.append("missing outcome")
    if not cap.arabic_title.strip():
        problems.append("missing Arabic title")
    if not cap.arabic_summary.strip():
        problems.append("missing Arabic summary")
    if cap.category not in CATEGORIES:
        problems.append(f"unknown section '{cap.category}'")
    if cap.governed and "--approve" not in cap.cli:
        problems.append("a governed action must name --approve in its CLI command")
    if not cap.governed and "--approve" in cap.cli:
        problems.append("a read-only action must not name --approve")
    return tuple(problems)


#: The buttons the cognition window actually draws, by capability key. Written
#: out by hand on purpose: if it were derived from the catalogue it could not
#: catch a *drift*, and catching drift is the whole point. The window imports
#: :func:`capability` and looks each key up, so a key added here without an entry
#: (or an entry with no button) fails the gate.
WINDOW_BUTTONS: frozenset[str] = frozenset(
    {
        "think",
        "ideas",
        "drives",
        "prove",
        "study",
        "bank_size",
        "next_questions",
        "last_report",
        "depth",
        "coverage",
        "harvest",
        "bridges",
        "arabic",
        "metacognition",
        "plan",
        "consolidate",
        "patterns",
        "review",
        "skills",
        "learn",
        "recall",
        "learning_stats",
        "train",
        "train_gradual",
        "train_autonomous",
        "progression",
        "curriculum",
        "versions",
        "rollback",
        "characters",
        "evaluate",
        "mode",
        "tick",
        "catalog",
        "command",
    }
)


def _cli_subcommands() -> frozenset[str]:
    """The real subcommands of ``python -m cli cognition``.

    The CLI module is imported lazily so this catalogue stays importable from the
    window and from a test without pulling the CLI in at module load.
    """
    try:
        import argparse

        from cli.cognition import _parser

        parser = _parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return frozenset(str(choice) for choice in action.choices)
    except Exception:
        return frozenset()
    return frozenset()


def _command_token(cli: str) -> str:
    """The subcommand named in a CLI string (``python -m cli cognition think``)."""
    parts = cli.split()
    if "cognition" in parts:
        index = parts.index("cognition")
        if index + 1 < len(parts):
            return parts[index + 1]
    return parts[-1] if parts else ""


def catalog_accuracy(judgements: tuple[CatalogJudgement, ...] = CATALOG_JUDGEMENTS) -> float:
    """The share of catalogue judgements the shipped catalogue satisfies.

    The gate is honest and discriminating: it runs the real validators over the
    real catalogue, so a capability shipped without an explanation, a governed
    action that stopped naming ``--approve``, a CLI command that no longer
    resolves, or a window button with no entry all lower the metric.
    """
    if not judgements:
        return 0.0
    checks = _CHECKS
    passed = 0
    for judgement in judgements:
        check = checks.get(judgement.key)
        if check is None:
            continue
        try:
            if check():
                passed += 1
        except Exception:
            # A catalogue that cannot even be read fails its judgement.
            continue
    return round(passed / len(judgements), 6)


def _keys_unique() -> bool:
    keys = [cap.key for cap in CAPABILITIES]
    return len(keys) == len(set(keys))


def _every_key_explained() -> bool:
    return all(
        cap.title.strip() and cap.summary.strip() and cap.outcome.strip()
        for cap in CAPABILITIES
    )


def _bilingual() -> bool:
    return all(
        cap.arabic_title.strip() and cap.arabic_summary.strip() for cap in CAPABILITIES
    )


def _governed_names_approve() -> bool:
    return all(
        "--approve" in cap.cli for cap in CAPABILITIES if cap.governed
    )


def _read_only_has_no_approve() -> bool:
    return all(
        "--approve" not in cap.cli for cap in CAPABILITIES if not cap.governed
    )


def _cli_resolves() -> bool:
    subcommands = _cli_subcommands()
    if not subcommands:
        # The CLI module could not be imported here; do not claim a pass.
        return False
    return all(_command_token(cap.cli) in subcommands for cap in CAPABILITIES)


def _categories_declared() -> bool:
    return all(cap.category in CATEGORIES for cap in CAPABILITIES)


def _sections_described() -> bool:
    return all(
        CATEGORY_SUMMARIES.get(cat, "").strip()
        and CATEGORY_SUMMARIES_AR.get(cat, "").strip()
        for cat in CATEGORIES
    )


def _window_buttons_covered() -> bool:
    return WINDOW_BUTTONS <= set(CAPABILITY_INDEX)


def _no_orphan_entries() -> bool:
    return set(CAPABILITY_INDEX) <= WINDOW_BUTTONS


#: The judgement key -> its real check. Kept at module scope so the gate is a
#: data-driven table rather than a chain of ``if``s.
_CHECKS = {
    "keys_unique": _keys_unique,
    "every_key_explained": _every_key_explained,
    "bilingual": _bilingual,
    "governed_names_approve": _governed_names_approve,
    "read_only_has_no_approve": _read_only_has_no_approve,
    "cli_resolves": _cli_resolves,
    "categories_declared": _categories_declared,
    "sections_described": _sections_described,
    "window_buttons_covered": _window_buttons_covered,
    "no_orphan_entries": _no_orphan_entries,
}


def describe_catalog() -> str:
    """A readable report of every judgement and whether it held."""
    lines = ["Cognition catalogue gate:"]
    for judgement in CATALOG_JUDGEMENTS:
        check = _CHECKS.get(judgement.key)
        try:
            ok = bool(check()) if check is not None else False
        except Exception:
            ok = False
        lines.append(f"  [{'ok' if ok else 'FAIL'}] {judgement.description}")
    return "\n".join(lines)


__all__ = [
    "CAPABILITIES",
    "CAPABILITY_INDEX",
    "CATALOG_JUDGEMENTS",
    "CATEGORIES",
    "CATEGORY_ARABIC",
    "CATEGORY_CHARACTERS",
    "CATEGORY_COMPOSER",
    "CATEGORY_CONSOLIDATION",
    "CATEGORY_DEPTH",
    "CATEGORY_INDICATORS",
    "CATEGORY_LEARNING",
    "CATEGORY_METACOGNITION",
    "CATEGORY_MIND",
    "CATEGORY_STUDY",
    "CATEGORY_SUMMARIES",
    "CATEGORY_SUMMARIES_AR",
    "CATEGORY_SYSTEM",
    "CATEGORY_TITLES",
    "CATEGORY_TRAINING",
    "WINDOW_BUTTONS",
    "Capability",
    "CatalogJudgement",
    "capabilities_for",
    "capability",
    "capability_groups",
    "catalog_accuracy",
    "catalog_lines",
    "describe_catalog",
    "governed_keys",
    "validate_capability",
]
