"""Arabic prose rendered from the project's own knowledge graph (Phase 50).

The project is Arabic-first by design, but Phase 49 closed only the *input* half
of that: an Arabic question could finally reach the ontology. The *output* half
was still missing. Measured on the released build, every generated text block was
100% English:

* ``ml.llm.corpus`` -- the language model's training text: **0% Arabic** across
  ``curriculum_sentences`` (1139), ``corpus_sentences`` (836), and
  ``procedural_sentences`` (146), so the in-project model could not speak Arabic
  at all;
* ``core.documents`` -- the citable documents the grounded assistant retrieves
  from: English only, so an Arabic answer's *body* was English even when its
  Arabic reasoning frame ("هذه خلاصة المصادر عن المسامية") was Arabic.

This module renders the same knowledge the graph already carries into Arabic
sentences, so both gaps close from one source of truth:

* :func:`arabic_label` resolves a concept label to its Arabic term, preferring the
  ontology's own aliases (the curated glossary of Phase 49 plus the corpus's
  glossary) and falling back to the English label -- it never invents a term;
* :func:`arabic_relation_sentence` renders one relation with
  :data:`RELATION_ARABIC`, a curated map of the seeds' relation vocabulary onto an
  Arabic clause;
* :func:`arabic_curriculum_sentences` and :func:`arabic_corpus_sentences` produce
  the training text, and :func:`arabic_document_lines` the citable lines.

It is **presentation-only and additive**: it reads the ontology and returns text,
it writes nothing, and every English block is left exactly as it was, so no
existing test or metric moves because of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.arabic_glossary import ARABIC_ALIASES, has_arabic

if TYPE_CHECKING:
    from core.ontology import Ontology

#: The curated English-label -> Arabic-term table, keyed for lookup. Kept as the
#: *fallback* behind the ontology's own aliases, because a corpus glossary term
#: (added by ``seed_petroleum_corpus``) may name a concept the table does not.
_LABEL_TO_ARABIC: dict[str, str] = dict(ARABIC_ALIASES)

#: The relation vocabulary of the seeds, rendered as an Arabic clause that takes
#: the target as its object (``"{source} {clause} {target}."``). Only the seeds'
#: own relation types are listed; anything else falls back to
#: :data:`DEFAULT_RELATION_ARABIC` rather than being dropped, so no edge is lost.
RELATION_ARABIC: dict[str, str] = {
    "is_a": "من نوع",
    "measures": "يقيس",
    "increases": "يزيد",
    "reduces": "يقلل",
    "causes": "يسبب",
    "caused_by": "ناتج عن",
    "prevents": "يمنع",
    "controls": "يتحكم في",
    "controlled_by": "يتحكم فيه",
    "contains": "يحتوي على",
    "requires": "يتطلب",
    "drives": "يدفع",
    "predicts": "يتنبأ بـ",
    "forecasts": "يتنبأ بـ",
    "supports": "يدعم",
    "forms": "يشكل",
    "has": "يمتلك",
    "separates": "يفصل",
    "depends_on": "يعتمد على",
    "affects": "يؤثر على",
    "produces": "ينتج",
    "generates": "يولد",
    "stores": "يخزن",
    "transports": "ينقل",
    "indicates": "يشير إلى",
    "estimates": "يقدّر",
    "improves": "يحسّن",
    "protects": "يحمي",
    "enables": "يمكّن",
    "creates": "ينشئ",
    "uses": "يستخدم",
    "used_in": "يُستخدم في",
    "includes": "يشمل",
    "part_of": "جزء من",
    "related_to": "مرتبط بـ",
    "relates": "يربط",
    "connects": "يربط",
    "connects_to": "يرتبط بـ",
    "links_to": "يرتبط بـ",
    "drills": "يحفر",
    "seals": "يغلق",
    "treats": "يعالج",
    "treated_by": "يعالج بواسطة",
    "triggers": "يحفّز",
    "induces": "يحفّز",
    "stimulates": "يحفّز",
    "maintains": "يحافظ على",
    "governs": "يحكم",
    "governed_by": "يحكمه",
    "verifies": "يتحقق من",
    "evaluates": "يقيّم",
    "assesses": "يقيّم",
    "quantifies": "يحدد كميًا",
    "computes": "يحسب",
    "designs": "يصمم",
    "plans": "يخطط",
    "optimizes": "يحسّن",
    "maximizes": "يزيد إلى أقصى حد",
    "limits": "يحد",
    "bounds": "يحد",
    "constrains": "يقيد",
    "regulates": "ينظم",
    "balances": "يوازن",
    "equalizes": "يوازن",
    "stabilizes": "يثبّت",
    "records": "يسجل",
    "documents": "يوثّق",
    "describes": "يصف",
    "described_by": "يوصفه",
    "defines": "يعرّف",
    "characterizes": "يميّز",
    "identifies": "يحدد",
    "detects": "يكتشف",
    "detected_by": "يكتشفه",
    "reveals": "يكشف",
    "shows": "يظهر",
    "informs": "يبلّغ",
    "guides": "يوجّه",
    "feeds": "يغذي",
    "fills": "يملأ",
    "drains": "يصرّف",
    "cools": "يبرد",
    "pressurizes": "يضغط",
    "releases": "يحرر",
    "removes": "يزيل",
    "adds_to": "يضاف إلى",
    "mixes_with": "يمتزج مع",
    "complements": "يكمل",
    "completes": "يكمل",
    "carries": "يحمل",
    "carried_by": "يحملها",
    "hosts": "يستضيف",
    "migrates_within": "يهاجر داخل",
    "accumulates_in": "يتراكم في",
    "transformed_by": "يتحول بواسطة",
    "derived_from": "مشتق من",
    "produced_from": "منتج من",
    "composed_of": "مكوّن من",
    "built_from": "مبني من",
    "results_from": "ينتج عن",
    "estimated_by": "يقدّره",
    "evaluated_by": "يقيّمه",
    "checked_by": "يفحصه",
    "analyzed_from": "يحلل من",
    "interpreted_with": "يفسّر بواسطة",
    "modeled_by": "ينمذج بواسطة",
    "sorted_by": "يفرز بواسطة",
    "held_by": "يحتفظ به",
    "opposed_by": "يعارضه",
    "compared_to": "يقارن بـ",
    "seeks": "يسعى إلى",
    "prioritizes": "يعطي الأولوية لـ",
    "allocates": "يخصص",
    "discounts": "يخصم",
    "projects": "يقدّر",
    "recovers": "يستخرج",
    "resists": "يقاوم",
    "threatens": "يهدد",
    "pollutes": "يلوث",
    "guards": "يحرس",
    "conserves": "يحفظ",
    "transition_to": "ينتقل إلى",
    "undergo": "يخضع لـ",
    "undergoes": "يخضع لـ",
    "qualifies": "يؤهل",
    "introduces": "يقدّم",
    "initiates": "يبدأ",
    "runs": "يشغّل",
    "handles": "يتعامل مع",
    "writes": "يكتب",
    "maps": "يرسم",
    "locates": "يحدد موقع",
    "images": "يصوّر",
    "builds": "يبني",
    "applies_to": "يُطبّق على",
    "applied_to": "يُطبّق على",
    "calibrates": "يعاير",
    "influence": "يؤثر في",
    "opens_under": "يفتح عند",
    "stands_in": "يقف في",
    "flow_together_as": "يتدفق معًا كـ",
    "precedes": "يسبق",
    "followed_by": "يتبعه",
    "minimizes": "يقلل إلى أدنى حد",
    "concludes": "يستنتج",
    "discretizes": "يفصل",
    "condenses": "يكثّف",
    "determines": "يحدد",
}

#: The clause used for a relation the map does not name, so an edge is never
#: silently dropped from the Arabic text.
DEFAULT_RELATION_ARABIC = "يرتبط بـ"

#: The Arabic words the model needs early: the same world the English ``_sentences``
#: block describes, in Arabic, so a small model sees Arabic function words and
#: sentence shape from the first step rather than only noun phrases.
CORE_ARABIC_SENTENCES: tuple[str, ...] = (
    "الماء يجعل التربة رطبة.",
    "الضوء يضيء الشمعة.",
    "المطر يسقي الحديقة.",
    "الحرارة تغلي الماء.",
    "البذرة تحتاج إلى تربة وماء.",
    "المعرفة تُبنى من الأدلة.",
    "الاعتقاد يتغير عندما يظهر دليل.",
    "النموذج يتنبأ بالحالة التالية.",
    "الفعل يغير العالم.",
    "التخطيط يختار الخطوات قبل التنفيذ.",
    "المكافأة تخبر الوكيل بما نجح.",
    "الملاحظة تسبق الاستنتاج.",
    "النموذج الجيد يتنبأ بما سيحدث.",
    "الخبرة تصبح قاعدة لما بعد.",
    "الفضول يدفع السؤال التالي.",
    "التعلم يصحح الخطأ خطوة بخطوة.",
    "الذاكرة تحفظ ما حدث.",
    "التفكير يربط السبب بالنتيجة.",
    "الأداة توسع ما تستطيعه اليد.",
    "التدريب يجعل الضعيف قويًا.",
    "المكمن يحتوي على الهيدروكربونات.",
    "الماء يملأ مسام الصخر.",
    "المسامية تقيس الفراغات في الصخر.",
    "النفاذية تقيس سهولة تدفق المائع.",
    "الضغط يدفع النفط نحو البئر.",
    "البئر يخترق طبقة المكمن.",
    "الحفر يصل إلى المكمن.",
    "الأسمنت يعزل البئر عن الطبقات الأخرى.",
    "السجل البئري يقيس المسامية والنفاذية.",
    "الإنتاج ينخفض عندما ينخفض الضغط.",
    "الهيدروكربونات توجد في صخر مسامي.",
    "الطبقة تحمل النفط والغاز.",
    "الضغط الأعلى يرفع معدل التدفق.",
    "الضغط الأقل يقلل الإنتاج.",
    "الماء يظهر مع النفط في بعض الآبار.",
    "الصخر غير المنفذ يمنع تدفق المائع.",
    "قراءة السجل تكشف نوع الصخر.",
    "الكسور تزيد نفاذية الطبقة.",
    "عينة اللب تكشف مسامية الصخر.",
    "الحرارة ترفع لزوجة النفط.",
)


#: Arabic names for the labels the *concept* glossary does not cover: the
#: disciplines, the courses, and the corpus entries. A discipline or course label
#: is a phrase, not a concept, so ``core.arabic_glossary`` never held it and
#: ``arabic_label`` used to return it in English. Curated by hand; a label absent
#: here still falls back to English rather than being guessed at.
TOPIC_ARABIC: dict[str, str] = {
    # Disciplines (core.domain_knowledge).
    "Reservoir Engineering": "هندسة المكامن",
    "Drilling Engineering": "هندسة الحفر",
    "Production Engineering": "هندسة الإنتاج",
    "Petrophysics and Formation Evaluation": "البتروفيزياء وتقييم الطبقات",
    "Well Logging": "تسجيل الآبار",
    "Well Completions": "إكمال الآبار",
    "Enhanced Oil Recovery": "الاستخلاص المعزز للنفط",
    "Well Testing": "اختبار الآبار",
    "Artificial Lift": "الرفع الصناعي",
    "Well Stimulation": "تحفيز الآبار",
    "Well Cementing": "تسميت الآبار",
    "Well Control": "التحكم في الآبار",
    "Petroleum Economics": "اقتصاديات النفط",
    "Health, Safety and Environment": "الصحة والسلامة والبيئة",
    "Facilities Engineering": "هندسة المنشآت",
    "Process Safety": "سلامة العمليات",
    "Unconventional Resources": "الموارد غير التقليدية",
    # Courses (core.curriculum_knowledge).
    "Mathematics": "الرياضيات",
    "Physics": "الفيزياء",
    "Chemistry": "الكيمياء",
    "Statistics and Data Analysis": "الإحصاء وتحليل البيانات",
    "Computing and Programming": "الحوسبة والبرمجة",
    "Numerical Methods": "الطرق العددية",
    "General Geology": "الجيولوجيا العامة",
    "Structural Geology": "الجيولوجيا البنيوية",
    "Sedimentology": "علم الرسوبيات",
    "Petroleum Geology": "جيولوجيا النفط",
    "Geophysics": "الجيوفيزياء",
    "Fluid Mechanics": "ميكانيكا الموائع",
    "Thermodynamics": "الديناميكا الحرارية",
    "Mechanics of Materials": "ميكانيكا المواد",
    "Rock Mechanics": "ميكانيكا الصخور",
    "Petrophysics": "البتروفيزياء",
    "Reservoir Fluid Properties": "خصائص موائع المكمن",
    "Facilities, Transport and Storage": "المنشآت والنقل والتخزين",
    "Offshore Drilling": "الحفر البحري",
    "Project Management": "إدارة المشاريع",
    "Ethics and Professionalism": "الأخلاقيات والمهنية",
    "Basin Analysis and Petroleum Systems": "تحليل الأحواض والأنظمة النفطية",
    "Seismic Interpretation": "تفسير البيانات السيزمية",
    "Petroleum Geomechanics": "الجيوميكانيكا النفطية",
    "Multiphase Flow": "التدفق متعدد الأطوار",
    "Reservoir Simulation and Modelling": "محاكاة المكامن ونمذجتها",
    "Advanced Enhanced Oil Recovery": "الاستخلاص المعزز المتقدم للنفط",
    "Unconventional Resources Engineering": "هندسة الموارد غير التقليدية",
    "Subsea and Deepwater Engineering": "الهندسة البحرية والمياه العميقة",
    "Flow Assurance": "ضمان التدفق",
    "Process Safety Engineering": "هندسة سلامة العمليات",
    "Petroleum Data Analytics and Machine Learning": "تحليلات بيانات النفط وتعلم الآلة",
    "Carbon Capture, Utilisation and Storage": "احتجاز الكربون واستخدامه وتخزينه",
    "Well Plugging and Decommissioning": "إغلاق الآبار وإخراجها من الخدمة",
    "Integrated Field Management": "الإدارة المتكاملة للحقل",
    # Petroleum-corpus entry titles (core.petroleum_corpus).
    "Reservoir rock properties": "خصائص صخور المكمن",
    "Reservoir drive mechanisms": "آليات دفع المكمن",
    "Volumetric reserves": "الاحتياطيات الحجمية",
    "Material balance and reservoir simulation": "موازنة المواد ومحاكاة المكمن",
    "Formation evaluation": "تقييم الطبقات",
    "Capillary pressure and saturation-height": "الضغط الشعري وارتفاع التشبع",
    "Well-logging tools": "أدوات تسجيل الآبار",
    "Log interpretation workflow": "منهجية تفسير السجلات",
    "Drilling fluids": "سوائل الحفر",
    "Casing design": "تصميم البطانة",
    "Directional and horizontal drilling": "الحفر الاتجاهي والأفقي",
    "Production optimisation and decline": "تحسين الإنتاج وانحداره",
    "Artificial-lift selection": "اختيار الرفع الصناعي",
    "Well completions": "إكمال الآبار",
    "Hydraulic fracturing": "التكسير الهيدروليكي",
    "Matrix acidizing": "التحميض المصفوفي",
    "Primary cementing": "التسميت الأولي",
    "Well control and kicks": "التحكم في الآبار والاندفاعات",
    "Enhanced oil recovery": "الاستخلاص المعزز للنفط",
    "Pressure-transient testing": "اختبار ضغط العابر",
    "Petroleum project economics": "اقتصاديات مشروع النفط",
    "Field development planning": "تخطيط تطوير الحقل",
    "Well integrity management": "إدارة سلامة البئر",
    "Process safety": "سلامة العمليات",
    "Reservoir fluid properties": "خصائص موائع المكمن",
    "Rock mechanics": "ميكانيكا الصخور",
    "Fluid flow in porous media": "تدفق الموائع في الأوساط المسامية",
    "Offshore operations": "العمليات البحرية",
    "Surface facilities and transport": "المنشآت السطحية والنقل",
    "Project management": "إدارة المشاريع",
    "Professional ethics": "الأخلاقيات المهنية",
    "The petroleum system": "النظام النفطي",
    "Seismic interpretation": "تفسير البيانات السيزمية",
    "Surface production facilities": "منشآت الإنتاج السطحية",
    "Gas processing and LNG": "معالجة الغاز والغاز المسال",
    "Process safety management": "إدارة سلامة العمليات",
    "Unconventional resources": "الموارد غير التقليدية",
    "Reservoir simulation and modelling": "محاكاة المكامن ونمذجتها",
    "Advanced enhanced oil recovery": "الاستخلاص المعزز المتقدم للنفط",
    "Waterflood management": "إدارة حقن الماء",
    "Integrated field development planning": "تخطيط تطوير الحقل المتكامل",
    "Geomechanics and wellbore stability": "الجيوميكانيكا واستقرار البئر",
    "Flow assurance": "ضمان التدفق",
    "Multiphase flow": "التدفق متعدد الأطوار",
    "Subsea and deepwater production": "الإنتاج البحري وفي المياه العميقة",
    "Petroleum data analytics and machine learning": "تحليلات بيانات النفط وتعلم الآلة",
    "Carbon capture, utilisation and storage": "احتجاز الكربون واستخدامه وتخزينه",
    "Well plugging and decommissioning": "إغلاق الآبار وإخراجها من الخدمة",
    # Corpus discipline keys (the short identifier each entry carries).
    "reservoir": "المكامن",
    "petrophysics": "البتروفيزياء",
    "well_logging": "تسجيل الآبار",
    "drilling": "الحفر",
    "production": "الإنتاج",
    "artificial_lift": "الرفع الصناعي",
    "completions": "إكمال الآبار",
    "stimulation": "تحفيز الآبار",
    "cementing": "تسميت الآبار",
    "well_control": "التحكم في الآبار",
    "eor": "الاستخلاص المعزز",
    "well_testing": "اختبار الآبار",
    "economics": "الاقتصاديات",
    "hse": "الصحة والسلامة والبيئة",
    "geoscience": "علوم الأرض",
    "facilities": "المنشآت",
    "process_safety": "سلامة العمليات",
    "unconventional": "الموارد غير التقليدية",
}


def arabic_label(label: str, ontology: Ontology | None = None) -> str:
    """The Arabic term for a concept label, or the label unchanged.

    Prefers the ontology's own aliases (the curated Phase 49 glossary plus the
    petroleum corpus's glossary, both folded in as aliases), then the curated
    table, then the English label. It never invents a term: an unknown label is
    returned as it is, so a Latin identifier the domain keeps (``SPE``) is left
    alone rather than guessed at.
    """
    if not isinstance(label, str):
        return str(label)
    if ontology is not None:
        from core.arabic_glossary import _concept_for_label

        concept = _concept_for_label(ontology, label)
        if concept is not None:
            for alias in sorted(concept.aliases):
                if has_arabic(alias):
                    return alias
    if label in _LABEL_TO_ARABIC:
        return _LABEL_TO_ARABIC[label]
    return TOPIC_ARABIC.get(label, label)


def relation_arabic(relation_type: str) -> str:
    """The Arabic clause for a relation type, or the honest default."""
    return RELATION_ARABIC.get(relation_type, DEFAULT_RELATION_ARABIC)


def arabic_relation_sentence(source: str, relation_type: str, target: str) -> str:
    """``"{source} {clause} {target}."`` in Arabic."""
    return f"{source} {relation_arabic(relation_type)} {target}."


def _arabic_for_ontology(ontology: Ontology | None) -> Ontology:
    """The ontology to read: the given one, else a seeded controller's."""
    if ontology is not None:
        return ontology
    from core.cognitive_controller import CognitiveController

    return CognitiveController().ontology


def arabic_curriculum_sentences(ontology: Ontology | None = None) -> list[str]:
    """Arabic training prose from the seeded curriculum and the graph.

    Mirrors :func:`ml.llm.corpus.curriculum_sentences`: every discipline and course
    contributes a sentence for each concept and each relation, but the concepts are
    named by their Arabic term (when one exists) and the relations by their Arabic
    clause, so the model is trained on the same knowledge in Arabic rather than on
    an English block beside an Arabic one.
    """
    from core.curriculum_knowledge import courses
    from core.domain_knowledge import disciplines

    graph = _arabic_for_ontology(ontology)
    sentences: list[str] = []
    sources: list[Any] = [*disciplines(), *courses()]
    for source in sources:
        source_ar = arabic_label(source.label, graph)
        for label, aliases in source.concepts:
            term = arabic_label(label, graph)
            sentences.append(f"{term} مفهوم من {source_ar}.")
            for alias in aliases:
                if alias != label and not has_arabic(alias):
                    sentences.append(f"{term} يُعرف أيضًا بـ {arabic_label(alias, graph)}.")
        for src, tgt, relation, _confidence in source.relations:
            sentences.append(
                arabic_relation_sentence(
                    arabic_label(src, graph), relation, arabic_label(tgt, graph)
                )
            )
    return sentences


def arabic_corpus_sentences(ontology: Ontology | None = None) -> list[str]:
    """Arabic training prose from the seeded petroleum corpus.

    Each entry contributes its title, its glossary (the term and its Arabic word),
    and its concept relations -- the parts the corpus itself holds in both
    languages. The English *definitions* are deliberately not translated: this
    module never invents prose it cannot derive from the graph, so the block stays
    honest rather than becoming a machine translation of the whole corpus.
    """
    from core.petroleum_corpus import entries

    graph = _arabic_for_ontology(ontology)
    sentences: list[str] = []
    for entry in entries():
        title = arabic_label(entry.title, graph)
        sentences.append(f"{title} موضوع من {arabic_label(entry.discipline, graph)}.")
        for term, arabic in entry.glossary:
            if arabic.strip():
                sentences.append(f"{arabic.strip()} هو المصطلح العربي لـ {term}.")
        for label, _aliases in entry.concepts:
            sentences.append(f"{arabic_label(label, graph)} مفهوم من {title}.")
        for source, target, relation, _confidence in entry.facts:
            sentences.append(
                arabic_relation_sentence(
                    arabic_label(source, graph), relation, arabic_label(target, graph)
                )
            )
    return sentences


def arabic_procedural_sentences(ontology: Ontology | None = None) -> list[str]:
    """Arabic numbered procedures, the twin of ``procedural_sentences``.

    The step *text* stays in the language the corpus recorded it in (the steps are
    English prose this module does not translate); what is Arabic is the frame --
    the task name and the count -- so the model learns the "خطوات" shape.
    """
    from core.petroleum_corpus import entries

    graph = _arabic_for_ontology(ontology)
    sentences: list[str] = []
    for entry in entries():
        steps = [step for step in entry.procedures if step.strip()]
        if not steps:
            continue
        title = arabic_label(entry.title, graph)
        sentences.append(f"إجراء {title} يتضمن {len(steps)} خطوات.")
    return sentences


def arabic_document_lines(ontology: Ontology | None = None) -> list[str]:
    """The Arabic lines a citable document should carry, in document order.

    One line per discipline and course (its Arabic label and each concept's
    Arabic term), then the corpus entries' Arabic glossary pairs, so a retrieved
    Arabic answer can cite Arabic text rather than only English.
    """
    from core.curriculum_knowledge import courses
    from core.domain_knowledge import disciplines
    from core.petroleum_corpus import entries

    graph = _arabic_for_ontology(ontology)
    lines: list[str] = []
    sources: list[Any] = [*disciplines(), *courses()]
    for source in sources:
        source_ar = arabic_label(source.label, graph)
        terms = [
            arabic_label(label, graph)
            for label, _aliases in source.concepts
        ]
        terms = [term for term in terms if has_arabic(term)]
        if not terms:
            continue
        lines.append(f"{source_ar}: {'، '.join(dict.fromkeys(terms))}.")
    for entry in entries():
        pairs = [
            f"{arabic.strip()} ({term})"
            for term, arabic in entry.glossary
            if arabic.strip()
        ]
        if pairs:
            lines.append(f"{arabic_label(entry.title, graph)}: {'، '.join(pairs)}.")
    return lines


def arabic_document_sections(
    ontology: Ontology | None = None,
) -> list[tuple[str, str, list[str]]]:
    """The Arabic citable documents as ``(doc_id, title, lines)`` triples.

    One section per discipline, per course, and per corpus entry -- the same
    shape as the English documents -- so retrieval ranks the *relevant* Arabic
    text first rather than one large block whose opening lines name an unrelated
    discipline. A section with no Arabic term is dropped, so no document is
    empty.
    """
    from core.curriculum_knowledge import courses
    from core.domain_knowledge import disciplines
    from core.petroleum_corpus import entries

    graph = _arabic_for_ontology(ontology)
    sections: list[tuple[str, str, list[str]]] = []

    def term_line(label: str, concept_pairs) -> str:
        terms = [
            arabic_label(concept, graph)
            for concept, _aliases in concept_pairs
        ]
        terms = [term for term in terms if has_arabic(term)]
        if not terms:
            return ""
        return f"{arabic_label(label, graph)}: {'، '.join(dict.fromkeys(terms))}."

    for discipline in disciplines():
        line = term_line(discipline.label, discipline.concepts)
        if line:
            sections.append((f"arabic:discipline:{discipline.key}", f"إطار: {arabic_label(discipline.label, graph)}", [line]))
    for course in courses():
        line = term_line(course.label, course.concepts)
        if line:
            sections.append((f"arabic:course:{course.key}", f"مقرر: {arabic_label(course.label, graph)}", [line]))
    for entry in entries():
        lines: list[str] = []
        title = arabic_label(entry.title, graph)
        pairs = [
            f"{arabic.strip()} ({term})"
            for term, arabic in entry.glossary
            if arabic.strip()
        ]
        if pairs:
            lines.append(f"{title}: {'، '.join(pairs)}.")
        concept_terms = [
            arabic_label(concept, graph)
            for concept, _aliases in entry.concepts
        ]
        concept_terms = [term for term in concept_terms if has_arabic(term)]
        if concept_terms:
            lines.append(f"مفاهيم {title}: {'، '.join(dict.fromkeys(concept_terms))}.")
        relation_lines = [
            arabic_relation_sentence(
                arabic_label(source, graph), relation, arabic_label(target, graph)
            )
            for source, target, relation, _confidence in entry.facts
        ]
        lines.extend(relation_lines)
        if lines:
            sections.append((f"arabic:corpus:{entry.key}", f"محتوى: {title}", lines))
    return sections


def arabic_sentence_blocks(ontology: Ontology | None = None) -> list[str]:
    """Every Arabic block this module produces, in a stable order."""
    graph = _arabic_for_ontology(ontology)
    return (
        list(CORE_ARABIC_SENTENCES)
        + arabic_curriculum_sentences(graph)
        + arabic_corpus_sentences(graph)
        + arabic_procedural_sentences(graph)
    )


__all__ = [
    "CORE_ARABIC_SENTENCES",
    "DEFAULT_RELATION_ARABIC",
    "RELATION_ARABIC",
    "TOPIC_ARABIC",
    "arabic_corpus_sentences",
    "arabic_curriculum_sentences",
    "arabic_document_lines",
    "arabic_document_sections",
    "arabic_label",
    "arabic_procedural_sentences",
    "arabic_relation_sentence",
    "arabic_sentence_blocks",
    "relation_arabic",
]
