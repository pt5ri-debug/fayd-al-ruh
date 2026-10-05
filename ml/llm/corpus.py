"""The in-project training corpus for the language model.

Keeping the corpus in the repository is what makes the model self-contained: no
download, no external dataset. It is small and domain-focused -- Arabic and
English sentences about the project's own world (water, light, soil, tools,
evidence, beliefs) and about its declared domain, petroleum engineering -- so
the trained model produces coherent text about the things this program reasons
about, rather than pretending to be general.
"""

from __future__ import annotations

import unicodedata


def _sentences() -> list[str]:
    ar = [
        "الماء يجعل التربة رطبة.",
        "التربة الرطبة تساعد النبات على النمو.",
        "الضوء يضيء الشمعة.",
        "المطر يسقي الحديقة.",
        "الشمس تدفئ الغرفة.",
        "الريح تحرك الأوراق.",
        "البرودة تحفظ الطعام.",
        "الحرارة تبخر الماء.",
        "الوقت يغيّر كل شيء ببطء.",
        "البذرة تحتاج إلى تربة وماء وضوء.",
        "المعرفة تُبنى من الأدلة لا من الظن.",
        "الاعتقاد يتغير عندما تظهر أدلة جديدة.",
        "الشك طريق إلى الفهم الأفضل.",
        "التجربة تكشف ما لا تكشفه الفكرة وحدها.",
        "السبب والنتيجة يظهران مع التكرار.",
        "التعلم يعني تصحيح الخطأ خطوة بخطوة.",
        "الذاكرة تحفظ ما جرى وتعيد ترتيبه.",
        "التفكير يربط الملاحظة بالنتيجة.",
        "الفضول يدفع العقل إلى السؤال.",
        "الصبر جزء من كل تجربة ناجحة.",
        "الأداة توسع ما تستطيع اليد فعله.",
        "الملاحظة الدقيقة تسبق الاستنتاج.",
        "الخطأ فرصة لتصحيح النموذج الداخلي.",
        "النموذج الجيد يتنبأ بما سيحدث.",
        "الهدف الواضح يوجه الإجراء التالي.",
        "الإجراء يغيّر الحالة إلى حالة جديدة.",
        "التخطيط يعني اختيار خطوات قبل التنفيذ.",
        "المكافأة تخبر الوكيل ما الذي نجح.",
        "الخبرة تُختصر في قاعدة تُستخدم لاحقاً.",
        "التكرار يقوي ما كان ضعيفاً.",
    ]
    en = [
        "Water makes the soil wet.",
        "Light lights the candle.",
        "Rain waters the garden.",
        "Heat boils the water.",
        "A seed needs soil and water.",
        "Knowledge is built from evidence.",
        "A belief changes when evidence appears.",
        "The model predicts the next state.",
        "An action changes the world.",
        "Planning chooses steps before acting.",
        "Reward tells the agent what worked.",
        "Observation comes before inference.",
        "A good model predicts what happens.",
        "Experience becomes a rule for later.",
        "Curiosity drives the next question.",
        "Learning corrects the error step by step.",
        "Memory keeps what happened.",
        "Thinking connects the cause to the effect.",
        "A tool extends what a hand can do.",
        "Practice makes the weak strong.",
    ]
    return ar + en


def _petroleum_sentences() -> list[str]:
    """Domain sentences: the project declares a petroleum-engineering identity."""
    ar = [
        "المكمن يحتوي على الهيدروكربون.",
        "الماء يملأ مسام الصخر.",
        "المسامية تقيس الفراغات في الصخر.",
        "النفاذية تقيس سهولة مرور السائل.",
        "الضغط يدفع النفط إلى البئر.",
        "البئر يخترق الطبقة الحاملة.",
        "الحفر يصل إلى المكمن.",
        "الاسمنت يعزل البئر عن الطبقات.",
        "السجل يقيس المسامية والنفاذية.",
        "الانتاج يتوقف عندما ينخفض الضغط.",
        "الهيدروكربون يوجد في الطبقة المسامية.",
        "الطبقة الحاملة تجمع النفط والغاز.",
        "رفع الضغط يرفع معدل الانتاج.",
        "انخفاض الضغط يقلل الانتاج.",
        "الماء يظهر مع النفط في بعض الابار.",
        "الصخر الكتيم يمنع حركة السائل.",
        "قراءة السجل تكشف نوع الصخر.",
        "التشقق يزيد نفاذية الطبقة.",
        "العينة تكشف مسامية الصخر.",
        "الحرارة ترفع لزوجة النفط.",
    ]
    en = [
        "A reservoir contains hydrocarbons.",
        "Water fills the pores of the rock.",
        "Porosity measures the voids in rock.",
        "Permeability measures how easily fluid flows.",
        "Pressure drives oil toward the well.",
        "A well penetrates the reservoir layer.",
        "Drilling reaches the reservoir.",
        "Cement seals the well from other layers.",
        "A well log measures porosity and permeability.",
        "Production declines when pressure drops.",
        "Hydrocarbons sit in porous rock.",
        "The formation holds oil and gas.",
        "Higher pressure raises the flow rate.",
        "Lower pressure reduces production.",
        "Water appears with oil in some wells.",
        "An impermeable rock blocks fluid flow.",
        "A log reading reveals the rock type.",
        "Fractures increase formation permeability.",
        "A core sample reveals rock porosity.",
        "Heat raises the viscosity of oil.",
    ]
    return ar + en


def build_corpus(*, repeats: int = 40, structured_repeats: int = 60) -> str:
    """Return the training text, repeated so the small model can fit it.

    Three blocks are concatenated: general prose about the project's world,
    petroleum-engineering prose matching the declared identity, and JSON objects
    in the shape ``MODEL_OUTPUT_SCHEMA`` expects. The JSON block teaches the
    model the field names and Arabic values used by schema-guided decoding.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if structured_repeats < 1:
        raise ValueError("structured_repeats must be at least 1.")
    prose = " ".join(_sentences())
    petroleum = " ".join(_petroleum_sentences())
    structured = " ".join(structured_examples())
    return (prose + " ") * repeats + (petroleum + " ") * repeats + (structured + " ") * structured_repeats


def structured_examples() -> list[str]:
    """JSON objects matching ``MODEL_OUTPUT_SCHEMA`` with Arabic content."""
    return [
        '{"concepts": [{"label": "الماء", "aliases": ["السائل"]}], '
        '"evidence": [{"description": "التربة رطبة", "source": "الملاحظة", '
        '"supports": true, "strength": 0.8}], '
        '"hypotheses": [{"statement": "الماء يجعل التربة رطبة", "confidence": 0.7}], '
        '"questions": [{"text": "هل يجف الماء بالحرارة", "topic": "السببية", "priority": 0.6}], '
        '"plans": [{"goal": "اسقاء النبات", "priority": 0.9, '
        '"steps": [{"description": "افحص التربة"}, {"description": "اسكب الماء"}]}]}',
        '{"concepts": [{"label": "الضوء", "aliases": ["النور"]}], '
        '"evidence": [{"description": "الشمعة مضاءة", "source": "الرؤية", '
        '"supports": true, "strength": 0.9}], '
        '"hypotheses": [{"statement": "الضوء يضيء الشمعة", "confidence": 0.8}], '
        '"questions": [{"text": "ما مصدر الضوء", "topic": "السببية", "priority": 0.5}], '
        '"plans": [{"goal": "اشعال الشمعة", "priority": 0.8, '
        '"steps": [{"description": "احضر عود الثقاب"}, {"description": "اضرم النار"}]}]}',
        '{"concepts": [{"label": "الحرارة", "aliases": ["الدفء"]}], '
        '"evidence": [{"description": "الماء يغلي", "source": "الملاحظة", '
        '"supports": true, "strength": 0.7}], '
        '"hypotheses": [{"statement": "الحرارة تبخر الماء", "confidence": 0.6}], '
        '"questions": [{"text": "متى يغلي الماء", "topic": "الفيزياء", "priority": 0.4}], '
        '"plans": [{"goal": "غلي الماء", "priority": 0.7, '
        '"steps": [{"description": "ضع الاناء على النار"}, {"description": "انتظر الفقاعات"}]}]}',
        '{"concepts": [{"label": "البذرة", "aliases": ["النواة"]}], '
        '"evidence": [{"description": "البذرة في التربة", "source": "الملاحظة", '
        '"supports": true, "strength": 0.6}], '
        '"hypotheses": [{"statement": "البذرة تنمو بالماء والضوء", "confidence": 0.7}], '
        '"questions": [{"text": "كم تحتاج البذرة لتنمو", "topic": "الزراعة", "priority": 0.5}], '
        '"plans": [{"goal": "زراعة البذرة", "priority": 0.8, '
        '"steps": [{"description": "احفر حفرة"}, {"description": "ضع البذرة واسقها"}]}]}',
        '{"concepts": [{"label": "الدليل", "aliases": ["البرهان"]}], '
        '"evidence": [{"description": "التجربة تكررت", "source": "السجل", '
        '"supports": true, "strength": 0.8}], '
        '"hypotheses": [{"statement": "التكرار يقوي الاعتقاد", "confidence": 0.6}], '
        '"questions": [{"text": "هل يكفي دليل واحد", "topic": "المعرفة", "priority": 0.7}], '
        '"plans": [{"goal": "التحقق من الفرضية", "priority": 0.9, '
        '"steps": [{"description": "اجمع الادلة"}, {"description": "قارن النتائج"}]}]}',
        '{"concepts": [{"label": "المكمن", "aliases": ["الخزان"]}], '
        '"evidence": [{"description": "الضغط مرتفع في البئر", "source": "قياس الضغط", '
        '"supports": true, "strength": 0.8}], '
        '"hypotheses": [{"statement": "الضغط يدفع النفط نحو البئر", "confidence": 0.7}], '
        '"questions": [{"text": "ما معدل هبوط الضغط", "topic": "المكمن", "priority": 0.8}], '
        '"plans": [{"goal": "تقييم انتاجية المكمن", "priority": 0.9, '
        '"steps": [{"description": "اقرأ سجل البئر"}, {"description": "احسب المسامية والنفاذية"}]}]}',
        '{"concepts": [{"label": "النفاذية", "aliases": ["قابلية التدفق"]}], '
        '"evidence": [{"description": "العينة صخرية مسامية", "source": "تحليل العينة", '
        '"supports": true, "strength": 0.7}], '
        '"hypotheses": [{"statement": "المسامية ترفع النفاذية", "confidence": 0.6}], '
        '"questions": [{"text": "كيف تؤثر التشققات في التدفق", "topic": "الطبقة", "priority": 0.6}], '
        '"plans": [{"goal": "تحسين الانتاج", "priority": 0.8, '
        '"steps": [{"description": "افحص التشققات"}, {"description": "ارفع الضغط"}]}]}',
    ]


def normalise(text: str) -> str:
    """Unicode-normalise text so the byte model sees consistent Arabic forms."""
    return unicodedata.normalize("NFC", text)


__all__ = ["build_corpus", "normalise"]
