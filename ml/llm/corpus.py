"""The in-project training corpus for the language model.

Keeping the corpus in the repository is what makes the model self-contained: no
download, no external dataset. It is small and domain-focused -- Arabic and
English sentences about the project's own world (water, light, soil, tools,
evidence, beliefs) -- so the trained model produces coherent text about the
things this program reasons about, rather than pretending to be general.
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


def build_corpus(*, repeats: int = 40) -> str:
    """Return the training text, repeated so the small model can fit it."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    block = " ".join(_sentences())
    return (block + " ") * repeats


def normalise(text: str) -> str:
    """Unicode-normalise text so the byte model sees consistent Arabic forms."""
    return unicodedata.normalize("NFC", text)


__all__ = ["build_corpus", "normalise"]
