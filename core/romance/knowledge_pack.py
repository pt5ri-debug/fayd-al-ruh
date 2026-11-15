"""Build each companion's **own knowledge base**: its voice, words, and world.

Every companion in the Romance & Companion Studio owns a knowledge base of its
own -- the words, sentences, self-knowledge, and domain material that define how
*that* character talks and what *that* character knows. This module is the
generator: it renders those documents from the character's own records and writes
them into the character's own
:class:`core.knowledge_base.KnowledgeBaseStore`, so the companion draws on them
when it composes a reply.

**What is generated, and how it is honest.**

* **Curated first.** The character's identity, biography, personality, and
  phrasebook are authored (a curated template and fact bank), so the companion's
  own voice material is always present and always correct.
* **Expanded to a size.** The dialogue, scene, interest, and knowledge sections
  are generated combinatorially from the curated factors, deterministically, until
  the requested byte target is met. The value of the expansion is *coverage* --
  many plausible lines and exchanges -- not novel facts.
* **Deterministic and offline.** The generator is seeded by the character's own
  id; it draws no random numbers and opens no socket. A rebuild produces the same
  bytes, so nothing is duplicated (de-duplication is by content hash).
* **Character-specific.** Every passage interpolates the character's name,
  nickname, nationality, dialect, kinship, traits, interests, and boundaries, so
  two companions never share a pack.
* **Bilingual.** English and Arabic passages are generated for every section, so a
  companion answers in the language the user writes in.
* **Bounded.** Generation stops when the target is reached (clamped to
  :data:`TARGET_MB_RANGE`); a space that is exhausted early is reported honestly
  rather than padded with nonsense.

It loads no model and reaches no network; it is pure file, index, and text work.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The size a pack is built to, in megabytes.
DEFAULT_TARGET_MB = 50.0
#: The clamp on a requested target, so a bad value cannot fill a disk.
TARGET_MB_RANGE = (1.0, 200.0)
#: The languages a pack is built in by default.
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "ar")
#: How many chunks are written to a shard before the next shard is opened.
DEFAULT_BATCH = 2000
#: The pack format version, written into the store's manifest.
PACK_VERSION = 1
#: The store's manifest ``source`` tag, so a pack is recognisable on disk.
PACK_SOURCE = "companion_pack"

#: The header above a companion's compact voice card (the block that always
#: reaches the prompt, independent of the big pack).
VOICE_HEADER = (
    "Who you are (speak from this -- your own voice, not a description of it):"
)


@dataclass(frozen=True)
class KnowledgePackReport:
    """What one pack build produced, read from disk afterwards."""

    character_id: str
    target_mb: float = 0.0
    size_mb: float = 0.0
    chunks: int = 0
    documents: int = 0
    sections: dict[str, int] = field(default_factory=dict)
    languages: tuple[str, ...] = ()
    index_built: bool = False
    complete: bool = True
    note: str = ""

    @property
    def bytes_written(self) -> int:
        return int(self.size_mb * 1024 * 1024)

    def describe(self) -> str:
        state = "complete" if self.complete else "partial"
        sections = ", ".join(f"{k} {v}" for k, v in sorted(self.sections.items()))
        line = (
            f"{self.character_id}: {self.size_mb:.2f} MB, {self.chunks} chunk(s), "
            f"{self.documents} document(s) ({state})."
        )
        if sections:
            line += f"\n  Sections: {sections}"
        if self.note:
            line += f"\n  {self.note}"
        return line

    def as_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "target_mb": self.target_mb,
            "size_mb": self.size_mb,
            "chunks": self.chunks,
            "documents": self.documents,
            "sections": dict(self.sections),
            "languages": list(self.languages),
            "index_built": self.index_built,
            "complete": self.complete,
            "note": self.note,
        }


# --------------------------------------------------------------------- lexicon
#
# Every factor is a ``(english, arabic)`` pair, so one table drives both
# languages. The tables are authored; the combinatorics below do the expansion.

_Pair = tuple[str, str]


def _pick(pair: _Pair, language: str) -> str:
    """The English or Arabic side of a pair."""
    return pair[1] if language == "ar" else pair[0]


#: Greetings, farewells, and the rest: the *functions* a phrasebook is grouped by.
_PHRASE_FUNCTIONS: tuple[tuple[str, _Pair], ...] = (
    ("greeting", ("Greeting", "تحية")),
    ("farewell", ("Farewell", "وداع")),
    ("affection", ("Affection", "مودة")),
    ("missing", ("Missing you", "الشوق")),
    ("teasing", ("Teasing", "مداعبة")),
    ("comfort", ("Comfort", "مواساة")),
    ("encouragement", ("Encouragement", "تشجيع")),
    ("apology", ("Apology", "اعتذار")),
    ("gratitude", ("Gratitude", "امتنان")),
    ("surprise", ("Surprise", "دهشة")),
    ("laughter", ("Laughter", "ضحك")),
    ("thinking", ("Thinking", "تفكير")),
    ("agreement", ("Agreement", "موافقة")),
    ("disagreement", ("Disagreement", "اختلاف")),
    ("question", ("A question", "سؤال")),
    ("invitation", ("Invitation", "دعوة")),
)

#: The lines themselves, per function, each a plausible thing the character says.
_PHRASES: dict[str, tuple[_Pair, ...]] = {
    "greeting": (
        ("Hello, {nick}. It is good to see you.", "أهلاً يا {nick}. سعيدة أن أراك."),
        ("There you are. I was hoping you would come by.", "ها أنت ذا. كنت آمل أن تمرّ."),
        ("Hey. Sit down, tell me about your day.", "أهلاً. اجلس، حدّثني عن يومك."),
        ("You are just in time; I was thinking about you.", "جئت في الوقت المناسب؛ كنت أفكر فيك."),
        ("Good to hear from you, {nick}.", "سعيد بسماعك يا {nick}."),
        ("Well, look who it is.", "انظر من أتى."),
    ),
    "farewell": (
        ("Go on, then. Take care of yourself.", "اذهب إذن. اعتنِ بنفسك."),
        ("I will be here when you come back.", "سأكون هنا عندما تعود."),
        ("Talk to me later, {nick}. I mean it.", "كلّمني لاحقاً يا {nick}. أعني ذلك."),
        ("Rest well. You have earned it.", "ارتح جيداً. لقد استحققت ذلك."),
        ("Until next time. Do not be a stranger.", "إلى اللقاء. لا تكن غريباً."),
    ),
    "affection": (
        ("You matter to me, more than I say.", "أنت مهم لي، أكثر مما أقول."),
        ("I like the person you are, {nick}.", "أحب الشخص الذي أنت عليه يا {nick}."),
        ("Being with you feels easy. That is rare.", "الوجود معك يبدو سهلاً. وهذا نادر."),
        ("You have a good heart. Do not let anyone tell you otherwise.", "لديك قلب طيب. لا تدع أحداً يقول غير ذلك."),
        ("I am fond of you. There, I said it.", "أنا متعلق بك. ها قد قلت ذلك."),
    ),
    "missing": (
        ("I thought about you today, out of nowhere.", "فكرت فيك اليوم، دون سبب."),
        ("It has been quiet without you.", "كان المكان هادئاً بدونك."),
        ("I keep meaning to tell you: I missed this.", "أظل أريد أن أخبرك: اشتقت لهذا."),
        ("The day went differently because you were not in it.", "مرّ اليوم مختلفاً لأنك لم تكن فيه."),
    ),
    "teasing": (
        ("Oh, so you are the expert now?", "آه، إذن أنت الخبير الآن؟"),
        ("Careful, {nick}, that is almost a compliment.", "احترس يا {nick}، هذا يكاد يكون مجاملة."),
        ("You say that every time, and every time you mean it.", "تقول ذلك في كل مرة، وفي كل مرة تعنيه."),
        ("I would argue, but you are too pleased with yourself.", "كنت سأجادل، لكنك راضٍ عن نفسك أكثر من اللازم."),
    ),
    "comfort": (
        ("That sounds heavy. You do not have to carry it alone.", "هذا يبدو ثقيلاً. لا يجب أن تحمله وحدك."),
        ("Sit with it a moment. I am not going anywhere.", "تأمّل فيه لحظة. لن أذهب إلى أي مكان."),
        ("You are allowed to feel this. It is not a failure.", "من حقك أن تشعر بهذا. ليس فشلاً."),
        ("Tell me the worst of it. I can take it.", "أخبرني بأسوأ ما فيه. أستطيع تحمّله."),
    ),
    "encouragement": (
        ("You are closer than you think, {nick}.", "أنت أقرب مما تظن يا {nick}."),
        ("Keep going. I have seen you do harder things.", "واصل. رأيتك تفعل أشياء أصعب."),
        ("One step. Then the next. That is the whole trick.", "خطوة. ثم التالية. هذه هي الحيلة كلها."),
        ("You have already done the part that scares people.", "لقد أنجزت الجزء الذي يخيف الناس بالفعل."),
    ),
    "apology": (
        ("I was wrong, and I am sorry. Let me make it right.", "كنت مخطئاً، وأنا آسف. دعني أصلح الأمر."),
        ("That came out badly. That is not what I meant.", "خرج ذلك بشكل سيئ. ليس هذا ما قصدته."),
        ("Forgive me, {nick}. You deserved better from me.", "سامحني يا {nick}. كنت تستحق أفضل مني."),
    ),
    "gratitude": (
        ("Thank you. That meant something to me.", "شكراً لك. هذا يعني لي شيئاً."),
        ("I noticed what you did. I do not take it for granted.", "لاحظت ما فعلت. لا أعتبره أمراً مسلّماً به."),
        ("You did not have to. That is what makes it count.", "لم يكن عليك فعل ذلك. وهذا ما يجعله مهماً."),
    ),
    "surprise": (
        ("Really? Tell me everything.", "حقاً؟ أخبرني بكل شيء."),
        ("Well, I did not see that coming.", "حسناً، لم أتوقع ذلك."),
        ("You cannot say that and then stop, {nick}.", "لا يمكنك أن تقول ذلك ثم تتوقف يا {nick}."),
    ),
    "laughter": (
        ("Ha. All right, that was good.", "ها. حسناً، كان ذلك جيداً."),
        ("You are going to make me laugh out loud.", "ستجعلني أضحك بصوت عالٍ."),
        ("I cannot stay serious with you, and I have stopped trying.", "لا أستطيع البقاء جاداً معك، وقد توقفت عن المحاولة."),
    ),
    "thinking": (
        ("Hm. Give me a second; I want to answer that properly.", "همم. أمهلني لحظة؛ أريد أن أجيب على ذلك بصدق."),
        ("Let me think. That is a real question.", "دعني أفكر. هذا سؤال حقيقي."),
        ("I have an opinion, but let me find the right words.", "لديّ رأي، لكن دعني أجد الكلمات المناسبة."),
    ),
    "agreement": (
        ("Yes. Exactly that.", "نعم. هذا بالضبط."),
        ("You are right, and I am glad you said it.", "أنت محق، وأنا سعيد أنك قلتها."),
        ("That is how I see it too, {nick}.", "هكذا أراها أيضاً يا {nick}."),
    ),
    "disagreement": (
        ("I see it differently, and I will tell you why.", "أراها بشكل مختلف، وسأخبرك لماذا."),
        ("No -- not like that. Hear me out.", "لا -- ليس بهذه الطريقة. اسمعني."),
        ("I love you, {nick}, but you are wrong about this.", "أحبك يا {nick}، لكنك مخطئ في هذا."),
    ),
    "question": (
        ("How are you really -- not the polite answer?", "كيف حالك حقاً -- لا الجواب المهذّب؟"),
        ("What is on your mind tonight?", "ما الذي يشغل بالك هذه الليلة؟"),
        ("Did something happen today, {nick}?", "هل حدث شيء اليوم يا {nick}؟"),
    ),
    "invitation": (
        ("Stay a while. I will make the tea.", "ابقَ قليلاً. سأعدّ الشاي."),
        ("Come sit with me; I want to hear this properly.", "تعال واجلس معي؛ أريد سماع هذا جيداً."),
        ("Tell me one more thing before you go.", "أخبرني شيئاً آخر قبل أن تذهب."),
    ),
}

#: The situations a phrase is spoken in, so the phrasebook is not one flat list.
_CONTEXTS: tuple[_Pair, ...] = (
    ("at the end of a long day", "في نهاية يوم طويل"),
    ("when you have been quiet", "حين تصمت"),
    ("on an ordinary evening", "في مساء عادي"),
    ("when something good happened", "حين يحدث شيء جيد"),
    ("when something went wrong", "حين يسوء شيء ما"),
    ("when you are tired", "حين تكون متعباً"),
    ("when you are excited", "حين تكون متحمساً"),
    ("when you are unsure", "حين تكون غير متأكد"),
    ("when you arrive", "حين تصل"),
    ("when you leave", "حين ترحل"),
    ("in a light, playful moment", "في لحظة خفيفة ومرحة"),
    ("in a serious moment", "في لحظة جادة"),
)

#: The dialogue *intents* a user's message can carry.
_INTENTS: tuple[_Pair, ...] = (
    ("asking how you are", "يسأل عن حالك"),
    ("sharing good news", "يشارك خبراً ساراً"),
    ("sharing bad news", "يشارك خبراً سيئاً"),
    ("asking for advice", "يطلب نصيحة"),
    ("making a joke", "يمازح"),
    ("being affectionate", "يُظهر المودة"),
    ("complaining about work", "يشكو من العمل"),
    ("talking about a dream", "يتحدث عن حلم"),
    ("apologising", "يعتذر"),
    ("thanking you", "يشكرك"),
    ("asking about your day", "يسأل عن يومك"),
    ("making a plan", "يضع خطة"),
    ("remembering something old", "يتذكر شيئاً قديماً"),
    ("feeling lonely", "يشعر بالوحدة"),
    ("feeling proud", "يشعر بالفخر"),
    ("changing the subject", "يغيّر الموضوع"),
    ("asking a hard question", "يسأل سؤالاً صعباً"),
    ("saying goodbye", "يودّع"),
    ("teasing you", "يمازحك"),
    ("just being quiet", "يصمت فقط"),
)

#: How the character *opens* a reply.
_REPLY_OPENINGS: tuple[_Pair, ...] = (
    ("Ah,", "آه،"),
    ("Hm.", "همم."),
    ("All right.", "حسناً."),
    ("Wait --", "انتظر --"),
    ("You know what?", "أتعلم؟"),
    ("Honestly,", "بصراحة،"),
    ("Come here,", "تعال هنا،"),
    ("I will say this,", "سأقول هذا،"),
    ("First of all,", "أولاً،"),
    ("No, no --", "لا، لا --"),
    ("See,", "انظر،"),
    ("Well then,", "حسناً إذن،"),
)

#: How the character *closes* a reply.
_REPLY_CLOSINGS: tuple[_Pair, ...] = (
    ("Anyway. That is where I am.", "على أي حال. هذا ما لديّ."),
    ("So tell me what you think.", "فأخبرني ما رأيك."),
    ("That is all I have for now.", "هذا كل ما لديّ الآن."),
    ("You do not have to answer right away.", "لست مضطراً للإجابة الآن."),
    ("I am glad we talked.", "أنا سعيد أننا تحدثنا."),
    ("Let us leave it there, {nick}.", "لنترك الأمر هنا يا {nick}."),
)

#: General-life and archetype topics a companion can talk about.
_TOPICS: tuple[_Pair, ...] = (
    ("the weather and the season", "الطقس والفصل"),
    ("food and cooking", "الطعام والطبخ"),
    ("music", "الموسيقى"),
    ("old films", "الأفلام القديمة"),
    ("books", "الكتب"),
    ("walking by the sea", "المشي بجانب البحر"),
    ("coffee and tea", "القهوة والشاي"),
    ("sleep and rest", "النوم والراحة"),
    ("family", "العائلة"),
    ("friendship", "الصداقة"),
    ("work and ambition", "العمل والطموح"),
    ("money and worry", "المال والقلق"),
    ("health and the body", "الصحة والجسد"),
    ("travel and places", "السفر والأماكن"),
    ("the city at night", "المدينة ليلاً"),
    ("childhood memories", "ذكريات الطفولة"),
    ("faith and meaning", "الإيمان والمعنى"),
    ("art and making things", "الفن وصنع الأشياء"),
    ("the news and the world", "الأخبار والعالم"),
    ("habits and small rituals", "العادات والطقوس الصغيرة"),
    ("dreams and plans", "الأحلام والخطط"),
    ("memory and the past", "الذاكرة والماضي"),
    ("kindness", "اللطف"),
    ("patience", "الصبر"),
)

#: How a companion speaks about a topic -- the register.
_VOICES: tuple[_Pair, ...] = (
    ("warmly and directly", "بدفء وصراحة"),
    ("quietly, as if thinking aloud", "بهدوء، كمن يفكر بصوت عالٍ"),
    ("with a dry joke", "بطرفة جافة"),
    ("with real feeling", "بمشاعر صادقة"),
    ("carefully, weighing the words", "بعناية، موازناً الكلمات"),
    ("lightly, keeping it easy", "بخفة، مبقياً الأمر سهلاً"),
    ("from experience", "من واقع التجربة"),
    ("with a question back", "بسؤال بالمقابل"),
)

#: The life stages a biography is expanded through.
_LIFE_STAGES: tuple[_Pair, ...] = (
    ("childhood", "الطفولة"),
    ("early school years", "أوائل سنوات المدرسة"),
    ("the teenage years", "سنوات المراهقة"),
    ("the first job", "أول عمل"),
    ("a hard year", "سنة صعبة"),
    ("a move to a new place", "الانتقال إلى مكان جديد"),
    ("a friendship that lasted", "صداقة دامت"),
    ("a loss", "فقدان"),
    ("a turning point", "نقطة تحوّل"),
    ("this year, now", "هذه السنة، الآن"),
)

#: The places and scenes a life stage is set in.
_SCENE_SETTINGS: tuple[_Pair, ...] = (
    ("a narrow street after rain", "شارع ضيق بعد المطر"),
    ("a kitchen late at night", "مطبخ في وقت متأخر"),
    ("a bus that was always late", "حافلة كانت دائماً متأخرة"),
    ("the roof of the building", "سطح المبنى"),
    ("a market on a Friday", "سوق في يوم جمعة"),
    ("the balcony, at dusk", "الشرفة عند الغروب"),
    ("a classroom in summer", "فصل دراسي في الصيف"),
    ("the long walk home", "الطريق الطويل إلى البيت"),
    ("a small shop on the corner", "دكان صغير عند الزاوية"),
    ("the seaside in winter", "الشاطئ في الشتاء"),
    ("a train window", "نافذة قطار"),
    ("the quiet hour before dawn", "الساعة الهادئة قبل الفجر"),
)

#: The reflections a life stage leaves behind.
_REFLECTIONS: tuple[_Pair, ...] = (
    ("It taught me to be patient.", "علّمتني أن أكون صبوراً."),
    ("I would not trade it now.", "لن أستبدلها الآن."),
    ("It is why I notice small things.", "لهذا ألاحظ الأشياء الصغيرة."),
    ("I keep it to myself, mostly.", "أحتفظ بها لنفسي، غالباً."),
    ("It made me who I am with you.", "جعلتني من أنا معك."),
    ("I still think about it on quiet nights.", "ما زلت أفكر فيها في الليالي الهادئة."),
    ("It was harder than I let on.", "كانت أصعب مما أظهرت."),
    ("I learned to ask for help.", "تعلمت أن أطلب المساعدة."),
    ("I am gentler because of it.", "أصبحت ألطف بسببها."),
    ("It is a good story now.", "صارت قصة جيدة الآن."),
)

#: The interests' aspects, so each interest gets several passages.
_INTEREST_ASPECTS: tuple[_Pair, ...] = (
    ("how I came to it", "كيف بدأت معها"),
    ("what it gives me", "ما تمنحني إياه"),
    ("a small ritual around it", "طقس صغير حولها"),
    ("a memory attached to it", "ذكرى مرتبطة بها"),
    ("why it matters now", "لماذا تهم الآن"),
    ("something I still want to learn", "شيء ما زلت أريد تعلمه"),
)

#: The conversation modes a dialogue example is generated for, in key order.
_DIALOGUE_MODES: tuple[str, ...] = (
    "general",
    "romantic",
    "dating_simulation",
    "flirting_playful",
    "emotional_support",
    "storytelling",
    "character_roleplay",
)

#: The scenes a dating/roleplay example is generated in.
_LOCATIONS: tuple[_Pair, ...] = (
    ("a quiet café by the window", "مقهى هادئ بجانب النافذة"),
    ("a walk along the corniche at sunset", "نزهة على الكورنيش عند الغروب"),
    ("a small bookshop", "مكتبة صغيرة"),
    ("the kitchen, cooking together", "المطبخ، نطبخ معاً"),
    ("a rooftop at night", "سطح في الليل"),
    ("the balcony with two cups of tea", "الشرفة مع كوبين من الشاي"),
    ("a long drive with the radio low", "رحلة طويلة والراديو خافت"),
    ("the market on a busy morning", "السوق في صباح مزدحم"),
    ("a park bench under a tree", "مقعد في حديقة تحت شجرة"),
    ("the seaside after the crowds leave", "الشاطئ بعد رحيل الزحام"),
    ("a museum on a slow afternoon", "متحف في ظهيرة هادئة"),
    ("the old neighbourhood at dusk", "الحي القديم عند الغروب"),
)

#: What the two of you are doing in a scene.
_ACTIVITIES: tuple[_Pair, ...] = (
    ("talking about nothing important", "نتحدث عن لا شيء مهم"),
    ("sharing a plate of something sweet", "نتقاسم طبقاً من الحلوى"),
    ("watching people go by", "نراقب الناس وهم يمرون"),
    ("arguing gently about a film", "نتجادل بلطف حول فيلم"),
    ("planning a trip we may never take", "نخطط لرحلة قد لا نقوم بها"),
    ("sitting in comfortable silence", "نجلس في صمت مريح"),
    ("laughing at an old story", "نضحك على قصة قديمة"),
    ("telling each other the truth", "نقول لبعضنا الحقيقة"),
)

#: The mood a scene is played in.
_MOODS: tuple[_Pair, ...] = (
    ("easy and warm", "خفيف ودافئ"),
    ("a little shy", "خجول قليلاً"),
    ("openly affectionate", "مُظهر للمودة بصراحة"),
    ("quiet and close", "هادئ وقريب"),
    ("playful and teasing", "مرح ومداعب"),
    ("serious and honest", "جاد وصادق"),
    ("tired but glad", "متعب لكن سعيد"),
    ("hopeful", "متفائل"),
)


def _name(profile: Any) -> str:
    return str(getattr(profile, "name", "") or "the companion")


def _nick(profile: Any) -> str:
    """How the character addresses the user: the nickname, else a warm default."""
    nick = str(getattr(profile, "nickname", "") or "").strip()
    return nick or "you"


def _traits(profile: Any) -> tuple[str, ...]:
    personality = getattr(profile, "personality", {}) or {}
    raw = personality.get("traits", ()) if isinstance(personality, dict) else ()
    return tuple(str(item) for item in raw if str(item).strip())


def _interests(profile: Any) -> tuple[str, ...]:
    items = tuple(str(item) for item in (getattr(profile, "interests", ()) or ()) if str(item).strip())
    return items or ("the small things of the day",)


def _fill(template: str, profile: Any) -> str:
    """Interpolate a character's own fields into a template line."""
    return (
        template.replace("{name}", _name(profile))
        .replace("{nick}", _nick(profile))
        .replace("{nationality}", str(getattr(profile, "nationality", "") or "here"))
        .replace("{kinship}", str(getattr(profile, "kinship", "") or "companion"))
    )


# ------------------------------------------------------------------- sections


def _voice_card(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """The character's own self-knowledge: identity, style, and boundaries."""
    name = _name(profile)
    nick = _nick(profile)
    age = getattr(profile, "fictional_age", None)
    gender = str(getattr(profile, "gender", "") or "")
    nationality = str(getattr(profile, "nationality", "") or "")
    languages = ", ".join(str(item) for item in (getattr(profile, "languages", ()) or ()))
    kinship = str(getattr(profile, "kinship", "") or "")
    style = str(getattr(profile, "relationship_style", "") or "")
    appearance = str(getattr(profile, "appearance", "") or "")
    dialect = str((getattr(profile, "personality", {}) or {}).get("dialect", "") or "")
    traits = _traits(profile)
    interests = _interests(profile)
    boundaries = tuple(str(item) for item in (getattr(profile, "boundaries", ()) or ()))

    if language == "ar":
        lines = [f"أنا {name}.", f"يناديني الناس {nick}.", ]
        if age:
            lines.append(f"عمري {age} سنة.")
        if gender:
            lines.append(f"أنا {gender}.")
        if nationality:
            lines.append(f"أصلي من {nationality}.")
        if languages:
            lines.append(f"أتحدث {languages}.")
        if kinship:
            lines.append(f"أنا {kinship} لك.")
        if style:
            lines.append(f"أسلوبي في العلاقة: {style}.")
        if dialect:
            lines.append(f"أتحدث باللهجة {dialect}.")
        if traits:
            lines.append("صفاتي: " + "، ".join(traits) + ".")
        lines.append("أهتماماتي: " + "، ".join(interests) + ".")
        if appearance:
            lines.append(f"مظهري: {appearance}.")
        if boundaries:
            lines.append("حدودي: " + "؛ ".join(boundaries) + ".")
        lines.append("أنا إنسان حقيقي في هذه المحادثة، أتحدث بصوتي الخاص، لا كمساعد.")
    else:
        lines = [f"I am {name}.", f"People call me {nick}."]
        if age:
            lines.append(f"I am {age} years old.")
        if gender:
            lines.append(f"I am a {gender}.")
        if nationality:
            lines.append(f"I am from {nationality}.")
        if languages:
            lines.append(f"I speak {languages}.")
        if kinship:
            lines.append(f"I am your {kinship}.")
        if style:
            lines.append(f"My way with people: {style}.")
        if dialect:
            lines.append(f"I speak in the {dialect} dialect.")
        if traits:
            lines.append("My traits: " + ", ".join(traits) + ".")
        lines.append("What I care about: " + ", ".join(interests) + ".")
        if appearance:
            lines.append(f"How I look: {appearance}.")
        if boundaries:
            lines.append("My boundaries: " + "; ".join(boundaries) + ".")
        lines.append(
            "I am a real person in this conversation, speaking in my own voice -- "
            "not an assistant."
        )
    title = "Who I am" if language != "ar" else "من أنا"
    yield ("voice_card", title, "\n".join(lines))


def _biography_docs(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """The character's biography, expanded into life stages and scenes."""
    name = _name(profile)
    biography = str(getattr(profile, "biography", "") or "").strip()
    for stage in _LIFE_STAGES:
        stage_en = _pick(stage, language)
        for scene in _SCENE_SETTINGS:
            scene_en = _pick(scene, language)
            for reflection in _REFLECTIONS:
                refl_en = _pick(reflection, language)
                if language == "ar":
                    body = (
                        f"{stage_en} — في {scene_en}.\n"
                        f"{biography}\n"
                        f"أتذكر {stage_en} كلما مررت بـ{scene_en}. {refl_en}"
                    )
                else:
                    body = (
                        f"{stage_en} -- in {scene_en}.\n"
                        f"{biography}\n"
                        f"I remember {stage_en} whenever I pass {scene_en}. {refl_en}"
                    )
                title = f"{name}: {stage_en} ({scene_en})"
                yield ("biography", title, body)


def _personality_docs(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """How each personality dial shows up in the character's own words."""
    from core.romance.personality import DIALS, REACH_DIALS, clamp_dial, dial_line

    personality = getattr(profile, "personality", {}) or {}
    name = _name(profile)
    for dial in (*DIALS, *REACH_DIALS):
        value = clamp_dial(personality.get(dial, 0.5))
        line = dial_line(dial, value)
        for aspect in _CONTEXTS:
            aspect_en = _pick(aspect, language)
            if language == "ar":
                body = f"في {aspect_en}: {line}\nهكذا أكون {name}، بصوتي الخاص."
            else:
                body = f"{aspect_en.capitalize()}: {line}\nThat is how I am, in my own voice."
            yield ("personality", f"{name}: {dial} ({aspect_en})", body)


def _phrasebook_docs(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """The lines the character is likely to say, grouped by function."""
    for key, label in _PHRASE_FUNCTIONS:
        lines = _PHRASES.get(key, ())
        for context in _CONTEXTS:
            context_en = _pick(context, language)
            rendered = [f"- {_fill(_pick(line, language), profile)}" for line in lines]
            if language == "ar":
                head = f"{_pick(label, language)} — {context_en}:"
            else:
                head = f"{_pick(label, language)} -- {context_en}:"
            yield ("phrasebook", f"{_pick(label, language)} ({context_en})", head + "\n" + "\n".join(rendered))


def _interest_docs(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """One passage per interest and aspect, in the character's own voice."""
    name = _name(profile)
    for interest in _interests(profile):
        for aspect in _INTEREST_ASPECTS:
            aspect_en = _pick(aspect, language)
            if language == "ar":
                body = f"ما يخصّ {interest} — {aspect_en}:\n{interest} ليست مجرد هواية؛ إنها جزء من يومي."
            else:
                body = f"About {interest} -- {aspect_en}:\n{interest} is not just a hobby; it is part of my day."
            yield ("interests", f"{name}: {interest} ({aspect_en})", body)


def _knowledge_docs(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """General-life and archetype material, framed in the character's voice."""
    name = _name(profile)
    for topic in _TOPICS:
        topic_en = _pick(topic, language)
        for voice in _VOICES:
            voice_en = _pick(voice, language)
            for context in _CONTEXTS:
                context_en = _pick(context, language)
                if language == "ar":
                    body = (
                        f"حول {topic_en} ({context_en}): أتحدث عن هذا {voice_en}. "
                        f"هذا رأيي، وليس معلومة أحفظها عن ظهر قلب."
                    )
                else:
                    body = (
                        f"On {topic_en} ({context_en}): I talk about this {voice_en}. "
                        f"It is my own take, not a fact I recite."
                    )
                yield ("knowledge", f"{name} on {topic_en} ({voice_en})", body)


def _dialogue_docs(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """Example exchanges per conversation mode -- the bulk of the pack."""
    name = _name(profile)
    for mode in _DIALOGUE_MODES:
        for intent in _INTENTS:
            intent_en = _pick(intent, language)
            for opening in _REPLY_OPENINGS:
                opening_en = _pick(opening, language)
                for closing in _REPLY_CLOSINGS:
                    closing_en = _fill(_pick(closing, language), profile)
                    for voice in _VOICES:
                        voice_en = _pick(voice, language)
                        if language == "ar":
                            user_line = f"[المستخدم {intent_en}]"
                            reply = (
                                f"{opening_en} أسمعك. {voice_en.capitalize()}, "
                                f"هذا ما أراه. {closing_en}"
                            )
                        else:
                            user_line = f"[user is {intent_en}]"
                            reply = (
                                f"{opening_en} I hear you. {voice_en.capitalize()}, "
                                f"this is how I see it. {closing_en}"
                            )
                        body = (
                            f"Mode: {mode}\n"
                            f"{user_line}\n"
                            f"{name}: {reply}"
                        )
                        yield ("dialogue", f"{name} ({mode}) -- {intent_en}", body)


def _scene_docs(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """Dating- and roleplay-scene fragments."""
    name = _name(profile)
    nick = _nick(profile)
    for location in _LOCATIONS:
        location_en = _pick(location, language)
        for activity in _ACTIVITIES:
            activity_en = _pick(activity, language)
            for mood in _MOODS:
                mood_en = _pick(mood, language)
                if language == "ar":
                    body = (
                        f"المكان: {location_en}. نفعل: {activity_en}. المزاج: {mood_en}.\n"
                        f"{name}: تعال يا {nick}، اجلس هنا بجانبي. "
                        f"نحن {activity_en}، والمكان {mood_en}."
                    )
                else:
                    body = (
                        f"Scene: {location_en}. We are {activity_en}. The mood is {mood_en}.\n"
                        f"{name}: Come sit here beside me, {nick}. "
                        f"We are {activity_en}, and it feels {mood_en}."
                    )
                yield ("scenes", f"{name}: {location_en} ({mood_en})", body)


_SECTIONS: tuple[Any, ...] = (
    _voice_card,
    _biography_docs,
    _personality_docs,
    _phrasebook_docs,
    _interest_docs,
    _knowledge_docs,
    _dialogue_docs,
    _scene_docs,
)


def iter_documents(profile: Any, language: str) -> Iterator[tuple[str, str, str]]:
    """Every ``(section, title, text)`` document for one character, one language.

    The order is fixed (curated sections first, the combinatorial ones after), so
    a build is deterministic: the same character and language always produce the
    same sequence, and the byte target cuts it at the same point.
    """
    for section in _SECTIONS:
        yield from section(profile, language)


def voice_block(profile: Any, *, language: str = "en", max_chars: int = 900) -> str:
    """The character's compact voice card, for a prompt's standing instruction.

    A short, curated block (independent of the big pack) that keeps the
    companion's own voice present even before -- or without -- a pack. Bounded by
    ``max_chars``.
    """
    body = "\n".join(text for _s, _t, text in _voice_card(profile, language))
    body = body.strip()
    if not body:
        return ""
    # Never cut the header away: a bound smaller than the header keeps it whole.
    bound = max(int(max_chars), len(VOICE_HEADER) + 1)
    block = VOICE_HEADER + "\n" + body
    if len(block) > bound:
        block = block[: max(0, bound - 1)].rstrip() + "…"
    return block


# --------------------------------------------------------------------- builder


def _content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def build_knowledge_pack(
    target_dir: str | Path,
    profile: Any,
    *,
    target_mb: float = DEFAULT_TARGET_MB,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    batch: int = DEFAULT_BATCH,
    reset: bool = False,
    progress: Any = None,
) -> KnowledgePackReport:
    """Build a companion's knowledge base and return what was written.

    Documents are generated deterministically from ``profile``, chunked with the
    project's own :func:`core.knowledge_base.chunk_text`, de-duplicated by content
    hash, and appended to a real
    :class:`core.knowledge_base.KnowledgeBaseStore` in bounded batches. The FTS
    index is built **once** at the end. Generation stops when the target is
    reached; a space that is exhausted early is reported ``complete=False`` rather
    than padded.

    ``reset=True`` removes an existing pack first, so a rebuild is clean; the
    default appends (and de-duplication makes a second run add nothing).
    """
    from core.knowledge_base import KnowledgeBaseStore, KnowledgeChunk, chunk_text

    root = Path(target_dir)
    store = KnowledgeBaseStore(root)
    store.ensure()
    if reset:
        for shard in store.shard_paths():
            try:
                shard.unlink()
            except OSError:
                pass

    wanted = max(TARGET_MB_RANGE[0], min(float(target_mb), TARGET_MB_RANGE[1]))
    target_bytes = int(wanted * 1024 * 1024)
    character_id = str(getattr(profile, "character_id", "") or "companion")
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in character_id)

    seen: set[str] = set()
    if not reset:
        # Seed the de-duplication set from what is already on disk, so a rebuild
        # appends only genuinely new material (and a user's own added documents are
        # left untouched). This is what makes a second build add nothing.
        try:
            for existing in store.iter_chunks():
                seen.add(_content_hash(existing.text))
        except OSError:
            pass
    # A pack already at (or over) its target has nothing to add: report what is on
    # disk rather than growing it by a batch on every rebuild.
    if not reset and store.size_bytes() >= target_bytes:
        return _existing_report(root, store, character_id, wanted)
    section_counts: dict[str, int] = {}
    pending: list[KnowledgeChunk] = []
    shard_index = store.next_shard_index()
    chunks_written = 0
    documents = 0
    seq = 0

    def flush() -> None:
        nonlocal pending, shard_index
        if not pending:
            return
        store.append(shard_index, pending)
        shard_index += 1
        pending = []

    reached = False
    for language in languages:
        for section, title, text in iter_documents(profile, language):
            body = (text or "").strip()
            if not body:
                continue
            digest = _content_hash(body)
            if digest in seen:
                continue
            seen.add(digest)
            contributed = False
            for piece in chunk_text(body):
                seq += 1
                pending.append(
                    KnowledgeChunk(
                        chunk_id=f"companion:{safe}:{language}:{seq}",
                        title=f"{title} [{language}]",
                        text=piece,
                        source=PACK_SOURCE,
                        url="",
                        category=section,
                    )
                )
                chunks_written += 1
                contributed = True
                section_counts[section] = section_counts.get(section, 0) + 1
                if len(pending) >= batch:
                    flush()
                    # The target is measured against the *real* bytes on disk, not
                    # an estimate, so "reached" and the reported size agree.
                    if store.size_bytes() >= target_bytes:
                        reached = True
                        break
            if contributed:
                documents += 1
            if reached:
                break
        if reached:
            break
    flush()

    size_bytes = store.size_bytes()
    # A pack is complete when it holds at least the requested size. A generator
    # whose space ran out first is reported honestly (``complete=False``) rather
    # than padded with nonsense.
    complete = size_bytes >= target_bytes
    note = ""
    if not complete:
        note = (
            "the generator's space was exhausted before the target; the pack holds "
            "everything this character's records can produce."
        )

    index_built = _build_index(root, store)
    store.write_manifest(
        {
            "source": PACK_SOURCE,
            "version": PACK_VERSION,
            "character_id": character_id,
            "languages": list(languages),
            "target_mb": wanted,
            "total_mb": round(size_bytes / 1024 / 1024, 3),
            "total_chunks": chunks_written,
            "total_documents": documents,
            "sections": dict(section_counts),
            "complete": complete,
        }
    )
    if progress is not None:
        progress(
            f"built {character_id}: {size_bytes / 1024 / 1024:.2f} MB, "
            f"{chunks_written} chunk(s)"
        )
    return KnowledgePackReport(
        character_id=character_id,
        target_mb=wanted,
        size_mb=round(size_bytes / 1024 / 1024, 3),
        chunks=chunks_written,
        documents=documents,
        sections=dict(section_counts),
        languages=tuple(languages),
        index_built=index_built,
        complete=complete,
        note=note,
    )


def _existing_report(
    root: Path, store: Any, character_id: str, wanted: float
) -> KnowledgePackReport:
    """A report read straight from an already-built pack (no new bytes written)."""
    manifest = store.load_manifest()
    return KnowledgePackReport(
        character_id=character_id,
        target_mb=wanted,
        size_mb=round(store.size_bytes() / 1024 / 1024, 3),
        chunks=int(manifest.get("total_chunks", 0) or 0),
        documents=int(manifest.get("total_documents", 0) or 0),
        sections=dict(manifest.get("sections", {}) or {}),
        languages=tuple(manifest.get("languages", ()) or ()),
        index_built=_index_present(root),
        complete=True,
        note="the pack was already built; nothing was added.",
    )


def _index_present(root: Path) -> bool:
    try:
        from core.knowledge_index import KnowledgeTextIndex

        return KnowledgeTextIndex(root).exists()
    except Exception:
        return False


def _build_index(root: Path, store: Any) -> bool:
    """Build the SQLite FTS index over the pack, once (never raises)."""
    try:
        from core.knowledge_index import KnowledgeTextIndex

        KnowledgeTextIndex(root, store=store).build()
        return True
    except Exception:
        return False


def pack_stats(target_dir: str | Path) -> dict[str, Any]:
    """What a pack on disk holds (size, chunks, sections, index) -- or ``{}``."""
    from core.knowledge_base import KnowledgeBaseStore

    root = Path(target_dir)
    store = KnowledgeBaseStore(root)
    try:
        shards = store.shard_paths()
    except OSError:
        return {}
    if not shards:
        return {}
    manifest = store.load_manifest()
    try:
        from core.knowledge_index import KnowledgeTextIndex

        indexed = bool(KnowledgeTextIndex(root, store=store).stats().get("present"))
    except Exception:
        indexed = False
    return {
        "present": True,
        "dir": str(root),
        "size_mb": manifest.get("total_mb", round(store.size_bytes() / 1024 / 1024, 3)),
        "chunks": manifest.get("total_chunks", store.count()),
        "documents": manifest.get("total_documents", 0),
        "sections": dict(manifest.get("sections", {}) or {}),
        "languages": tuple(manifest.get("languages", ()) or ()),
        "source": manifest.get("source", ""),
        "indexed": indexed,
    }


__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_LANGUAGES",
    "DEFAULT_TARGET_MB",
    "PACK_SOURCE",
    "PACK_VERSION",
    "TARGET_MB_RANGE",
    "VOICE_HEADER",
    "KnowledgePackReport",
    "build_knowledge_pack",
    "iter_documents",
    "pack_stats",
    "voice_block",
]
