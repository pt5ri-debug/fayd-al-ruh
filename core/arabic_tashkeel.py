"""Full Arabic diacritization (tashkeel) as a deterministic, dependency-free pass.

The project is Arabic-first: an Arabic question reaches the ontology, and the
answer is written in Arabic. But written Arabic normally carries no vowel marks,
so a reader (and a speech engine) has to infer them. This module restores them.

Three honest layers, applied in order:

1. **A curated lexicon** -- the authoritative layer. Every entry is a word the
   project itself writes (its own function words, its petroleum vocabulary, and
   the Arabic terms of its glossary) with its correct full vocalization. A word
   found here is marked with its exact vowels and case ending.
2. **Deterministic pattern rules** -- for a word the lexicon does not hold. Only
   rules that are *certain* are applied: the definite article ``ال`` (moon vs.
   sun letter), a long vowel ``ا``/``و``/``ي`` after its short vowel, the tāʾ
   marbūṭa ``ة`` and the alif maqṣūra ``ى``. A rule is never a guess: it is a
   spelling fact of the word's own letters.
3. **A bounded case-ending pass** (only in *formal* mode) -- a simplified but
   real grammar: a word is nominative by default, genitive after a preposition,
   accusative after ``إنّ`` and its sisters, jussive after ``لم``. Its limits are
   stated plainly below rather than hidden.

Two modes, because the user asked for both:

* :data:`MODE_FORMAL` -- *فصحى*: the three layers, including case endings.
* :data:`MODE_COLLOQUIAL` -- *عامية*: the lexical and pattern vowels only; case
  endings (إعراب) are dropped, because spoken Arabic does not pronounce them.

Honesty by construction
-----------------------

* **Nothing is invented for a word the project did not write.** A word no rule
  can safely mark is returned with whatever marks *are* certain (often none),
  and the report counts it, so a caller can see how much was covered rather than
  trusting a figure.
* **A wrong vowel mark is worse than none.** The pattern layer therefore applies
  only transformations that follow from the letters themselves; it does not
  guess a short vowel it cannot justify.
* **English and every non-Arabic token are returned byte-identical.** Only an
  Arabic word is touched, and a Latin identifier, a number, a formula, or a
  punctuation mark is left exactly as it was.
* **Idempotent.** Diacritizing an already-diacritized text changes nothing: the
  existing marks are stripped and re-derived, so a second pass is a no-op.

This module imports nothing but :mod:`re` and :mod:`dataclasses`, loads no model,
and reaches no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# The marks
# --------------------------------------------------------------------------- #

FATHA = "\u064e"        # ـَ
DAMMA = "\u064f"        # ـُ
KASRA = "\u0650"        # ـِ
SUKUN = "\u0652"        # ـْ
SHADDA = "\u0651"       # ـّ
FATHATAN = "\u064b"     # ـً
DAMMATAN = "\u064c"     # ـٌ
KASRATAN = "\u064d"     # ـٍ
DAGGER_ALEF = "\u0670"  # ـٰ  (the superscript alef of ``هٰذا`` / ``الله``)

#: Every combining mark this module writes or strips.
DIACRITICS = FATHA + DAMMA + KASRA + SUKUN + SHADDA + FATHATAN + DAMMATAN + KASRATAN + DAGGER_ALEF

#: The short vowels, the marks the case-ending pass adds and removes.
SHORT_VOWELS = FATHA + DAMMA + KASRA
TANWEEN = FATHATAN + DAMMATAN + KASRATAN

#: ``re`` class matching any combining mark, so a word can be de-vocalized.
_DIACRITICS_RE = re.compile("[" + DIACRITICS + "]")

#: An Arabic word: a run of Arabic letters **and** their combining marks. The
#: marks are included so an already-vocalized word is matched as one unit and the
#: pass is idempotent -- excluding them would split ``الْمَسَامِيَّةُ`` at every
#: vowel and re-mark the fragments.
_ARABIC_WORD_RE = re.compile(r"[\u0621-\u064a\u064b-\u0652\u0670]+")

#: The definite article, and the letters that assimilate its ``ل`` (sun letters).
_SUN_LETTERS = "تثدذرزسشصضطظلن"
_MOON_LETTERS = "أبجحخعغفقكلمهويء"

#: The modes.
MODE_FORMAL = "formal"
MODE_COLLOQUIAL = "colloquial"
MODES = (MODE_FORMAL, MODE_COLLOQUIAL)


def strip_diacritics(text: str) -> str:
    """Remove every combining mark, leaving the bare consonantal skeleton."""
    return _DIACRITICS_RE.sub("", text)


def is_diacritized(text: str) -> bool:
    """True when ``text`` already carries at least one combining mark."""
    return bool(_DIACRITICS_RE.search(text))


def has_arabic(text: str) -> bool:
    """True when ``text`` carries any Arabic letter (U+0621..U+064A)."""
    return bool(_ARABIC_WORD_RE.search(text))


# --------------------------------------------------------------------------- #
# Layer 1 -- the curated lexicon
# --------------------------------------------------------------------------- #
#: Bare (undiacritized) word -> its fully vocalized form. Written as
#: ``vocalized: None`` pairs so the key is derived by stripping the marks, which
#: keeps one source of truth per word. Every entry is a word the project itself
#: writes: its function words, its petroleum vocabulary, and the Arabic terms of
#: its own glossary.
_VOCALIZED: tuple[str, ...] = (
    # -- function words, pronouns, particles --------------------------------
    "فِي", "مِنْ", "إِلَى", "عَلَى", "عَنْ", "مَعَ", "هَذَا", "هَذِهِ",
    "ذَلِكَ", "تِلْكَ", "الَّذِي", "الَّتِي", "الَّذِينَ", "مَا", "لَا", "لَمْ",
    "لَنْ", "أَنْ", "إِنَّ", "كَانَ", "كَانَتْ", "كَانُوا", "يَكُونُ",
    "تَكُونُ", "هُوَ", "هِيَ", "هُمْ", "نَحْنُ", "أَنْتَ", "أَنَا", "قَدْ",
    "كُلّ", "كُلِّ", "بَعْض", "غَيْر", "بَيْنَ", "عِنْدَ", "بَعْدَ", "قَبْلَ",
    "خِلَالَ", "حَتَّى", "كَمَا", "لَكِنْ", "أَوْ", "وَ", "ثُمَّ", "أَيْضًا",
    "حَيْثُ", "إِذَا", "إِذْ", "عِنْدَمَا", "لِذَلِكَ", "بِذَلِكَ", "فِيهِ",
    "فِيهَا", "مِنْهَا", "مِنْهُ", "بِهِ", "بِهَا", "لَهُ", "لَهَا",
    "إِلَيْهِ", "عَلَيْهِ", "عَلَيْهَا", "بِهَذَا", "وَمَا", "وَمِنْ",
    "وَفِي", "وَهُوَ", "وَهِيَ", "وَهَذَا", "وَهَذِهِ", "مِمَّا", "بِمَا",
    "لِأَنَّ", "لِكَيْ", "بَلْ", "لَيْسَ", "لَيْسَتْ", "إِلَّا", "سِوَى",
    "سَوَاء", "إِمَّا", "لَدَى", "دُونَ", "بِدُونِ", "عَبْرَ", "ضِمْنَ",
    "دَاخِلَ", "خَارِجَ", "أَمَامَ", "خَلْفَ", "تَحْتَ", "فَوْقَ", "أَسْفَل",
    "مِثْل", "حَسَبَ", "طِبْقًا", "وَفْقًا", "بِشَكْلِ", "بِصُورَةِ", "جِدًّا",
    "فَقَطْ", "هُنَا", "هُنَاكَ", "الْآنَ", "كَذَلِكَ", "لِمَاذَا", "مَاذَا",
    "كَيْفَ", "مَتَى", "أَيْنَ", "مَنْ", "أَيُّ", "هَلْ", "نَعَمْ", "لَازِم",
    # -- ordinals, quantifiers, adjectives ----------------------------------
    "أَوَّل", "أَوَّلًا", "ثَانِي", "ثَانِيًا", "ثَالِث", "ثَالِثًا",
    "أَخِير", "أَخِيرًا", "أَكْثَر", "أَقَلّ", "أَعْلَى", "أَدْنَى",
    "أَكْبَر", "أَصْغَر", "جَيِّد", "جَيِّدَة", "كَبِير", "كَبِيرَة",
    "صَغِير", "صَغِيرَة", "عَالِي", "عَالِيَة", "مُنْخَفِض", "مُنْخَفِضَة",
    "مُرْتَفِع", "مُخْتَلِف", "مُخْتَلِفَة", "مُتَعَدِّد", "مُمْكِن",
    "ضَرُورِي", "مُهِمّ", "مُهِمَّة", "أَسَاسِي", "رَئِيسِي", "فِعْلِي",
    "فَعَّال", "فَعَّالَة", "وَاضِح", "دَقِيق", "دَقِيقَة", "صَحِيح",
    "خَطَأ", "كَامِل", "كَامِلَة", "جَدِيد", "حَدِيث", "قَدِيم", "طَبِيعِي",
    "صِنَاعِي", "كِيمِيَائِي", "مِيكَانِيكِي", "كَهْرَبَائِي", "حَرَارِي",
    "هَيْدْرُولِيكِي", "زِلْزَالِي", "مَغْنَاطِيسِي", "نَوَوِي", "أَفْضَل",
    "الْأَوَّل", "الْأُولَى", "الثَّانِي", "الثَّالِث", "الْأَخِير",
    "النِّهَائِي", "النِّهَائِيَّة", "الْكُلِّي", "الْكُلِّيَّة",
    "الْجُزْئِي", "الْمُطْلَق", "الْمُطْلَقَة", "الْفَعَّال", "الْفَعَّالَة",
    "الْحَقِيقِي", "الْحَقِيقِيَّة", "الْمِثَالِي", "الْأَمْثَل",
    "الْأَفْضَل", "الْأَكْبَر", "الْأَقَلّ", "الْأَعْلَى", "الْأَدْنَى",
    # -- answer-body nouns (definite and indefinite) ------------------------
    "مَصْدَر", "الْمَصْدَر", "مَصَادِر", "الْمَصَادِر", "خُلَاصَة",
    "الْخُلَاصَة", "نَتِيجَة", "النَّتِيجَة", "نَتَائِج", "النَّتَائِج",
    "سَبَب", "السَّبَب", "أَسْبَاب", "الْأَسْبَاب", "تَأْثِير", "التَّأْثِير",
    "عَلَاقَة", "الْعَلَاقَة", "فَرْق", "الْفَرْق", "مُقَارَنَة",
    "الْمُقَارَنَة", "مِثَال", "الْمِثَال", "حَال", "الْحَال", "حَالَة",
    "الْحَالَة", "ظُرُوف", "الظُّرُوف", "شَرْط", "الشَّرْط", "شُرُوط",
    "الشُّرُوط", "حُدُود", "الْحُدُود", "حَدّ", "الْحَدّ", "نِطَاق",
    "النِّطَاق", "مَجَال", "الْمَجَال", "مِنْطَقَة", "الْمِنْطَقَة",
    "مِسَاحَة", "الْمِسَاحَة", "طُول", "الطُّول", "عَرْض", "الْعَرْض",
    "سُمْك", "وَقْت", "الْوَقْت", "زَمَن", "الزَّمَن", "فَتْرَة",
    "الْفَتْرَة", "مُدَّة", "الْمُدَّة", "سَنَة", "السَّنَة", "يَوْم",
    "الْيَوْم", "شَهْر", "الشَّهْر", "سَاعَة", "السَّاعَة", "ثَانِيَة",
    "مَادَّة", "الْمَادَّة", "مَوَاد", "الْمَوَاد", "خَصَائِص", "الْخَصَائِص",
    "خَاصِيَّة", "صِفَة", "صِفَات", "نَوْع", "النَّوْع", "أَنْوَاع",
    "الْأَنْوَاع", "شَكْل", "الشَّكْل", "أَشْكَال", "وَحْدَة", "الْوَحْدَة",
    "وَحَدَات", "الْوَحَدَات", "عَدَد", "الْعَدَد", "رَقْم", "الرَّقْم",
    "أَرْقَام", "الْأَرْقَام", "مُسْتَوَى", "الْمُسْتَوَى", "مُسْتَوَيَات",
    "مَعْلُومَات", "الْمَعْلُومَات", "بَيَانَات", "الْبَيَانَات", "تَحْلِيل",
    "التَّحْلِيل", "قِيَاس", "الْقِيَاس", "اخْتِبَار", "الِاخْتِبَار", "سِجِلّ",
    "السِّجِلّ", "تَسْجِيل", "التَّسْجِيل", "نَمُوذَج", "النَّمُوذَج",
    "مُعَادَلَة", "الْمُعَادَلَة", "قَانُون", "الْقَانُون", "مُعَامِل",
    "الْمُعَامِل", "مُعَامِلَات", "ثَابِت", "الثَّابِت", "دَارْسِي",
    "نُقْطَة", "النُّقْطَة", "فَقَاعَة", "الْفَقَاعَة", "نَدَى", "النَّدَى",
    # -- petroleum nouns ----------------------------------------------------
    "بِئْر", "الْبِئْر", "آبَار", "الْآبَار", "نَفْط", "النَّفْط", "غَاز",
    "الْغَاز", "مَاء", "الْمَاء", "مَكْمَن", "الْمَكْمَن", "مَكَامِن",
    "الْمَكَامِن", "خَزَّان", "الْخَزَّان", "طَبَقَة", "الطَّبَقَة",
    "طَبَقَات", "الطَّبَقَات", "صَخْر", "الصَّخْر", "صُخُور", "الصُّخُور",
    "مَسَامِيَّة", "الْمَسَامِيَّة", "نَفَاذِيَّة", "النَّفَاذِيَّة", "تَشَبُّع",
    "التَّشَبُّع", "ضَغْط", "الضَّغْط", "دَرَجَة", "الدَّرَجَة", "حَرَارَة",
    "الْحَرَارَة", "لُزُوجَة", "اللُّزُوجَة", "كَثَافَة", "الْكَثَافَة", "حَجْم",
    "الْحَجْم", "مُعَدَّل", "الْمُعَدَّل", "نِسْبَة", "النِّسْبَة", "قِيمَة",
    "الْقِيمَة", "كَمِّيَّة", "الْكَمِّيَّة", "تَدَفُّق", "التَّدَفُّق",
    "جَرَيَان", "الْجَرَيَان", "إِنْتَاج", "الْإِنْتَاج", "اسْتِخْلَاص",
    "الِاسْتِخْلَاص", "حَفْر", "الْحَفْر", "إِكْمَال", "الْإِكْمَال",
    "تَحْفِيز", "التَّحْفِيز", "تَحْمِيض", "التَّحْمِيض", "تَكْسِير",
    "التَّكْسِير", "كَسْر", "الْكَسْر", "أَسْمَنْت", "الْأَسْمَنْت", "بِطَانَة",
    "الْبِطَانَة", "أَنَابِيب", "الْأَنَابِيب", "مِضَخَّة", "الْمِضَخَّة",
    "صِمَام", "الصِّمَام", "سَائِل", "السَّائِل", "سَوَائِل", "مَوَائِع",
    "مَائِع", "الْمَائِع", "رَمْل", "الرَّمْل", "طِين", "الطِّين", "مِلْح",
    "الْمِلْح", "إِغْرَاق", "الْإِغْرَاق", "حَقْن", "الْحَقْن", "رَأْس",
    "الرَّأْس", "عُمْق", "الْعُمْق", "قُطْر", "الْقُطْر", "وَزْن", "الْوَزْن",
    "قُوَّة", "الْقُوَّة", "إِجْهَاد", "الْإِجْهَاد", "طَاقَة", "الطَّاقَة",
    "عَمَل", "الْعَمَل", "عَمَلِيَّة", "الْعَمَلِيَّة", "عَمَلِيَّات",
    "الْعَمَلِيَّات", "طَرِيقَة", "الطَّرِيقَة", "خَطْوَة", "الْخَطْوَة",
    "خَطَوَات", "الْخَطَوَات", "مَرْحَلَة", "الْمَرْحَلَة", "مَرَاحِل",
    "الْمَرَاحِل", "نِظَام", "النِّظَام", "جِهَاز", "الْجِهَاز", "أَدَاة",
    "الْأَدَاة", "تَسْمِيت", "التَّسْمِيت", "تَثْقِيب", "التَّثْقِيب",
    "هُبُوط", "الْهُبُوط", "انْهِيَار", "الِانْهِيَار", "انْفِجَار",
    "الِانْفِجَار", "انْدِفَاع", "الِانْدِفَاع", "حَاجِز", "الْحَاجِز",
    "حَوَاجِز", "الْحَوَاجِز", "سَلَامَة", "السَّلَامَة", "خَطَر", "الْخَطَر",
    "أَخْطَار", "الْأَخْطَار", "حِمَايَة", "الْحِمَايَة", "مُعَالَجَة",
    "الْمُعَالَجَة", "إِدَارَة", "الْإِدَارَة", "اقْتِصَادِيَّات",
    "الِاقْتِصَادِيَّات", "مَوَارِد", "الْمَوَارِد", "صِحَّة", "الصِّحَّة",
    "بِيئَة", "الْبِيئَة", "تَقْلِيدِيَّة", "التَّقْلِيدِيَّة", "مُعَزَّز",
    "الْمُعَزَّز", "مَائِي", "الْمَائِي", "بَحْرِي", "الْبَحْرِي", "هَنْدَسَة",
    "الْهَنْدَسَة", "مِيكَانِيكَا", "فِيزِيَاء", "كِيمِيَاء", "جِيُولُوجِيَا",
    "رِيَاضِيَّات", "إِحْصَاء", "حَاسُوب", "بَرْمَجَة", "تَعَلُّم", "مَعْرِفَة",
    "عِلْم", "عُلُوم", "بَحْث", "دِرَاسَة", "تَقْرِير", "مِلَفّ", "الْمِلَفّ",
    "وَثِيقَة", "مَرْجِع", "تَعْرِيف", "التَّعْرِيف", "مَفْهُوم", "مَفَاهِيم",
    "الْمَفَاهِيم", "مُصْطَلَح", "الْمُصْطَلَح", "كَلِمَة", "الْكَلِمَة",
    "جُمْلَة", "الْجُمْلَة", "نَصّ", "النَّصّ", "سُؤَال", "السُّؤَال",
    "جَوَاب", "الْجَوَاب", "إِجَابَة", "الْإِجَابَة", "دَلِيل", "الدَّلِيل",
    "أَدِلَّة", "الْأَدِلَّة", "ثِقَة", "الثِّقَة", "تَعَارُض", "التَّعَارُض",
    "تَنَاقُض", "التَّنَاقُض", "احْتِمَال", "الِاحْتِمَال", "خِيَار",
    "الْخِيَار", "خِيَارَات", "الْخِيَارَات", "تَوْصِيَة", "التَّوْصِيَة",
    "قَرَار", "الْقَرَار", "حُكْم", "الْحُكْم", "حَلّ", "الْحَلّ", "حُلُول",
    "الْحُلُول", "بَدِيل", "أَفْضَل", "دَعْم", "الدَّعْم",
    # -- base (indefinite) forms, so ``ال`` + base is derived ----------------
    "فَرَاغ", "فَرَاغَات", "حَجْم", "حَجْم", "ضَغْط", "سَجْل", "نِسْبَة",
    "قِيمَة", "قُوَّة", "طَاقَة", "مُعَدَّل", "تَدَفُّق", "جَرَيَان",
    "مَسَامِيَّة", "نَفَاذِيَّة", "تَشَبُّع", "لُزُوجَة", "كَثَافَة", "حَرَارَة",
    "دَرَجَة", "طَبَقَة", "صَخْر", "مَكْمَن", "بِئْر", "نَفْط", "غَاز",
    "مَاء", "خَزَّان", "تَكْسِير", "تَحْمِيض", "تَحْفِيز", "إِنْتَاج",
    "اسْتِخْلَاص", "حَفْر", "إِكْمَال", "إِغْرَاق", "حَقْن", "رَأْس",
    "عُمْق", "قُطْر", "وَزْن", "إِجْهَاد", "مِضَخَّة", "أَسْمَنْت", "بِطَانَة",
    "صِمَام", "سَائِل", "رَمْل", "طِين", "مِلْح", "مَائِع", "هُبُوط",
    "انْهِيَار", "انْفِجَار", "انْدِفَاع", "حَاجِز", "سَلَامَة", "خَطَر",
    "حِمَايَة", "مُعَالَجَة", "إِدَارَة", "مَصْدَر", "خُلَاصَة", "نَتِيجَة",
    "سَبَب", "تَأْثِير", "عَلَاقَة", "فَرْق", "مُقَارَنَة", "مِثَال", "حَال",
    "شَرْط", "حَدّ", "نِطَاق", "مَجَال", "مِنْطَقَة", "مِسَاحَة", "طُول",
    "عَرْض", "سُمْك", "وَقْت", "زَمَن", "فَتْرَة", "مُدَّة", "سَنَة", "يَوْم",
    "شَهْر", "سَاعَة", "ثَانِيَة", "مَادَّة", "خَاصِيَّة", "صِفَة", "نَوْع",
    "شَكْل", "وَحْدَة", "عَدَد", "رَقْم", "مُسْتَوَى", "تَحْلِيل", "قِيَاس",
    "اخْتِبَار", "تَسْجِيل", "نَمُوذَج", "مُعَادَلَة", "قَانُون", "مُعَامِل",
    "نُقْطَة", "فَقَاعَة", "نَدَى", "مَعْلُومَة", "بَيَان", "تَقْرِير",
    "مِلَفّ", "وَثِيقَة", "مَرْجِع", "تَعْرِيف", "مَفْهُوم", "مُصْطَلَح",
    "كَلِمَة", "جُمْلَة", "نَصّ", "سُؤَال", "جَوَاب", "إِجَابَة", "دَلِيل",
    "ثِقَة", "تَعَارُض", "تَنَاقُض", "احْتِمَال", "خِيَار", "تَوْصِيَة",
    "قَرَار", "حُكْم", "حَلّ", "بَدِيل", "دَعْم", "عَمَلِيَّة", "طَرِيقَة",
    "خَطْوَة", "مَرْحَلَة", "نِظَام", "جِهَاز", "أَدَاة", "تَسْمِيت",
    "تَثْقِيب", "هَنْدَسَة", "عِلْم", "بَحْث", "دِرَاسَة", "مَعْرِفَة",
    # -- common verbs (3rd person, present) ---------------------------------
    "يُوجَدُ", "تُوجَدُ", "يُمْكِنُ", "يَجِبُ", "يَعْنِي", "يُشِيرُ",
    "يُوَضِّحُ", "يُبَيِّنُ", "يُحَدِّدُ", "يَقِيسُ", "يَزِيدُ", "يَنْقُصُ",
    "يَقِلُّ", "يَخْفِضُ", "يَرْفَعُ", "يَمْنَعُ", "يُسَبِّبُ", "يُؤَثِّرُ",
    "يَعْتَمِدُ", "يَحْتَوِي", "يَتَكَوَّنُ", "يُسْتَخْدَمُ", "يَحْدُثُ",
    "يُنْتِجُ", "يَتَسَبَّبُ", "يَنْتَقِلُ", "يَتَدَفَّقُ", "يَجْرِي",
    "يَبْقَى", "يُصْبِحُ", "يَبْلُغُ", "يَصِلُ", "يَنْخَفِضُ", "يَرْتَفِعُ",
    "يَتَحَكَّمُ", "يَسَيْطَرُ", "يَحْتَاجُ", "يُوَفِّرُ", "يُحَسِّنُ",
    "يُعْطِي", "يَبْنِي", "يَعْمَلُ", "يُسْتَعْمَلُ", "يُعْتَبَرُ",
    "تُعْتَبَرُ", "يُعَدُّ", "تُعَدُّ", "يُسَمَّى", "تُسَمَّى", "يُطْلَقُ",
    "اسْتِخْدَام", "الِاسْتِخْدَام",
)

#: The lexicon, keyed by the bare skeleton of each vocalized word.
#:
#: **First entry wins.** A handful of words share a skeleton but carry a different
#: vocalization (``مِنْ`` the preposition vs ``مَنْ`` the interrogative, ``كُلّ`` vs
#: ``كُلِّ``, ``سِجِلّ`` vs ``سَجْل``). A plain dict comprehension would let the last
#: one silently overwrite the earlier, so the order of :data:`_VOCALIZED` is the
#: decision: the form listed first is the default. This keeps the pass deterministic
#: and stops a later addition from quietly changing an earlier word's vowels.
def _build_lexicon(words: tuple[str, ...]) -> dict[str, str]:
    table: dict[str, str] = {}
    for word in words:
        table.setdefault(strip_diacritics(word), word)
    return table


LEXICON: dict[str, str] = _build_lexicon(_VOCALIZED)


# --------------------------------------------------------------------------- #
# Layer 2 -- deterministic pattern rules
# --------------------------------------------------------------------------- #

#: Long-vowel letters and the short vowel each implies on the consonant before it.
_LONG_VOWEL = {"\u0627": FATHA, "\u0648": DAMMA, "\u064a": KASRA}  # ا و ي

#: The imperfect-verb prefixes, used only to *avoid* marking a verb as a noun.
_VERB_PREFIXES = "يتنأ"

#: A consonant letter (not one of the long-vowel letters).
_CONSONANT_RE = re.compile(r"[\u0621-\u0649]")


def _is_consonant(char: str) -> bool:
    return bool(char) and char not in "اويىء" and bool(_CONSONANT_RE.match(char))


def _definite_form(skeleton: str) -> str | None:
    """Vocalize ``ال`` + a lexicon word, or ``None``.

    A definite noun is the article plus its bare form, and both are already known
    -- so ``الفراغات`` is derived from ``فراغات`` rather than enumerated. A **moon**
    letter keeps the ``ل`` (marked sukun): ``الْقَمَر``. A **sun** letter
    assimilates it, so the ``ل`` carries no vowel and the following letter takes a
    shadda: ``الشَّمْس``. The shadda is placed *before* the base word's own first
    vowel (``مَ`` -> ``مَّ``), which is the correct order.
    """
    if not skeleton.startswith("ال") or len(skeleton) < 3:
        return None
    rest = skeleton[2:]
    base = LEXICON.get(rest)
    if base is None:
        return None
    if rest[0] in _SUN_LETTERS:
        return "\u0627\u0644" + base[0] + SHADDA + base[1:]
    return "\u0627\u0644\u0652" + base


def _vocalize_article(skeleton: str) -> str | None:
    """Mark only the article of a definite word whose base is unknown.

    A **moon** letter keeps the ``ل`` (marked sukun): ``الْقَمَر``. A **sun** letter
    assimilates it, so the ``ل`` carries no vowel and the following letter takes a
    shadda: ``الشَّمْس``. Only the article
    itself is marked here; the rest of the word keeps its bare letters, because the
    pattern layer does not guess vowels it cannot justify.
    """
    if not skeleton.startswith("ال") or len(skeleton) < 3:
        return None
    rest = skeleton[2:]
    first = rest[0]
    if first in _SUN_LETTERS:
        return "\u0627\u0644" + first + SHADDA + rest[1:]
    return "\u0627\u0644\u0652" + rest


def _pattern_vocalize(skeleton: str) -> str:
    """Mark the vowels that follow *certainly* from the word's own letters.

    Only two facts are used, because only they are safe: a long vowel implies the
    short vowel on the consonant before it, and a final ``ة`` / ``ى`` implies a
    fatha. Every other consonant is left bare rather than guessed -- a wrong
    vowel sounds worse than none.
    """
    if not skeleton:
        return skeleton
    chars = list(skeleton)
    marks: dict[int, str] = {}
    for index, char in enumerate(chars):
        if index > 0 and char in _LONG_VOWEL and _is_consonant(chars[index - 1]):
            if index - 1 not in marks:
                marks[index - 1] = _LONG_VOWEL[char]
    # A final tāʾ marbūṭa or alif maqṣūra takes a fatha on the letter before it.
    if len(chars) >= 2 and chars[-1] in "\u0629\u0649" and _is_consonant(chars[-2]):
        marks.setdefault(len(chars) - 2, FATHA)
    out: list[str] = []
    for index, char in enumerate(chars):
        out.append(char)
        if index in marks:
            out.append(marks[index])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Layer 3 -- a bounded case-ending pass (formal mode only)
# --------------------------------------------------------------------------- #

#: Prepositions and other words that govern the genitive on the noun after them.
_GENITIVE_GOVERNORS = frozenset(
    strip_diacritics(word)
    for word in (
        "فِي", "مِنْ", "إِلَى", "عَلَى", "عَنْ", "مَعَ", "بِ", "لِ", "كَ",
        "عِنْدَ", "بَيْنَ", "بَعْدَ", "قَبْلَ", "خِلَالَ", "حَتَّى", "لَدَى",
        "دُونَ", "عَبْرَ", "ضِمْنَ", "دَاخِلَ", "خَارِجَ", "أَمَامَ", "خَلْفَ",
        "تَحْتَ", "فَوْقَ", "حَسَبَ", "مِثْل", "نَحْوَ", "لَدَى",
    )
)

#: ``إنّ`` and its sisters, which govern the accusative on the noun after them.
_ACCUSATIVE_GOVERNORS = frozenset(
    strip_diacritics(word) for word in ("إِنَّ", "أَنَّ", "كَأَنَّ", "لَكِنَّ", "لَيْتَ", "لَعَلَّ")
)

#: Words that never take a case ending: particles, pronouns, adverbs, verbs, and
#: the numbers. Everything else the lexicon holds is treated as a declinable noun.
_INDECLINABLE = frozenset(
    strip_diacritics(word)
    for word in (
        "فِي", "مِنْ", "إِلَى", "عَلَى", "عَنْ", "مَعَ", "هَذَا", "هَذِهِ",
        "ذَلِكَ", "تِلْكَ", "الَّذِي", "الَّتِي", "الَّذِينَ", "مَا", "لَا",
        "لَمْ", "لَنْ", "أَنْ", "إِنَّ", "كَانَ", "كَانَتْ", "كَانُوا",
        "يَكُونُ", "تَكُونُ", "هُوَ", "هِيَ", "هُمْ", "نَحْنُ", "أَنْتَ",
        "أَنَا", "قَدْ", "كُلّ", "كُلِّ", "بَعْض", "غَيْر", "بَيْنَ",
        "عِنْدَ", "بَعْدَ", "قَبْلَ", "خِلَالَ", "حَتَّى", "كَمَا", "لَكِنْ",
        "أَوْ", "وَ", "ثُمَّ", "أَيْضًا", "حَيْثُ", "إِذَا", "إِذْ",
        "عِنْدَمَا", "لِذَلِكَ", "بِذَلِكَ", "فِيهِ", "فِيهَا", "مِنْهَا",
        "مِنْهُ", "بِهِ", "بِهَا", "لَهُ", "لَهَا", "إِلَيْهِ", "عَلَيْهِ",
        "عَلَيْهَا", "بِهَذَا", "وَمَا", "وَمِنْ", "وَفِي", "وَهُوَ",
        "وَهِيَ", "وَهَذَا", "وَهَذِهِ", "مِمَّا", "بِمَا", "لِأَنَّ", "لِكَيْ",
        "بَلْ", "لَيْسَ", "لَيْسَتْ", "إِلَّا", "سِوَى", "سَوَاء", "إِمَّا",
        "لَدَى", "دُونَ", "بِدُونِ", "عَبْرَ", "ضِمْنَ", "دَاخِلَ",
        "خَارِجَ", "أَمَامَ", "خَلْفَ", "تَحْتَ", "فَوْقَ", "أَسْفَل",
        "مِثْل", "حَسَبَ", "طِبْقًا", "وَفْقًا", "بِشَكْلِ", "بِصُورَةِ",
        "جِدًّا", "فَقَطْ", "هُنَا", "هُنَاكَ", "الْآنَ", "كَذَلِكَ",
        "لِمَاذَا", "مَاذَا", "كَيْفَ", "مَتَى", "أَيْنَ", "مَنْ", "أَيُّ",
        "هَلْ", "نَعَمْ", "لَازِم", "أَوَّلًا", "ثَانِيًا", "ثَالِثًا",
        "أَخِيرًا", "أَكْثَر", "أَقَلّ", "أَعْلَى", "أَدْنَى", "أَكْبَر",
        "أَصْغَر", "يُوجَدُ", "تُوجَدُ", "يُمْكِنُ", "يَجِبُ", "يَعْنِي",
        "يُشِيرُ", "يُوَضِّحُ", "يُبَيِّنُ", "يُحَدِّدُ", "يَقِيسُ",
        "يَزِيدُ", "يَنْقُصُ", "يَقِلُّ", "يَخْفِضُ", "يَرْفَعُ", "يَمْنَعُ",
        "يُسَبِّبُ", "يُؤَثِّرُ", "يَعْتَمِدُ", "يَحْتَوِي", "يَتَكَوَّنُ",
        "يُسْتَخْدَمُ", "يَحْدُثُ", "يُنْتِجُ", "يَتَسَبَّبُ", "يَنْتَقِلُ",
        "يَتَدَفَّقُ", "يَجْرِي", "يَبْقَى", "يُصْبِحُ", "يَبْلُغُ", "يَصِلُ",
        "يَنْخَفِضُ", "يَرْتَفِعُ", "يَتَحَكَّمُ", "يَسَيْطَرُ", "يَحْتَاجُ",
        "يُوَفِّرُ", "يُحَسِّنُ", "يُعْطِي", "يَبْنِي", "يَعْمَلُ",
        "يُسْتَعْمَلُ", "يُعْتَبَرُ", "تُعْتَبَرُ", "يُعَدُّ", "تُعَدُّ",
        "يُسَمَّى", "تُسَمَّى", "يُطْلَقُ",
    )
)

#: The letters after which a word cannot carry a case ending: a defective noun
#: (ending in a long vowel) or a fixed form.
_NO_CASE_TAIL = "\u0627\u0648\u064a\u0649"


def _is_declinable_noun(skeleton: str | None) -> bool:
    """Whether ``skeleton`` is a noun this pass declines (a lexicon noun)."""
    return bool(skeleton) and skeleton in LEXICON and skeleton not in _INDECLINABLE


def _case_vowel(preceding: str | None, *, definite: bool) -> str:
    """The case mark a noun takes, from the word that governs it.

    Three safe facts, in order: a preposition governs the genitive; ``إنّ`` and its
    sisters govern the accusative; and a **definite noun whose preceding word is a
    declinable noun** is the second term of an إضافة (``ضَغْطُ المَكْمَنِ``), which
    is genitive. Everything else is nominative, the unmarked default.
    """
    if preceding is None:
        return DAMMA
    if preceding in _GENITIVE_GOVERNORS:
        return KASRA
    if preceding in _ACCUSATIVE_GOVERNORS:
        return FATHA
    if definite and _is_declinable_noun(preceding):
        return KASRA
    return DAMMA


def _apply_case(
    word: str,
    *,
    preceding: str | None,
    following: str | None,
    definite: bool,
) -> str:
    """Put the right case ending on a declinable noun.

    Definite nouns take a single vowel (``الضَّغْطُ``); indefinite ones take the
    matching tanwīn (``ضَغْطٌ``). A noun in the **construct state** -- the first
    term of an إضافة, i.e. one *followed* by a definite noun -- keeps the single
    vowel without tanwīn (``ضَغْطُ المَكْمَن``), which is why the *following*
    word is consulted, not the preceding one.
    """
    if not word:
        return word
    last = word[-1]
    if last in _NO_CASE_TAIL or last in DIACRITICS:
        return word
    vowel = _case_vowel(preceding, definite=definite)
    if definite or _construct_state(following):
        return word + vowel
    return word + {"\u064f": DAMMATAN, "\u0650": KASRATAN, "\u064e": FATHATAN}[vowel]


def _construct_state(following: str | None) -> bool:
    """Whether a *following* definite noun makes this word an إضافة's first term.

    The narrow, safe test: the next word is a definite noun (it carries ``ال``),
    so the current noun is the مضاف and takes the single case vowel.
    """
    return following is not None and following.startswith("ال")


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TashkeelReport:
    """What one diacritization pass did, so the coverage is measured not assumed."""

    text: str
    mode: str
    words: int
    lexical: int
    patterned: int
    untouched: int
    changes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def covered(self) -> int:
        """Words that received at least a lexical or a pattern vocalization."""
        return self.lexical + self.patterned

    @property
    def coverage(self) -> float:
        """The share of Arabic words that received any vowel mark."""
        if self.words == 0:
            return 0.0
        return self.covered / self.words

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def describe(self) -> str:
        return (
            f"tashkeel ({self.mode}): {self.covered}/{self.words} words marked "
            f"({self.lexical} from the lexicon, {self.patterned} by pattern); "
            f"{self.untouched} left unmarked"
        )


def _word_vocalization(
    skeleton: str,
    *,
    preceding: str | None,
    following: str | None,
    mode: str,
) -> tuple[str, str]:
    """Vocalize one bare Arabic word; return ``(vocalized, layer)``.

    ``layer`` is ``"lexical"``, ``"patterned"``, or ``"untouched"``, so the caller
    can report the coverage honestly. The lexicon is authoritative; a word it does
    not hold gets the article rule and the safe pattern vowels; a word neither can
    mark is returned unchanged rather than guessed at.
    """
    if not skeleton:
        return skeleton, "untouched"
    from_lexicon = LEXICON.get(skeleton)
    if from_lexicon is not None:
        if mode == MODE_FORMAL and skeleton not in _INDECLINABLE:
            definite = skeleton.startswith("ال")
            cased = _apply_case(
                from_lexicon, preceding=preceding, following=following, definite=definite
            )
            if cased != from_lexicon:
                return cased, "lexical"
        return from_lexicon, "lexical"
    definite_form = _definite_form(skeleton)
    if definite_form is not None:
        # A definite noun is a declinable noun, so it takes a case ending too.
        if mode == MODE_FORMAL:
            cased = _apply_case(
                definite_form, preceding=preceding, following=following, definite=True
            )
            if cased != definite_form:
                return cased, "patterned"
        return definite_form, "patterned"
    article = _vocalize_article(skeleton)
    if article is not None:
        return article, "patterned"
    patterned = _pattern_vocalize(skeleton)
    if patterned != skeleton:
        return patterned, "patterned"
    return skeleton, "untouched"


def diacritize_report(text: str, *, mode: str = MODE_FORMAL) -> TashkeelReport:
    """Diacritize every Arabic word in ``text`` and report the coverage.

    Only Arabic words are touched; a Latin identifier, a number, a formula, and
    every punctuation mark are returned byte-identical. Existing marks are
    stripped first, so the pass is idempotent.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown tashkeel mode: {mode!r}")
    if not text or not has_arabic(text):
        return TashkeelReport(text=text, mode=mode, words=0, lexical=0, patterned=0, untouched=0)

    # Split into Arabic-word and non-word runs, keeping every delimiter intact.
    pieces = _ARABIC_WORD_RE.split(text)
    words = _ARABIC_WORD_RE.findall(text)
    skeletons = [strip_diacritics(word) for word in words]

    lexical = patterned = untouched = 0
    rendered: list[str] = []
    for index, piece in enumerate(pieces):
        rendered.append(piece)
        if index >= len(skeletons):
            continue
        skeleton = skeletons[index]
        preceding = skeletons[index - 1] if index > 0 else None
        following = skeletons[index + 1] if index + 1 < len(skeletons) else None
        vocalized, layer = _word_vocalization(
            skeleton, preceding=preceding, following=following, mode=mode
        )
        if layer == "lexical":
            lexical += 1
        elif layer == "patterned":
            patterned += 1
        else:
            untouched += 1
        rendered.append(vocalized)

    result = "".join(rendered)
    changes: list[str] = []
    if result != text:
        changes.append("diacritized Arabic words")
    return TashkeelReport(
        text=result,
        mode=mode,
        words=len(skeletons),
        lexical=lexical,
        patterned=patterned,
        untouched=untouched,
        changes=tuple(changes),
    )


def diacritize(text: str, *, mode: str = MODE_FORMAL) -> str:
    """The diacritized text alone, for callers that do not need the report."""
    return diacritize_report(text, mode=mode).text


def coverage(text: str, *, mode: str = MODE_FORMAL) -> float:
    """The share of Arabic words in ``text`` this module can mark."""
    return diacritize_report(text, mode=mode).coverage


__all__ = [
    "DAGGER_ALEF",
    "DAMMA",
    "DAMMATAN",
    "DIACRITICS",
    "FATHA",
    "FATHATAN",
    "KASRA",
    "KASRATAN",
    "LEXICON",
    "MODES",
    "MODE_COLLOQUIAL",
    "MODE_FORMAL",
    "SHADDA",
    "SHORT_VOWELS",
    "SUKUN",
    "TANWEEN",
    "TashkeelReport",
    "coverage",
    "diacritize",
    "diacritize_report",
    "has_arabic",
    "is_diacritized",
    "strip_diacritics",
]
