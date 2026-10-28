# تقييم طبقة الاستدلال في «فيض الروح»

**الإصدار:** 2.3.0 · **تاريخ التقييم:** 2026-10-23
**النطاق:** طبقة الاستدلال (reasoning layer) وصلتها بمسار الإجابة، مع قائمة
النواقص القابلة للقياس وما يجب إضافته.

---

## 1. الخلاصة التنفيذية

المشروع يملك **طبقة استدلال غنية فعلًا** (69 وحدة استدلالية في `core/`،
وكلها مُختبرة اختبارًا فرديًا)، لكن هذه الطبقة كانت **معزولة عن مسار الإجابة**:
المساعد (`GroundedAssistant`) و`Jarvis` والواجهة الرسومية **لا يبنون
`CognitiveController` إطلاقًا**، وهو الحامل الوحيد لكل محرّكات الاستدلال
(السببية، التخطيط، القرار، مراجعة الاعتقاد، الاستدلال الزمني، الاستدلال
الاستنباطي/الاستقرائي/الاستبعادي…).

بمعنى آخر: **الطبقة موجودة ومختبَرة، لكنها لا تُستخدَم عند الإجابة.** الجواب
المعروض للمستخدم كان يأتي من الاسترجاع (RAG) والنموذج اللغوي، لا من الاستدلال.

أنجزنا في المراحل 28 و36 و37 و38 و39 و40 و41 و42 و43 و44 و45 الجسور التالية:
- **المرحلة 28:** العلاقات التي *يستنتجها* الرسم (entailment) تصل إلى الإجابة.
- **المرحلة 36:** سلاسل العلاقات متعددة الخطوات بين مفاهيم السؤال تصل إلى الإجابة.
- **المرحلة 37:** **السلاسل السببية الموجّهة** تصل إلى الإجابة، مع تصحيح
  عيب قائم في عرضها.
- **المرحلة 38:** **عدم اليقين** (التعارض والضعف) يُحسب داخل ثقة الإجابة.
- **المرحلة 39:** **بوابة الرسوخ** تُقوّى (اتجاه السبب + عدم التناقض + دعم كل جملة).
- **المرحلة 40:** **أثر استدلالي مرئي للمستخدم** («كيف وصلت إلى هذه الإجابة»)
  يصل إلى الواجهة والـ CLI ووحدة `why`.
- **المرحلة 41:** **التخطيط** يصل الإجابة (خطوات المتن عند سؤال «كيف؟»).
- **المرحلة 42:** **القرار** يصل الإجابة (توصية الرسم عند سؤال «ماذا أفعل؟»).
- **المرحلة 43:** **الاعتقاد** يصل الإجابة (الحكم المرجَّح بالدليل عند سؤال «هل يصح؟»).
- **المرحلة 44:** **الاستدلال يصل متن الإجابة نفسه** (لا الأثر وحده) عبر
  `core/answer_lead.py`.
- **المرحلة 45:** **الزمن** يصل الإجابة (جدول زمني مرتّب لما تغيّر في المحادثة عند
  سؤال «ما الذي تغيّر؟») عبر `core/answer_temporal.py`.
- **المرحلة 46:** **التناظر** يصل الإجابة (مقارنة بنيوية بين مفهومين عند سؤال
  «هل X يشبه Y؟») عبر `core/answer_analogy.py`.

بهذا **سُدَّ النقص 7 كاملًا**: كل محرّك استدلالي كان معزولًا (التخطيط، القرار،
الاعتقاد، الزمن، التناظر) صار يصل مسار الإجابة، ويظهر في متن الإجابة نفسه لا في
الأثر وحده.

يبقى **النقص البنيوي الأساسي**: تراكم المعرفة بين الجلسات، ومقاييس التقييم التي ما
زالت تقيس المعزول أكثر من الموصول **ما زالت معزولة**.

---

## 2. منهجية القياس

- عدد الملفات: 745 ملف Python؛ وحدات `core/` الأساسية: 253؛ ملفات الاختبار: 288.
- الوحدات الاستدلالية (بالاسم): **69**.
- الوحدات الاستدلالية **التي يصل إليها مسار الإجابة** قبل هذه السلسلة:
  `ontology`, `ontology_reasoner`, `ontology_retrieval`, `relation_consistency`,
  `relation_inference` — أي **5 من 69**.
- الرسم البياني المزروع: **495 مفهومًا، 474 علاقة**، لكن **0 روابط سببية**
  في `CausalGraph` و**0 معتقدات** — أي أن الحاويات تُبنى فارغة في مسار الإجابة.

---

## 3. النواقص المؤكَّدة (بأدلة من الكود)

### نقص 1 — العزل البنيوي (الأخطر)
`CognitiveController` لا يُبنى إلا في: `core/relation_paths.py`, `core/analogy.py`,
`core/relation_inference.py`, `core/relation_extraction.py`, `cli/analogy.py`,
`cli/relations.py`, `evaluation/runner.py`.
**لا وجود له** في `core/assistant.py` ولا `core/jarvis.py` ولا `gui/`.
⇒ كل محرّكات الاستدلال المرتبطة به غير قابلة للوصول من سؤال.

### نقص 2 — الطبقة السببية معطّلة في مسار الإجابة
`CausalGraph` فيه 0 روابط افتراضيًا، و`CausalReasoner`/`CausalExplainer`
لا يُستخدمان إلا في تقييم السيناريوهات. لكن **الأنطولوجيا تحمل 117 حافة سببية
موجّهة** (`causes`, `drives`, `controls`, `reduces`, `increases`, `prevents`…)
غير مستثمرة في أي إجابة. *(عولج في المرحلة 37.)*

### نقص 3 — عرض السلسلة السببية كان معطوبًا (عيب قائم)
`CausalExplainer.explain_chain` كان يمرّر `chain.nodes` (الذي يشمل الطرفين)
إلى `causal_chain_sentence` الذي يضيف الطرفين مرة أخرى، فيظهر المسار مكرَّرًا:
`a → a → b → c → c`. لم يُلاحظ قبل الآن لأن الطبقة لم تصل إلى إجابة قط.
*(صُحّح في المرحلة 37 مع اختبار انحدار.)*

### نقص 4 — عدم اليقين لا يُحسب في ثقة الإجابة
`_confidence()` في `core/assistant.py` = `0.7 * top_score + 0.3 * coverage`.
لا تدخل فيها: التعارضات (`relation_conflicts`)، ولا الاستنتاجات (`inferred`)،
ولا السلاسل (`connections`/`causal_paths`)، ولا معايرة الاحتمالات.
⇒ إجابة مبنية على مسار ضعيف أو متناقض قد تُعرض بثقة عالية.
*(عولج في المرحلة 38.)*

**الحل (المرحلة 38):** `core/answer_confidence.py` هو الآن الموضع الوحيد الذي
تُحسب فيه الثقة. الأساس هو حدّ الأدلة (`0.7 * top + 0.3 * coverage`)، ثم
يُضيف **أقوى إشارة استدلالية** (الاستنتاجات/السلاسل/المسارات السببية) نسبةً
محدودة (`GRAPH_SUPPORT_WEIGHT = 0.25`) من المسافة المتبقية إلى اليقين — ولا
تُصنع ثقة من الرسم البياني إن لم توجد أدلة مُسترجَعة (`evidence == 0` ⇒
`final == 0`). ثم يُطبَّق **خصم التعارض** مضروبًا، فيخفض الثقة ولا يمكن أن
يُغالَبه الدعم المشتق (`CONFLICT_PENALTY_WEIGHT = 0.5`). ويُعاد التفصيل
الكامل (`evidence` / `graph_support` / `conflict_penalty` / `reasons`) عبر
`AssistantAnswer.confidence_report` و`JarvisTurn.confidence_note`، فيظهر
للقارئ أي حدّ حرّك الرقم وبكم. بوّابة مقيسة `answer_confidence_accuracy`
(8 أحكام مع ضوابط سلبية) = `1.0`، ومقياس جديد في `evaluation/benchmark.py`.

### نقص 5 — بوابة الرسوخ (groundedness gate) ضعيفة
`_is_grounded()` تكتفي بمشاركة **مصطلح واحد** بين الإجابة والأدلة. لا تتحقق من
اتجاه السبب، ولا من عدم التناقض، ولا من أن كل جملة مدعومة بمصدر.
*(عولج في المرحلة 39.)*

**الحل (المرحلة 39):** `core/grounding.py` هو الآن الموضع الوحيد للبوابة،
ويجري **ثلاثة فحوص حتمية** ويفشل مُغلَقًا (fallback إلى الإجابة الاستخلاصية):

1. **تغطية المصطلحات** — يجب أن تظهر نسبة كافية من مصطلحات الإجابة في الأدلة
   (`DEFAULT_MIN_COVERAGE = 0.3`)، لا مصطلح واحد فقط.
2. **دعم كل جملة** — كل جملة ذات مضمون يجب أن تشترك في مصطلح مع الأدلة؛ وأغلبية
   الجمل غير المدعومة تُسقط البوابة (`DEFAULT_MIN_SENTENCE_SUPPORT = 0.5`)، فلا
   تعبر فقرة كاملة تتعلّق بكلمة عرضية.
3. **اتجاه السبب** — إن ذكرت الإجابة رابطًا سببيًا بين مفهومين **بعكس** ما يخزّنه
   الرسم البياني (يوجد العكس ولا يوجد الاتجاه)، تُرفض الإجابة. هذه قاعدة
   الأمانانية التي تطبّقها طبقة السببية على مساراتها، مطبَّقةً الآن على كلمات
   النموذج.

`_is_grounded` بقي كغلافٍ رقيق للتوافق. البوابة المقيسة `grounding_accuracy`
(8 أحكام محجوزة، منها ضوابط سالبة: فقرة بكلمة عرضية، أغلبية جمل غير مدعومة،
ميكانيزم معكوس) = `1.0`، ومقياس جديد في `evaluation/benchmark.py` (overall
0.9797 → 0.9804).

### نقص 6 — لا أثر استدلالي مرئي للمستخدم
`ReasoningTrace`/`ExplanationEngine`/`ReasoningMonitor` موجودة، لكن
`ReasoningTrace` **لا يصل إلى أي سطح للمستخدم** (لا CLI ولا GUI) — الاستدلال
يحدث بلا تفسير.

**المعالجة (المرحلة 40):** وحدة `core/reasoning_report.py` تجمع **من البيانات
الموجودة أصلًا على `AssistantAnswer`** (الفقرات المسترجَعة + علاقات الرسم +
السلاسل المتعددة الخطوات + المسارات السببية + تقرير الثقة) في تقرير واحد
مقروء بعنوان «How this answer was reached»، بخطوات مرتّبة ثابتة ومعلَّمة:
المخزون (`stored`) مقابل المشتق (`derived`). يصل إلى:
`AssistantAnswer.reasoning` ← `JarvisTurn.reasoning_trace` ←
`ai/chat.format_turn` و`turn_payload` (فتظهر في الواجهة الرسومية مجانًا)،
والأمر `python -m cli relations why "<question>" [--json]`، ووحدة تحكم
`why`/«لماذا»/«اشرح» في Jarvis التي تعيد عرض الأثر عند الطلب. لا يخترع التقرير
شيئًا: كل سطر مأخوذ من بيانات موجودة، وخطوة فارغة تُعلَن `(nothing)`.

### نقص 7 — الاعتقاد والتخطيط والقرار والزمن معزولة
`planner`, `decision_engine`, `belief_revision`, `temporal_reasoner`,
`counterfactual_reasoner`, `uncertainty_propagator`, `analogy` —
لا تُستدعى من أي مسار إجابة (عدد المستخدمين خارج وحدة التحكم = 0 أو 1).
*(عولج التخطيط في المرحلة 41 والقرار في المرحلة 42 والاعتقاد في المرحلة 43؛ والزمن في المرحلة 45.)*

### نقص 8 — لا تراكم معرفي بين الجلسات في مسار الإجابة
لا يوجد حفظ لحالة الرسم/الاعتقادات عبر الجلسات في مسار الإجابة (الحفظ موجود
نظريًا في `KnowledgeStore`/`Snapshot` لكنه غير موصول بالإجابة).
*(عولج في المرحلة 47: العلاقات **المتعلَّمة** تُلتقط في نهاية الجلسة وتُعاد إلى
رسم الجلسة التالية قبل أول سؤال — انظر القسم 15.)*

### نقص 9 — التقييم يقيس المعزول لا الموصول
معظم مقاييس التقييم تُبنى على سيناريوهات تُنشئ وحدة تحكم خاصة، فلا تكشف أن
الطبقة لا تصل إلى الإجابة. (عالجنا ذلك جزئيًا بمقياسَي
`connection_reasoning_accuracy` و`causal_reasoning_accuracy`.)

---

## 4. ما يجب إضافته (مرتَّبًا بالأولوية)

| # | الإضافة | المبرر | الحالة |
|---|---------|--------|--------|
| 1 | وصل السلاسل السببية بالإجابة (`causal_paths`) | 117 حافة سببية غير مستثمرة | ✅ المرحلة 37 |
| 2 | تصحيح عرض السلسلة السببية | عيب `a → a → … → c → c` | ✅ المرحلة 37 |
| 3 | مقياس `causal_reasoning_accuracy` بعناصر ضبط سالبة | يمنع قراءة الاقتران كسبب | ✅ المرحلة 37 |
| 4 | حساب عدم اليقين داخل ثقة الإجابة (خصم عند التعارض/الضعف) | النقص 4 | ✅ المرحلة 38 |
| 5 | تقوية بوابة الرسوخ (اتجاه السبب + عدم التناقض + دعم كل جملة) | النقص 5 | ✅ المرحلة 39 |
| 6 | إظهار أثر استدلالي للمستخدم (لماذا هذه الإجابة) | النقص 6 | ✅ المرحلة 40 |
| 7 | وصل التخطيط بالإجابة عند سؤال «كيف؟» (خطوات المتن) | النقص 7 | ✅ المرحلة 41 |
| 8 | وصل القرار بالإجابة عند سؤال «ماذا أفعل؟» | النقص 7 | ✅ المرحلة 42 |
| 9 | وصل الاعتقاد بالإجابة عند سؤال «هل يصح؟» | النقص 7 | ✅ المرحلة 43 |
| 10 | وصل الاستدلال الزمني (سؤال «ما تغيّر؟») | النقص 7 | ✅ المرحلة 45 |
| 11 | إظهار الاستدلال في **متن الإجابة** لا في الأثر فقط | النقص 6/7 | ✅ المرحلة 44 |
| 12 | حفظ حالة الرسم/الاعتقادات عبر الجلسات | النقص 8 | ✅ المرحلة 47 |
| 13 | مقاييس تقييم تقيس **المسار الموصول** لا المعزول | النقص 9 | ✅ المرحلة 48 |

---

## 5. المعالجات المُنجَزة في المرحلة 37

**وحدة جديدة:** `core/causal_paths.py`
- `CAUSAL_RELATION_TYPES`: مجموعة مغلقة لأنواع العلاقات السببية الموجّهة،
  تستبعد الاقتران (`measures`, `contains`, `is_a`) — قاعدة الأمانانية الأهم.
- `causal_graph_from_ontology(ontology)`: تبني `CausalGraph` من الحواف السببية
  فقط، بثقة العلاقة وآليتها (نوع العلاقة).
- `CausalExplanation`: حاملة (cause, effect, hops, confidence, nodes,
  mechanisms, sentence) مع `describe()`.
- `causal_explanations(...)`: تبحث عن **أقوى سلسلة موجّهة** لكل زوج مفاهيم
  يسمّيه السؤال، **دون عكس أي حافة** (السببية غير متماثلة)، مرتّبة بالقوة ثم
  قلة الخطوات — حسّاسة وحتمية.
- `describe_causal_explanations(...)`: كتلة مُعنونة «موجّهة، سبب → أثر».
- `CAUSAL_JUDGEMENTS` + `causal_reasoning_accuracy(...)`: كتالوج محجوز فيه
  **عناصر ضبط سالبة** (زوج مقترن يجب ألّا يُنتج سببًا).

**الوصل:** `AssistantAnswer.causal_paths` و`JarvisTurn.causal_paths`،
وعرضهما في `ai/chat.format_turn` وحقل `causal_paths` في `turn_payload`
(فتظهر في الواجهة الرسومية مجانًا لأنها تعرض عبر `format_turn`).

**CLI:** `python -m cli relations causal "<question>" [--limit] [--max-depth] [--json]`.

**التقييم:** مقياس `causal_reasoning_accuracy` مضاف إلى
`evaluation/benchmark.py` و`runner.py` و`baseline.json`.

**مثال مُتحقَّق منه:**
```
$ python -m cli relations causal "why does waterflood increase production?"
Causal paths the graph states (directed, cause → effect):
  - waterflood → reservoir_pressure → production (confidence 0.49)
    -- Waterflood can influence production along the path
       waterflood → reservoir_pressure → production, with an overall
       confidence of 0.49. The apparent mechanism is maintains; drives.
```

---

## 6. المعالجة المُنجَزة في المرحلة 40 (الأثر الاستدلالي المرئي)

**وحدة جديدة:** `core/reasoning_report.py`
- `ReasoningStep`: خطوة واحدة (label, lines, derived) مع `describe()` يعدّ
  سطورها، ويعلن `(nothing)` إن كانت فارغة.
- `ReasoningReport`: سؤال + خطوات مرتّبة، مع `derived_steps` (الخطوات المشتقة
  التي **تحمل محتوى** فقط)، و`is_empty`، و`describe()` الذي يفتح بعنوان
  «How this answer was reached» ثم يسرد كل خطوة.
- `reasoning_report(question, *, passages, relation_facts, relation_conflicts,
  inferred, connections, causal_paths, confidence)`: تبني التقرير **من بيانات
  موجودة أصلًا** على `AssistantAnswer`؛ لا تستدعي أي محرك جديد ولا تخترع سطرًا.
- `STEP_ORDER` وثوابت المصادر (`SOURCE_EVIDENCE`، `SOURCE_STORED`،
  `SOURCE_INFERRED`، `SOURCE_CONNECTIONS`، `SOURCE_CAUSAL`، `SOURCE_CONFLICTS`،
  `SOURCE_CONFIDENCE`) — ترتيب ثابت معلن.

**الوصل:**
- `AssistantAnswer.reasoning: ReasoningReport` (تُبنى في `GroundedAssistant.ask`
  من البيانات التي جمعها أصلًا).
- `JarvisTurn.reasoning_trace: str` + وحدة تحكم `WHY` (`why`/«لماذا»/«اشرح»/
  «show your reasoning») في `core/jarvis_controls.py`، و`JarvisMode` يحفظ آخر
  أثر ويُعيد عرضه عند الطلب (`_last_reasoning`، يُصفَّر مع `new conversation`).
- `ai/chat.format_turn` و`turn_payload` يعرضان الأثر (فتظهر في الواجهة
  الرسومية مجانًا لأنها تعرض عبر `format_turn`).
- `HELP_TEXT`/`TEXT_HELP_TEXT` يذكران وحدة `why`.

**CLI:** `python -m cli relations why "<question>" [--limit] [--json]`.

**التقييم:** مقياس `reasoning_trace_accuracy` (7 أحكام محجوزة، منها ضوابط
سالبة: أثر بلا شيء يجب أن يقرأ فارغًا، وأثر بالدليل وحده يجب أن يُبقي كل خطوة
أخرى موجودة-فارغة لا محذوفة، والاشتقاق يُعلَن `derived` لا مخزونًا) مضاف إلى
`evaluation/benchmark.py` و`runner.py` و`baseline.json`
(overall 0.9804 → 0.9810).

**الاختبارات:** `tests/test_reasoning_report.py` — 36 اختبارًا: كائن الخطوة
والتقرير، الترتيب الثابت، الصدق عند الفراغ، خطوة الثقة، أثر المساعد، وحدة
`why` (إيجابًا وسلبًا: سؤال «why does…» لا يُختطَف)، التصفير مع محادثة جديدة،
والعرض في `format_turn`/`turn_payload`.

**مثال مُتحقَّق منه:**
```
$ python -m cli relations why "how does waterflood affect production?"
Here is what the available sources say about waterflood affect production.
...
How this answer was reached (how does waterflood affect production?):
Retrieved evidence:
  - [petroleum_corpus:corpus:waterflood_management] Waterflood management ...
Stored in the knowledge graph:
  - waterflood —maintains→ reservoir_pressure
Walked connections (multi-hop, not stored):
  - waterflood → production in 2 step(s) (confidence 0.49): ...
Directed causal paths the graph states:
  - waterflood → reservoir_pressure → production (confidence 0.49)
How the confidence was reached:
  - evidence 0.80
  - graph support 0.49
  - final 0.83
```

---

## 7. البوابات الأربع (الوضع الحالي)

| البوابة | النتيجة |
|---------|---------|
| `python -m pytest -q` | ✅ المجموعة الكاملة: 3997 نجحت، 200 متخطّاة، 0 فشل |
| `python -m ruff check .` | ✅ All checks passed |
| `python -m mypy` | ✅ Success: no issues found in 735 source files |
| `python -m evaluation` | ✅ PASS — `grounding_accuracy=1.0000`, `reasoning_trace_accuracy=1.0000`, `plan_accuracy=1.0000`, `decision_recommendation_accuracy=1.0000`, `belief_accuracy=1.0000`, `overall=0.9827` |

---

## 8. المعالجة المُنجَزة في المرحلة 41 (خطوات المتن تصل الإجابة)

**وحدة جديدة:** `core/answer_plan.py`
- `is_procedural(question)`: هل السؤال يطلب «كيف نفعل شيئًا» (عربي/إنجليزي)؟
  السؤال التعريفي («ما هي المسامية؟») لا يُقرأ كإجرائي، فلا يُقحم في إجابة تعريفية
  إجراءٌ لا يطلبه القارئ.
- `match_entry(question)`: يطابق السؤال مع مدخل المتن الأكثر صلة به، بمشاركة
  مصطلحات المحتوى (المفاهيم + أطراف العلاقات + نص الإجراء نفسه)، مع كسر التعادل
  بمفتاح المدخل (حتمي). يعيد `None` لسؤال لا يشترك في أي مصطلح — لا تخمين.
- `order_procedure(steps, goal_description=...)`: يرتّب خطوات الإجراء عبر
  **محرّك التخطيط الحقيقي** `Planner.order_steps`؛ دورة أو اعتماد على خطوة مجهولة
  **يُرفَض** بـ `ValueError` ولا يُعاد ترتيبه بصمت. لا يُنفَّذ شيء.
- `answer_plan(question)`: يبني `AnswerPlan` فقط إن كان السؤال إجرائيًا **و**
  وُجد مدخل مطابق له إجراء؛ وإلا `None`.
- `AnswerPlan` (topic/source/steps + `describe()` مرقّم).

**الوصل:** `AssistantAnswer.plan` ← خطوة `SOURCE_PLAN` في `core.reasoning_report`
(موضعها ثابت: بعد المخزون وقبل المشتق، لأنها خطوة مشتقة) ← `JarvisTurn.plan` ←
`ai/chat.format_turn` وحقل `plan` في `turn_payload`، فيصل إلى الواجهة الرسومية
مجانًا. وأمر `python -m cli relations why` يعرضها داخل الأثر وفي `--json`.

**التقييم:** مقياس `plan_accuracy` بعناصر ضبط سالبة (سؤال تعريفي/غير ذي صلة
يجب ألّا يُنتج خطة؛ دورة أو اعتماد مجهول يجب أن يُرفَض؛ خطوات معطاة بغير ترتيب
يجب أن تعود بترتيب التبعية). أُضيف إلى `evaluation/benchmark.py` و`runner.py`
و`baseline.json` (overall 0.9810 → 0.9816).

**مثال مُتحقَّق منه:**
```
$ python -c "from core.answer_plan import answer_plan; print(answer_plan('how do I measure porosity and permeability on a core sample?').describe())"
The procedure for Reservoir rock properties (reservoir_rock_properties):
  1. Measure porosity and permeability on a core sample.
  2. Compare core values with well-log estimates to calibrate the model.
```

---

## 9. المعالجة المُنجَزة في المرحلة 42 (توصية القرار تصل الإجابة)

**وحدة جديدة:** `core/answer_decision.py`
- `is_decisional(question)`: هل السؤال يطلب «ماذا نفعل» (عربي/إنجليزي)؟ السؤال
  التعريفي («ما هي النفاذية؟») لا يُقرأ كقراري، فلا تُقحم توصية لا يطلبها القارئ.
- `direction_of(question)`: الاتجاه الذي يطلبه السؤال (increase/reduce/prevent/
  maintain/control) أو `""`؛ يُمسَح بترتيب ثابت، فالسؤال نفسه يعطي الاتجاه نفسه.
- `_options(...)`: يقرأ **حواف الرسم الموجّهة** إلى المفهوم الذي يسأل عنه السؤال
  (`acid_stimulation —increases→ permeability`)، ويبني لكل خيار `Action` يضيف
  الهدف بتكلفة `1 - الثقة`. مصدر يخدم أكثر من هدف يندمج في خيار **واحد** يخدمها
  جميعًا — وهي حالة الأهداف المتعددة التي وُجد نموذج المنفعة لأجلها.
- `answer_decision(question, ontology=...)`: يشغّل **محرّك القرار الحقيقي**
  `DecisionEngine.decide(state, goal, actions)` ويرتّب الخيارات عبر
  `DecisionExplainer.explain`. يعيد `AnswerDecision` فقط إن كان السؤال قراريًا
  **و** للرسم خيار باتجاه ما سُئل؛ وإلا `None` — لا تخمين. لا يُنفَّذ شيء.
- **لا يخترع خيارًا**: الخيارات هي حواف الرسم المخزَّنة حصرًا، ولا يُعرَض خيار
  يسير بعكس الاتجاه المطلوب (سؤال «كيف أقلّل النفاذية؟» يعيد `None` لأن الرسم
  لا يحمل حافة تُنقص النفاذية، بدل أن يُعرَض خيار يزيدها).

**الوصل:** `AssistantAnswer.decision` ← خطوة `SOURCE_DECISION` في
`core.reasoning_report` (موضعها ثابت: بعد الخطة وقبل المشتق) ← `JarvisTurn.decision`
← `ai/chat.format_turn` وحقل `decision` في `turn_payload`، فيصل إلى الواجهة
الرسومية مجانًا. وأمر `python -m cli relations why` يعرضها في الأثر وفي `--json`.

**التقييم:** مقياس `decision_recommendation_accuracy` بعناصر ضبط سالبة (سؤال
تعريفي/غير ذي صلة/اتجاه لا يخدمه الرسم يجب ألّا يُنتج توصية). أُضيف إلى
`evaluation/benchmark.py` و`runner.py` و`baseline.json` (overall 0.9816 → 0.9821).

**مثال مُتحقَّق منه:**
```
$ python -m cli relations why "what should I do to increase permeability?"
The options the graph states (ranked by the decision engine):
  - acid_stimulation (increases, confidence 0.70)
  - hydraulic_fracturing (increases, confidence 0.70)
  - porosity (increases, confidence 0.60)
```

---

## 10. المعالجة المُنجَزة في المرحلة 43 (الاعتقاد يصل الإجابة)

**وحدة جديدة:** `core/answer_belief.py`
- `is_evaluative(question)`: هل السؤال يطلب **حكمًا** على قضية («هل يزيد الحمض
  النفاذية؟» / "does acid stimulation increase permeability?")؟ الاختبار **مثبَّت
  على مطلع السؤال** (أو عبارة حكم صريحة)، فالسؤال التعريفي («ما هي النفاذية؟») لا
  يُقرأ كتقييمي، وسؤال يحتوي «is» في وسطه («what is the graph?») لا يُفعِّل شيئًا.
- `answer_belief(question, ontology=..., prior=...)`: يقرأ **حواف الرسم الموجّهة**
  بين المفاهيم التي يسمّيها السؤال حصرًا (طرفا الحافة كلاهما مذكور)، ويحوّل كل حافة
  إلى سجل `Evidence` يحمل نوع العلاقة في بياناته الوصفية — وهو ما يمكّن
  `ContradictionDetector` من تمييز علاقتين متضادّتين على الزوج نفسه كتعارض حقيقي —
  ثم يشغّل **محرّك مراجعة الاعتقاد الحقيقي** `BeliefRevisionEngine.revise`. الثقة
  والحالة التي تصل القارئ هما اللتان أنتجهما المحرّك فعلًا (بعديُّه المرجَّح
  بالدليل وكشفه للتعارض). يعيد `AnswerBelief` فقط إن كان السؤال تقييميًا **و**
  للرسم حافة بين مفهومين يسمّيهما؛ وإلا `None` — لا تخمين.
- **صادق بالبناء**: لا يخترع دليلًا لم يذكره الرسم، وينسب الاعتقاد إلى الحواف التي
  يستند إليها، و**لا يعدّل الرسم إطلاقًا** (قراءة فقط). الاعتقاد **مُبلَّغ**، لا
  التزام.
- `AnswerBelief` (statement/confidence/status/supporting/contradicting + `describe()`).

**الوصل:** `AssistantAnswer.belief` ← خطوة `SOURCE_BELIEF` في
`core/reasoning_report` (موضعها ثابت: بعد القرار وقبل المشتق، لأنها خطوة مشتقة) ←
`JarvisTurn.belief` ← `ai/chat.format_turn` وحقل `belief` في `turn_payload`، فيصل
إلى الواجهة الرسومية مجانًا. وأمر `python -m cli relations why` يعرضها داخل الأثر
وفي `--json`.

**التقييم:** مقياس `belief_accuracy` بعناصر ضبط سالبة (سؤال تعريفي، جملة عادية،
سؤال تقييمي يسمّي مفهومًا واحدًا، أو سؤال تقييمي لا يستطيع الرسم الكلام عنه —
كلها يجب ألّا تُنتج اعتقادًا). أُضيف إلى `evaluation/benchmark.py` و`runner.py`
و`baseline.json` (overall 0.9821 → 0.9827).

**مثال مُتحقَّق منه:**
```
$ python -m cli relations why "does acid stimulation increase permeability?"
The evidence-weighed belief the graph supports:
  - does acid stimulation increase permeability? — confidence 0.71 (provisional)
  - supports: acid_stimulation —increases→ permeability (confidence 0.70)
```

---

## 11. المعالجة المُنجَزة في المرحلة 44 (الاستدلال يصل متن الإجابة)

**المشكلة التي كشفها المراجعة:** المراحل 41–43 أوصلت التخطيط والقرار والاعتقاد إلى
الإجابة، لكن الوصل توقّف عند **الأثر** (trace) وحده: **متن الإجابة** بقي قالب
الاسترجاع («إليك ما تقوله المصادر المتاحة عن…»)، فجاء الاستدلال في تذييل قد لا
يفتحه القارئ. وكان التقييم يمرّ بينما النثر الذي يقرأه المستخدم فعلًا لم يتغيّر.

**وحدة جديدة:** `core/answer_lead.py`
- `answer_lead(question, plan=, decision=, belief=)`: يبني المقدّمة **بالأقوى أولًا**
  — الإجراء، ثم التوصية، ثم الحكم — ويأخذ **أول ما ينطبق** فقط، فلا يُحمَّل السؤال
  مقدّمتين متنافستين.
- كل مقدّمة **منسوبة لمصدرها بالاسم**: عنوان الإجراء يسمّي مدخل المتن الذي جاء منه
  (`plan.source`)، والتوصية تسرد خيارات الرسم مرتّبة، والحكم يذكر القضية وثقتها
  والحواف التي يستند إليها.
- `AnswerLead` (kind/heading/body/language + `describe()`)، و`None` تعني أن السؤال
  لم يستحق استدلالًا — فلا يُختلق شيء لملء مقدّمة.

**الوصل:** `core.composition.with_lead(question, lead, body)` يصل المقدّمة بالمتن
الموثّق تحت عنوان «ما تضيفه المصادر عن…»، و`grounded_answer(..., lead=...)` أضاف
الوسيط. المقدّمة تفتح الإجابة في الحالتين: عندما يكتب النموذج المتن وعندما يكتب
المستخلِص الاحتياطي — فلا يطمس أحدهما الآخر. وتُحمَل على الدور المقروء آليًا:
`JarvisTurn.lead` و`turn_payload["lead"]`.

**إصلاحان مصاحبان:**
- `core/answer_plan.py`: `match_entry` صار يقرأ **معجم المتن** حيث يوجد المرادف
  العربي لكل مصطلح (`porosity / مسامية`)، فالسؤال الإجرائي العربي
  («كيف أقيس المسامية؟») يطابق المدخل ويصل إلى المخطّط.
- `core/composition.py`: `topic_phrase` لم يعد يُنتج عبارة مكسورة — يُسقط فعلًا
  مجرّدًا وضميرًا حذفه السؤال («how does porosity relate to permeability» →
  «porosity permeability»)، مع إبقاء كل كلمة قد تكون اسمًا.

**التقييم:** مقياس `lead_accuracy` بعناصر ضبط سالبة (سؤال تعريفي/غير ذي صلة/فارغ
يجب ألّا ينال مقدّمة)، **وأحكام المتن تشغّل مسار الإجابة الحقيقي عبر `ask`**
(`answer_for=`) — فانحدار يترك الاستدلال في الأثر ويُبقي المتن على القالب يخفض
المقياس. تحقّقنا من تمييزه: متن أثر-فقط 0.667، نوع خاطئ 0.778، المسار الحقيقي
1.000. أُضيف إلى `benchmark.py`/`runner.py`/`baseline.json` (overall 0.9827 → 0.9832).

**مثال مُتحقَّق منه:**
```
$ python -m cli chat --mind  # (أو أي واجهة)
س: how do I measure porosity and permeability on a core sample?
The corpus states this procedure for Reservoir rock properties (reservoir_rock_properties):
  1. ...
  2. ...
What the sources add about porosity and permeability:
  ... [1] ...
```

---

## 12. المعالجة المُنجَزة في المرحلة 45 (الزمن يصل الإجابة)

**المشكلة التي كشفها المراجعة:** المشروع يحمل طبقة زمنية كاملة (`TemporalReasoner`
و`Timeline` و`TemporalEvent` و`TemporalPattern` و`TimePrediction`)، لكنها كانت
مُتاحة للاختبارات وحدها؛ فسؤال مثل «ما الذي تغيّر خلال الجلسة؟» كان يأتي بتعريف، لا
بحساب مرتّب لما جرى فعلًا — وهو **النصف الزمني من النقص 7**.

**وحدة جديدة:** `core/answer_temporal.py`
- `answer_temporal(question, *, conversation=, events=)`: يقرأ الجدول الزمني الوحيد
  الحقيقي المتاح في مسار الإجابة — **أدوار المحادثة المختومة بالوقت** — ويُجريها عبر
  `TemporalReasoner` الحقيقي: الترتيب زمني، والعلاقة بين آخر دورين علاقة المحرّك نفسه
  (`before`/`during`/`simultaneous`…)، وأي نمط متكرر وتوقّعه القادم هما ما أنتجه
  `find_pattern`/`predict_next` فعلًا.
- **مُبوَّب** إلى سؤال زمني فقط (`is_temporal`، `TEMPORAL_CUES` بالعربية والإنجليزية)،
  ويتخطّى أدوار `SYSTEM`/`CONTEXT` (تعليمة نظام وملف مرفق ليسا شيئًا جرى في المحادثة)،
  ويُعيد `None` بدل التخمين حين لا يوجد جدول زمني (أقل من دورين، أو لا محادثة).
- `AnswerTemporal` (entries/relation/change/pattern/prediction/language +
  `span_seconds` + `describe()`)، و`None` تعني أن السؤال لم يستحق جدولًا زمنيًا.

**الوصل:** الجدول الزمني يمكن أن يفتح الإجابة كمقدّمة
(`core.answer_lead`'s `KIND_TIMELINE`، بعد الإجراء والتوصية والحكم)، ويُحمَل على
الدور المقروء آليًا (`JarvisTurn.temporal` و`turn_payload["temporal"]`) وفي أثر
الاستدلال (`core.reasoning_report.SOURCE_TEMPORAL`).

**التقييم:** مقياس `temporal_accuracy` بعناصر ضبط سالبة (سؤال تعريفي أو غير ذي صلة،
أو سؤال زمني بحدث واحد، يجب ألّا يُنتج جدولًا زمنيًا). أُضيف إلى
`benchmark.py`/`runner.py`/`baseline.json` (overall 0.9832 → 0.9836).

**مثال مُتحقَّق منه:**
```
$ python -m cli chat --mind  # (أو أي واجهة)
س: what is porosity?          (دور أول)
س: what changed over the session?
What changed over this conversation (span 1 s):
  - Q: what is porosity?
  - A: Here is what the available sources say about porosity. First...
  - Q: what changed over the session?
  Latest change: the discussion moved from "A: ..." to "Q: ..." (temporal relation: before).
```

---

## 13. المعالجة المُنجَزة في المرحلة 46 (التناظر يصل الإجابة)

**المشكلة التي كشفها المراجعة:** المشروع يحمل طبقة تناظر كاملة (`AnalogyEngine`
و`StructureMappingEngine` و`Structure` و`StructureMapping`)، لكنها كانت مُتاحة
للمتحكّم وواجهة `cli/analogy.py` وسيناريوهات التقييم وحدها؛ فسؤال مثل «هل المسامية
تشبه النفاذية؟» كان يأتي بتعريف — والأسوأ: جسر الاعتقاد كان يقرأ فاتحته `is …?`
كحكم على **دعوى مختلفة** («هل تزيد المسامية النفاذية؟») — وهو **النصف التناظري من
النقص 7، وآخر محرّك معزول فيه**.

**وحدة جديدة:** `core/answer_analogy.py`
- `answer_analogy(question, *, ontology=, hops=, limit=)`: يقرأ المفهومين اللذين
  يسمّيهما السؤال، ويبني لكلٍّ جواره العلائقي، ويحاذيهما بـ`AnalogyEngine` الحقيقي
  (عبر `StructureMappingEngine`) — فالمقارنة التي تصل القارئ هي ما أنتجه محرّك
  المطابقة البنيوية فعلًا: تطابقاته، وتماسكه، واستنتاجاته المرشَّحة (كلٌّ منها
  **مرشَّح** مخصوم بمقدار ما تطابق من البنية).
- **مُبوَّب** إلى سؤال تناظري فقط (`is_analogical`، `ANALOGY_CUES` بالعربية
  والإنجليزية)، ويُعيد `None` بدل التخمين حين لا يستحق السؤال مقارنة (سؤال تعريفي،
  أو غير ذي صلة، أو تناظري بسمّى مفهومًا واحدًا).
- **صادق بالبناء:** يُبلّغ بحكم المحرّك نفسه (`meaningful`) حتى لو كان `False` — زوج
  لا يشترك في نظام علاقات يُقال عنه ذلك، لا يُزيَّن كتناظر — ويُبلّغ بما **لم**
  يتطابق كما يُبلّغ بما تطابق. لا يُثبِّت شيئًا في الرسم.
- `AnswerAnalogy` (base/target/meaningful/score/matched/transferred/unmapped_*/language
  + `describe()`).

**الوصل:** المقارنة يمكن أن تفتح الإجابة كمقدّمة (`core.answer_lead`'s
`KIND_ANALOGY`، بين التوصية والحكم)، وتُحمَل على الدور المقروء آليًا
(`JarvisTurn.analogy` و`turn_payload["analogy"]`)، وفي أثر الاستدلال
(`core.reasoning_report.SOURCE_ANALOGY`)، وفي `cli/relations.py why --json`. **ويمنع
الجسر جسرَ الاعتقاد من الإجابة عن سؤال مقارنة كدعوى** (`is_analogical` → لا اعتقاد).

**التقييم:** مقياس `analogy_answer_accuracy` بعناصر ضبط سالبة (سؤال تعريفي، أو غير
ذي صلة، أو تناظري بمفهوم واحد، يجب ألّا يُنتج مقارنة؛ وزوج بلا نظام علاقات مشترك
يجب أن يُبلَّغ أنه ليس تناظرًا). أُضيف إلى `benchmark.py`/`runner.py`/`baseline.json`
(overall 0.983628 → 0.984071).

**مثال مُتحقَّق منه:**
```
$ python -m cli relations why "is porosity like permeability?" --json
"analogy": {
  "base": "porosity", "target": "permeability",
  "meaningful": true, "score": 23.8,
  "matched": ["core_sample --measures--> porosity  ==  core_sample --measures--> permeability", ...],
  "transferred": [{"source": "rock", "relation": "contains", "target": "porosity", "confidence": 0.4}, ...]
}
```

---

## 14. المعالجة المُنجَزة في المرحلة 47 (المعرفة تتراكم بين الجلسات — سدّ النقص 8)

**المشكلة التي كشفها المراجعة:** المشروع يحمل طبقة حفظ كاملة
(`KnowledgeStore`/`Snapshot`/`Persistence`/`JsonStore`)، لكنها **لم تكن موصولة
بمسار الإجابة**: الرسم الذي يقرأ منه الجواب يُعاد بناؤه من البذور عند كل تشغيل،
وكل علاقة **تعلّمها** النظام في جلسة (علاقة مقروءة من جملة المستخدم، رابط مُعزَّز،
اعتقاد عُدِّل) تُرمى عند انتهاء العملية. فمعرفة تعلّمها Jarvis اليوم كانت تضيع غدًا.
هذا هو **النقص 8** من `docs/REASONING_ASSESSMENT.md`.

**وحدة جديدة:** `core/knowledge_carry.py`
- `LearnedRelation` — سجل علاقة **مفتاحه وسوم المفهومين لا معرّفاتها** (معرّف
  `CON-…` يُولَّد من جديد عند كل تشغيل، فالوسم هو ما يصمد بين جلسة وأخرى)، مع
  `to_primitive`/`from_primitive`/`describe` ومفتاح حالة موحَّد للمنع من التكرار.
- `capture_learned(ontology, learner)` — العلاقات **المتعلَّمة فقط**، والبذور
  مستثناة: علاقة تُعَدّ متعلَّمة فقط حين أنشأها المعلِّم (`is_learned`)؛ فبذرة
  مدقَّقة لا تُلتقط أبدًا (هي في كل جلسة أصلًا، ووصلها لا يغيّر جوابًا).
- `restore_learned(ontology, learner, records)` — يُعيد تشغيل السجلات عبر
  `OntologyLearner` **نفسه**، فيبقى كل رابط مقيَّدًا بسقف المعلِّم وقاعدة المفهوم
  المعروف؛ سجل باسم مفهوم غير معروف يُبلَّغ عنه **مُتجاوَزًا لا مُختلَقًا**، وسجل
  لزوج موجود يُعَدّ حاضرًا لا مُكرَّرًا. `CarryReport` (restored/already_present/
  skipped + `changed`/`total`/`describe`).
- `LearnedKnowledgeStore` — غلاف متين فوق `Persistence` (وثيقة واحدة مُصدَّرة
  بإصدار، مفتاح واحد `CARRY_KEY` تحت `CARRY_NAMESPACE`)، مع `capture`/`restore`/
  `save`/`load`. سجل غير قابل للقراءة يُسقَط بدل أن يُسقط الجلسة.
- **صادق بالبناء:** لا يُلتقط إلا المتعلَّم، ولا يُختلَق مفهوم، ولا يُتجاوَز على
  بذرة، ولا شيء يعمل تلقائيًا — الالتقاط والإعادة نداءان صريحان، فجلسة للقراءة فقط
  لا تُمسّ.

**الوصل:** `GroundedAssistant` يكتسب `remember_relation()` (يكتب علاقة، يُبطل ذاكرة
الاسترجاع، ويلتقط المتعلَّم للمخزن) و`recall_learned()` (يُعيد تشغيل ما حمله السابق
قبل أول سؤال) و`learn_from_text()` (يقرأ جملة بـ`RelationExtractor` الحقيقي ويلتقط
النتيجة). و`build_default_assistant(learned=…)` يستدعي `recall_learned()` عند البناء.
مسار التغذية الراجعة في `JarvisMode` يمرّر التصحيح/التأكيد إلى `learn_from_text`،
و`ai/jarvis.build_session` يفتح المخزن المتين تحت `data/learned-knowledge`
(`_learned_store()`)، فجلسة CLI/GUI تتراكم فعلًا. وأمر
`python -m cli relations carry "<statement>" [--show] [--json]` هو التوأم غير المرئي.

**التقييم:** مقياس `knowledge_carry_accuracy` بعناصر ضبط سالبة (بذرة مدقَّقة لا
تُلتقط أبدًا كمُتعلَّمة؛ سجل باسم مفهوم غير معروف يُتجاوَز ولا يُختلَق؛ إعادة تشغيل
السجلات نفسها مرتين لا تُكرِّر). أُضيف إلى `benchmark.py`/`runner.py`/`baseline.json`
(overall 0.984071 → 0.984516).

**مثال مُتحقَّق منه (جلسة ← جلسة، عبر عمليتين):**
```
$ FAYD_DATA_DIR=/tmp/carry python -m cli relations carry "Waterflood increases water saturation."
  learned: waterflood --increases--> saturation
Carried 1 learned relation(s) for the next session:
  waterflood --increases--> saturation (0.45)

$ FAYD_DATA_DIR=/tmp/carry python -m cli relations carry --show
Carried 1 learned relation(s); 0 already present, 0 skipped.
```
وعلى مسار الإجابة: علاقة تعلّمها `assistant.remember_relation(...)` في جلسة تظهر
موجودة في رسم `build_default_assistant(learned=…)` في الجلسة التالية.

---

## 15. المعالجة المُنجَزة في المرحلة 48 (التقييم يقيس المسار الموصول لا المعزول — سدّ النقص 9)

**المشكلة التي كشفها المراجعة:** المراحل 37–47 جسرت كل طبقة استدلالية إلى الإجابة
(السببية، عدم اليقين، بوابة الرسوخ، الأثر، التخطيط، القرار، الاعتقاد، الزمن،
التناظر، وحفظ المعرفة)، وأعطت كل جسر مقياسه المحجوز. لكن تلك المقاييس كانت تقيس
الجسر **معزولًا**: تستدعي `answer_plan`/`answer_decision`/`answer_belief`/… مباشرةً،
أو تسلّم المساعد سؤالًا واحدًا، فيمكن لطبقة **توقّف وصولها إلى كائن الإجابة الذي
يستلمه المستخدم فعلًا** أن تظل تنجح. هذا بالضبط الانحدار الذي يحذّر منه النقص 9:
تقييم يقيس الوحدة المعزولة لا المسار الموصول.

**وحدة جديدة:** `core/answer_audit.py`
- `answer_layers(answer)` — الطبقات التي يحملها **كائن الإجابة** فعلًا: الكائنات
  الاختيارية (plan/decision/belief/analogy/temporal/lead/reasoning/confidence)
  والمجموعات غير الفارغة (facts/inferred/connections/causal). لا تخمين: `None`
  أو مجموعة فارغة ليست طبقة، فطبقة أنتجها الجسر وأسقطتها الإجابة تكون غائبة.
- `AuditExpectation` — توقّع واحد: `required` (طبقات يجب أن يحملها كائن الإجابة)،
  `forbidden` (طبقات يجب ألّا يحملها — عناصر الضبط السالبة)، `body` (أن يكون
  **رأس** الـlead في متن الإجابة لا في الأثر وحده)، و`timeline` (أن تُسلَّم جلسة
  تحقيقات قصيرة لأن الخط الزمني يحتاج دورين على الأقل).
- `AUDIT_EXPECTATIONS` — كتالوج محجوز بعناصر ضبط سالبة: سؤال إجرائي يجب أن يحمل
  الخطة وlead الإجراء، وسؤال قراري التوصية، وتقييمي الحكم، وتناظري المقارنة،
  وزمني الخط الزمني، ووصل السلاسل للاتصالات، وتعريفـي للعلاقات المخزَّنة — كل
  إيجابي يشترط أيضًا الأثر (`reasoning`)؛ وسؤال تعريفـي/غير ذي صلة/مبهم **لا
  يحمل أي طبقة استدلالية**.
- `audit_answer(expectation, answer_for)` — يشغّل `ask` **الحقيقي**، يقرأ الطبقات
  عن كائن الإجابة، ويفحص المجموعتين، ومتى طُلب أن الـlead يفتتح المتن. إجابة
  غائبة أو خطأ = فشل لا تخطٍّ.
- `answer_path_accuracy(expectations, answer_for)` — نسبة التوقعات التي يحققها
  **المسار الموصول**. بلا `answer_for` لا يُقاس المسار، فالجواب الصادق `0.0` بدل
  ادّعاء نجاح لم يُكتسب.

**التقييم:** مقياس `answer_path_accuracy` أُضيف إلى `benchmark.py`/`runner.py`/
`baseline.json` ويدخل في `overall` (0.984516 → 0.984911). **مُثبَت أنه يميّز:**
بحذف الـlead عن الإجابة → `0.0`؛ وبإدخال طبقة في عنصر ضبط سالب → `0.9`؛ وبلا مسار
→ `0.0`. أي أن مقياسًا يقيس الطبقة المعزولة كان سيمرّ من هذه الانحدارات الثلاثة.

**مثال مُتحقَّق منه (مقاييس المسار الموصول):**
```
$ python -m evaluation
  ok  knowledge_carry_accuracy       1.0000 -> 1.0000 (+0.0000)
  ok  answer_path_accuracy           1.0000 -> 1.0000 (+0.0000)
  ok  overall                        0.9849 -> 0.9849 (+0.0000)
PASS: no performance regression
```

---

## 16. الحكم النهائي

المشروع **متقدّم معماريًا** (طبقة استدلال من 70 وحدة، كلها مختبَرة، مع بوابات
أمان: الرمل، بوابة الرسوخ، التغذية الراجعة الصريحة، مراقب الانحدار)، لكن **قوته
الاستدلالية كانت حبيسة**: موجودة ولا تظهر في الإجابة. الاتجاه الصحيح — الذي
بدأته المراحل 28 و36 و37 و38 و39 و40 و41 و42 و43 و44 و45 و46 و47 و48 — هو **جسر كل طبقة
استدلالية إلى مسار الإجابة، مع مقياس محجوز بعناصر ضبط سالبة لكل جسر**. سُدَّت حتى الآن
النواقص 2–6 (السببية، عدم اليقين، بوابة الرسوخ، والأثر المرئي)، **وسُدَّ النقص 7
كاملًا** (التخطيط وصل الإجابة في المرحلة 41، والقرار في المرحلة 42، والاعتقاد في
المرحلة 43، والزمن في المرحلة 45، **والتناظر في المرحلة 46**)، **وصار الاستدلال يظهر
في متن الإجابة نفسه لا في الأثر وحده** في المرحلة 44، **وسُدَّ النقص 8** في المرحلة 47
(العلاقات المتعلَّمة تُلتقط وتُعاد إلى رسم الجلسة التالية، فلا تعود المعرفة تضيع بين
الجلسات)، **وسُدَّ النقص 9** في المرحلة 48 (مقياس `answer_path_accuracy` يقيس
**المسار الموصول**: يشغّل `ask` الحقيقي ويفحص ما يحمله كائن الإجابة فعلًا، فطبقة
تتوقف عن الوصول إلى الإجابة تفشل حتى لو نجح جسرها المعزول). بهذا اكتملت قائمة
النواقص التسعة. الخطوة التالية ليست سدّ نقص آخر، بل **توسيع** الكتالوج المحجوز:
أسئلة عربية موازية، وعناصر ضبط سالبة أوسع، ومقاييس عبر مسارات متعددة الخطوات.
