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

أنجزنا في المراحل 28 و36 و37 و38 و39 و40 الجسور التالية:
- **المرحلة 28:** العلاقات التي *يستنتجها* الرسم (entailment) تصل إلى الإجابة.
- **المرحلة 36:** سلاسل العلاقات متعددة الخطوات بين مفاهيم السؤال تصل إلى الإجابة.
- **المرحلة 37:** **السلاسل السببية الموجّهة** تصل إلى الإجابة، مع تصحيح
  عيب قائم في عرضها.
- **المرحلة 38:** **عدم اليقين** (التعارض والضعف) يُحسب داخل ثقة الإجابة.
- **المرحلة 39:** **بوابة الرسوخ** تُقوّى (اتجاه السبب + عدم التناقض + دعم كل جملة).
- **المرحلة 40:** **أثر استدلالي مرئي للمستخدم** («كيف وصلت إلى هذه الإجابة»)
  يصل إلى الواجهة والـ CLI ووحدة `why`.

يبقى **النقص البنيوي الأساسي**: معظم المحرّكات الأخرى (التخطيط، القرار،
الاعتقاد، الزمن، عدم اليقين، التناظر) **ما زالت معزولة**.

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
*(عولج التخطيط في المرحلة 41 والقرار في المرحلة 42؛ يتبقّى الاعتقاد والزمن.)*

### نقص 8 — لا تراكم معرفي بين الجلسات في مسار الإجابة
لا يوجد حفظ لحالة الرسم/الاعتقادات عبر الجلسات في مسار الإجابة (الحفظ موجود
نظريًا في `KnowledgeStore`/`Snapshot` لكنه غير موصول بالإجابة).

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
| 9 | وصل الاعتقاد والزمن (سؤال «ما تغيّر؟») | النقص 7 | ⏳ لاحقًا |
| 10 | حفظ حالة الرسم/الاعتقادات عبر الجلسات | النقص 8 | ⏳ لاحقًا |
| 11 | مقاييس تقييم تقيس **المسار الموصول** لا المعزول | النقص 9 | ⏳ لاحقًا |

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
| `python -m pytest -q` | ✅ المجموعة الكاملة: 3964 نجحت، 200 متخطّاة، 0 فشل |
| `python -m ruff check .` | ✅ All checks passed |
| `python -m mypy` | ✅ Success: no issues found in 734 source files |
| `python -m evaluation` | ✅ PASS — `grounding_accuracy=1.0000`, `reasoning_trace_accuracy=1.0000`, `plan_accuracy=1.0000`, `decision_recommendation_accuracy=1.0000`, `overall=0.9821` |

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

## 10. الحكم النهائي

المشروع **متقدّم معماريًا** (طبقة استدلال من 69 وحدة، كلها مختبَرة، مع بوابات
أمان: الرمل، بوابة الرسوخ، التغذية الراجعة الصريحة، مراقب الانحدار)، لكن **قوته
الاستدلالية كانت حبيسة**: موجودة ولا تظهر في الإجابة. الاتجاه الصحيح — الذي
بدأته المراحل 28 و36 و37 و38 و39 و40 و41 و42 — هو **جسر كل طبقة استدلالية إلى
مسار الإجابة، مع مقياس محجوز بعناصر ضبط سالبة لكل جسر**. سُدَّت حتى الآن النواقص
2–6 (السببية، عدم اليقين، بوابة الرسوخ، والأثر المرئي)، **وسُدَّ النقص 7 كاملًا**
(التخطيط وصل الإجابة في المرحلة 41، والقرار وصل الإجابة في المرحلة 42).
**المتبقّي هو النواقص 8–10** — الاعتقاد والزمن والتناظر ما زالت معزولة، ولا تراكم
معرفي بين الجلسات في مسار الإجابة، ومقاييس التقييم ما زالت تقيس المعزول أكثر من
الموصول. تُعالج بالترتيب نفسه، وكل جسر يُقاس ويُختبر قبل الدمج.
