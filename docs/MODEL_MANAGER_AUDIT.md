# تقييم «مدير النماذج» (Model Manager) — تقرير فني

**الإصدار:** 2.3.0 · **تاريخ التقييم:** 2026-10-31
**النطاق:** مراجعة فنية لبنية مدير النماذج القائم فعلًا في المستودع، وتحديد ما
يجب إضافته، ومقارنة ذلك بالبنية الموجودة، مع خطة تنفيذ مرتّبة على مراحل صغيرة
مستقلة قابلة للاختبار.

**هذا التقرير لا يعدّل أي ملف برمجي.** كل الأرقام والأسماء هنا مقروءة من
المستودع نفسه (`git`, `grep`, `python -m cli models …`)، لا من افتراضات.

---

## 0. الخلاصة التنفيذية (أهم ما يجب أن يعرفه القارئ)

**مدير النماذج موجود بالفعل، وليس مجرد فكرة.** المستودع يحتوي على منظومة كاملة
مكوّنة من **12 وحدة في `core/`** و**3 واجهات أمامية** (`gui/model_manager.py`,
`gui/model_manager_dialog.py`, `cli/models.py`) و**~614 اختبارًا** تمرّ محليًا،
وقد دُمجت المراحل 4–7 منها في `main`.

لذلك، **ليست مهمة هذا التقرير أن يقترح مدير نماذج جديدًا من الصفر** — بل أن:

1. يوثّق البنية القائمة بدقّة (الأقسام أ، ب).
2. يميّز بوضوح بين **ما هو موجود فعلًا** و**ما هو ناقص فعلًا** (القسم ج).
3. يقترح أصغر إضافة صيانة ممكنة لسدّ النواقص الحقيقية فقط (الأقسام د–ي).
4. يضع خطة مراحل صغيرة قابلة للاختبار (القسم الأخير).

### الحالة المقيسة على هذه الآلة (وقت التقرير)

```
$ python -m cli models status
Model Manager
  registry: <repo>/data/model_registry
  1 registered, 27 in the build's catalog, 1 ready to run.
  operation log: 0 event(s)

Runtimes
  llama.cpp: not available — llama-cpp-python is not installed
      next step: python -m pip install -r requirements-ultron.txt
  ollama:    not available — cannot reach http://localhost:11434
      next step: ollama pull qwen2.5-coder:7b
  fayd-inprocess: ready (cpu only).
```

```
$ python -m cli models resources
  cpu cores:  4
  memory:     15.6 GB total, 13.5 GB available
  disk free:  24.2 GB
  GPU: none detected that this program can report on
```

**ملاحظة جوهرية حول العتاد:** تقرير المهمة يفترض AMD Radeon R7 M360 و Intel HD
Graphics 520 و 16 GB RAM. الآلة التي جرى عليها هذا التدقيق **تصرّح بـ 15.6 GB
وبلا GPU يمكن للمشروع رصده** (`nvidia-smi` غير موجود، و`sysfs` لم يُبلّغ عن
جهاز). لذلك:

- **لا يجوز اعتبار أي من كرتَي الرسوميات مدعومًا** — وهو ما يطابق تحذير المهمة.
- **مسار المعالج (CPU) هو المسار الذي يعمل دائمًا**، وهو ما يفرضه التصميم القائم
  أصلًا (`fayd-inprocess: ready (cpu only)`).

---

## أ. البنية الحالية (Current Architecture)

### أ-1 كيف يُكتشَف النموذج ويُختار ويُشغَّل؟

يوجد **مساران متوازيان**، ولا يتناقضان، بل أحدهما يقرأ الآخر:

**المسار الأول — الكتالوج (الكتالوج الموحّد، موجود قبل مدير النماذج):**

```
core/language_models.py        (all_language_models / models_for_choice)
   ↑ يقرأ من
core/ultron.py, core/vision_model.py, core/banner.py,
core/presentation_model.py, companion/model.py, hakeem/model.py,
core/cloud_model.py, ai/mind.py (نقطة التحقق المدرَّبة)
```

هذا المسار **للقراءة فقط**: لا يخزّن شيئًا، ويجيب على سؤال «ما الذي يعرفه هذا
البناء، وهل كل نموذج موجود على القرص؟» بإعادة بناء القوائم في كل نداء، مع
**إزالة التكرار بحسب هوية الأوزان** (`repo/filename`)، فيصبح نفس الملف الظاهر
من محرّكين سجلًّا واحدًا يذكر المحرّكين.

**المسار الثاني — مدير النماذج (الطبقة الجديدة، المراحل 4–7):**

```
core/model_manager.py          ← المنسّق (ModelManager)
   ├── core/language_models.py      «ما يجلبه البناء» (للقراءة فقط)
   ├── core/model_registry.py       «ما سجّله المستخدم» (سجل دائم)
   ├── core/model_files.py          «ما هو الملف فعلًا» (GGUF/npz probe)
   ├── core/model_runtime.py        «ما يشغّله، وهل يستطيع»
   ├── core/model_process.py        «العمليات التي بدأها هذا البرنامج»
   ├── core/model_settings.py       «كيف يجب تشغيله» (إعدادات لكل نموذج)
   ├── core/model_selection.py      «أي نموذج لهذه المهمة، ولماذا»
   ├── core/model_preflight.py      «هل سيتّسع؟» (تقدير، لا ضمان)
   ├── core/model_resources.py      «ما تملكه هذه الآلة»
   ├── core/model_sources.py        «من أين يأتي النموذج» (مهايئات مصادر)
   ├── core/model_download.py       «جلبه بأمان، خلف البوابات»
   └── core/model_log.py            «ماذا حدث» (سجل عمليات دائم محدود)
```

القاعدة المعمارية الحاكمة (مذكورة صراحةً في `core/model_manager.py`):
**المنسّق لا يملك أي حقيقة بنفسه** — كل حقيقة تعود إلى الوحدة التي كانت تملكها
أصلًا. ما يضيفه المدير هو **الرَبْط (the join)**: نداء واحد يجيب «أرِني كل نموذج
يستطيع هذا البرنامج تشغيله، وحالة كل واحد، والخطوة التالية».

### أ-2 تتبّع مسار النداء من الواجهة إلى التنفيذ

**من الـ GUI (النافذة الرئيسية):**

```
gui/app.py  _open_model_manager()                     (سطر ~2824)
   → gui/model_manager_dialog.py  ModelManagerDialog
        → gui/model_manager.py  ModelManagerService   (Tk-free)
             → core/model_manager.py  ModelManager
                  → core/model_registry.py   ModelRegistry
                  → core/model_runtime.py    adapter_for(...) → RuntimeAdapter
                       → LlamaCppAdapter        (in-process, llama-cpp-python)
                       → OllamaAdapter          (يُعيد استخدام core/coder_model)
                       → InProcessTrainedAdapter (نموذج المشروع المدرَّب)
```

المفتاح: `ModelManagerService` في `gui/model_manager.py` **خالٍ من Tk**، لذا
يمكن اختباره بلا شاشة، وهو ما يفعله `tests/test_model_manager_service.py`.

**من الـ CLI:**

```
cli/menu.py  run_models(argv)  → _run_module("cli.models", argv)   (سطر ~376)
   → cli/models.py  (argparse, 35+ أمرًا فرعيًا)
        → gui/model_manager.py  ModelManagerService   ← نفس الخدمة تمامًا
             → core/model_manager.py  ModelManager
```

**حقيقة مهمة:** الـ CLI والـ GUI يقودان **نفس** `ModelManagerService`، فلا
يوجد منطق مُكرَّر بينهما، ولا يمكن أن يختلفا في «أي نموذج محمَّل الآن».

### أ-3 مسار الإجابة العادي (للمقارنة — يجب عدم المساس به)

```
gui/app.py / ai/chat.py
   → ai/jarvis.build_session  → core/jarvis.JarvisMode
        → ai/chat.ModelSwitcher
             → Mouth (GroundedAssistant / UltronAssistant / VisionAssistant /
                      Banner / Cloud / Natasha / Hakeem / MindAssistant)
```

هذا المسار **مستقلّ تمامًا** عن مدير النماذج، وهو المطلوب. مدير النماذج
**لا يعترض** مسار الإجابة، ولا يبدّل الموديل تلقائيًا، ولا يكتب في إعدادات أي
فم (mouth). هذا فصل مقصود ويجب الحفاظ عليه.

---

## ب. القدرات الموجودة (Existing Capabilities)

الجدول التالي يسرد **ما هو موجود فعلًا** في المستودع، بمكانه، وهل يمكن للمدير
إعادة استخدامه.

| القدرة | مكان التنفيذ | إعادة الاستخدام؟ |
|---|---|---|
| كتالوج موحّد لكل النماذج، بلا تكرار | `core/language_models.py` | ✅ يُقرأ كما هو |
| سجل دائم بنموذج واحد لكل ملف JSON | `core/model_registry.py` (`ModelRegistry`, `JsonStore`) | ✅ موجود ومختبَر |
| قراءة حقائق ملف GGUF/`.npz` بلا تحميل | `core/model_files.py` (`probe_file`, `read_gguf_info`) | ✅ موجود |
| كشف التلف/البتر/عدم تطابق السحر | `core/model_files.py` (`INTEGRITY_*`) | ✅ موجود |
| اكتشاف ملفات النماذج في مجلدات المشروع | `core/model_files.discover_models` | ✅ موجود |
| مهايئ وقت تشغيل بوصف قدرات | `core/model_runtime.py` (`RuntimeAdapter`, `RuntimeCapabilities`) | ✅ موجود |
| مهايئ llama.cpp داخل العملية | `core/model_runtime.LlamaCppAdapter` | ✅ موجود (كسول الاستيراد) |
| مهايئ Ollama (اختياري) | `core/model_runtime.OllamaAdapter` | ✅ موجود (يعيد استخدام `core/coder_model`) |
| مهايئ النموذج المدرَّب داخليًا | `core/model_runtime.InProcessTrainedAdapter` | ✅ موجود |
| كشف دعم GPU بلا تخمين | `core/model_runtime.llama_cpp_gpu_offload()` → `bool \| None` | ✅ موجود |
| إدارة العمليات التي بدأها البرنامج فقط | `core/model_process.py` (`ProcessRegistry`) | ✅ موجود |
| إعدادات استنتاج لكل نموذج مع أسبقية واضحة | `core/model_settings.py` (`effective_settings`) | ✅ موجود |
| اختيار نموذج لمهمة مع الأسباب | `core/model_selection.py` (`rank_for_task`, `RankedModel`) | ✅ موجود |
| خط رجوع محدود عند الفشل | `core/model_selection.plan_fallback` | ✅ موجود |
| تقدير الذاكرة كنطاق لا رقم مؤكد | `core/model_preflight.estimate_memory` | ✅ موجود |
| فحص ما قبل التحميل (ملف/توافق/ذاكرة/سياق/GPU) | `core/model_preflight.preflight` | ✅ موجود |
| موارد الآلة بالمكتبة القياسية فقط | `core/model_resources.py` (`system_resources`, `gpu_info`) | ✅ موجود |
| مهايئات مصادر (محلي/URL/Hugging Face) بلا تنزيل | `core/model_sources.py` (`ModelSource`, `CLAIM_*`) | ✅ موجود |
| تنزيل آمن قابل للاستئناف خلف البوابات | `core/model_download.py` | ✅ موجود |
| سجل عمليات دائم محدود ومُنقّى من الأسرار | `core/model_log.py` (`ModelLog`) | ✅ موجود |
| واجهة خدمة خالية من Tk | `gui/model_manager.py` (`ModelManagerService`) | ✅ موجود |
| نافذة GUI مدمجة في الشريط الجانبي | `gui/model_manager_dialog.py` | ✅ موجود |
| CLI بـ 35+ أمرًا فرعيًا | `cli/models.py` | ✅ موجود |
| تسجيل النموذج المدرَّب | `core/model_manager.register_trained` | ✅ موجود |
| تشغيل/إيقاف/اختبار نموذج | `ModelManager.load/unload/test` | ✅ موجود |
| تشخيص وقراءة سجل | `ModelManager.diagnostics/log_tail` | ✅ موجود |
| تخزين ذرّي للبيانات | `storage/json_store.py` (`JsonStore`) | ✅ يُعاد استخدامه |
| مجلد بيانات محترم لـ `FAYD_DATA_DIR` | `core/paths.py` | ✅ يُعاد استخدامه |
| عمليات «لا تفشّل المستدعي» | `core/best_effort.best_effort` | ✅ يُعاد استخدامه |
| خيوط عمل في الواجهة + طابور نتائج | `gui/model_manager.run_in_background` | ✅ موجود |

**الخلاصة:** كل بند في قائمة المهمة المطلوبة (2) يقابل وحدة قائمة فعلًا.

---

## ج. القدرات الناقصة (Missing Capabilities)

هذا أهم قسم، لأن المهمة تشترط التمييز بوضوح بين الناقص والموجود.

### ج-1 ما هو **موجود فعلًا** ولا يجب إضافته (وإلا صار تكرارًا)

- كشف النماذج، تسجيلها، فحصها، تحميلها، إيقافها، اختبارها.
- تكامل llama.cpp و Ollama.
- تسجيل النموذج المدرَّب.
- فحص التوافق وتقدير الموارد.
- إعدادات الاستنتاج لكل نموذج.
- API مشترك للـ GUI والـ CLI.
- اختيار النموذج للمهمة.
- التشخيص وسجل العمليات.

### ج-2 النواقص الحقيقية (مقيسة، لا مفترضة)

| # | النقص | الدليل | الأثر |
|---|---|---|---|
| N1 | **لا سجل تشخيصي عام** للنظام | `grep -rn "import logging"` يعيد **0** نتيجة في كل المستودع | لا يوجد مسار موحّد لتشخيص أخطاء وقت التشغيل خارج `ModelLog` الخاص بالنماذج |
| N2 | **`ai/local_model.py` قديم وغير موصول** | 83 سطرًا فقط؛ غير مُستورَد من `ai/mouths.py` ولا من المدير | مسار تحميل قديم قد يوحي بازدواجية؛ يجب توثيقه كـ«مهجور» أو وصله صراحةً |
| N3 | **لا تصدير/استيراد لسجل النماذج** | لا توجد دالة `export`/`import` في `core/model_registry.py` | لا يمكن نسخ احتياطي أو نقل الإعدادات بين آلتين |
| N4 | **لا فحص صحة دوري (health probe) للنموذج المحمَّل** | `ModelManager.test` يدوي فقط | لا كشف تلقائي لنموذج ماتت عمليته |
| N5 | **GPU: لا كشف AMD/Intel** | `gpu_info()` يعتمد `nvidia-smi` و `sysfs`؛ أعاد «none» هنا | كرتا المهمة (AMD/Intel) غير مرئيين، لذا القرار دائمًا CPU — وهذا صحيح أمنيًا لكن يجب توثيقه لا إخفاؤه |
| N6 | **لا مقاييس أداء (latency/tokens/s)** | لا يوجد قياس زمن في `InferenceResult` | لا يمكن مقارنة نموذجين بأرقام حقيقية |
| N7 | **لا «وضع افتراضي لكل مهمة» محفوظ** | `select_for_task` يحسب لحظيًا؛ لا حفظ للاختيار الافتراضي | يختار المستخدم من جديد كل مرة |
| N8 | **لا توثيق معماري لمدير النماذج** | لا يوجد `docs/MODEL_MANAGER*.md` | البنية موجودة لكن غير مشروحة لغير الكاتب |

**ملاحظة حاسمة:** النواقص N1–N8 كلها **صغيرة وإضافية**. لا يوجد نقص يبرّر
إعادة كتابة أي شيء قائم.

---

## د. البنية المقترحة (Proposed Architecture)

**المبدأ الحاكم: أصغر إضافة صيانة ممكنة. لا وحدة جديدة إلا لسدّ نقص مُثبت.**

### د-1 إضافات مقترحة (3 وحدات فقط)

```
core/model_health.py     ← جديد: فحص صحة دوري للمحمَّل + مقاييس أداء (N4، N6)
core/model_profiles.py   ← جديد: «النموذج الافتراضي لكل مهمة» محفوظ (N7)
docs/MODEL_MANAGER.md    ← جديد: توثيق معماري (N8)
```

مع **تعديلات طفيفة** على وحدتين قائمتين:

```
core/model_registry.py   ← إضافة export_records() / import_records() (N3)
ai/local_model.py        ← توثيق كـ«مهجور» أو حذف مُبرَّر (N2)
```

### د-2 المسؤوليات والعلاقات

| الوحدة | المسؤولية الواحدة | تعتمد على | لا تفعل |
|---|---|---|---|
| `core/model_health.py` | قياس زمن الاستدلال وسرعته، والتحقق الدوري من أن النموذج المحمَّل ما زال يستجيب | `core/model_runtime` | لا تحمّل نموذجًا، لا تقرر أي نموذج |
| `core/model_profiles.py` | حفظ «النموذج المفضل لكل مهمة» | `storage.json_store`, `core/model_selection` | لا ترتّب ولا تخمّن؛ تقرأ فقط |
| `core/model_registry.py` (معدَّل) | + نسخ احتياطي/نقل للسجل | كما هو | لا تغيّر سلوك التسجيل القائم |

**قاعدة الفصل:** كل وحدة جديدة تقرأ من `ModelManager` ولا تكتب فيه، إلا عبر
واجهته العامة. هذا يمنع ظهور مصدر حقيقة ثانٍ.

### د-3 لماذا لا نضيف `core/model_manager.py` ثانيًا؟

لأنه موجود (1506 سطرًا) ويعمل. أي «مدير نماذج جديد» سيكون تكرارًا صريحًا
يناقض شرط المهمة («دون تكرار الوظائف»).

---

## هـ. خطة التنفيذ على مستوى الملفات (File-Level Plan)

**لا يُنشَأ أي ملف الآن.** الجدول للتوثيق فقط.

| الملف | موجود؟ | الإجراء | المسؤولية | التبعيات | الواجهات العامة | الاختبارات |
|---|---|---|---|---|---|---|
| `core/model_manager.py` | ✅ | لا تغيير | المنسّق | الكل | `ModelManager`, `model_manager` | `test_model_manager.py` (88) |
| `core/model_registry.py` | ✅ | **تعديل طفيف** (N3) | السجل + نسخ/نقل | `model_files`, `json_store` | `+export_records`, `+import_records` | `test_model_registry.py` (65) |
| `core/model_files.py` | ✅ | لا تغيير | فحص الملف | stdlib | `probe_file`, `discover_models` | `test_model_files.py` (38) |
| `core/model_runtime.py` | ✅ | لا تغيير | مهايئ التشغيل | `coder_model` | `adapter_for`, `RuntimeAdapter` | `test_model_runtime.py` (54) |
| `core/model_process.py` | ✅ | لا تغيير | إدارة العمليات | stdlib | `ProcessRegistry` | (مُغطّى في runtime) |
| `core/model_settings.py` | ✅ | لا تغيير | الإعدادات | `json_store`, `best_effort` | `effective_settings` | `test_model_settings.py` (26) |
| `core/model_selection.py` | ✅ | لا تغيير | الترتيب والرجوع | — | `rank_for_task`, `plan_fallback` | `test_model_selection.py` (51) |
| `core/model_preflight.py` | ✅ | لا تغيير | التقدير والفحص | `model_resources` | `estimate_memory`, `preflight` | `test_model_preflight.py` (52) |
| `core/model_resources.py` | ✅ | لا تغيير | موارد الآلة | stdlib | `system_resources`, `gpu_info` | `test_model_resources.py` (10) |
| `core/model_sources.py` | ✅ | لا تغيير | مهايئات المصادر | — | `ModelSource`, `CLAIM_*` | (مُغطّى في download) |
| `core/model_download.py` | ✅ | لا تغيير | التنزيل الآمن | `model_sources` | `download_model` | `test_model_download.py` (61) |
| `core/model_log.py` | ✅ | لا تغيير | سجل العمليات | `json_store` | `ModelLog`, `ModelEvent` | `test_model_log.py` (31) |
| `core/model_health.py` | ❌ | **إنشاء** (N4، N6) | صحة + مقاييس | `model_runtime` | `HealthReport`, `probe_health`, `InferenceMetrics` | `tests/test_model_health.py` (جديد) |
| `core/model_profiles.py` | ❌ | **إنشاء** (N7) | المفضّل لكل مهمة | `json_store`, `model_selection` | `ModelProfiles`, `default_for_task` | `tests/test_model_profiles.py` (جديد) |
| `ai/local_model.py` | ✅ | **توثيق/إزالة** (N2) | مهجور | — | — | `tests/test_local_model.py` |
| `gui/model_manager.py` | ✅ | لا تغيير | خدمة الواجهة | `core/model_manager` | `ModelManagerService` | `test_model_manager_service.py` (56) |
| `gui/model_manager_dialog.py` | ✅ | لا تغيير | النافذة | `gui/*` | `ModelManagerDialog` | `test_model_manager_window.py` (25) |
| `cli/models.py` | ✅ | لا تغيير | الـ CLI | `gui/model_manager` | `main` | `test_models_cli.py` (56) |
| `docs/MODEL_MANAGER.md` | ❌ | **إنشاء** (N8) | التوثيق | — | — | (لا اختبار — وثيقة) |

---

## و. خطة دمج الواجهة الرسومية (GUI Integration Plan)

### و-1 الإطار القائم

- **الشريط الجانبي:** `gui/sidebar.py` (`SidebarColumn`, `SidebarSection`) —
  أقسام قابلة للطي، تُحفظ حالتها لكل `scope`.
- **مجموعات الإجراءات:** `gui/actions.py` (`ACTIONS`, `ACTION_GROUPS`,
  `grouped_visible_actions()`).
- **فتح النوافذ:** `gui/app.py` يحفظ مرجعًا لكل نافذة ويعيد استخدامها
  (`_open_model_manager` → `_forget_model_manager`).
- **النافذة:** تُبنى بـ `Theme` و `RoundedBox` و `gui/rich_text` و
  `gui/chat_composer` و `gui/window_controls`.

### و-2 الموقع الموصى به

**لا موقع جديد.** مدير النماذج **موجود بالفعل** في الشريط الجانبي:

```
gui/actions.py  → ACTION_GROUPS  → ("System", ...)
gui/app.py      → _open_model_manager()   (سطر ~2824)
```

عند إضافة الوحدات الجديدة، **لا تُضاف نافذة جديدة**. بدلًا من ذلك:

1. `core/model_health.py` → زر **Check health** داخل النافذة القائمة (صف
   التفاصيل)، يعمل على خيط عمل عبر `run_in_background`.
2. `core/model_profiles.py` → قائمة **Default model per task** داخل قسم
   «Tasks» القائم في النافذة (`_tasks_label` موجود أصلًا).

هذا يحقّق شرط المهمة: «إعادة استخدام اصطلاحات الواجهة القائمة وتجنّب إعادة
تصميم صفحات لا علاقة لها».

---

## ز. خطة دمج الـ CLI (CLI Integration Plan)

### ز-1 الإطار القائم

- `cli/__main__.py`: إعادة توجيه UTF-8 (مهم للعربية على ويندوز) وتوزيع الأوامر.
- `cli/menu.py`: `MENU` (قائمة نصية) + `run_*` لكل وحدة عبر `_run_module`.
- `cli/models.py`: `argparse` بأوامر فرعية (35+).

### ز-2 التسجيل دون كسر القائم

الأوامر الحالية مسجّلة هكذا:

```python
("Model Manager: every language model the program can run", run_models, ("list",)),
```

لإضافة الأوامر الجديدة:

```python
# في cli/models.py — أضف parserين فرعيين فقط:
health  = sub.add_parser("health",  help="check that a loaded model still answers")
profiles = sub.add_parser("profiles", help="the default model for each task")
```

ثم سطران في `MENU` (`cli/menu.py`). **لا تغيير في أي أمر قائم**، ولا في
`cli/__main__.py`.

**شرط الحفاظ على السلوك:** أبقِ رموز الخروج الحالية كما هي
(`0` نجاح، `2` وسائط خاطئة، `3` غير متاح، `4` بوّابة رفضت، `5` فشل غير متوقع).

---

## ح. تصميم مهايئ وقت التشغيل (Runtime Adapter Design)

### ح-1 الواجهة المشتركة — موجودة

`core/model_runtime.py` يعرّف:

```python
class RuntimeCapabilities:      # can_load/unload/stop/discover/validate/infer,
    gpu_offload: bool | None    # streaming/vision/resident
class RuntimeStatus:            # can it answer now + الخطوة التالية الحقيقية
class RuntimeAdapter(Protocol):
    capabilities, status, load, unload, infer, ...
```

**القاعدة الذهبية:** عملية لا يدعمها المهايئ **تُرفَض** عبر
`RuntimeAdapter.require` مع السبب، ولا تُنفَّذ كأنها نجحت
(`UnsupportedCapability`).

### ح-2 llama.cpp

- **موجود فعلًا** في `LlamaCppAdapter` (داخل العملية).
- **الاستيراد كسول** ولا يُفترض وجوده:
  `llama_cpp_available()` يستخدم `importlib.util` بلا استيراد فعلي.
- **دعم GPU غير مفترض:**
  `llama_cpp_gpu_offload()` يعيد `True`/`False`/`None`، و`None` يُعامَل
  كـ`False` في القرار مع **الإبلاغ عنه بأمانة**.
- **لا مسار تنفيذي مفترض، ولا وضع خادم مفترض، ولا backend GPU مفترض.**

### ح-3 Ollama — تكامل اختياري

- **موجود فعلًا** في `OllamaAdapter`.
- **يعيد استخدام** `core/coder_model.OllamaCoderClient` (عميل `urllib` قياسي)
  فلا يُكتب عميل HTTP ثانٍ ولا تُضاف تبعية إلزامية.
- غير متاح هنا (لا خادم) ويُبلّغ عن الخطوة التالية: `ollama pull …`.

### ح-4 متى يلزم عمل جديد؟

**لا يلزم أي عمل جديد على المهايئات.** النقص الوحيد (N6) هو **قياس** الأداء،
وهو يُضاف في `core/model_health.py` كطبقة قراءة فوق `InferenceResult` بلا
تعديل المهايئات.

---

## ط. البيانات والاستمرارية (Data and Persistence)

### ط-1 ما هو موجود ومُعاد استخدامه

| البيانات | المخزن | الملف |
|---|---|---|
| سجل النماذج (سجل لكل نموذج) | `JsonStore` (كتابة ذرّية) | `<data>/model_registry/` |
| إعدادات كل نموذج | `JsonStore` | namespace `model_settings` |
| سجل العمليات | `JsonStore` (حلقة محدودة) | namespace `model_log` |
| إعدادات كل فم (mouth) | `JsonStore` | namespace `mouths` |
| تفضيلات المستخدم | `JsonStore` | `core/preferences.py` |
| اختيار النموذج في الواجهة | `JsonStore` | `ModelManagerService.select` |

**مصدر حقيقة واحد لكل شيء:** لا يوجد تخزين مزدوج. `core/model_manager.py`
يقرأ الكتالوج ولا ينسخه.

### ط-2 الإضافة المقترحة (N3، N7)

```python
# core/model_registry.py
def export_records(self) -> dict[str, Any]: ...   # كل السجلات كوثيقة واحدة
def import_records(self, data, *, replace=False) -> int: ...  # تحقق أولًا، ثم كتابة ذرّية

# core/model_profiles.py  (جديد)
class ModelProfiles:
    def default_for_task(self, task: str) -> str: ...   # "" عندما لا يوجد اختيار
    def set_default(self, task: str, model_id: str) -> None: ...
```

كلاهما يكتب عبر `JsonStore` (ذرّي)، والكتابة `best_effort` مع تسجيل الفشل،
وقراءة سجل تالف تُعيد الافتراضي بلا انهيار.

---

## ي. فحوص الموارد والتوافق (Resource and Compatibility Checks)

### ي-1 ما هو موجود فعلًا (مقيس)

| الفحص | التنفيذ | ملاحظة الصدق |
|---|---|---|
| وجود الملف وسلامته | `core/model_files.probe_file` → `FileProbe.integrity` | `ok` فقط يسمح بالتحميل |
| تطابق السحر (GGUF/npz) | `INTEGRITY_MAGIC_MISMATCH` | يكشف ملفًا أُعيد تسميته |
| البتر | `INTEGRITY_TRUNCATED` | يكشف تنزيلًا ناقصًا |
| توافق وقت التشغيل | `RuntimeStatus` + `runtime_for_format` | — |
| تقدير الذاكرة | `core/model_preflight.estimate_memory` | **نطاق** لا رقم مؤكد |
| قيد طول السياق | `preflight` يقارن بالمدعوم | — |
| توفّر إزاحة GPU | `llama_cpp_gpu_offload()` | `None` = غير معروف، يُعامل كـ`False` |
| الرجوع إلى المعالج | مسار `fayd-inprocess` | **المسار الذي يعمل دائمًا** |
| فشل وقت التشغيل | `ModelLog` (`OP_FAILURE`) + `OperationResult` | — |

### ي-2 صياغة الصدق الإلزامية

يجب أن يظهر مع **كل** رقم مورد الجملة القائمة في `core/model_resources.py`:

> «هذه إجماليات الآلة نفسها كما يبلّغ عنها نظام التشغيل. حاجة النموذج للذاكرة
> **تقدير** مُشتق من ملفه، وليست قياسًا.»

و`fits` تعني «التقدير داخل ما يبلّغ عنه النظام»، وهي **سبب للمحاولة لا وعد
بالنجاح**.

### ي-3 حدود GPU (يجب توثيقها)

`gpu_info()` تعتمد `nvidia-smi` و `sysfs`. على آلة المهمة (AMD Radeon R7 M360،
Intel HD 520) **لن يُكتشف أي منهما**، لذا:

- القرار سيكون **CPU دائمًا**، وهذا هو السلوك الصحيح والآمن.
- **لا يجوز** إضافة كشف AMD/Intel هشّ (استدعاء `rocm-smi` أو `clinfo` بلا
  ضمان) في المرحلة الأولى. يُترك كبند مؤجَّل مع تحقّق مستقل.

---

## ك. استراتيجية الاختبار (Testing Strategy)

### ك-1 الموجود فعلًا

**~614 اختبارًا** تمرّ محليًا في ~10 ثوانٍ، **بلا تنزيل نموذج وبلا GPU**:

```
614 passed in 10.15s
```

| الملف | العدد | الطبقة |
|---|---|---|
| `test_model_manager.py` | 88 | وحدة/تكامل |
| `test_model_registry.py` | 65 | وحدة |
| `test_model_download.py` | 61 | وحدة (نقل مزيف) |
| `test_model_manager_service.py` | 56 | خدمة (Tk-free) |
| `test_models_cli.py` | 56 | CLI |
| `test_model_runtime.py` | 54 | وحدة |
| `test_model_preflight.py` | 52 | وحدة |
| `test_model_selection.py` | 51 | وحدة |
| `test_model_files.py` | 38 | وحدة |
| `test_model_log.py` | 31 | وحدة |
| `test_model_settings.py` | 26 | وحدة |
| `test_model_manager_phases.py` | 26 | خدمة+CLI |
| `test_model_manager_window.py` | 25 | GUI (Tk) |
| `test_model_resources.py` | 10 | وحدة |

### ك-2 الاختبارات المطلوبة للإضافات الجديدة

- **`tests/test_model_health.py`:** الصحة قبل التحميل (`HealthReport.ok=False`
  مع الخطوة التالية)، بعد التحميل، القياس بمهايئ محقون (زمن ثابت)، التعامل مع
  مهلة، ولا استدعاء شبكة.
- **`tests/test_model_profiles.py`:** الحفظ/القراءة، الافتراضي الفارغ، مهمة
  مجهولة، سجل تالف يُعيد الافتراضي، لا كتابة في ملفات الأوضاع (mouths).
- **`test_model_registry.py` (إضافة):** تصدير/استيراد، الاستيراد يتحقق أولًا،
  الاستيراد لا يمسّ ملفات النماذج.

**قاعدة صارمة:** كل اختبار جديد يعمل بمهايئ/نقل محقون، وبلا تنزيل، وبلا GPU.
الاختبارات التي تحتاج Tk تُتخطّى تلقائيًا بلا شاشة (كما يفعل المشروع الآن).

---

## ل. المخاطر والتوصيات (Risks and Recommendations)

### ل-1 مخاطر معمارية

| المخاطرة | الاحتمال | الأثر | التوصية |
|---|---|---|---|
| ظهور «مصدر حقيقة ثانٍ» لسجل النماذج | متوسط | عالٍ | كل قراءة تمرّ بـ`ModelManager`؛ لا تُنسخ القوائم |
| تكرار منطق بين `ai/local_model.py` و`model_runtime` | متوسط | متوسط | توثيق/إزالة `ai/local_model.py` (N2) |
| إضافة كشف GPU هشّ | متوسط | عالٍ | لا كشف AMD/Intel في المرحلة الأولى |
| كسر مسار الإجابة | منخفض | عالٍ | لا تعديل في `core/jarvis.py`, `ai/chat.py`, `ai/mouths.py` |
| تعارض `ModelRegistry` مع `custom_models`/`mouth_profile` | منخفض | متوسط | الطبقة تقرأ فقط؛ `import_custom_models` أحادي الاتجاه |

### ل-2 مخاوف الترحيل

- المراحل 4–7 **مدموجة فعلًا** في `main` (‏`fdb5c27`). لا ترحيل مطلوب.
- ملفات `data/` (السجل، الإعدادات، السجل الزمني) **مُتجاهَلة في git**، لذا
  الترقية لا تمسّ بيانات المستخدم.
- إضافة `export_records`/`import_records` **لا تغيّر** صيغة السجلات القائمة.

### ل-3 الأمن

- **التنزيل يمرّ ببوّابتين:** سياسة الصلاحيات (`Capability.EXTERNAL_TOOL`)
  **و** الموافقة البشرية، وكلتاهما **مرفوضة افتراضيًا**.
- **تقييد المسارات:** كل مسار تنزيل يُحلّ ويُتحقق من بقائه داخل مجلد التنزيل.
- **تنقية الأسرار:** `core/model_log.py` ينقّي الرسائل قبل التخزين، وكل رسالة
  خطأ في السجل مُنقّاة (`redact`).
- **لا تنفيذ لمحتوى النموذج:** الفحص يقرأ الرأس فقط، ولا ينفّذ بايتًا.
- **لا «إيقاف بالاسم»:** `ProcessRegistry` يوقف فقط معرّفات العمليات التي بدأها.
- **توصية إضافية:** عند إضافة `export_records`، تأكّد أنه **لا يصدّر** أي قيمة
  تشبه سرًّا (نفس `redact` المستخدم في السجل).

### ل-4 افتراضات تحتاج تحقّقًا

1. **العتاد:** الآلة الحالية 15.6 GB وبلا GPU مرئي — يخالف افتراض المهمة.
   **يجب تأكيد** مواصفات آلة التطوير المستهدفة قبل أي قرار إزاحة GPU.
2. **`ai/local_model.py`:** يجب تأكيد أنه مهجور فعلًا (لا مستدعي) قبل حذفه.
3. **دعم llama.cpp لـ AMD R7 M360:** غير مؤكد، ولا يُفترض. يلزم تحقّق مستقل.

---

## م. خطة التنفيذ المرتّبة (Prioritized Implementation Plan)

مراحل صغيرة، كل واحدة **مستقلة وقابلة للاختبار**، وكل واحدة تُبقي البوّابات
الأربع خضراء (`pytest`, `ruff`, `mypy`, `python -m evaluation`).

### المرحلة 1 — التوثيق المعماري (N8) · مخاطرة: منخفضة جدًا
- **الملف:** `docs/MODEL_MANAGER.md` (جديد فقط).
- **المحتوى:** رسم البنية من القسم أ، جدول القدرات من القسم ب، جدول النواقص.
- **الاختبار:** لا شيء (وثيقة).
- **لماذا أولًا:** صفر مخاطرة، ويجعل كل مرحلة لاحقة مفهومة.

### المرحلة 2 — توضيح `ai/local_model.py` (N2) · مخاطرة: منخفضة
- **الملف:** `ai/local_model.py`.
- **الإجراء:** إضافة توثيق «مهجور» مع اسم البديل، **أو** حذف مُبرَّر بعد تأكيد
  عدم وجود مستدعٍ (`grep -rn "local_model"`).
- **الاختبار:** `tests/test_local_model.py` يبقى أخضر أو يُحدَّث.
- **لماذا ثانيًا:** يمنع الالتباس قبل إضافة أي شيء.

### المرحلة 3 — قياس الأداء والصحة (N4، N6) · مخاطرة: متوسطة
- **الملف:** `core/model_health.py` (جديد) + اختبار جديد.
- **الواجهة:** `HealthReport`, `probe_health(adapter, model_id)`,
  `InferenceMetrics` (زمن، رموز/ثانية عند توفّرها).
- **الاختبار:** `tests/test_model_health.py` بمهايئ محقون، بلا نموذج.
- **ملاحظة:** قراءة فقط فوق `model_runtime`؛ لا تعديل المهايئات.

### المرحلة 4 — النموذج الافتراضي لكل مهمة (N7) · مخاطرة: منخفضة
- **الملف:** `core/model_profiles.py` (جديد) + اختبار جديد.
- **الواجهة:** `ModelProfiles.default_for_task`, `set_default`.
- **التخزين:** `JsonStore`، كتابة `best_effort`، قراءة تالفة تُعيد الافتراضي.
- **الاختبار:** `tests/test_model_profiles.py`.

### المرحلة 5 — نسخ/نقل سجل النماذج (N3) · مخاطرة: منخفضة
- **الملف:** `core/model_registry.py` (تعديل طفيف).
- **الواجهة:** `export_records`, `import_records` (تحقّق أولًا، كتابة ذرّية،
  بلا أسرار).
- **الاختبار:** إضافات في `tests/test_model_registry.py`.

### المرحلة 6 — توصيل الوحدات الجديدة بالواجهتين · مخاطرة: متوسطة
- **الملفات:** `gui/model_manager.py` (زر Check health + قائمة الافتراضي)،
  `cli/models.py` (`health` + `profiles`)، `cli/menu.py` (سطران).
- **الاختبار:** إضافات في `test_model_manager_service.py`, `test_models_cli.py`,
  `test_model_manager_window.py`.
- **الشرط:** لا نافذة جديدة، لا أمر قائم يتغيّر، رموز الخروج كما هي.

### المرحلة 7 (مؤجَّلة) — كشف GPU لـ AMD/Intel · مخاطرة: عالية
- **لا تبدأ** قبل تأكيد مواصفات العتاد المستهدف (افتراض ل-4-1).
- تتطلب تحقّقًا مستقلًا من أن بناء llama.cpp يدعم الكرت فعلًا.

---

## ن. ملحق: أدلة الأوامر المستخدمة في هذا التقرير

```bash
# عدد وحدات مدير النماذج
ls core/model*.py                     # 12 وحدة
# حجمها
wc -l core/model_*.py gui/model_manager*.py cli/models.py
# لا سجل تشخيصي عام
grep -rn "import logging" --include="*.py" . | wc -l     # → 0
# الاختبارات تمرّ بلا نموذج
python -m pytest tests/test_model_manager*.py tests/test_model_*.py tests/test_models_cli.py -q
# → 614 passed in 10.15s
# الحالة الحيّة
python -m cli models status
python -m cli models resources
python -m cli models diagnostics
```

---

**انتهى التقرير.** هذا المستند **لا يعدّل أي ملف برمجي**؛ هو توثيق وتحليل
لتمكين قرار تنفيذ مبني على المستودع الحقيقي.
