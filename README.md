# Fayd Al-Ruh

مشروع Fayd Al-Ruh هو **معمارية معرفية رمزية واعية بالأدلة** (evidence-aware symbolic
cognitive architecture). يحوّل الرسم المعرفي من مخزن للكيانات والعلاقات إلى نموذج
معرفي قابل للتفسير والتطور: كل معرفة تحمل دليلها، ودرجة الثقة، ومصدرها، وتاريخها،
وحالتها (مؤقتة أم مستقرة).

المشروع يعرّف نفسه كـ**عقل مهندس بترول** (`core/self_model.py`): يستدلّ في مجال هندسة
البترول — المكامن، الحفر، الانتاج، البتروفيزياء، وسجلات الآبار — من الأدلة لا من الظن.
الهوية إطار مجال، لا ادّعاء وعي. المعرفة الأولية للمجال بذرة صغيرة قابلة للفحص في
`core/domain_knowledge.py`، وتُزرع في أنطولوجيا المشروع عند الإقلاع.

> المشروع لا يدّعي وعياً أو ذكاءً بشرياً عاماً. هو طبقات عملية قابلة للتطوير،
> وكل استنتاج فيه قابل للتفسير والمراجعة.

## المبادئ الأساسية

لكل اعتقاد في النظام:

- ماذا أعرف؟ — `core/belief.py`
- لماذا أعتقد ذلك؟ — `core/evidence.py`, `core/evidence_processor.py`
- متى عرفته ومن أين؟ — `core/cognitive_context.py`
- ما درجة ثقتي؟ — `core/confidence_model.py`, `core/uncertainty.py`
- هل توجد أدلة تعارضه؟ — `core/contradiction_detector.py`, `core/belief_conflict.py`
- هل المعرفة مؤقتة أم مستقرة؟ — `core/belief.py` (`BeliefStatus`)

## هوية المجال: عقل مهندس بترول

المشروع يعرّف نفسه في `core/self_model.py`، والمعرفة الأولية للمجال في
`core/domain_knowledge.py`:

```python
from core.cognitive_controller import CognitiveController

controller = CognitiveController()
controller.identity          # إطار الهوية الذي يُمرَّر مع كل طلب نموذج
controller.ontology.find_concepts("reservoir")   # مكمن، طبقة، مسامية، نفاذية...
```

- `SelfModel`: الاسم والهوية والمجال والقدرات والحدود ومجالات الخبرة.
- `seed_ontology`: يزرع مفاهيم البترول وعلاقاتها (مثل `المكمن contained_in الطبقة`)
  بثقة متواضعة قابلة للمراجعة، بشكل **خامل التكرار** (idempotent).
- `CognitiveController`: يزرع الأنطولوجيا عند الإقلاع، ويضيف إطار الهوية إلى سياق كل
  طلب نموذج، ويكشفه عبر `controller.identity`.

هذا إطار مجال فقط: لا يمنح النظام وعياً، ولا يغيّر الاعتقادات تلقائياً، وكل علاقة
بذرة قابلة للتصحيح بالأدلة مثل أي معرفة أخرى.

## قياس الهوية: مجموعة تقييم بترولية

الهوية ليست ادّعاءً غير قابل للفحص؛ حزمة `evaluation/` تقيسها. أُضيف مجال
`Domain.PETROLEUM` مع خمسة سيناريوهات ثابتة:

- **سببية** (`petroleum-causal`): سلسلة `ضغط ← انتاج ← عائد` تُكتشَف وتُفسَّر.
- **استدلال/بتروفيزياء** (`petroleum-petrophysics`): استرجاع قراءات السجل النووي
  وربطها بمفاهيم الأنطولوجيا المزروعة (`السجل measures المسامية/النفاذية`).
- **تخطيط** (`petroleum-planning`): تسلسل «اكمل البئر» (تغليف ثم اسمنت) وتسلسل
  «عالج هبوط الانتاج» (حقن ماء ثم تحفيز).
- **عدم يقين** (`petroleum-uncertainty`): «هل ينخفض الانتاج؟» يُراجَع من أدلة
  الحقل ويبقى معايراً.
- **هوية** (`petroleum-identity`): يتحقق أن الأنطولوجيا المزروعة تحمل إطار المجال
  المعلن (مثل `المكمن contained_in الطبقة`، `الاسمنت seals البئر`).

تُقاس ثلاثة مقاييس بترولية في الأساس: `petroleum_causal_accuracy`،
`petroleum_retrieval_accuracy`، و`petroleum_ontology_accuracy`، إضافة إلى
`domain_scores["petroleum"]`. فتصبح الهوية قابلة للقياس ومنع التراجع في CI.

## البنية

```
core/        النواة المعرفية: الاستدلال، السببية، الذاكرة، التعلم، التخطيط، التفسير
             (self_model.py وdomain_knowledge.py: هوية «عقل مهندس بترول» وبذرة مجالها)
storage/     محوّلات التخزين (JSON وSQLite وGraph)
ai/          النموذج اللغوي الداخلي (transformer) وعميل Ollama اختياري
tests/       اختبارات العقود والتكامل والدورات الكاملة
main.py      عرض محلي آمن لدورة معرفية كاملة
```

## التشغيل

```bash
pip install -r requirements.txt
python main.py
```

## الاختبارات

```bash
python -m pytest -q
```

## التعلّم من التفاعل

المكافأة في بيئات الرمل شرط نجاح مكتوب يدوياً (مكافأة بديلة). وللتعلّم من نتائج
مرصودة فعلاً:

```bash
python -m ml --episodes 400 --real-reward
```

- `core/interaction_reward.py`: يتعلّم `InteractionRewardModel` معايرة (ثقة →
  نسبة نجاح مرصودة) وقيم إجراءات سياقية، من متن التقييم (`evaluation.scenarios`)
  ومن مراجعة بشرية (`outcomes_from_reviews`).
- `ml/interaction_learner.py`: نفس وكيل Q-learning، لكن مكافأته النسبة المرصودة
  للإجراء، لا شرط المهمة؛ وتُمنح فقط عندما يُغيّر الإجراء الحالة فعلاً.

المخرج يعرض `outcomes/forecasts/ece/brier` ومعدل نجاح كل مهمة، فتبقى جودة المكافأة
المتعلَّمة قابلة للفحص بدل إخفائها.

## التخطيط بنموذج متعلَّم

يُتعلَّم نموذج عالم رمزي من الانتقالات المرصودة، ثم يُستخدم للتخطيط وللتعلّم
بالتخيّل (Dyna-Q)، مع قياس النجاح دائماً بالهدف الحقيقي:

```bash
python -m ml --episodes 40 --model-based --seeds 3
```

- `ml/world_model.py`: `SymbolicWorldModel` يستنتج تأثيرات الإجراءات من الرصد
  فقط، ويعيد `unknown` لما لم يُرَ قط.
- `ml/model_based_planner.py`: `plan_with_world_model` (بحث داخل النموذج المتعلَّم)
  و`DynaQAgent` (تعلّم من انتقالات حقيقية ومتخيَّلة).

المقارنة عند 40 حلقة (متوسط 3 بذور): المكسب يظهر في المهام متعددة الخطوات —
`plant-seed` 0.42→0.71، `boil-water` 0.67→0.92، `recharge-lamp` 0.83→0.96.

## النموذج اللغوي الداخلي

المشروع يحتوي نموذجاً لغوياً خاصاً به، مكتوباً من الصفر بـ`numpy`، ومُدرَّباً داخل
المستودع، فلا حاجة إلى خدمة خارجية ولا مفاتيح API.

- `ml/llm/tokenizer.py`: ترميز على مستوى البايت (آمن للعربية، بلا تدريب).
- `ml/llm/transformer.py`: محوّل decoder-only (انتباه سببي متعدد الرؤوس، GELU،
  LayerNorm) مع backprop يدوي وAdam — تحقّقنا من التدرجات عددياً.
- `ml/llm/corpus.py`: متن تدريب داخل المستودع (عربي/إنجليزي حول عالم المشروع،
  وعن هندسة البترول، مع أمثلة JSON بنسق المخطط).
- `ml/llm/generator.py` و`ai/local_model.py`: توليد نص، وتكييف مع واجهة
  `CognitiveModel` فيحلّ محل Ollama في أي مكان يُقبل فيه نموذج.
- `ml/llm/schema.py`: فك ترميز موجَّه بالمخطط. البنية تُثبَّت من
  `MODEL_OUTPUT_SCHEMA`، والنموذج يقترح محتوى الأوراق فقط، فيعود JSON صالحاً
  دائماً ويخضع بعدها لتحقق `LLMReasoner`.

```bash
python -m ml.llm                       # تدريب نقطة التحقق الافتراضية وحفظها
python -m ml.llm --steps 4000          # تدريب أطول
python -m ml.llm --generate "الماء"    # تدريب ثم توليد تكملة
```

استخدامه كنموذج معرفي:

```python
from ai.local_model import LocalLanguageModel

model = LocalLanguageModel(temperature=0.0)   # يقرأ نقطة التحقق المُلتزَمة
response = controller.reason(model, request)
```

ربطه بمسار القرار: عندما يحمل الطلب `response_schema`، يقترح النموذج مفاهيم
وفرضيات وأسئلة وخططاً، ثم يتحقق منها `LLMReasoner` ولا يستطيع تعديل الاعتقادات
ولا تنفيذ أدوات:

```python
analysis = controller.analyze_with_model(model, request)   # نفس مسار التحقق
```

> تنويه صريح: هذا محوّل حقيقي لكنه صغير، مُدرَّب على متن صغير داخل المستودع.
> يُنتج نصاً متماسكاً حول عالم المشروع، وليس نموذجاً عاماً بحجم النماذج الحدودية.

## ربط نموذج لغوي حقيقي

النموذج يُمرَّر من التطبيق ولا تُحفظ مفاتيح API داخل المشروع:

```python
from core.model_interface import CallableModel

model = CallableModel("my-model", lambda request: client.generate(request))
response = controller.reason(model, request)
```

الغلاف يعيد نصاً فقط؛ أما الأدوات والإجراءات الخارجية فتبقى تحت تحكم التطبيق والمستخدم.

## الحالة

- الإصدار: 2.1.0
- بايثون: 3.12 فأحدث
- الترخيص: MIT — راجع `LICENSE.txt`
