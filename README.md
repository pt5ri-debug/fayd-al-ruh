# Fayd Al-Ruh

مشروع Fayd Al-Ruh هو **معمارية معرفية رمزية واعية بالأدلة** (evidence-aware symbolic
cognitive architecture). يحوّل الرسم المعرفي من مخزن للكيانات والعلاقات إلى نموذج
معرفي قابل للتفسير والتطور: كل معرفة تحمل دليلها، ودرجة الثقة، ومصدرها، وتاريخها،
وحالتها (مؤقتة أم مستقرة).

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

## البنية

```
core/        النواة المعرفية: الاستدلال، السببية، الذاكرة، التعلم، التخطيط، التفسير
storage/     محوّلات التخزين (JSON وSQLite وGraph)
ai/          عميل Ollama محلي اختياري
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
