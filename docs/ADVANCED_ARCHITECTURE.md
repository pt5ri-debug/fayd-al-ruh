# Advanced cognitive architecture

هذا المشروع الآن يوفّر طبقات عملية قابلة للتطوير، وليست ادعاءً بامتلاك وعي أو ذكاء بشري عام.

## المكوّنات

- `core/model_interface.py`: عقد موحّد لربط أي نموذج لغوي أو متعدد الوسائط. يمرَّر العميل من التطبيق ولا تُحفظ مفاتيح API داخل المشروع.
- `core/semantic_memory.py`: فهرس دلالي محلي حتمي (cosine similarity). يمكن تبديل دالة `hashed_embedding` بمزوّد embeddings حقيقي لاحقاً.
- `core/memory.py`: ذاكرة عرضية مع استرجاع يجمع الصلة اللفظية والدلالية والأهمية.
- `core/goals.py`: أهداف وخطط بتبعيات صريحة؛ المخطط لا ينفّذ خطواته تلقائياً.
- `core/safe_learning.py`: تشغيل محاكاة مسجّلة فقط؛ لا يوجد وصول إلى shell أو الشبكة أو الملفات.
- `core/evaluation_suite.py`: قياس اتساق الرسم المعرفي، استخدام الذاكرة، ومعايرة التنبؤات.

## ربط نموذج حقيقي

أنشئ `CallableModel` يغلّف عميل النموذج الذي تختاره، ثم مرّره إلى `CognitiveController.reason`. يجب أن يعيد الغلاف نصاً فقط؛ أما الأدوات والإجراءات الخارجية فتبقى تحت تحكم التطبيق والمستخدم.

```python
model = CallableModel("my-model", lambda request: client.generate(request))
response = controller.reason(model, request)
```

## التحقق

شغّل `python main.py` لعرض دورة محلية آمنة. وعند تثبيت متطلبات المشروع، شغّل `python -m pytest -q`.
