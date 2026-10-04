اسم المشروع:
Fayd Al-Ruh

Fayd Al-Ruh v2.0
Knowledge Graph 2.0

هدف الإصدار:

تحويل الـ Knowledge Graph من مخزن للكيانات والعلاقات إلى نموذج معرفي قابل للتفسير والتطور.

سنضيف إلى كل معرفة:

ماذا أعرف؟
↓
لماذا أعتقد ذلك؟
↓
متى عرفته؟
↓
ما درجة ثقتي؟
↓
ما الأدلة؟
↓
هل توجد أدلة تعارضه؟
↓
هل المعرفة مؤقتة أم مستقرة؟
1. البنية الجديدة

سنقسمها إلى:

core/
│
├── entity.py
├── relation.py
├── knowledge_graph.py
│
├── evidence.py
├── belief.py
└── provenance.py

وأهم إضافة في v2.0 هي:

Evidence

الدليل.

Belief

الاعتقاد الناتج من الأدلة.

Provenance

تاريخ ومصدر وكيفية تكوين المعرفة.

2. العلاقة في v2.0

في الإصدار الأول كانت العلاقة:

Robot → SEES → Ball

لكن الإصدار 2.0 سيجعلها:

Robot
   │
   │ SEES
   ▼
Ball

Confidence: 0.94
Evidence: EVID-001
Created: ...
Source: sensory_observation
Status: active

وهذا فارق جوهري.