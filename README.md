# ⚙️ Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction

## 📖 نظرة عامة
يحتوي المجلد على البيئة التكويدية ومجموعات البيانات لتدريب نموذج الذكاء الاصطناعي للتنبؤ بالعمر المتبقي (RUL) بناءً على بحث: "Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps".

هذا المشروع يعتمد على معمارية معقدة حيث تتمركز طبقة الانتباه (Attention Layer) مباشرة بعد طبقة BiLSTM. كما يتم الاعتماد في جوانب أخرى من تجاربنا على البيانات المستخرجة من نظام متغيرات NASA POWER (أو CMAPSS حسب السياق الفني)، مع تطبيق نافذة بيانات منزلقة بمقدار 3 خطوات زمنية (Sliding Window of 3 Time Steps) لضمان المعالجة الزمنية المثلى.

## 📐 معمارية النموذج
النموذج الهجين مطور باستخدام TensorFlow ويحتوي على:
- `Twin-Stage 1D-CNN` لتقليل الضوضاء الفضائية واستخراج الميزات.
- `BiLSTM (128 units)` لاستخلاص أطول سياق زمني ممكن.
- طبقة الانتباه المخصصة `Additive Attention Layer` التي تقوم بإنتاج أوزان الانتباه القابلة للتفسير (Heatmaps).
- يتم تطبيق دالة خسارة غير متماثلة `Asymmetric Loss` لمعاقبة المبالغة في تقدير العمر المتبقي أكثر من التقليل لضمان السلامة في بيئة العمل.

## 🛠 الاستخدام
التثبيت:
```bash
pip install -r requirements.txt
```
تشغيل عملية التدريب واستخراج النماذج والرسومات:
```bash
python nasa_rul_prediction.py
```

## 🔖 الاقتباس المرجعي (APA 7th Edition)
Abdullah, M. E. B. (2026). *Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps*. arXiv. https://arxiv.org/abs/2604.13459

---
© محفوظة رسميًا للباحث المهندس / م. محمد عزالدين بابكر عبدالله - 2026
