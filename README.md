# Concrete Strength Prediction Models
## نماذج تنبؤ قوة الخرسانة

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ==================== 1. بناء قاعدة البيانات ====================
print("=" * 60)
print("خطوة 1: بناء قاعدة بيانات واقعية")
print("=" * 60)

np.random.seed(42)
n_samples = 300

# إنشاء المتغيرات المستقلة
data = {
    'Cement': np.random.uniform(250, 450, n_samples),      # الإسمنت (كغ/م³)
    'FlyAsh': np.random.uniform(50, 150, n_samples),        # الرماد المتطاير
    'Water': np.random.uniform(150, 200, n_samples),        # الماء (كغ/م³)
    'Plasticizer': np.random.uniform(5, 15, n_samples),    # المحسنات
    'Age': np.random.choice([7, 14, 28], n_samples)        # عمر الخرسانة (يوم)
}
df = pd.DataFrame(data)

# إضافة ميزات هندسية
df['WC_Ratio'] = df['Water'] / (df['Cement'] + df['FlyAsh'])  # نسبة الماء/الإسمنت
df['Cementitious'] = df['Cement'] + df['FlyAsh']             # المواد الإسمنتية الكلية

# توليد قيمة القوة بمعادلة فيزيائية
df['Strength'] = (
    (df['Cement'] * 0.12) + 
    (df['Age'] * 0.9) - 
    (df['WC_Ratio'] * 45) + 
    (df['Plasticizer'] * 0.5) +
    np.random.normal(0, 2.5, n_samples)  # ضوضاء واقعية
)

print(f"\nعدد العينات: {len(df)}")
print(f"\nإحصائيات البيانات:")
print(df.describe().round(2))

# ==================== 2. تحضير البيانات ====================
print("\n" + "=" * 60)
print("خطوة 2: تحضير البيانات")
print("=" * 60)

X = df.drop('Strength', axis=1)
y = df['Strength']

# تطبيع البيانات
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"عدد عينات التدريب: {len(X_train)}")
print(f"عدد عينات الاختبار: {len(X_test)}")

# ==================== 3. النموذج الأول: Random Forest ====================
print("\n" + "=" * 60)
print("خطوة 3: تدريب نموذج Random Forest مع ضبط المعاملات")
print("=" * 60)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_

print(f"\nأفضل معاملات Random Forest:")
print(f"  - عدد الأشجار: {best_model_rf.n_estimators}")
print(f"  - عمق الشجرة الأقصى: {best_model_rf.max_depth}")
print(f"  - أفضل درجة CV: {grid_search_rf.best_score_:.4f}")

# التنبؤ والتقييم
y_pred_rf = best_model_rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\nنتائج Random Forest على بيانات الاختبار:")
print(f"  - R² Score: {r2_rf:.4f} (أقرب لـ 1 أفضل)")
print(f"  - MAE: {mae_rf:.4f} (متوسط الخطأ المطلق)")
print(f"  - RMSE: {rmse_rf:.4f} (جذر متوسط مربعات الخطأ)")

# أهمية الميزات
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nأهمية الميزات (Random Forest):")
print(feature_importance_rf.to_string(index=False))

# ==================== 4. النموذج الثاني: Gradient Boosting ====================
print("\n" + "=" * 60)
print("خطوة 4: تدريب نموذج Gradient Boosting مع ضبط المعاملات")
print("=" * 60)

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search_gb = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid_gb,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search_gb.fit(X_train, y_train)
best_model_gb = grid_search_gb.best_estimator_

print(f"\nأفضل معاملات Gradient Boosting:")
print(f"  - عدد المراحل: {best_model_gb.n_estimators}")
print(f"  - معدل التعلم: {best_model_gb.learning_rate}")
print(f"  - عمق الشجرة: {best_model_gb.max_depth}")
print(f"  - أفضل درجة CV: {grid_search_gb.best_score_:.4f}")

# التنبؤ والتقييم
y_pred_gb = best_model_gb.predict(X_test)

r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"\nنتائج Gradient Boosting على بيانات الاختبار:")
print(f"  - R² Score: {r2_gb:.4f}")
print(f"  - MAE: {mae_gb:.4f}")
print(f"  - RMSE: {rmse_gb:.4f}")

# أهمية الميزات
feature_importance_gb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model_gb.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nأهمية الميزات (Gradient Boosting):")
print(feature_importance_gb.to_string(index=False))

# ==================== 5. مقارنة النماذج ====================
print("\n" + "=" * 60)
print("خطوة 5: مقارنة أداء النماذج")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'R² Score': [r2_rf, r2_gb],
    'MAE': [mae_rf, mae_gb],
    'RMSE': [rmse_rf, rmse_gb]
})

print("\n" + comparison.to_string(index=False))

best_model_name = "Gradient Boosting" if r2_gb > r2_rf else "Random Forest"
print(f"\n🏆 النموذج الأفضل: {best_model_name}")

# ==================== 6. التصور البياني ====================
print("\n" + "=" * 60)
print("خطوة 6: رسوم بيانية توضيحية")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# الرسم 1: المتنبأ به مقابل الفعلي (Random Forest)
axes[0, 0].scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('القيمة الفعلية')
axes[0, 0].set_ylabel('القيمة المتنبأ بها')
axes[0, 0].set_title(f'Random Forest (R²={r2_rf:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# الرسم 2: المتنبأ به مقابل الفعلي (Gradient Boosting)
axes[0, 1].scatter(y_test, y_pred_gb, alpha=0.6, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('القيمة الفعلية')
axes[0, 1].set_ylabel('القيمة المتنبأ بها')
axes[0, 1].set_title(f'Gradient Boosting (R²={r2_gb:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# الرسم 3: أهمية الميزات (Random Forest)
top_features_rf = feature_importance_rf.head(6)
axes[1, 0].barh(top_features_rf['Feature'], top_features_rf['Importance'], color='skyblue')
axes[1, 0].set_xlabel('الأهمية')
axes[1, 0].set_title('أهمية الميزات - Random Forest')
axes[1, 0].invert_yaxis()

# الرسم 4: أهمية الميزات (Gradient Boosting)
top_features_gb = feature_importance_gb.head(6)
axes[1, 1].barh(top_features_gb['Feature'], top_features_gb['Importance'], color='lightgreen')
axes[1, 1].set_xlabel('الأهمية')
axes[1, 1].set_title('أهمية الميزات - Gradient Boosting')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ تم حفظ الرسوم البيانية في: model_comparison.png")

# ==================== 7. مثال على التنبؤ ====================
print("\n" + "=" * 60)
print("خطوة 7: مثال على التنبؤ بقوة خرسانة جديدة")
print("=" * 60)

# عينة جديدة
new_sample = pd.DataFrame({
    'Cement': [350],
    'FlyAsh': [100],
    'Water': [175],
    'Plasticizer': [10],
    'Age': [28],
    'WC_Ratio': [175 / (350 + 100)],
    'Cementitious': [450]
})

new_sample_scaled = scaler.transform(new_sample)
prediction_rf = best_model_rf.predict(new_sample_scaled)[0]
prediction_gb = best_model_gb.predict(new_sample_scaled)[0]

print(f"\nمتغيرات العينة الجديدة:")
print(f"  - الإسمنت: 350 كغ/م³")
print(f"  - الرماد: 100 كغ/م³")
print(f"  - الماء: 175 كغ/م³")
print(f"  - المحسنات: 10 لتر/م³")
print(f"  - العمر: 28 يوم")

print(f"\nالقوة المتنبأ بها:")
print(f"  - Random Forest: {prediction_rf:.2f} MPa")
print(f"  - Gradient Boosting: {prediction_gb:.2f} MPa")
print(f"  - المتوسط: {(prediction_rf + prediction_gb) / 2:.2f} MPa")

print("\n" + "=" * 60)
print("✅ انتهى التحليل بنجاح!")
print("=" * 60)
```

## شرح المقاييس:

| المقياس | المعنى | الأفضل |
|--------|--------|--------|
| **R² Score** | نسبة التباين المشروح (0-1) | أقرب لـ 1 أفضل |
| **MAE** | متوسط الخطأ المطلق | أقل أفضل |
| **RMSE** | جذر متوسط مربعات الخطأ | أقل أفضل |

## النتائج المتوقعة:
- **R² عالي** (> 0.9): النموذج ممتاز
- **R² متوسط** (0.7-0.9): جيد
- **R² منخفض** (< 0.7): يحتاج تحسين
