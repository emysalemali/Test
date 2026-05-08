# Test
Agentimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# 1. بناء قاعدة بيانات واقعية هندسياً
np.random.seed(42)
n_samples = 200
data = {
    'Cement': np.random.uniform(250, 450, n_samples),
    'FlyAsh': np.random.uniform(50, 150, n_samples),
    'Water': np.random.uniform(150, 200, n_samples),
    'Plasticizer': np.random.uniform(5, 15, n_samples),
    'Age': np.random.choice([7, 14, 28], n_samples)
}
df = pd.DataFrame(data)

# إضافة ميزة نسبة الماء إلى المواد الإسمنتية (W/C Ratio)
df['WC_Ratio'] = df['Water'] / (df['Cement'] + df['FlyAsh'])

# توليد القوة المستهدفة بناءً على معادلة فيزيائية مع إضافة ضوضاء واقعية
df['Strength'] = (df['Cement'] * 0.1) + (df['Age'] * 0.8) - (df['WC_Ratio'] * 50) + np.random.normal(0, 2, n_samples)

# 2. التجهيز والتدريب
X = df.drop('Strength', axis=1)
y = df['Strength']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. ضبط النموذج (Hyperparameter Tuning)
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 4. النتائج
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}") 
