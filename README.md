# Importar librerías
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Cargar el dataset
car_data = pd.read_csv('car_data.csv')

# 1. Análisis exploratorio
# Gráfico de correlaciones entre variables numéricas
sns.heatmap(car_data.corr(), annot=True, cmap="coolwarm")
plt.show()

# Histograma para analizar la distribución de 'selling_price'
plt.hist(car_data['selling_price'], bins=20)
plt.xlabel('Precio de Venta')
plt.ylabel('Frecuencia')
plt.show()

# 2. Preprocesamiento
# Aplicar One Hot Encoding a las variables categóricas
car_data_encoded = pd.get_dummies(car_data, drop_first=True)

# 3. Selección de características
selector = SelectKBest(score_func=f_regression, k=5)
X = car_data_encoded.drop('selling_price', axis=1)  # Variables predictoras
y = car_data_encoded['selling_price']  # Variable objetivo
X_new = selector.fit_transform(X, y)

# 4. División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento de los modelos
# Regresión Lineal
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Árbol de decisión
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# 6. Evaluación de los modelos
# Predicciones
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)

# Evaluación Regresión Lineal
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Evaluación Árbol de Decisión
r2_dt = r2_score(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)

print(f"R² (Regresión Lineal): {r2_lr}, MSE: {mse_lr}")
print(f"R² (Árbol de Decisión): {r2_dt}, MSE: {mse_dt}")

# 7. Visualización de los resultados
plt.scatter(y_test, y_pred_lr, color='blue', label='Regresión Lineal')
plt.scatter(y_test, y_pred_dt, color='green', label='Árbol de Decisión')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal')
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.legend()
plt.show()
