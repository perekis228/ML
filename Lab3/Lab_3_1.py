from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Загрузка
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

#Отрисовка отношений
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

sl0 = df['sepal length (cm)'][:50]
sw0 = df['sepal width (cm)'][:50]
axes[0, 0].scatter(sl0, sw0, color='purple', marker='o')
pl0 = df['petal length (cm)'][:50]
pw0 = df['petal width (cm)'][:50]
axes[0, 1].scatter(pl0, pw0, color='purple', marker='o')
axes[0, 0].set_title('setosa')

sl1 = df['sepal length (cm)'][50:100]
sw1 = df['sepal width (cm)'][50:100]
axes[1, 0].scatter(sl1, sw1, color='red', marker='o')
pl1 = df['petal length (cm)'][50:100]
pw1 = df['petal width (cm)'][50:100]
axes[1, 1].scatter(pl1, pw1, color='red', marker='o')
axes[1, 0].set_title('versicolor')

sl2 = df['sepal length (cm)'][100:]
sw2 = df['sepal width (cm)'][100:]
axes[2, 0].scatter(sl2, sw2, color='green', marker='o')
pl2 = df['petal length (cm)'][100:]
pw2 = df['petal width (cm)'][100:]
axes[2, 1].scatter(pl2, pw2, color='green', marker='o')
axes[2, 0].set_title('virginica')

plt.tight_layout()
plt.show()

# Построение pairplot
sns.pairplot(df, hue='target', palette='viridis', markers=['o', 's', 'D'], plot_kws={'alpha': 0.7})
plt.show()

#Подготовка датасета
df1 = df.iloc[:100]
df2 = df.iloc[50:]

#Делим на тестовые и обучающие
from sklearn.model_selection import train_test_split
X0_train, X0_test, y0_train, y0_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=0)
X1_train, X1_test, y1_train, y1_test = train_test_split(df1.drop('target', axis=1), df1['target'], test_size=0.2, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(df2.drop('target', axis=1), df2['target'], test_size=0.2, random_state=0)

#Обучение + fit
from sklearn.linear_model import LogisticRegression
regr0 = LogisticRegression(random_state=0)
regr0.fit(X1_train, y1_train)
regr1 = LogisticRegression(random_state=0)
regr1.fit(X1_train, y1_train)
regr2 = LogisticRegression(random_state=0)
regr2.fit(X2_train, y2_train)

#Предсказания
y0_pred = regr0.predict(X0_test)
y1_pred = regr1.predict(X1_test)
y2_pred = regr2.predict(X2_test)

#Точность
score0 = regr0.score(X0_test, y0_test)
score1 = regr1.score(X1_test, y1_test)
score2 = regr2.score(X2_test, y2_test)

print(f'Точность для df1: {score1}')
print(f'Точность для df2: {score2}')

#Случайный датасет
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
print(f'Точность для df: {score0}')

# Отрисовка точек (разные цвета для разных классов)
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)
plt.title('Сгенерированный датасет для классификации', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regr = LogisticRegression(random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
score = regr.score(X_test, y_test)
print(f'Точность для случайного датасета: {score}')
