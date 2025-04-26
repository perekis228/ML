import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Загрузка данных
data = pd.read_csv('diabetes.csv', encoding='utf-8')
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Качество от глубины
depths = range(1, 21)
f1_scores = []

for depth in depths:
    regr = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    f1_scores.append(f1)

plt.figure(figsize=(10, 5))
plt.plot(depths, f1_scores, marker='o', label='F1-score')
plt.xlabel('Глубина деревьев')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от глубины деревьев')
plt.grid(True)
plt.legend()
plt.show()
print('1)', sum(f1_scores)/len(f1_scores))

#Качество от количества признаков
import numpy as np

max_features_range = np.arange(1, X.shape[1] + 1)
f1_scores = []

for max_features in max_features_range:
    regr = RandomForestClassifier(max_features=max_features, n_estimators=100, random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    f1_scores.append(f1)

plt.figure(figsize=(10, 5))
plt.plot(max_features_range, f1_scores, marker='o', label='F1-score')
plt.xlabel('Количество признаков')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от количества признаков')
plt.grid(True)
plt.legend()
plt.show()
print('2)', sum(f1_scores)/len(f1_scores))


#Качество от числа деревьев
import time
n_estimators_range = [10, 50, 100, 150, 200, 300, 500]
f1_scores = []
train_times = []

for n in n_estimators_range:
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=n, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    end_time = time.time()
    train_time = end_time - start_time

    f1_scores.append(f1)
    train_times.append(train_time)

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, f1_scores, marker='o', label='F1-score')
plt.xlabel('Число деревьев')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от числа деревьев')
plt.grid(True)
plt.legend()
plt.show()
print('3)', sum(f1_scores)/len(f1_scores))
print('4)', sum(train_times)/len(train_times))

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, train_times, marker='o', color='red', label='Время обучения (сек)')
plt.xlabel('Число деревьев')
plt.ylabel('Время обучения (сек)')
plt.title('Зависимость времени обучения от числа деревьев')
plt.grid(True)
plt.legend()
plt.show()

'''
Средний F1_score в зависимости от: 
глубин               - 0.624
количества признаков - 0.694
числа деревьев       - 0.647

Среднее время на обучение моделей с разным числом деревьев:
0.318 сек
'''