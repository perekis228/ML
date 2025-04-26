from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time

# Загрузка данных
data = pd.read_csv('diabetes.csv', encoding='utf-8')
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Качество от глубины
depths = range(1, 21)
f1_scores = []

for depth in depths:
    bst = XGBClassifier(n_estimators=100, max_depth=depth, learning_rate=0.1, objective='binary:logistic', random_state=0)
    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    f1_scores.append(f1)

plt.plot(depths, f1_scores, marker='o')
plt.xlabel('Глубина деревьев')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от глубины деревьев')
plt.grid()
plt.show()
print('1)', sum(f1_scores)/len(f1_scores))

#Качество от learning_rate
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
f1_scores = []

for lr in learning_rates:
    bst = XGBClassifier(n_estimators=100, max_depth=depth, learning_rate=lr, objective='binary:logistic', random_state=0)
    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    f1_scores.append(f1)

plt.plot(learning_rates, f1_scores, marker='o')
plt.xlabel('Learning_rate')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от learning_rate')
plt.grid()
plt.show()
print('2)', sum(f1_scores)/len(f1_scores))

#Качество от n_estimators
n_estimators_range = [10, 50, 100, 150, 200, 300, 500]
f1_scores = []
train_times = []

for ne in n_estimators_range:
    start_time = time.time()
    bst = XGBClassifier(n_estimators=ne, max_depth=depth, learning_rate=lr, objective='binary:logistic', random_state=0)
    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    end_time = time.time()
    train_time = end_time - start_time

    f1_scores.append(f1)
    train_times.append(train_time)

plt.plot(n_estimators_range, f1_scores, marker='o')
plt.xlabel('N_estimators')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от n_estimators')
plt.grid()
plt.show()
print('3)', sum(f1_scores)/len(f1_scores))
print('4)', sum(train_times)/len(train_times))

# График времени обучения
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
глубин            - 0.694
скорости обучения - 0.681
числа деревьев    - 0.661

Среднее время на обучение моделей с разным числом деревьев:
0.1 сек

                         Random Forest         XGBoost:
F1 от глубины                0.624              0.694
F1 от числа деревьев         0.647              0.661
Среднее время обучения     0.318 сек           0.1 сек

По всем показателям лучше XGBoost
'''