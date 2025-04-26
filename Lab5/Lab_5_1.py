import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

#Загрузка данных
data = pd.read_csv('diabetes.csv', encoding='utf-8')

#Деление на тестовые и обучающие
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Обучение моделей
rerg_lg = LogisticRegression(max_iter=1000)
rerg_lg.fit(X_train, y_train)
y_pred_log = rerg_lg.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

#Метрики
def metrics(y_true, y_pred, model_name):
    print(f'\nМетрики для модели {model_name}:')
    print(f'Precision: {precision_score(y_true, y_pred):.3f}')
    print(f'Recall: {recall_score(y_true, y_pred):.3f}')
    print(f'F1-score: {f1_score(y_true, y_pred):.3f}')

metrics(y_test, y_pred_log, "Logistic Regression")
metrics(y_test, y_pred_tree, "Decision Tree")

'''
Средние баллы для:
LG:   0.608  |
Tree: 0.597  | => LG лучше подходит для данного датасета
'''

#F1 в зависимости от глубины дерева (т.к. F1 является балансированной мерой между precision и recall)
import matplotlib.pyplot as plt

f1_scores = []

#Проходим по глубинам 1-21
for depth in range(1, 21):
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

#Отрисовка графика
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), f1_scores, marker='o', linestyle='-', color='b')
plt.title("Зависимость F1 от глубины дерева")
plt.xlabel("Глубина дерева")
plt.ylabel("F1")
plt.grid(True)
plt.xticks(range(1, 21))
plt.show()

# Оптимальная глубина
optimal_depth = f1_scores.index(max(f1_scores)) + 1
print(f'\nОптимальная глубина дерева: {optimal_depth} (F1 = {max(f1_scores):.3f})')

#Дерево с оптимальной глубиной
tree = DecisionTreeClassifier(max_depth=optimal_depth)
tree.fit(X_train, y_train)

#Отрисовка дерева (через graphviz не получилось)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["No Diabetes", "Diabetes"],
    filled=True,
    rounded=True
)
plt.show()

#Важность признаков
feature_importances = pd.Series(tree.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.sort_values().plot(kind='barh', color='skyblue')
plt.title("Важность признаков (Feature Importances)")
plt.xlabel("Важность")
plt.ylabel("Признак")
plt.grid(True, axis='x')
plt.show()

#PR-кривая
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
y_proba = tree.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(tree, X_test, y_test)
plt.title(f"PR-кривая (AUC = {pr_auc:.3f})")
plt.grid(True)
plt.show()

# 4. ROC-кривая (Receiver Operating Characteristic)
from sklearn.metrics import roc_curve, RocCurveDisplay
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(tree, X_test, y_test)
plt.title(f"ROC-кривая (AUC = {roc_auc:.3f})")
plt.grid(True)
plt.show()