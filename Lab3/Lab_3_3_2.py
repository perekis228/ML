import Lab_3_2_1
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


regr_lin, X_test, y_test, X_train, y_train, y_pred_lin, y_prob_lin = Lab_3_2_1.main()

#Метод опорных векторов (SVM)
regr_svm = SVC(kernel='rbf', probability=True, random_state=1)
regr_svm.fit(X_train, y_train)
y_pred_svm = regr_svm.predict(X_test)
y_prob_svm = regr_svm.predict_proba(X_test)[:, 1]

#K-ближайших соседей (KNN)
regr_knn = KNeighborsClassifier(n_neighbors=5)
regr_knn.fit(X_train, y_train)
y_pred_knn = regr_knn.predict(X_test)
y_prob_knn = regr_knn.predict_proba(X_test)[:, 1]

#Метрики
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y, y_pred, model):
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f'\nМетрики для {model}:')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1-score: {f1:.3f}')

evaluate_model(y_test, y_pred_lin, 'Logistic Regression')
evaluate_model(y_test, y_pred_svm, 'SVM')
evaluate_model(y_test, y_pred_knn, 'KNN')

#Матрицы ошибок
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def matrix(name, pred):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(f'Confusion Matrix ({name})')
    plt.show()

matrix('Logistic Regression', y_pred_lin)
matrix('SVM', y_pred_svm)
matrix('KNN', y_pred_knn)

#ROC-кривые
from sklearn.metrics import roc_curve, roc_auc_score
plt.figure(figsize=(10, 6))
plt.plot(*roc_curve(y_test, y_prob_lin)[:2], label=f'LR (AUC = {roc_auc_score(y_test, y_prob_lin):.3f})')
plt.plot(*roc_curve(y_test, y_prob_svm)[:2], label=f'SVM (AUC = {roc_auc_score(y_test, y_prob_svm):.3f})')
plt.plot(*roc_curve(y_test, y_prob_knn)[:2], label=f'KNN (AUC = {roc_auc_score(y_test, y_prob_knn):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

'''
Средняя оценка метрик:
LR:  0.898   |
SVM: 0.902   | => По метрикам лучше SVM
KNN: 0.853   |

Матрицы:
LR:  6/37=0.162   |
SVM: 7/37=0.189   | => По доли ошибок матриц лучше LR
KNN: 9/37=0.243   |

По AUC ROC-кривых:
LR:  0.905   |
SVM: 0.667   | => По AUC лучше LR, KNN ведёт себя на уровне случайного выбора
KNN: 0.500   |

Победитель: Logistic Regression!!!
'''