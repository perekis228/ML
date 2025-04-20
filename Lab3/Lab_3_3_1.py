from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import Lab_3_2_1

regr, X_test, y_test, _, _, y_pred, _ = Lab_3_2_1.main()

#Вычисление метрик
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}')

#Матрица ошибок
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Кривая Precision-Recall
y_proba = regr.predict_proba(X_test)[:, 1]
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall_curve, precision_curve, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()

#Кривая ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.show()

'''
Precision: 0.873
Recall: 0.865
F1: 0.868
Метрики между 0.8 и 0.9 => очень хороший результат

По матрице ошибок 5/37 = 13.5% угадано неверно, близко к 10%  => очень хороший результат

Кривая PR близка к (1,1) => близка к отличному результату

Кривая ROC близка к (0,1) => близка к отличному результату
'''