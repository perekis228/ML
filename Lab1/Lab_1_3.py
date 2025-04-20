from Lab1 import Lab_1_2_ipynb, Lab_1_2
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def metrics(func):
    def mean_absolute_percentage_error(y, y_pred):
        return np.mean(np.abs((y - y_pred) / y)) * 100

    y, y_pred = func

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)

    print(f'MAE: {mae}\nR2: {r2}\nMAPE: {mape}')

metrics(Lab_1_2_ipynb.main())
metrics(Lab_1_2.main())

#Для 1.2_ipynb: R2 = 0.945, очень близко к 1 - отличный результат
#               MAPE = 12.569, больше 10%, но меньше 20%, не отличный, но очень хороший результат
#Для 1.2:       R2 = 0.344, меньше 0.5 - средний результат, но близок к плохому
#               MAPE = 47.687, почти 50% - средний результат, но близок к плохому