import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def main():
    #Чтение
    data = datasets.load_diabetes()
    file = pd.DataFrame({
        'bmi': data.data[:, 2],
        'target': data.target
    })
    columns = file.columns.tolist()

    #Отрисовка точек
    plt.figure(figsize=(8, 6))
    plt.scatter(file[columns[0]], file[columns[1]], color='red', marker='o')

    #МНК
    n = file.shape[0]
    sum_x = file[columns[0]].sum()
    sum_y = file[columns[1]].sum()
    sum_xy = sum([file.iloc[i, 0] * file.iloc[i, 1] for i in range(n)])
    sum_x2 = sum([file.iloc[i, 0]**2 for i in range(n)])
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    #Отрисовка регрессии
    x_values = np.linspace(min(file[columns[0]]), max(file[columns[0]]), 100)
    y_values = a * x_values + b
    plt.plot(x_values, y_values, color='blue', label=f'Регрессия: y = {a:.2f}x + {b:.2f}')

    #Отрисовка графика
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()

    #Таблица
    y_pred = [a * file['bmi'][i] + b for i in range(file.shape[0])]
    res = pd.DataFrame({
        'bmi': file['bmi'],
        'y': file['target'],
        'y_pred': y_pred,
        'y-y_pred': file['target'] - y_pred
    })
    print(res.head())

    return file[columns[1]], [a*i+b for i in file[columns[0]]]

if __name__ == '__main__':
    main()