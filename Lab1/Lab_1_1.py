import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    #Чтение
    file = pd.read_csv('student_scores.csv', encoding='utf-8')
    columns = file.columns.tolist()

    #Выбор столбцов
    print(f'X: {columns[0]}; Y: {columns[1]}\nПоменять местами(y/n)?')
    if input() == 'y':
        columns[0], columns[1] = columns[1], columns[0]

    #Данные по столбцам
    for column in columns:
        print(f"Для '{column}':")
        print(f'Количество строк: {file.shape[0]}')
        print(f'Сумма: {file[column].sum()}')
        print(f'Минимум: {file[column].min()}')
        print(f'Максимум: {file[column].max()}')
        print(f'Среднее: {file[column].mean()}\n')

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

    #Квадраты ошибок
    for xi, yi in zip(file[columns[0]], file[columns[1]]):
        yi_pred = a * xi + b
        left = xi - 0.07
        right = xi + 0.07
        bottom = min(yi, yi_pred)
        top = max(yi, yi_pred)

        plt.fill_between([left, right], [bottom, bottom], [top, top],
                         color='gray', alpha=0.3, edgecolor='black')

    #Отрисовка графика
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()

    return file[columns[1]], [a*i+b for i in file[columns[0]]]

if __name__ == '__main__':
    main()