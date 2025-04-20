import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    dataset = pd.read_csv('student_scores.csv')

    dataset.plot(x='Hours', y='Scores', style='o')
    plt.title('Hours vs Percentage')
    plt.xlabel('Hours Studied')
    plt.ylabel('Percentage Score')
    plt.show()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    print(regressor.intercept_)
    print(regressor.coef_)

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return y_test, y_pred

if __name__ == '__main__':
    main()