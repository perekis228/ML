import pandas as pd

def main():
    df = pd.read_csv('Titanic.csv', encoding='utf-8') #Чтение
    start_size = df.shape[0]
    df = df.dropna() #Убираем NaN
    df = df.drop(columns=['Name', 'Cabin', 'Ticket']) #Убираем столбцы
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) #Переименовываем пол
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}) #Переименовывем порты посадки
    df = df.drop(columns=['PassengerId']) #Убираем PassengerId
    end_size = df.shape[0]
    print(f'{(1 - end_size / start_size) * 100:.3f}% данных потеряно.') #Процент потерянных данных

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    #Нормализация
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, df['Survived'], test_size=0.2, random_state=0)
    regr = LogisticRegression(random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score1 = regr.score(X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(
        df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']],
        df['Survived'],
        test_size=0.2,
        random_state=0)
    regr = LogisticRegression(random_state=0)
    regr.fit(X_train, y_train)
    y_pred2 = regr.predict(X_test)
    y_prob = regr.predict_proba(X_test)[:, 1]
    score2 = regr.score(X_test, y_test)

    print(f'Точность с Embarked: {score1*100:.3f}%\nТочность без Embarked: {score2*100:.3f}%\nРазница: {(score1 - score2)*100:.3f}%')

    return regr, X_test, y_test, X_train, y_train, y_pred, y_prob

if __name__ == '__main__':
    main()