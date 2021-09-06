import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def RandomForest():
    dataset = pd.read_csv('Position_Salaries.csv')
    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())
    plt.scatter(dataset['Level'], dataset['Salary'], color='pink')
    plt.show()
    x = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values


    # Training the Random Forest Regression model on the whole dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(x, y)

    # Predicting a new result
    regressor.predict([[6.5]])

    # Visualising the Random Forest Regression results(higher resolution)
    x_grid = np.arange(min(x), max(x), 0.01)
    x_grid = x_grid.reshape((len(x_grid), 1))
    plt.scatter(x, y, color='red')
    plt.plot(x_grid, regressor.predict(x_grid), color='black')
    plt.title('Truth or Bluff (Random Forest Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()



if __name__ == '__main__':
    RandomForest()

