import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

data = pd.read_csv("data.csv")

def linear_regression(alpha, iteration):
    theta = [0, 0]
    a = alpha
    n = data.shape[0]
    for k in range(iteration):
        i = 0
        sum1 = 0
        sum2 = 0
        while(i < n):
            sum1 += theta[0] + theta[1] * data["km_transform"][i] - data["price_transform"][i]
            sum2 += (theta[0] + theta[1] * data["km_transform"][i] - data["price_transform"][i]) * data["km_transform"][i]
            i += 1
        theta[0] -= (a / n) * sum1
        theta[1] -= (a / n) * sum2
    return theta

def J(theta):
    n = data.shape[0]
    sum = 0
    for i in range(n):
        sum += (theta[0] + theta[1] * data["km_transform"][i] - data["price_transform"][i]) ** 2
    return (1 / (2 * n)) * sum

def plot_data(a, theta):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # plt.figure(1)
    axs[0].plot(data["km"], data["price"], 'ro', markersize=4)
    axs[0].set_title("Car price vs kilometers")

    # plt.figure(2)
    axs[1].plot(data.km_transform, data.price_transform, 'ro', markersize=4)
    axs[1].plot([0, 1], [theta[0], theta[0] + theta[1]], linestyle='--')
    axs[1].set_title("Normalized datas and linear regression")

    # plt.figure(3)
    axs[2].plot(J_epoch_arr(a, 50)[0], J_epoch_arr(a, 50)[1], '-')
    axs[2].set_title("Cost evolution")
    plt.tight_layout()
    plt.show()

def print_data(a, theta):
    print("Dataframe km vs price:\n", data)
    print("\n\nOptimal learning rate: ", a)
    print("\n\nNormalized Linear Regression Parameters θ: ", theta)
    print("\n\n")

def J_epoch_arr(a, iter):
    array = np.empty((2, iter))
    for i in range(iter):
        array[0][i] = i
        array[1][i] = J(linear_regression(a, i))
    return array

def find_best_learning_rate():
    Jlim = 1
    alpha = np.arange(1.3, 1.7, 0.005)
    best_a = 1.3
    for a in alpha:
        array = J_epoch_arr(a, 11)
        if (array[1][10] < Jlim):
            Jlim = array[1][10]
            best_a = a
    return best_a


if __name__ == '__main__':
    data["km_transform"] = data["km"].apply(lambda x: normalize(x, data["km"].min(), data["km"].max()))
    data["price_transform"] = data["price"].apply(lambda x: normalize(x, data["price"].min(), data["price"].max()))
    a = find_best_learning_rate()
    theta = linear_regression(a, 50)
    with open('theta.csv', "w") as file:
        file.write(f"{theta[1]},{theta[0]}")
    print_data(a, theta)
    plot_data(a, theta)

