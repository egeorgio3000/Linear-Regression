import pandas as pd
import sys

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def denormalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min

data = pd.read_csv("data.csv")

def calculate_predicted_price(mileage):
    normalized_mileage = normalize(mileage, data["km"].min(), data["km"].max())
    normalized_predicted_price = theta[0] + theta[1] * normalized_mileage
    predicted_price = denormalize(normalized_predicted_price, data["price"].min(), data["price"].max())
    return predicted_price


if __name__ == '__main__':
    try:
        theta = pd.read_csv('theta.csv', header=None)
    except Exception as e:
        print(f"{e}: please regenerate csv file")
        sys.exit(1)
    mileage = input("This program predict the price for a given mileage\n\nPlease provide a mileage: ")
    while (not mileage.isdigit()):
        mileage = input("Wrong entry, please provide a positive integer for mileage: ")

    theta = theta.iloc[0]
    try:
        theta[0] = float(theta[0])
        theta[1] = float(theta[1])
    except Exception as e:
        print(f"{e}: please regenerate csv file")
        exit(1)
    predicted_price = calculate_predicted_price(float(mileage))
    print(f"\nThe predicted price for your mileage is {predicted_price} euros")
    
    