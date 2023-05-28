import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tkinter import Tk
from sklearn.impute import SimpleImputer
from tkinter.filedialog import askopenfilename


def load_and_correct_data(file_path):
    data=pd.read_csv(file_path)
    mapping = {}

    for column in data.columns:
        # Check if the column contains non-numeric values
        if data[column].dtype == object:
            # Create a mapping for the column
            unique_values = data[column].unique()
            mapping[column] = {value: i for i, value in enumerate(unique_values)}
            # Replace the non-numeric values with numeric values
            data[column] = data[column].map(mapping[column])
        else:
            imputer = SimpleImputer(strategy="mean")  # or "median", "most_frequent", etc.
            data[column] = imputer.fit_transform(data[[column]])

    return data,mapping


def separate(data):
    # separate the training data from the target data

    x=data.drop(columns=data.columns[data.shape[1]-1])
    y=data[data.columns[data.shape[1]-1]]

    return x,y

def divide_data(x,y,test_percent):
    x_learn, x_test, y_learn, y_test = train_test_split(x, y, test_size=test_percent,
                                                        random_state=0)  # here we made the test data set is 30% of the original data
    return x_learn,x_test,y_learn,y_test

def make_it_learn(x_learn,y_learn,lr):
    lr.fit(x_learn,y_learn)
    c=lr.intercept_
    thetas=lr.coef_
    return c,thetas

def prediction_equation(c,thetas):

    equation = str(c)
    for i, theta in enumerate(thetas):
        equation += f" + {theta}*X{i+1}"
    return equation

def accuracy(lr,x_test,y_test):
#now we want to check wether it's predicting correct or not
    y_predict=lr.predict(x_test)

    plt.scatter(y_test,y_predict)
    plt.xlabel("Actual Data")
    plt.ylabel("Predicted Data from testing data")

    plt.show()

    Acc=r2_score(y_test,y_predict)*100

    return Acc

def Check(data,c,thetas,mapping):
    p = int(input("Would you like to check? (Enter 1 for yes): "))
    if p == 1:
        print("Mapping information:")
        for column, mapping_info in mapping.items():
            print(f"{column}:")
            for value, code in mapping_info.items():
                print(f"{value} -> {code}")
            print()
        a = []
        for column in data.columns[:-1]:
            g = float(input(f"Enter the value of {column}: "))
            a.append(g)
        total_sum = c

        for i in range(len(thetas)):
            total_sum += a[i] * thetas[i]
        return total_sum

##############################################################################################


root = Tk()

root.withdraw()

file_path = askopenfilename(initialdir='Desktop/', title='Select a csv  File')

corrected_data,mapping=load_and_correct_data(file_path)

x,y=separate(corrected_data)

# dividing the data in 2 groups , one for learning and the other for testing
test_percent=float(input('how many percent would u like to be the test data be from the original data ?n\n\t\trange(0->1) : '))

x_learn,x_test,y_learn,y_test=divide_data(x,y,test_percent)

lr=LinearRegression()

constant,coefficients=make_it_learn(x_learn,y_learn,lr)

print(prediction_equation(constant,coefficients))

Accuracy=accuracy(lr,x_test,y_test)

print("Accuracy = ",Accuracy)

print(corrected_data)

print(Check(corrected_data,constant,coefficients,mapping))



