import pandas as pd
import numpy as np

def fix_y_test_arr(y_arr):
    df = pd.DataFrame(y_arr)
    df[1] = df[1].apply(lambda x: 2 if x == 1 else 0)
    df[2] = df[2].apply(lambda x: 3 if x == 1 else 0)
    df[3] = df[3].apply(lambda x: 4 if x == 1 else 0)
    df[4] = df[4].apply(lambda x: 5 if x == 1 else 0)
    arr = df.sum(axis = 1).astype(int).values
    return arr - 1

def accuracy(y_test, y_pred):
    correct = sum([1 for a, b in zip(y_test, y_pred) if a == b])
    return float(correct) / len(y_test)

def get_random_prob(y_test):
    random = np.random.randint(0, 5, len(y_test))
    return accuracy(y_test, random)
