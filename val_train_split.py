import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_csv(file_to_split, ratio):
    data = pd.read_csv(file_to_split)
    X_train, X_test, y_train, y_test = train_test_split(data, data.Image,test_size=ratio)
    # pd.DataFrame(np.array(X_train)).to_csv("train.csv")
    # pd.DataFrame(np.array(X_test)).to_csv("val.csv")
    return np.array(X_train), np.array(X_test)
# split_csv("labels.csv")