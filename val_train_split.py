import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_csv(file_to_split):
    data = pd.read_csv("labels.csv")
    X_train, X_test, y_train, y_test = train_test_split(data, data.Image,test_size=0.2)
    pd.DataFrame(np.array(X_train)).to_csv("train.csv")
    pd.DataFrame(np.array(X_test)).to_csv("val.csv")
