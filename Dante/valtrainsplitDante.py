import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_csv(file_to_split, ratio):
    data = pd.read_csv(file_to_split)
    X_train, X_val, y_train, y_test = train_test_split(data, data.Image,test_size=ratio)
    train = open("train.txt", "w+")
    val = open("val.txt", "w+")
    # pd.DataFrame(np.array(X_train)).to_csv("train.csv")
    # pd.DataFrame(np.array(X_val)).to_csv("val.csv")
    for i in range(len(np.array(X_train))):
        if np.array(X_train)[i][1] == 1:
            train.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            train.write(np.array(X_train)[i][0])
            train.write(" PW\n")
        if np.array(X_train)[i][2] == 1:
            train.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            train.write(np.array(X_train)[i][0])
            train.write(" EP\n")
        if np.array(X_train)[i][3] == 1:
            train.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            train.write(np.array(X_train)[i][0])
            train.write(" KW\n")
        if np.array(X_train)[i][4] == 1:
            train.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            train.write(np.array(X_train)[i][0])
            train.write(" NO\n")
    train.close()

    for i in range(len(np.array(X_val))):
        if np.array(X_val)[i][1] == 1:
            val.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            val.write(np.array(X_val)[i][0])
            val.write(" PW\n")
        if np.array(X_val)[i][2] == 1:
            val.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            val.write(np.array(X_val)[i][0])
            val.write(" EP\n")
        if np.array(X_val)[i][3] == 1:
            val.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            val.write(np.array(X_val)[i][0])
            val.write(" KW\n")
        if np.array(X_val)[i][4] == 1:
            val.write("C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/Images/")
            val.write(np.array(X_val)[i][0])
            val.write(" NO\n")
    val.close()
    return

print(split_csv("labels.csv", 0.3))
