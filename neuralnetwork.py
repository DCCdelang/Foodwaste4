import leren
import preprocessing
import val_train_split

training_set, validation_set = val_train_split.split_csv("labels.csv", 0.2)
for row in training_set:
    image_matrix = preprocessing.image_to_matrix(row[0])
    
    print(leren.n_layer_output(image_matrix, leren.n_layer_init(1920,1)))

x_train, y_train_raw = x_y_split(training_set)
y_train = transform_y(y_train_raw)