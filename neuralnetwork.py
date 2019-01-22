import leren
import preprocessing
import val_train_split



training_set, validation_set = val_train_split.split_csv("labels.csv", 0.2)
train_data_x = []
train_category = []
for row in training_set:
    image_matrix = preprocessing.image_to_matrix(row[0]).tolist()
    merged_list = []
    for l in image_matrix:
        merged_list += l
    train_data_x.append(merged_list)
    train_category.append(row[1:5])

val_data_x = []
val_category = []
for row in validation_set:
    image_matrix = preprocessing.image_to_matrix(row[0]).tolist()
    merged_list = []
    for l in image_matrix:
        merged_list += l
    val_data_x.append(merged_list)
    val_category.append(row[1:5])

    # print(leren.n_layer_output(image_matrix, leren.n_layer_init(1920,1)))


# y_train = leren.transform_y(y_train_raw)
# y_val = leren.transform_y(y_val_raw)

Theta_one = leren.one_layer_init(1920, 4)
Theta_one_learned = leren.one_layer_training_no_plot(train_data_x, train_category, Theta_one, iters=1000, rate=0.7)
Result_one = leren.one_layer_output(val_data_x, Theta_one_learned)


