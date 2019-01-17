import numpy as np

def sigmoid_func(X):
    return 1 / (1 + np.exp(-X))

def add_bias(X):
    N = len(X[:,0])
    return np.c_[np.ones(N), X]

def one_layer_init(input_size, output_size):
    return np.random.uniform(-0.3, 0.3, (output_size, input_size+1))    

def compute_layer(A_j, Theta_j):
    Z_j = np.dot(A_j, Theta_j.T)
    return sigmoid_func(Z_j)

def one_layer_output(X, Theta_0):
    X_bias = add_bias(X)
    return compute_layer(X_bias, Theta_0)


def n_layer_init(*layer_sizes):
    Theta_array = []
    for i in range(len(layer_sizes) - 1):
        Theta_array.append(one_layer_init(layer_sizes[i], layer_sizes[i+1]))
    return Theta_array
                
def n_layer_output(X, Theta):
    for theta in Theta:
        X = one_layer_output(X, theta)
    return X

def cost_function(A, Y):
    cost = - np.sum((Y * np.log(A)) + (1-Y) * np.log(1-A))
    return cost

def output_delta(A, Y):
    return A-Y


def weight_update(A_j, Delta_next, Theta_j, rate):
    New_Theta = Theta_j - rate * np.dot(Delta_next.T, A_j)
    return New_Theta

def one_layer_training(X, Y, Theta_0, iters=1000, rate=0.9):
    errors = []
    Theta = Theta_0    
    for i in range(iters):
        Output = one_layer_output(X, Theta)

        Cost = cost_function(Output, Y)
        errors.append(Cost)
        Delta = output_delta(Output, Y)

        Theta = weight_update(add_bias(X), Delta, Theta, rate)
    return Theta


def hidden_delta(A_j, Delta_next, Theta_j):
    return (1 - A_j)*A_j*np.dot(Delta_next, Theta_j)


def two_layer_training(X, Y, Theta_0, Theta_1, iters=5000, rate=0.9):
    errors = []    
    for i in range(iters):
        Output_0 = one_layer_output(X, Theta_0)
        Output_1 = one_layer_output(Output_0, Theta_1)

        Cost = cost_function(Output_1, Y)
        errors.append(Cost)
        
        Delta_1 = output_delta(Output_1, Y)
        Delta_0 = hidden_delta(add_bias(Output_0), Delta_1, Theta_1)
        Delta_0 = Delta_0[:,1:]

        Theta_1 = weight_update(add_bias(Output_0), Delta_1, Theta_1, rate)
        Theta_0 = weight_update(add_bias(X), Delta_0, Theta_0, rate)
    return Theta_0, Theta_1


def one_layer_training_no_plot(X, Y, Theta_0, iters=1000, rate=0.9):
    Theta = Theta_0    
    for i in range(iters):
        Output = one_layer_output(X, Theta)
        Delta = output_delta(Output, Y)
        Theta = weight_update(add_bias(X), Delta, Theta, rate)
    return Theta

def two_layer_training_no_plot(X, Y, Theta_0, Theta_1, iters=5000, rate=0.9):
    for i in range(iters):
        Output_0 = one_layer_output(X, Theta_0)
        Output_1 = one_layer_output(Output_0, Theta_1)
        
        Delta_1 = output_delta(Output_1, Y)
        Delta_0 = hidden_delta(add_bias(Output_0), Delta_1, Theta_1)
        Delta_0 = Delta_0[:,1:]

        Theta_1 = weight_update(add_bias(Output_0), Delta_1, Theta_1, rate)
        Theta_0 = weight_update(add_bias(X), Delta_0, Theta_0, rate)
        
    return Theta_0, Theta_1


def validation_split(data, ratio):
    np.random.shuffle(data)
    split_index = int(round(ratio*len(data)))
    set1 = data[:split_index]
    set2 = data[split_index:]
    return set1,set2

# SCIKIT 

from sklearn.neural_network import MLPClassifier

# Create training and validation data
train, val = validation_split(digits, 0.7)
x_train, y_train_raw = x_y_split(train)
x_val, y_val_raw = x_y_split(val)
y_train = transform_y(y_train_raw)
y_val = transform_y(y_val_raw)

# Create MLPClassifier and calculate validation score
classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100), activation='relu', solver='adam', max_iter=5000, alpha=0.0001,learning_rate_init=0.001)
classifier.fit(x_train, y_train) 

prediction_train = classifier.predict(x_train)
prediction_val = classifier.predict(x_val)

print('Classifier training score:', validate(prediction_train, y_train))
print('Classifier validation score:', validate(prediction_val, y_val))