from sklearn.neural_network import MLPClassifier
from jochem_preprocessing import *

classifier = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=5000, alpha=0.001,learning_rate_init=0.001)
# train_X = train_X.flatten()
train_X = train_X.reshape(1144, -1)

# valid_X = valid_X.flatten()
valid_X = valid_X.reshape(491, -1)

classifier.fit(train_X, train_Y)

prediction_train = classifier.predict_proba(train_X)
prediction_val = classifier.predict_proba(valid_X)

def transform_result(result):
    transformed_result = []
    for label in result:
        label = list(label)

        # if max(label) < 0.5:
        #     transformed_result.append([0, 0, 0, 0])
            
        i = label.index(max(label))
        if i == 0:
            transformed_result.append([1, 0, 0, 0])

        elif i == 1:
            transformed_result.append([0, 1, 0, 0])
        
        elif i == 2:
            transformed_result.append([0, 0, 1, 0])

        else:
            transformed_result.append([0, 0, 0, 1])
    return np.asarray(transformed_result)

result_val = transform_result(prediction_val)

def validate(result, labels):
    N = len(result)

    if N != len(labels):
        return 'Error: not equal length'
    classify_count = 0
    reject_count = 0

    final_N = N

    for i in range(N):
        if np.array_equal(result[i], labels[i]):
            classify_count+=1

        elif np.array_equal(result[i], [0, 0, 0, 0]):
            final_N -= 1
            reject_count += 1
    return (classify_count/final_N) * 100, reject_count

    
valscore, rejectscore = validate(result_val, valid_Y)
print(result_val)
print(valscore)
print(rejectscore)