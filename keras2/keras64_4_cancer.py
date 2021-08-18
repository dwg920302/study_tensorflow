from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, GlobalAvgPool2D

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from icecream import ic


x, y = load_breast_cancer(return_X_y=True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=57)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

optimizer = Adam()

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(30, ), name='input')
    x = Dense(256, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(16, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    return model

def hyperparameter():
    batches = [1, 3, 5, 10, 25]
    optimizers = ['adam', 'rmsprop']
    # learning_rate = [0.1, 0.01, 0.001]
    dropout = [0, 0.125, 0.25, 0.375, 0.5]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}

hyperparameters = hyperparameter()
print(hyperparameters)
# model2 = build_model()

# tensorflow 모델을 그대로 넣을 순 없어서 변환함

model2 = KerasClassifier(build_fn=build_model, verbose=1)

model3 = GridSearchCV(model2, hyperparameters, cv=5)

model3.fit(x_train, y_train, verbose=1, epochs=5, validation_split=0.2, shuffle=True) 


# TF model to SK model (with Wrapping)

print(model3.best_params_)
print(model3.best_estimator_)
print(model3.best_score_)
acc = model3.score(x_test, y_test)
print("Final Score : ", acc)

'''
{'batch_size': 2, 'drop': 0.125, 'optimizer': 'rmsprop'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000014ECE072940>
0.894505500793457
57/57 [==============================] - 0s 2ms/step - loss: 0.5772 - acc: 0.6316
Final Score :  0.6315789222717285
'''
'''
{'batch_size': 1, 'drop': 0, 'optimizer': 'adam'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E2B48D94C0>
0.6417582631111145
114/114 [==============================] - 0s 2ms/step - loss: 0.2459 - acc: 0.5702    
Final Score :  0.5701754093170166
'''