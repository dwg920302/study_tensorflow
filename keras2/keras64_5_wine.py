from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, GlobalAvgPool2D

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from icecream import ic
import pandas as pd


dataset = pd.read_csv('../_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

ic(dataset.shape, dataset.describe)

dataset = dataset.values

ic(type(dataset))

x = dataset[:, :11]
y = dataset[:, 11]

print(x.shape, y.shape)

ic(y)

y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

ic(y[:, :5], y[:, -5:])

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=57)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

optimizer = Adam(learning_rate=0.01)

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(11, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(7, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

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

'''