# 실습 : CNN으로 바꾸기
# Parameter 변경해보기 (Node의 개수, activation, learning_rate)
# epochs = [1, 2, 3]

# 추후에, 레이어로 파라미터로 만들어볼 것 (Dense, Conv)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, GlobalAvgPool2D

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_test.shape)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

optimizer = Adam()

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(512, kernel_size=(1, 1), padding='same', activation='relu', name='D1-CNN')(inputs) # (N, 4, 4, 10)
    x = Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu', name='D2-CNN')(x)
    x = GlobalAvgPool2D()(x)
    x = Flatten()(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='D3')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu', name='D4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    return model

def hyperparameter():
    batches = [5, 10, 15, 20, 25]
    optimizers = ['adam', 'rmsprop', 'adadelta']
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

model3.fit(x_train, y_train, verbose=1, epochs=5, validation_split=0.2) 


# TF model to SK model (with Wrapping)

print(model3.best_params_)
print(model3.best_estimator_)
print(model3.best_score_)
acc = model3.score(x_test, y_test)
print("Final Score : ", acc)