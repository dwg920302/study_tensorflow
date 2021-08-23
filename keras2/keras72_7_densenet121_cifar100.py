from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAvgPool2D
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])).reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
x_test = scaler.transform(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])).reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])


def model_1(pre_model):
        model = Sequential()
        model.add(pre_model)
        model.add(Flatten())
        model.add(Dropout(1/4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(1/4))
        model.add(Dense(100, activation='softmax'))
        return model

def model_2(pre_model):
        model = Sequential()
        model.add(pre_model)
        model.add(GlobalAvgPool2D())
        model.add(Dropout(1/4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(1/4))
        model.add(Dense(100, activation='softmax'))
        return model

trainables = [True, False]

model_names = [[model_1, 'Flatten'], [model_2, 'GlobalAvgPool']]

es = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

for trainable in trainables:
        for loop in model_names:
                model = loop[0]
                bc = loop[1]

                pre_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
                ad = ''
                
                if trainable == True:
                        pre_model.trainable = True
                        ad = 'Trainable'
                        model = model_1(pre_model)
                else:
                        pre_model.trainable = False
                        ad = 'Non-Trainable'
                        model = model_2(pre_model)

                model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
                model.fit(x_train, y_train, batch_size=128, epochs=25, verbose=1, validation_split=1/8, shuffle=True, callbacks=es)

                loss = model.evaluate(x_test, y_test)
                print('[Condition : ', ad, ' ', bc, ']')
                print('loss = ', loss[0])
                print('accuracy = ', loss[1])


'''
[Condition :  Trainable   Flatten ]
loss =  1.8178496360778809
accuracy =  0.520799994468689

[Condition :  Trainable   GlobalAvgPool ]
loss =  1.8612323999404907
accuracy =  0.5274999737739563

[Condition :  Non-Trainable   Flatten ]
loss =  2.4092280864715576
accuracy =  0.3801000118255615

[Condition :  Non-Trainable   GlobalAvgPool ]
loss =  2.405886173248291
accuracy =  0.3871999979019165
'''