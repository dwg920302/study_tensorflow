import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

dataset = load_iris()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

# One-Hot Encoding

# 0 → [1, 0, 0]

# 1 → [0, 1, 0]

# 2 → [0, 0, 1]

# [0, 1, 2, 1]

# → [ [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0] ]

# (4, ) → (4, 3)

print(x.shape, y.shape)  # (150, 4) (150,)

input_1 = Input(shape=(4, ))
dense_1 = Dense(8)(input_1)
dense_2 = Dense(150)(dense_1)
output_1 = Dense(3, activation='softmax')(dense_2)
model = Model(inputs = input_1, outputs = output_1)

y = to_categorical(y)
print(y[:5]) # [0, 0, 0, 0, 0] 이 [[1, 0. 0], [1, 0. 0], [1, 0. 0], [1, 0. 0], [1, 0. 0]] 가 되었음

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

es = EarlyStopping(monitor='loss', patience=25, mode='min', verbose=1)

# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=250, verbose=1, validation_split=1/4, shuffle=True, callbacks=[es])
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0], ', accuracy = ', loss[1])

y_pred = model.predict(x_test[-5:])
print(y_pred, '\n', y_test[-5:])

'''
[Best Fit]
With No Scaler, Tree 3 > 8 > 150 > 3(softmax)
monitor='loss', patience=50, mode='min'
batch_size=1, epochs=250

loss =  0.03411758318543434 , accuracy =  0.9666666388511658

loss =  0.020856058225035667 , accuracy =  1.0

[Better Fit]
With No Scaler, Tree 3 > 8 > 150(relu) > 3(softmax)
monitor='loss', patience=50, mode='min'
batch_size=1, epochs=250

loss =  0.09576485306024551 , accuracy =  0.9666666388511658
'''

'''
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica

    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

.. topic:: References

   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
'''