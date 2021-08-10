# Machine Learning #2 [machine_learning model with iris_dataset]

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]

# Model
model = LinearSVC()

# compile 없음
model.fit(x, y)

y_pred = model.predict(x)
print(x, '의 예측 결과 : ', y_pred)

acc = accuracy_score(y, y_pred)
print('accuracy_score : ', acc)