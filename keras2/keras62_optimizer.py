import numpy as np
from icecream import ic

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam


# Data

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Model

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# Compile

# optimizer = Adam(lr=0.1)
optimizer = SGD(lr=0.0001)   # 0.001를 넘는 lr에서는 항상 nan이 터짐

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

# Eval and Pred

loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
ic(loss, y_pred)

'''
[optimizer = Adam] -> default lr 0.001
lr 0.1
ic| loss: 5.082046072857338e-07
    y_pred: array([[10.999145]], dtype=float32)
ic| loss: 5.200785744818859e-05
    y_pred: array([[10.986877]], dtype=float32)
ic| loss: 231.5110626220703
    y_pred: array([[12.707781]], dtype=float32)

lr 0.01
ic| loss: 0.01597857102751732
    y_pred: array([[10.743713]], dtype=float32)
ic| loss: 9.521571309178967e-10
    y_pred: array([[10.999957]], dtype=float32)
ic| loss: 2.0406788238391815e-12
    y_pred: array([[11.000002]], dtype=float32)

lr 0.001
ic| loss: 0.0007016159361228347
    y_pred: array([[10.950053]], dtype=float32)
ic| loss: 1.4210854715202004e-14
    y_pred: array([[10.999999]], dtype=float32)
ic| loss: 0.0014828576240688562
    y_pred: array([[11.067769]], dtype=float32)

lr 0.0001
ic| loss: 3.217468110960908e-05
    y_pred: array([[10.994981]], dtype=float32)
ic| loss: 4.424630333232926e-06
    y_pred: array([[11.002692]], dtype=float32)
ic| loss: 1.1386092637621914e-06
    y_pred: array([[10.997826]], dtype=float32)
'''

'''
[Optimizer = Adagrad] -> default lr 0.001
lr 0.1
ic| loss: 2.5168557167053223
    y_pred: array([[12.524421]], dtype=float32)
ic| loss: 40.9775276184082
    y_pred: array([[2.4355173]], dtype=float32)
ic| loss: 3.9339919090270996
    y_pred: array([[14.503111]], dtype=float32)

lr 0.01
ic| loss: 3.2474781619384885e-05
    y_pred: array([[10.987723]], dtype=float32)
ic| loss: 4.645104745293338e-09
    y_pred: array([[11.000143]], dtype=float32)
ic| loss: 7.055060535776647e-08
    y_pred: array([[11.00024]], dtype=float32)

lr 0.001
ic| loss: 9.240956569556147e-05
    y_pred: array([[10.998308]], dtype=float32)
ic| loss: 2.0065828721271828e-05
    y_pred: array([[10.9930315]], dtype=float32)
ic| loss: 2.6965852157445624e-05
    y_pred: array([[10.990601]], dtype=float32)

lr 0.0001
ic| loss: 0.004032640252262354
    y_pred: array([[10.920678]], dtype=float32)
ic| loss: 0.00463426299393177
    y_pred: array([[10.914501]], dtype=float32)
ic| loss: 0.004656040109694004
    y_pred: array([[10.914832]], dtype=float32)
'''

'''
[Adadelta] -> default lr=0.001
lr 0.1
ic| loss: 2.810928094731935e-07
    y_pred: array([[10.999031]], dtype=float32)
ic| loss: 0.036973677575588226
    y_pred: array([[11.33062]], dtype=float32)
ic| loss: 0.0014602572191506624
    y_pred: array([[11.053967]], dtype=float32)

lr 0.01
ic| loss: 4.4962300307815894e-05
    y_pred: array([[10.986486]], dtype=float32)
ic| loss: 4.377225195639767e-06
    y_pred: array([[11.001489]], dtype=float32)
ic| loss: 0.0008331878343597054
    y_pred: array([[10.942026]], dtype=float32)

lr 0.001
ic| loss: 3.1269683837890625
    y_pred: array([[7.789152]], dtype=float32)
ic| loss: 10.930150032043457
    y_pred: array([[5.0800962]], dtype=float32)
ic| loss: 5.661940574645996
    y_pred: array([[6.721482]], dtype=float32)

lr 0.0001
ic| loss: 34.67642593383789
    y_pred: array([[0.5540937]], dtype=float32)
ic| loss: 21.811803817749023
    y_pred: array([[2.7147257]], dtype=float32)
ic| loss: 40.78833770751953
    y_pred: array([[-0.32800454]], dtype=float32)

[Adamax] -> default lr = 0.001

lr 0.1
ic| loss: 0.05630273371934891
    y_pred: array([[10.626773]], dtype=float32)
ic| loss: 4.890679150548749e-08
    y_pred: array([[10.999659]], dtype=float32)
ic| loss: 164.33335876464844
    y_pred: array([[23.096083]], dtype=float32)

lr 0.01
ic| loss: 4.2099657093785936e-13
    y_pred: array([[11.]], dtype=float32)
ic| loss: 2.9178437010307645e-12
    y_pred: array([[11.000003]], dtype=float32)
ic| loss: 8.300559892204795e-12
    y_pred: array([[11.000005]], dtype=float32)

lr 0.001
ic| loss: 8.684083923071739e-07
    y_pred: array([[11.000204]], dtype=float32)
ic| loss: 6.853492884317802e-09
    y_pred: array([[11.000026]], dtype=float32)
ic| loss: 1.0830251362392573e-08
    y_pred: array([[10.999941]], dtype=float32)

lr 0.0001
ic| loss: 0.0031456693541258574
    y_pred: array([[10.930381]], dtype=float32)
ic| loss: 0.0014535828959196806
    y_pred: array([[10.954727]], dtype=float32)
ic| loss: 0.0019620952662080526
    y_pred: array([[10.942871]], dtype=float32)
'''

'''
[Nadam] -> default lr 0.001
lr 0.1
ic| loss: 1.2866825827018147e-08
    y_pred: array([[11.000149]], dtype=float32)
ic| loss: 55.61162185668945
    y_pred: array([[-0.00574417]], dtype=float32)
ic| loss: 2901031.5, y_pred: array([[3528.7524]], dtype=float32)

lr 0.01
ic| loss: 10843.0107421875
    y_pred: array([[-182.25464]], dtype=float32)
ic| loss: 117552.703125, y_pred: array([[682.89764]], dtype=float32)
ic| loss: 1.2850087841798086e-05
    y_pred: array([[11.005852]], dtype=float32)

lr 0.001
ic| loss: 3.365925635989697e-07
    y_pred: array([[11.001055]], dtype=float32)
ic| loss: 2.589928326055674e-13, y_pred: array([[11.]], dtype=float32)
ic| loss: 2.912821628342499e-06
    y_pred: array([[10.996878]], dtype=float32)

lr 0.0001
ic| loss: 1.5097667528607417e-05
    y_pred: array([[10.998835]], dtype=float32)
ic| loss: 2.7285659598419443e-05
    y_pred: array([[11.00135]], dtype=float32)
ic| loss: 2.217790824943222e-05
    y_pred: array([[10.992648]], dtype=float32)
'''

'''
[RMSProp] -> default lr 0.001
lr 0.1
ic| loss: 100294.7109375, y_pred: array([[507.04208]], dtype=float32)
ic| loss: 4214736128.0, y_pred: array([[29515.129]], dtype=float32)
ic| loss: 1996334848.0, y_pred: array([[-73657.805]], dtype=float32)


lr 0.01
ic| loss: 149.4123992919922
    y_pred: array([[-10.689501]], dtype=float32)
ic| loss: 7.880348205566406
    y_pred: array([[4.9361577]], dtype=float32)
ic| loss: 1520.047119140625
    y_pred: array([[78.55736]], dtype=float32)

lr 0.001
ic| loss: 0.017665155231952667
    y_pred: array([[11.285364]], dtype=float32)
ic| loss: 0.00031435166602022946
    y_pred: array([[11.007831]], dtype=float32)
ic| loss: 0.025428276509046555
    y_pred: array([[10.765632]], dtype=float32)

lr 0.0001
ic| loss: 4.058228660142049e-05
    y_pred: array([[11.013101]], dtype=float32)
ic| loss: 0.00021863782603759319
    y_pred: array([[10.9727125]], dtype=float32)
ic| loss: 4.069809801876545e-05
    y_pred: array([[11.011837]], dtype=float32)
'''

'''
[SGD] -> default lr 0.001
lr 0.1
ic| loss: nan, y_pred: array([[nan]], dtype=float32)

lr 0.01
ic| loss: nan, y_pred: array([[nan]], dtype=float32)

lr 0.001
ic| loss: 1.7534152618736698e-07
    y_pred: array([[11.000059]], dtype=float32)
ic| loss: 1.0626204272057294e-07
    y_pred: array([[10.999451]], dtype=float32)
ic| loss: 1.3224024542068946e-06
    y_pred: array([[10.998941]], dtype=float32)

lr 0.0001
ic| loss: 0.0012084818445146084
    y_pred: array([[10.969496]], dtype=float32)
ic| loss: 0.0016384398331865668
    y_pred: array([[10.949491]], dtype=float32)
ic| loss: 0.0013355419505387545
    y_pred: array([[10.954828]], dtype=float32)
'''