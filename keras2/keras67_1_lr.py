weight = 0.5
in_put = 0.5
goal_prediction = 0.8
# lr = 0.1
lr = 0.1
# lr = 0.001
epochs = 1000

# 구동되는 원리 (추정)

for iteration in range(epochs):
  prediction = in_put * weight
  error = (prediction - goal_prediction) ** 2   # mean squared error!

  print(f"[Epoch {iteration+1}] Error : " + str(error) + "\tPrediction : " + str(prediction))

  up_prediction = in_put * (weight + lr)
  up_error = (goal_prediction - up_prediction) ** 2
  
  down_prediction = in_put * (weight - lr)
  down_error = (goal_prediction - down_prediction) ** 2

  if(down_error < up_error) : 
    weight = weight - lr
  if(down_error > up_error) : 
    weight = weight + lr


'''
(lr 0.1)
[Epoch 1] Error : 0.30250000000000005   Prediction : 0.25
[Epoch 2] Error : 0.25  Prediction : 0.3
[Epoch 3] Error : 0.20250000000000007   Prediction : 0.35
[Epoch 4] Error : 0.16000000000000006   Prediction : 0.39999999999999997
[Epoch 5] Error : 0.12250000000000007   Prediction : 0.44999999999999996
[Epoch 6] Error : 0.09000000000000007   Prediction : 0.49999999999999994
[Epoch 7] Error : 0.06250000000000006   Prediction : 0.5499999999999999
[Epoch 8] Error : 0.04000000000000003   Prediction : 0.6
[Epoch 9] Error : 0.022500000000000006  Prediction : 0.65
[Epoch 10] Error : 0.009999999999999995 Prediction : 0.7000000000000001
[Epoch 11] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 12] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 13] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 14] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 15] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 16] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 17] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 18] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 19] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 20] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 21] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 22] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 23] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 24] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 25] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 26] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 27] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 28] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002
[Epoch 29] Error : 0.0024999999999999935        Prediction : 0.7500000000000001
[Epoch 30] Error : 1.232595164407831e-32        Prediction : 0.8000000000000002

(lr 0.01)
[Epoch 1] Error : 0.30250000000000005   Prediction : 0.25
[Epoch 2] Error : 0.29702500000000004   Prediction : 0.255
[Epoch 3] Error : 0.2916        Prediction : 0.26
[Epoch 4] Error : 0.286225      Prediction : 0.265
[Epoch 5] Error : 0.28090000000000004   Prediction : 0.27
[Epoch 6] Error : 0.275625      Prediction : 0.275
[Epoch 7] Error : 0.27040000000000003   Prediction : 0.28
[Epoch 8] Error : 0.265225      Prediction : 0.28500000000000003
[Epoch 9] Error : 0.2601        Prediction : 0.29000000000000004
[Epoch 10] Error : 0.255025     Prediction : 0.29500000000000004
[Epoch 11] Error : 0.25 Prediction : 0.30000000000000004
[Epoch 12] Error : 0.245025     Prediction : 0.30500000000000005
[Epoch 13] Error : 0.24009999999999998  Prediction : 0.31000000000000005
[Epoch 14] Error : 0.235225     Prediction : 0.31500000000000006
[Epoch 15] Error : 0.2304       Prediction : 0.32000000000000006
[Epoch 16] Error : 0.225625     Prediction : 0.32500000000000007
[Epoch 17] Error : 0.22089999999999999  Prediction : 0.33000000000000007
[Epoch 18] Error : 0.21622499999999997  Prediction : 0.3350000000000001
[Epoch 19] Error : 0.21159999999999995  Prediction : 0.3400000000000001
[Epoch 20] Error : 0.20702499999999996  Prediction : 0.3450000000000001
[Epoch 21] Error : 0.20249999999999996  Prediction : 0.3500000000000001
[Epoch 22] Error : 0.19802499999999995  Prediction : 0.3550000000000001
[Epoch 23] Error : 0.19359999999999997  Prediction : 0.3600000000000001
[Epoch 24] Error : 0.18922499999999995  Prediction : 0.3650000000000001
[Epoch 25] Error : 0.18489999999999995  Prediction : 0.3700000000000001
[Epoch 26] Error : 0.18062499999999995  Prediction : 0.3750000000000001
[Epoch 27] Error : 0.17639999999999995  Prediction : 0.3800000000000001
[Epoch 28] Error : 0.17222499999999993  Prediction : 0.3850000000000001
[Epoch 29] Error : 0.16809999999999994  Prediction : 0.3900000000000001
[Epoch 30] Error : 0.16402499999999992  Prediction : 0.39500000000000013
[Epoch 31] Error : 0.15999999999999992  Prediction : 0.40000000000000013
[Epoch 32] Error : 0.1560249999999999   Prediction : 0.40500000000000014
[Epoch 33] Error : 0.15209999999999993  Prediction : 0.41000000000000014
[Epoch 34] Error : 0.1482249999999999   Prediction : 0.41500000000000015
[Epoch 35] Error : 0.14439999999999992  Prediction : 0.42000000000000015
[Epoch 36] Error : 0.14062499999999992  Prediction : 0.42500000000000016
[Epoch 37] Error : 0.1368999999999999   Prediction : 0.43000000000000016
[Epoch 38] Error : 0.1332249999999999   Prediction : 0.43500000000000016
[Epoch 39] Error : 0.1295999999999999   Prediction : 0.44000000000000017
[Epoch 40] Error : 0.12602499999999991  Prediction : 0.4450000000000002
[Epoch 41] Error : 0.1224999999999999   Prediction : 0.4500000000000002
[Epoch 42] Error : 0.11902499999999991  Prediction : 0.4550000000000002
[Epoch 43] Error : 0.1155999999999999   Prediction : 0.4600000000000002
[Epoch 44] Error : 0.11222499999999991  Prediction : 0.4650000000000002
[Epoch 45] Error : 0.1088999999999999   Prediction : 0.4700000000000002
[Epoch 46] Error : 0.1056249999999999   Prediction : 0.4750000000000002
[Epoch 47] Error : 0.1023999999999999   Prediction : 0.4800000000000002
[Epoch 48] Error : 0.0992249999999999   Prediction : 0.4850000000000002
[Epoch 49] Error : 0.0960999999999999   Prediction : 0.4900000000000002
[Epoch 50] Error : 0.0930249999999999   Prediction : 0.4950000000000002
[Epoch 51] Error : 0.0899999999999999   Prediction : 0.5000000000000002
[Epoch 52] Error : 0.0870249999999999   Prediction : 0.5050000000000002
[Epoch 53] Error : 0.0840999999999999   Prediction : 0.5100000000000002
[Epoch 54] Error : 0.0812249999999999   Prediction : 0.5150000000000002
[Epoch 55] Error : 0.07839999999999989  Prediction : 0.5200000000000002
[Epoch 56] Error : 0.07562499999999989  Prediction : 0.5250000000000002
[Epoch 57] Error : 0.0728999999999999   Prediction : 0.5300000000000002
[Epoch 58] Error : 0.07022499999999988  Prediction : 0.5350000000000003
[Epoch 59] Error : 0.06759999999999988  Prediction : 0.5400000000000003
[Epoch 60] Error : 0.06502499999999989  Prediction : 0.5450000000000003
[Epoch 61] Error : 0.06249999999999989  Prediction : 0.5500000000000003
[Epoch 62] Error : 0.06002499999999989  Prediction : 0.5550000000000003
[Epoch 63] Error : 0.05759999999999989  Prediction : 0.5600000000000003
[Epoch 64] Error : 0.05522499999999989  Prediction : 0.5650000000000003
[Epoch 65] Error : 0.05289999999999989  Prediction : 0.5700000000000003
[Epoch 66] Error : 0.05062499999999989  Prediction : 0.5750000000000003
[Epoch 67] Error : 0.04839999999999989  Prediction : 0.5800000000000003
[Epoch 68] Error : 0.04622499999999989  Prediction : 0.5850000000000003
[Epoch 69] Error : 0.04409999999999989  Prediction : 0.5900000000000003
[Epoch 70] Error : 0.042024999999999896 Prediction : 0.5950000000000003
[Epoch 71] Error : 0.0399999999999999   Prediction : 0.6000000000000003
[Epoch 72] Error : 0.03802499999999989  Prediction : 0.6050000000000003
[Epoch 73] Error : 0.036099999999999896 Prediction : 0.6100000000000003
[Epoch 74] Error : 0.034224999999999894 Prediction : 0.6150000000000003
[Epoch 75] Error : 0.0323999999999999   Prediction : 0.6200000000000003
[Epoch 76] Error : 0.0306249999999999   Prediction : 0.6250000000000003
[Epoch 77] Error : 0.0288999999999999   Prediction : 0.6300000000000003
[Epoch 78] Error : 0.027224999999999902 Prediction : 0.6350000000000003
[Epoch 79] Error : 0.025599999999999904 Prediction : 0.6400000000000003
[Epoch 80] Error : 0.024024999999999904 Prediction : 0.6450000000000004
[Epoch 81] Error : 0.022499999999999905 Prediction : 0.6500000000000004
[Epoch 82] Error : 0.02102499999999991  Prediction : 0.6550000000000004
[Epoch 83] Error : 0.01959999999999991  Prediction : 0.6600000000000004
[Epoch 84] Error : 0.01822499999999991  Prediction : 0.6650000000000004
[Epoch 85] Error : 0.016899999999999915 Prediction : 0.6700000000000004
[Epoch 86] Error : 0.015624999999999917 Prediction : 0.6750000000000004
[Epoch 87] Error : 0.01439999999999992  Prediction : 0.6800000000000004
[Epoch 88] Error : 0.01322499999999992  Prediction : 0.6850000000000004
[Epoch 89] Error : 0.012099999999999923 Prediction : 0.6900000000000004
[Epoch 90] Error : 0.011024999999999925 Prediction : 0.6950000000000004
[Epoch 91] Error : 0.009999999999999929 Prediction : 0.7000000000000004
[Epoch 92] Error : 0.009024999999999932 Prediction : 0.7050000000000004
[Epoch 93] Error : 0.008099999999999934 Prediction : 0.7100000000000004
[Epoch 94] Error : 0.007224999999999937 Prediction : 0.7150000000000004
[Epoch 95] Error : 0.0063999999999999405        Prediction : 0.7200000000000004
[Epoch 96] Error : 0.005624999999999943 Prediction : 0.7250000000000004
[Epoch 97] Error : 0.004899999999999947 Prediction : 0.7300000000000004
[Epoch 98] Error : 0.00422499999999995  Prediction : 0.7350000000000004
[Epoch 99] Error : 0.003599999999999953 Prediction : 0.7400000000000004
[Epoch 100] Error : 0.0030249999999999565       Prediction : 0.7450000000000004
[Epoch 101] Error : 0.00249999999999996 Prediction : 0.7500000000000004
[Epoch 102] Error : 0.0020249999999999635       Prediction : 0.7550000000000004
[Epoch 103] Error : 0.0015999999999999673       Prediction : 0.7600000000000005
[Epoch 104] Error : 0.0012249999999999711       Prediction : 0.7650000000000005
[Epoch 105] Error : 0.0008999999999999749       Prediction : 0.7700000000000005
[Epoch 106] Error : 0.0006249999999999789       Prediction : 0.7750000000000005
[Epoch 107] Error : 0.00039999999999998294      Prediction : 0.7800000000000005
[Epoch 108] Error : 0.00022499999999998706      Prediction : 0.7850000000000005
[Epoch 109] Error : 9.999999999999129e-05       Prediction : 0.7900000000000005
[Epoch 110] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 111] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 112] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 113] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 114] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 115] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 116] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 117] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 118] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 119] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 120] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 121] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 122] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 123] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 124] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 125] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 126] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 127] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 128] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 129] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 130] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 131] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 132] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 133] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 134] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 135] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 136] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 137] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 138] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 139] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 140] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 141] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 142] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 143] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 144] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 145] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 146] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 147] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 148] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005
[Epoch 149] Error : 1.9721522630525295e-31      Prediction : 0.8000000000000005
[Epoch 150] Error : 2.4999999999995603e-05      Prediction : 0.7950000000000005

(weight 1))
[Epoch 1] Error : 0.09000000000000002   Prediction : 0.5
[Epoch 2] Error : 0.0625        Prediction : 0.55
[Epoch 3] Error : 0.03999999999999998   Prediction : 0.6000000000000001
[Epoch 4] Error : 0.022499999999999975  Prediction : 0.6500000000000001
[Epoch 5] Error : 0.009999999999999974  Prediction : 0.7000000000000002
[Epoch 6] Error : 0.0024999999999999823 Prediction : 0.7500000000000002
[Epoch 7] Error : 4.930380657631324e-32 Prediction : 0.8000000000000003
[Epoch 8] Error : 0.0024999999999999823 Prediction : 0.7500000000000002
[Epoch 9] Error : 4.930380657631324e-32 Prediction : 0.8000000000000003
[Epoch 10] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 11] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 12] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 13] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 14] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 15] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 16] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 17] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 18] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 19] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 20] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 21] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 22] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 23] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 24] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 25] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 26] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 27] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 28] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
[Epoch 29] Error : 4.930380657631324e-32        Prediction : 0.8000000000000003
[Epoch 30] Error : 0.0024999999999999823        Prediction : 0.7500000000000002
'''