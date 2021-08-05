from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, GRU, LSTM, Bidirectional, Reshape


def use_model():
    model = model_0_1()
    return model

def model_0():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, ))
    dense_1_1 = Dense(64, activation='relu')(input_1)
    output_1 = Dense(128, activation='relu')(dense_1_1)

    # input_2
    input_2 = Input(shape=(13, ))
    dense_2_1 = Dense(256, activation='relu')(input_2)
    output_2 = Dense(512, activation='relu')(dense_2_1)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256, activation='relu')(merge_1)
    merge_3 = Dense(16, activation='relu')(merge_2)
    last_output = Dense(1, name='output-1')(merge_3)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model


def model_0_1():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, ))
    dense_1_1 = Dense(64, activation='relu')(input_1)
    output_1 = Dense(128, activation='relu')(dense_1_1)

    # input_2
    input_2 = Input(shape=(13, ))
    dense_2_1 = Dense(256, activation='relu')(input_2)
    output_2 = Dense(512, activation='relu')(dense_2_1)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256, activation='relu')(merge_1)
    merge_3 = Dense(16, activation='relu')(merge_2)
    last_output = Dense(1, name='output-1')(merge_3)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model


def model_1():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, ))
    dense_1_1 = Dense(64, activation='relu')(input_1)
    dropout_1 = Dropout(1/3)(dense_1_1)
    output_1 = Dense(128, activation='relu')(dropout_1)


    # input_2
    input_2 = Input(shape=(13, ))
    dense_2_1 = Dense(256, activation='relu')(input_2)
    dropout_2 = Dropout(1/3)(dense_2_1)
    output_2 = Dense(512, activation='relu')(dropout_2)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256, activation='relu')(merge_1)
    merge_3 = Dense(64)(merge_2)
    dropout_3 = Dropout(1/2)(merge_3)
    merge_4 = Dense(16)(dropout_3)
    merge_5 = Dense(4, activation='relu')(merge_4)
    last_output = Dense(1, name='output-1')(merge_5)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model


def model_1_1():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, ))
    dense_1_1 = Dense(64, activation='relu')(input_1)
    dropout_1 = Dropout(1/2)(dense_1_1)
    output_1 = Dense(128, activation='relu')(dropout_1)


    # input_2
    input_2 = Input(shape=(13, ))
    dense_2_1 = Dense(256, activation='relu')(input_2)
    dropout_2 = Dropout(1/2)(dense_2_1)
    output_2 = Dense(512, activation='relu')(dropout_2)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256, activation='relu')(merge_1)
    dropout_3 = Dropout(3/4)(merge_2)
    merge_3 = Dense(64)(dropout_3)
    merge_4 = Dense(16)(merge_3)
    dropout_4 = Dropout(3/4)(merge_4)
    merge_5 = Dense(4, activation='relu')(dropout_4)
    last_output = Dense(1, name='output-1')(merge_5)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model


def model_1_2():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, ))
    dense_1_1 = Dense(64, activation='relu')(input_1)
    dropout_1 = Dropout(1/2)(dense_1_1)
    output_1 = Dense(128)(dropout_1)


    # input_2
    input_2 = Input(shape=(13, ))
    dense_2_1 = Dense(256, activation='relu')(input_2)
    dropout_2 = Dropout(1/2)(dense_2_1)
    output_2 = Dense(512)(dropout_2)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256)(merge_1)
    dropout_3 = Dropout(3/4)(merge_2)
    merge_3 = Dense(64)(dropout_3)
    merge_4 = Dense(16)(merge_3)
    dropout_4 = Dropout(3/4)(merge_4)
    merge_5 = Dense(4, activation='relu')(dropout_4)
    last_output = Dense(1, name='output-1')(merge_5)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model


def model_2():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, ))
    dense_1_1 = Dense(64)(input_1)
    output_1 = Dense(128, activation='relu')(dense_1_1)


    # input_2
    input_2 = Input(shape=(13, ))
    dense_2_1 = Dense(256)(input_2)
    output_2 = Dense(512, activation='relu')(dense_2_1)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256, activation='relu')(merge_1)
    dropout_1 = Dropout(1/2)(merge_2)
    merge_3 = Dense(64, activation='relu')(dropout_1)
    merge_4 = Dense(16, activation='relu')(merge_3)
    dropout_2 = Dropout(1/2)(merge_4)
    merge_5 = Dense(4, activation='relu')(dropout_2)
    last_output = Dense(1, name='output-1')(merge_5)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model


def model_3():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, 1))
    # dense_1_1 = Dense(64)(input_1)
    dense_1_1 = GRU(32)(input_1)
    output_1 = Dense(128, activation='relu')(dense_1_1)

    # input_2
    input_2 = Input(shape=(13, 1))
    dense_2_1 = GRU(128)(input_2)
    output_2 = Dense(512, activation='relu')(dense_2_1)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256, activation='relu')(merge_1)
    dropout_1 = Dropout(1/2)(merge_2)
    merge_3 = Dense(64, activation='relu')(dropout_1)
    merge_4 = Dense(16, activation='relu')(merge_3)
    dropout_2 = Dropout(1/2)(merge_4)
    merge_5 = Dense(4, activation='relu')(dropout_2)
    last_output = Dense(1, name='output-1')(merge_5)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model


def model_4():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, 1))
    # dense_1_1 = Dense(64)(input_1)
    dense_1_1 = Bidirectional(GRU(32))(input_1)
    output_1 = Dense(128, activation='relu')(dense_1_1)

    # input_2
    input_2 = Input(shape=(13, 1))
    dense_2_1 = Bidirectional(GRU(128))(input_2)
    output_2 = Dense(512, activation='relu')(dense_2_1)

    merge_1 = concatenate([output_1, output_2])
    merge_2 = Dense(256, activation='relu')(merge_1)
    dropout_1 = Dropout(1/2)(merge_2)
    merge_3 = Dense(64, activation='relu')(dropout_1)
    merge_4 = Dense(16, activation='relu')(merge_3)
    dropout_2 = Dropout(1/2)(merge_4)
    merge_5 = Dense(4, activation='relu')(dropout_2)
    last_output = Dense(1, name='output-1')(merge_5)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model

def model_5():
    model = Model()

    # input_1
    input_1 = Input(shape=(5, 1))
    # dense_1_1 = Dense(64)(input_1)
    dense_1_1 = Bidirectional(GRU(32))(input_1)
    output_1 = Dense(128, activation='relu')(dense_1_1)

    # input_2
    input_2 = Input(shape=(13, 1))
    dense_2_1 = Bidirectional(GRU(128))(input_2)
    output_2 = Dense(384, activation='relu')(dense_2_1)

    merge_1 = concatenate([output_1, output_2])
    reshape_1 = Reshape((64, 8))(merge_1)
    dense_3_1 = Bidirectional(GRU(256, activation='relu'))(reshape_1)
    merge_2 = Dense(256, activation='relu')(dense_3_1)
    dropout_1 = Dropout(1/2)(merge_2)
    merge_3 = Dense(64, activation='relu')(dropout_1)
    merge_4 = Dense(16, activation='relu')(merge_3)
    dropout_2 = Dropout(1/2)(merge_4)
    merge_5 = Dense(4, activation='relu')(dropout_2)
    last_output = Dense(1, name='output-1')(merge_5)

    model = Model(inputs=[input_1, input_2], outputs=last_output)

    return model