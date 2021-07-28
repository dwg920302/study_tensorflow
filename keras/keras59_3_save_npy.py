from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)

x_train = np.save('../_save/_npy/k59_brain_x_train.npy', arr=xy_train[0][0])
y_train = np.save('../_save/_npy/k59_brain_y_train.npy', arr=xy_train[0][1])
x_test = np.save('../_save/_npy/k59_brain_x_test.npy', arr=xy_test[0][0])
y_test = np.save('../_save/_npy/k59_brain_y_test.npy', arr=xy_test[0][1])

# batch size가 image 수보다 커야 함.