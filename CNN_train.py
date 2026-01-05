# ======================================================
# GPU / CUDA CONFIG
# ======================================================
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/quanghuy/cuda-11.8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import cv2
from imutils import paths
from sklearn.utils import shuffle


from keras.models import Model
from keras.utils import to_categorical, load_img, array_to_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D
from keras.applications.resnet import preprocess_input      # chuan hoa chuan de pretrain tren resnet
from keras.applications import ResNet50
from keras.optimizers import Adam

#Duong dan thu muc
path = 'TRASH_CLASSIFICATION/trash_dataset'

# waste_labels = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
taget_size = (224,224)

# # Load du lieu tu thu muc
# def load_datasets(path):
#     x = []
#     labels = []
#     images_path = sorted(list(paths.list_images(path)))  #load tung anh

#     for img_path in images_path:
#         img = cv2.imread(img_path)   #cv2 dang doc theo bgr 
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # chuyen sang rgb de pretrain resnet
#         img = cv2.resize(img, taget_size)
#         x.append(img)
#         label = img_path.split(os.path.sep)[-2] # lay ten thu muc la nhan
#         labels.append(waste_labels[label])      #chuyen thanh so

        
#     return x, labels

# x, y = load_datasets(path)      # load du lieu
# x, y = shuffle(x, y, random_state=42)  # tron ngau nhien

# x = np.array(x)
# y = np.array(y)

# print('kich thuoc cua x: ')
# print(x.shape)
# print('kich thuoc cua y: ')
# print(y.shape)

input_shape = (224,224,3)

# chuan hoa va tang cuong du lieu
train = ImageDataGenerator(
    preprocessing_function=preprocess_input,  #chuan hoa chuan cho resnet pretrain
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.1,   #train 90- val 10
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1
)

train_generator = train.flow_from_directory(
    directory=path,
    target_size=taget_size,
    class_mode='categorical',
    subset='training'
)
val_generator = val.flow_from_directory(
    directory=path,
    target_size=taget_size,
    class_mode='categorical',
    subset='validation'
)


# load lai model resnet_50 pretrain imagenet
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False


#xay them lop phan loai moi
num_classes = train_generator.num_classes
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)
model  = Model(inputs = base_model.input, outputs = output)

model.compile(optimizer= Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


#====================================================
#state 1, hoc co ban
print('STATE 1 TRAIN')
h1 = model.fit(train_generator, 
               validation_data=val_generator, 
               epochs=15,
               callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
               )




#=====================================================
#state 2
for layer in base_model.layers[-30:]:  # mo 30 layer cuoi de resnet hoc
    layer.trainable = True

model.compile(optimizer= Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3),     #LR moi = LR cu × 0.3
    ModelCheckpoint(
        "resnet50_trash_best.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

print('STATE 2 TRAIN')
h2 = model.fit(train_generator, 
               validation_data=val_generator, 
               epochs=8,
               callbacks=callbacks 
               )

#=====================================================
#state 3 finetune lai toan bo mang 
for layer in base_model.layers:
    layer.trainable = True
model.compile(
    optimizer=Adam(learning_rate=5e-6),loss='categorical_crossentropy',metrics=['accuracy']
)



print('STATE 3 TRAIN')
h3 = model.fit(train_generator, 
               validation_data= val_generator, 
               epochs=5,
               callbacks=callbacks
               )

# save model

finanly_model_path = "trach_classification.h5"
model.save(finanly_model_path)
print('done====')

