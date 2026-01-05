import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# 
# CONFIG
# 
test_path = "/home/quanghuy/Documents/TRASH_CLASSIFICATION/trash_dataset_test"
model_path = "/home/quanghuy/Documents/resnet50_trash_best.h5"

target_size = (224, 224)
batch_size = 32


model = load_model(model_path)

# 
# TEST DATA GENERATOR
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False   
)

# 
# EVALUATE MODEL (LOSS + ACC)
test_steps = test_generator.samples // batch_size
if test_generator.samples % batch_size != 0:
    test_steps += 1

loss, acc = model.evaluate(
    test_generator,
    steps=test_steps,
    verbose=1
)

print("\n" + "="*40)
print(f"TEST LOSS     : {loss:.4f}")
print(f"TEST ACCURACY : {acc*100:.2f}%")
print("="*40)

# 
# PREDICTION
# 
y_pred = model.predict(
    test_generator,
    steps=test_steps,
    verbose=1
)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# 
# CLASSIFICATION REPORT
# 
print("\n CLASSIFICATION REPORT:")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names
))

# 
# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred_classes)

print(" CONFUSION MATRIX:")
print(cm)
