import keras
from keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

IMG_SIZE = 224

base_model = VGG19(input_shape = [IMG_SIZE, IMG_SIZE, 3], weights = "imagenet", include_top = False)

for layers in base_model.layers:
    layers.trainable = False
    
x = base_model.output
x = GlobalAveragePooling2D()(x)

# adding a fully-connected layer
x = Dense(1024, activation='relu')(x)

prediction = Dense(8,activation="softmax")(x) #Adding final output layer of 8 neurons

model = Model(inputs=base_model.input, outputs=prediction)

print(model.summary())

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

r = model.fit(X, y, batch_size=20, epochs=10, validation_split=0.2)

# Loss graph
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Accuracy graph
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


# Perparing Test Data and Evaluating

def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath)[:,:,::-1]
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)

test_img_path = "/content/drive/My Drive/pickle_files/dance_test_images/"

Image = []
target = []

for img in os.listdir(test_img_path):
    path = test_img_path + img
    print(path)
    pred = model.predict([prepare(path)])
    pred_list = pred[0].tolist()
    print(pred_list)
    print("Class of image : ",CATEGORIES[pred_list.index(max(pred_list))])
    Image.append(img)
    target.append(CATEGORIES[pred_list.index(max(pred_list))])
    print("---------------------------------------\n")



d = {'Image' : Image, 'target' : target}
df = pd.DataFrame(d)
print(df.target.value_counts())
df.to_csv("dance_vgg19.csv")


