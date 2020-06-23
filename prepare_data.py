
import numpy as np
import pickle


CATEGORIES = ['bharatanatyam','kathak','kuchipudi','kathakali','odissi','manipuri','sattriya','mohiniyattam']

training_data = []
IMG_SIZE = 224

DATADIR = "/home/user/Documents/dance_ml/imagedata"

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))[:,:,::-1] #converting image to array
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)

create_training_data()             

import random
random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

   
    
X = np.array(X).resize(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array(y)
   
X = X / 255 

pickle_out = open("A_224.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("b_224.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
