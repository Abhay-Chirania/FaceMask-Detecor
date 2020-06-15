import numpy as np
from numpy import save
from numpy import load
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def data_preprocess(imagePath,loadData=False):
    
    #Load data if already saved as Numpy array
    if loadData==True:
        print("Loading Data....")
        data=load('numpy_array/data.npy')
        label=load('numpy_array/label.npy')
        print("Data Loaded")
        return data,label
    
    #Data Preprocessing for first time
    category=['Mask','No_Mask']
    data,label=[],[]
    print("Starting Data preprocessing....")
    for i in category:
        path=os.path.join(imagePath,i)
        c=0
        for file in os.listdir(path):
            imgPath=os.path.join(path,file)
            image = load_img(imgPath, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            lbl=[1.,0.] if i=='Mask' else [0.,1.]
            label.append(lbl)


    data=np.array(data,dtype='float32')
    label=np.array(label)
    save('numpy_array/data.npy',data)
    save('numpy_array/label.npy',label)
    return data,label



                                        ####MODEL####
def modelMobileNetV2(lr,epochs):
    base=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))
    head=base.output
    head=AveragePooling2D(pool_size=(5,5))(head)
    head=Flatten(name="flatten")(head)
    head=Dense(128,activation='relu')(head)
    head=Dropout(0.4)(head)
    head=Dense(2,activation='softmax')(head)
    model = Model(inputs=base.input, outputs=head)
    for layer in base.layers:
        layer.trainable = False
    optimizer= Adam(lr=lr, decay=lr/ epochs)
    model.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=["accuracy"])
    print("Model Compiled")
    return model

def modelCustomCNN(lr,epochs):
    classifier=Sequential()
    classifier.add(Conv2D(32,(3,3),input_shape=(224,224,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(64,(3,3),input_shape=(128,128,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(128,(3,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten(name="flatten"))
    classifier.add(Dense(128,activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2,activation='softmax'))
    optimizer= Adam(lr=lr, decay=lr/ epochs)
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
                                            ####TRAINING####





lr = 1e-3                                                   #learning rate
epochs = 10                                                 #Epochs
batch_size = 24                                             #Batch Size

trainPath="trainData"                                       #Training folder should contain two folders One for with 'Mask' images and one for 'No_Mask'
data,label=data_preprocess(trainPath,loadData=False)         #loadData=True if you have already preprocessed data once and is saved as numpy array
print("Splitting Data...")


(X, testX, Y, testY) = train_test_split(data, label,test_size=0.20, stratify=label, random_state=42)
print("Data Splitted. Starting training...")
imageDataGen = ImageDataGenerator(zoom_range=0.15,shear_range=0.15,horizontal_flip=True,width_shift_range=0.2,height_shift_range=0.2,rotation_range=15,fill_mode="nearest")


m=modelMobileNetV2(lr,epochs)                               #calling model function we can choose either one of two models

final=m.fit(imageDataGen.flow(X,Y,batch_size=batch_size),
            validation_data=(testX,testY),
            steps_per_epoch=len(X)/batch_size,
            validation_steps=len(testX)/batch_size,
            epochs=epochs)

print("\n\nModel Trained\n\n")
print("Saving the model...")
m.save("mask_detector.h5")
print("Done")
