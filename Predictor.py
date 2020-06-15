import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os
classifier=tf.keras.models.load_model("mask_detector(MobileNet).h5")    #loading saved model after training

                #Prediction
def predict(test_image):
    test_image=cv2.cvtColor(test_image,COLOR_BGR2RGB)
    test_image=cv2.resize(test_image,(224,224))
    test_image=img_to_array(test_image)
    test_image=preprocess_input(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=classifier.predict(test_image)
    (Mask,noMask)=result[0]
    return Mask,noMask

    
image_path="00026.jpg"                                                  #fullpath of image

prototxt="face_detector/deploy.prototxt.txt"                            #fullpath to prototxt
model="face_detector/res10_300x300_ssd_iter_140000.caffemodel"          #fullpath to caffemodel
confidenceThresh=0.15                                                   #confidence thrsholf to detech faces
extraFaceMargin=30
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

image=cv2.imread(image_path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confidenceThresh:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        y = startY - 10 if startY - 10 > 10 else startY + 10
        (startX,startY)=(max(0,startX-extraFaceMargin),max(0,startY-extraFaceMargin))
        (endX,endY)=(min(w-1,endX+extraFaceMargin),min(h-1,endY+extraFaceMargin))
        face=image[startY:endY,startX:endX]
        x,y=predict(face)
        label='Mask' if x>y else 'No Mask'
        color = (0,255,0) if x>y else (0,0,255)
        cv2.rectangle(image, (startX, startY), (endX, endY),color, 2)
        cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
cv2.imshow("Output",image)
cv2.waitKey(0)
print("Done")
