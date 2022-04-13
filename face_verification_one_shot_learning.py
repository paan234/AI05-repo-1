# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:18:18 2022

@author: ACER
Credits to: 
1. https://github.com/foo290/Face-verification-using-One-shot-learning
2. https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X
"""

#Import all the libraries we are going to use
from mtcnn import MTCNN
import tensorflow as tf
import os #handling path
import numpy as np
import cv2
from architecture_facenet import InceptionResNetV2

#1. Create the face detection model
#We are using MTCNN
face_detector = MTCNN()

#2. Create FaceNet for face recognition
weight_path = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\face_recognition\weight\facenet_keras_weights.h5"
model = InceptionResNetV2()
model.load_weights(weight_path)

#3. Specify the path to our face database
face_database_path = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\face_recognition\database"

#Global variables for colour and thickness of the rectangle and points we are going to draw on the image
colour = (0,255,0) #Green in BGR
thickness = 2 #drawing box

#4. Create a function to obtain the bouding box result(in the form of x1,y1,x2,y2)
def get_xy(box):
    x1,y1,width,height = box    
    x1,y1 = abs(x1),abs(y1)
    x2,y2 = x1+width,y1+height    
    return x1,y1,x2,y2

#5. Create a function that will output face result
def get_face(image,resize_scale=(160,160)):
    face_list = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #We are using the MTCNN to detect all the faces appeared in an image    
    faces_detected = face_detector.detect_faces(image)
    #Create a for loop to loop through each individual face detection result    
    for detected_face in faces_detected:
        x1,y1,x2,y2 = get_xy(detected_face['box'])
        #Crop out the face        
        final_face = image[y1:y2,x1:x2]
        #Resize the image       
        face_array = cv2.resize(final_face,resize_scale)
        face_list.append(face_array)
        
    #Return a list of detected face image, and output of MTCNN    
    return face_list, faces_detected

#6. Create a function that will use FaceNet to produce embeddings
def get_face_embeddings(face_image):
    #We need to change the data type of the pixels    
    face_image = face_image.astype('float32')
    #Normalize the pixel values    
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std    
    #Expand dimension so that the model can take in    
    samples = np.expand_dims(face_image, axis=0)
    #Use the FaceNet to produce embeddings    
    embeddings = model.predict(samples)
    return embeddings

#7. Create a function that will load the database
def load_saved_user():
    #Empty list to store the embeddings of the face in database and the name    
    saved_faces = []
    saved_faces_names = []
    #List down all the images in the database    
    face_database = os.listdir(face_database_path)
    #Read the images, then use FaceNet to produce the embeddings for comparison later    
    if face_database:
        for face_img in face_database:
            #Read the image with OpenCV            
            image_np = cv2.imread(os.path.join(face_database_path,face_img))
            face_list, detected_faces = get_face(image_np)
            #We are assuming our database are tightly controlled (means that each image in the database only has one face)
            face_embedding = get_face_embeddings(face_list[0])
            #Append to the empty list            
            saved_faces.append(face_embedding)
            saved_faces_names.append(face_img.split('.')[0])
    else:
        print("No face available in the database")
        
    return saved_faces,saved_faces_names

#8. Create a function that will draw all the detected faces (bounding boxes and keypoints)
def mark_face(detected_face,image,x1,x2,y1,y2):
    #Get the face keypoints    
    left_eye = detected_face['keypoints']['left_eye']
    right_eye = detected_face['keypoints']['right_eye']
    nose = detected_face['keypoints']['nose']
    mouth_left = detected_face['keypoints']['mouth_left']
    mouth_right = detected_face['keypoints']['mouth_right']
    
    #Draw the detected face with a rectangle (for the bounding box)    
    image = cv2.rectangle(image, (x1,y1), (x2,y2), colour,thickness)
    #Draw the facial keypoints with circles    
    image = cv2.circle(image, left_eye, radius=2, color=colour,thickness=-1)
    image = cv2.circle(image, right_eye, radius=2, color=colour,thickness=-1)
    image = cv2.circle(image, nose, radius=2, color=colour,thickness=-1)
    image = cv2.circle(image, mouth_left, radius=2, color=colour,thickness=-1)
    image = cv2.circle(image, mouth_right, radius=2, color=colour,thickness=-1)
    
    return image

#9 Create a function that will perform face verification
def verify(target_image,threshold=10):
    #(a) Load the data from database    
    saved_faces,saved_faces_names = load_saved_user()
    #(b) Perform face detection on target image, only if there's an entry in database    
    if saved_faces:
        target_faces, detected_faces = get_face(target_image)
        #(c) If there are face detected, we will perform recognition on that face        
        if target_faces:
            for target_face, detected_face in zip(target_faces,detected_faces):
                #(d) Get embeddings from the target face                
                target_face_embedding = get_face_embeddings(target_face)
                #(e) We are going to compare the target face with all the data in database                
                for every_face, name in zip(saved_faces,saved_faces_names):
                    #(f) Measure similarity with Euclidean distance                    
                    dist = np.linalg.norm(every_face - target_face_embedding)
                    #(g) If distance lower than threshold, two images would be considered similar,                    
                    #that would be the identity of the face                    
                    if dist < threshold:
                        #(h) Display the face recognition result                        
                        x1,y1,x2,y2 = get_xy(detected_face['box'])
                        target_image = mark_face(detected_face, target_image, x1, x2, y1, y2)
                        target_image = cv2.putText(target_image,
                                                   name+f"__distance:{dist:.2f}",
                                                   (x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,
                                                   0.6,colour,thickness)
        else:
            print("No face detected in the image")
    return target_image

#%%
#10. Open webcam and perform face recognition
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break    
    
    #Convert to numpy array    
    image_np = np.array(frame)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    #Run inference on the image frame    
    drawn_image = verify(image_np)
    #Display result    
    display_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Face Verification", display_image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()

#%%
#For those with Google Collab, upload an image, and perform verification on image
file_path = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\face_recognition\face_to_verify"
faces_to_verify = os.listdir(file_path)

#We are going to loop through all these images for verification
for face_to_verify in faces_to_verify:
    complete_path = os.path.join(file_path,face_to_verify)
    image_np = cv2.imread(complete_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    #Run inference on the image    
    drawn_image = verify(image_np)
    #Display result    
    display_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Face Verification", display_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
