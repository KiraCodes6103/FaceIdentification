# KNN ALGORITHM
# This is pseudocode for implementing the KNN algorithm from scratch:

# Load the training data.
# Prepare data by scaling, missing value treatment, and dimensionality reduction as required.
# Find the optimal value for K:
# Predict a class value for new data:
# Calculate distance(X, Xi) from i=1,2,3,….,n.
# where X= new data point, Xi= training data, distance as per your chosen distance metric.
# Sort these distances in increasing order with corresponding train data.
# From this sorted list, select the top ‘K’ rows.
# Find the most frequent class from these chosen ‘K’ rows. This will be your predicted class.
from re import T
import cv2
import numpy as np
import os

def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5): #calculating how many faces are near to which of the dataset
    dist = []

    for i in range(train.shape[0]):  #shape has two digits the first one representing the total no. of datasets and th 2 representing no.of datas in dataset
        #get the vector and lable
        ix = train[i, :-1]
        iy = train[i, -1]

        #compute the distance from test point
        d = distance(test, ix)
        dist.append([d,iy])

    #sort on basis of distance and get the maximum k

    dk= sorted(dist , key = lambda x: x[0])[:k]

    #Retrieve only the lables 
    labels = np.array(dk)[:, -1]

    #get frequencies of each lables
    output = np.unique(labels, return_counts=T)
    #Find max freq for the label
    index = np.argmax(output[1])
    return output[0][index]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
dataset_path = './face_data/'

face_data = []
labels = []
class_id = 0    #labels for every given file
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones ((data_item.shape[0],))
        class_id +=1
        labels.append(target)
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))
# print(labels)
# print (face_labels)

trainset = np.concatenate((face_dataset,face_labels), axis =1)

font = cv2.FONT_HERSHEY_COMPLEX

while 1:
    ret, frame = cap.read() #ret is a boolean variable

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5) #returns the coordinates of your face

    if len(faces)==0:
        continue
    k=1

    faces = sorted(faces, key = lambda x : x[2]*x[3], reverse = True)


    for face in faces[ :1]:
        x,y,w,h = face
        offset = 5

        face_offset = frame[y-offset: y+h+offset, x-offset:x+w+offset]
        face_selection = cv2.resize(face_offset, (100,100))

        out = knn(trainset, face_selection.flatten())
        cv2.putText(frame, names[int (out)], (x,y-10), font, 1, (0,255,0),2,cv2.LINE_AA)
    

        cv2.rectangle(frame, (x,y), (x+w , y+h), (0,255,0), 2)



    cv2.imshow("web cam", frame)
    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

