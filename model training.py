import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

data_path='Training Samples/'
only_files=[f for f in listdir(data_path) if isfile(join(data_path,f))]

training_data,labels=[],[]

for i, files in enumerate(only_files):
    image_path=data_path+only_files[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append((i))

labels=np.asarray(labels,dtype=np.int32)

model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data),np.asarray(labels))

print('model traning complete')

face_classifier=cv2.CascadeClassifier(r'C:\Users\hp\PycharmProjects\opencv work\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# Function to detect face in an image
def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi

#Loading all the images from a folder
def load_images(folder):
    images=[]
    for img_name in listdir(folder):
        img=cv2.imread(join(folder,img_name))
        if img is not None:
            images.append(img)
    return images
folder='Source/'

images=load_images(folder)
for i in range(len(images)):
    image,face=face_detector(images[i])

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)
        print(result)

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            display_string=str(confidence)+'% confidence it is user'
            print(display_string)

        if confidence>75:
            print('Matched')
        else:
            print('Not Matched')

    except:
        print('face not found')
        pass


