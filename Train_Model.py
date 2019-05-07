"""
Facial Recognition in Opencv using Deep Learning

File Name : Train_model.py
Description : This program will train the model
Author: Shiyaz T
Reference : https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

"""

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

BASE_DIR = os.path.dirname(__file__)
print("[INFO] BASE DIR: ", BASE_DIR)

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open((os.path.join(BASE_DIR, 'output/embeddings.pickle')), "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open((os.path.join(BASE_DIR, 'output/recognizer.pickle')), "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open((os.path.join(BASE_DIR, 'output/le.pickle')), "wb")
f.write(pickle.dumps(le))
f.close()