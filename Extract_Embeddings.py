"""
Facial Recognition in Opencv using Deep Learning

File Name : Extract_embeddings.py
Description : This program is responsible for deep learning feature extractor to generate a 128-D vector describing a face
Author: Shiyaz T
Reference : https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

"""
# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

BASE_DIR = os.path.dirname(__file__)
print("[INFO] BASE DIR: ", BASE_DIR)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.join(BASE_DIR, "face_detection_model/deploy.prototxt")
modelPath = os.path.join(BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

embedding_model = os.path.join(BASE_DIR, 'openface_nn4.small2.v1.t7')
dataset = os.path.join(BASE_DIR, 'dataset')
embeddings = os.path.join(BASE_DIR, 'output/embeddings.pickle')
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# grab the paths to the input images in our dataset
print("[INFO] Load image dataset..")
imagePaths = list(paths.list_images(dataset))
print("[DEBUG] Image Paths: ", imagePaths)

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		print("[DEBUG] Confidence: ", confidence)

		# ensure that the detection with the 50% probabilty thus helping filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# dump the facial embeddings + names to disk
data = {"embeddings": knownEmbeddings, "names": knownNames}
print("[DEBUG] Total Faces:", total)
print("[DEBUG] Data: ", data['names'])
f = open(embeddings, 'wb')
f.write(pickle.dumps(data))
f.close()