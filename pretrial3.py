from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from numpy import load
import numpy as np
from scipy.spatial import distance

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB') #would only need it if the images are in B/W
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)#correction
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    #make face into one sample
    samples = expand_dims(face_pixels, axis=0)
    #make a prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

test_face = extract_face('te.PNG')
print(test_face.shape)
model = load_model('facenet_keras.h5')
test_embedding = get_embedding(model, test_face)
print(test_embedding.shape)


data = load('People-embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']
min_dist = 100
identity = None
ind = 0

for p in trainX:
    #dist = np.linalg.norm(test_embedding - p)
    dist = distance.euclidean(test_embedding, p)
    dist = sigmoid(dist)

    if dist < min_dist:
        min_dist = dist
        identity = trainy[ind]

    print('Distance:', dist)
    print('Current check:', trainy[ind])
    ind = ind + 1
    print('Min dist:', min_dist)
    print("Identity:", identity)

id = identity.split('.')

if min_dist <1:
    print('Face has been found')
    print('identity is ', id[0])
else:
    print('No match')
