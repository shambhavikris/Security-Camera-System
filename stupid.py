from keras.models import load_model
import cv2
import mtcnn
#print(mtcnn.__version__)
##face detection for live video, with bounding box for each face, has provision for adding label
from os import listdir
from os.path import isdir
from PIL import Image
from PIL import ImageDraw
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from numpy import load
import numpy as np
from scipy.spatial import distance
from keras import backend as K
from keras.models import load_model
from scipy.spatial import distance
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def extract_face(image, required_size=(160, 160)):
    #image = Image.open(filename)
    #image = image.convert('RGB')
    pixels = asarray(image)
    detector =  MTCNN()
    results = detector.detect_faces(pixels)
    #print('results are:', results)
    face_array = asarray(image)
    #return face_array
    return results

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    #make face into one sample
    samples = expand_dims(face_pixels, axis=0)
    #make a prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

def get_dist1(a, b):
    dist = distance.euclidean(test_embedding, p)
    return dist

def get_dist2(a,b):
    dist = np.linalg.norm(a-b)
    return dist

model = load_model('facenet_keras.h5')
data = load('People-embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']

cap = cv2.VideoCapture(0)

while True:
  ret, image_np = cap.read()
  results = extract_face(image_np)
  image = Image.fromarray(image_np)
  if results != []:
      for r in results:
          x1, y1, width, height = r['box']
          x1, y1 = abs(x1), abs(y1)
          x2, y2 = x1 + width, y1 + height
          test_face = image_np[y1:y2, x1:x2]
          image = Image.fromarray(test_face)
          image = image.resize((160, 160))
          face_array = asarray(image)
          test_embedding = get_embedding(model, face_array)
          min_dist = 100
          identity = None
          ind = 0
          for p in trainX:
              #dist = np.lingalg.norm(test_embedding - p)
              dist = get_dist2(test_embedding,p)
              dist = sigmoid(dist)
              #dist = distance.euclidean(test_embedding, p)
              if dist < min_dist:
                  min_dist = dist
                  identity = trainy[ind]
              ind = ind + 1

          if min_dist < 15:
              id = identity.split('.')
              label = id[0]
          else:
              label = 'Unknown'
          #id = identity.split('.')
          #label=id[0]
          print('dist:', dist, '\nlabel:', label, '\n')
          #draw_bounding_box_on_image(image_np, y1, x1, y2, x2)
          image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), (1, 1, 1), 4)
          image_np = cv2.putText(image_np, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

  cv2.imshow('fig', image_np)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
