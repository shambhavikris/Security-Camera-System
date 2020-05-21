from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

#extract a single face from a given photo
def extract_face(directory, filename, required_size=(160, 160)):
    x = directory+'/'+filename
    image = Image.open(x)
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

def load_dataset(directory):
    X,y = list(), list()
    for file in listdir(directory):
        if file == '.DS_Store':
            continue
        print(file)
        face = extract_face(directory, file)
        label = file
        X.append(face)
        y.append(label)
    return asarray(X), asarray(y)

trainX, trainy = load_dataset('Images')
print(trainX.shape, trainy.shape)

#save arrays to one file in compressed format
savez_compressed('People.npz', trainX, trainy)
