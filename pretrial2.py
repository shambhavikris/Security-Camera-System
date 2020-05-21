from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

#get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    #make face into one sample
    samples = expand_dims(face_pixels, axis=0)
    #make a prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]
#load the face dataset
#this can only work with detected faces-->bounding box is important
data = load('People.npz')
trainX, trainy = data['arr_0'], data['arr_1']
print('Loaded: ', trainX.shape, trainy.shape)#test statement
#loading facenet
model = load_model('facenet_keras.h5')
print('Loaded Model')
#convert each face in the train to an embedding
newTrainX = list()
print(trainy)

for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)


savez_compressed('People-embeddings.npz', newTrainX, trainy)
