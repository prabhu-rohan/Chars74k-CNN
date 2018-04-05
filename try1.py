import keras
import numpy as np
import h5py

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model



batch_size = 128
nb_classes = 62
nb_epoch = 12
img_width=128
img_height=128

train_dataset=np.load('train_dataset1.npy')
train_labels=np.load('train_labels1.npy')
test_dataset=np.load('test_dataset1.npy')
test_labels=np.load('test_labels1.npy')
valid_dataset=np.load('valid_dataset1.npy')
valid_labels=np.load('valid_labels1.npy')



def reformat(dataset, labels):
  dataset = dataset.reshape((-1, 128, 128 ,1)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(nb_classes) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

#def randomize(dataset, labels):
#  permutation = np.random.permutation(labels.shape[0])
#  shuffled_dataset = dataset[permutation,:,:]
#  shuffled_labels = labels[permutation]
#  return shuffled_dataset, shuffled_labels
#train_dataset, train_labels = randomize(train_dataset, train_labels)
#test_dataset, test_labels = randomize(test_dataset, test_labels)
#valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
#print(test_dataset)
#print(test_labels)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(img_width, img_height ,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(img_width, img_height ,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(img_width, img_height ,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(62))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_dataset, train_labels, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(valid_dataset, valid_labels))
score = model.evaluate(test_dataset, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('my_model.h5') 




