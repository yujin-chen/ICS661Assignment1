import pandas as pd
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#Load the Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#print(f'Train dataset shape: {train_df.shape}')
#print(f'Test dataset shape: {test_df.shape}')
#X_test = test_df.iloc[:,:].values
#print(X_test[0])

# preprocess the Data
# Separate features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

#print(X_train.shape)
#print(X_test.shape)
#print(X_test[0])

#convert the data type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#print(X_test[0])


# Normalize the features (scaling pixel values to [0, 1])
X_train /= 255.0
X_test /=255.0
#print(X_test[0])

nb_classes = 10 #b/c 0-9 totally of 10 unique digits

#convert labels to categorical format
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

#Define the Model, Initialize sequential model
model = Sequential()

#add layers
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('sigmoid'))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#Compile the model
sgd=keras.optimizers.SGD(learning_rate=0.01) #lr is learning rate
model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Train the model
history = model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test))

#Evaluate the Model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score or loss:', score[0])
print('Test accuracy:', score[1])


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()





