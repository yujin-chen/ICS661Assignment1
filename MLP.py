import pandas as pd
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from keras.src.layers import BatchNormalization
from keras.src.metrics import F1Score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import Precision, Recall

#Load the Data
train_df = pd.read_csv('train.csv', header=None)
test_df = pd.read_csv('test.csv', header=None)


# preprocess the Data
# Separate features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values


#convert the data type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# Normalize the features (scaling pixel values to [0, 1])
X_train /= 255.0
X_test /=255.0

#10 labeles in range 0-9
nb_classes = 10

#convert labels to categorical format
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

#Define the Model, Initialize sequential model
model = Sequential()

model.add(Input(shape=(784,)))
#add layers
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(nb_classes, activation='softmax'))


#Compile the model
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', Precision(), Recall(), F1Score()])

#Train the model
training_history = model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1)

# Define the F1 score metric
def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))

#Evaluate the Model
score = model.evaluate(X_test, Y_test, verbose=0)
# Print the evaluation results
print(f'Test score or loss: {score[0]:.4f}')   # Loss
print(f'Test accuracy: {score[1]:.4f}')         # Accuracy
print(f'Test Precision: {score[2]:.4f}')        # Precision
print(f'Test Recall: {score[3]:.4f}')           # Recall

# Calculate F1 score based on precision and recall
precision = score[2]
recall = score[3]
f1 = f1_score(precision, recall)
print(f'Test F1 Score: {f1:.4f}')

# Plot training accuracy
plt.plot(training_history.history['accuracy'])
plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot training loss
plt.plot(training_history.history['loss'])
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot F1 score
plt.plot(training_history.history['f1_score'])
plt.title('Train F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.show()

# Plot precision and recall
plt.plot(training_history.history['precision'])
plt.plot(training_history.history['recall'])
plt.title('Training Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend(['Precision', 'Recall'], loc='upper left')
plt.show()
