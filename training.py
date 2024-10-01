import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import random
from sklearn.model_selection import train_test_split
import optuna

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    units = [trial.suggest_int(f'units_l{i}', 32, 256) for i in range(num_layers)]
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

    model = Sequential()
    model.add(Dense(units[0], input_shape=(len(X_train[0]),), activation='relu'))
    model.add(Dropout(dropout_rate))

    for i in range(1, num_layers):
        model.add(Dense(units[i], activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(len(y_train[0]), activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )

    return history.history['val_accuracy'][-1]

def build_model(input_shape, output_shape, params):
    model = Sequential()
    model.add(Dense(params['units_l0'], input_shape=input_shape, activation='relu'))
    model.add(Dropout(params['dropout_rate']))

    for i in range(1, params['num_layers']):
        model.add(Dense(params[f'units_l{i}'], activation='relu'))
        model.add(Dropout(params['dropout_rate']))

    model.add(Dense(output_shape, activation='softmax'))

    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_val, y_val, callbacks=None):
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history

if __name__ == "__main__":
    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(np.array(train_x), np.array(train_y), test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # Train final model with best hyperparameters
    final_model = build_model((len(X_train[0]),), len(y_train[0]), best_params)
    
    tensorboard_callback = TensorBoard(log_dir="./logs")
    
    history = train_model(final_model, X_train, y_train, X_val, y_val, callbacks=[tensorboard_callback])

    # Evaluate on test set
    test_loss, test_accuracy = final_model.evaluate(X_test, y_test)

    print(f"Test accuracy: {test_accuracy}")

    # Save the model
    final_model.save('chatbot_model.h5')
    
    # Save words and classes
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    print("Model and data saved successfully.")