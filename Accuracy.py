from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras_sequential_ascii import keras2ascii


def accuracy(X, X_train, X_test, y_train, y_test):
    # Define the model with 3 layers of neural network

    model = Sequential()
    model.add(Dense(2
                    , input_dim=len(X.columns), activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    plot_model(model, to_file='model_plot.png',
               show_shapes=True, show_layer_names=True, show_layer_activations=True)
    keras2ascii(model)
  # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=3, batch_size=5)
    # Evaluate the model
    acc = model.evaluate(X_test, y_test)
    print("\n\n-------------------------")
    print('\nAccuracy is: %.2f' % (acc[1] * 100))
    print("\n-------------------------")
