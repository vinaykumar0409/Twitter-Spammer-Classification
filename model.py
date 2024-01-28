from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, SimpleRNN, LSTM, concatenate
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('twitter_spammer.csv')
X = df.drop(['UserID', 'CreatedAt', 'CollectedAt', 'FollowingsSeries', 'spammer'], axis=1)
y = df['spammer']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, SimpleRNN, LSTM, concatenate
from tensorflow.keras.optimizers import Adam

input_shape = (X_train.shape[1], 1)

# Branches for 1D CNN, RNN, and LSTM
cnn_branch = Sequential()
cnn_branch.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
cnn_branch.add(MaxPooling1D(pool_size=2))
cnn_branch.add(Conv1D(32, kernel_size=3, activation='relu'))
cnn_branch.add(MaxPooling1D(pool_size=2))
cnn_branch.add(Flatten())

rnn_branch = Sequential()
rnn_branch.add(SimpleRNN(64, activation='relu', input_shape=input_shape))

lstm_branch = Sequential()
lstm_branch.add(LSTM(64, activation='relu', input_shape=input_shape))

# Combine the branches
combined_model = concatenate([cnn_branch.output, rnn_branch.output, lstm_branch.output])

# Add dense layers for further processing
combined_model = Dense(128, activation='relu')(combined_model)
combined_model = Dense(64, activation='relu')(combined_model)

# Output layer for binary classification
output_layer = Dense(1, activation='sigmoid')(combined_model)

# Create the final model
model = Model(inputs=[cnn_branch.input, rnn_branch.input, lstm_branch.input], outputs=output_layer)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Reshape the input data
X_train_cnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_train_rnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))

# Train the model
model.fit([X_train_cnn, X_train_rnn, X_train_lstm], y_train, epochs=10, batch_size=32, validation_split=0.2)

X_test_cnn = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_rnn = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_lstm = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))

# Evaluate the model on the test data
evaluation_result = model.evaluate([X_test_cnn, X_test_rnn, X_test_lstm], y_val)

# Print the evaluation result (accuracy and loss)
print("Test Accuracy:", evaluation_result[1])
print("Test Loss:", evaluation_result[0])

# Save the entire model (architecture, weights, and optimizer state)
model.save('full_model.h5')
