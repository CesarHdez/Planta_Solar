from tensorflow.keras.models import load_model

model = load_model('LSTM_u.h5')
model.summary()