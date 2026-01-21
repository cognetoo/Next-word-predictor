from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
import pickle

##load files
model = load_model("next_word_lstm.keras")

with open('tokenizer.pickle','rb') as handle:
   tokenizer = pickle.load(handle)

##prediction function
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list],padding='pre',maxlen=max_sequence_len - 1)
    predicted = model.predict(token_list)
    
    predicted_word_index = np.argmax(predicted, axis = 1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

##streamlit app
st.title("Next word prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter sequence of words","To be or not to")
if st.button("predict next word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word : {next_word}")