import pandas as pd
import joblib
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.data.path.append("C:\Users\hp\Desktop\Symbiosis\PBL\spy-eye\nltk_data")
# LSTM 
max_len = 150

# Load the machine learning models
model = joblib.load('naive_bayes_classifier.pkl')
model_1 = joblib.load('random_forest_cl.pkl')

# Load the Keras model with the custom optimizer
model_2 = tf.keras.models.load_model('LSTM_retrained.h5')

# Load the vectorizers
vectorizer = joblib.load('vectorizer.pkl')
vectorizer_1 = joblib.load('vectorizer_random.pkl')
tok = joblib.load('tokenizer_retrained.pickle')

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Defining the lemma function
def lemma(test_pred):
    lemmatizer = WordNetLemmatizer()
    test_data_lemma = []
    for i in range(test_pred.shape[0]):
        sp = test_pred.iloc[i]
        sp = re.sub('[^A-Za-z]', ' ', sp)
        sp = sp.lower()
        tokenized_sp = word_tokenize(sp)
        sp_processed = []
        for word in tokenized_sp:
            if word not in set(stopwords.words('english')):
                sp_processed.append(lemmatizer.lemmatize(word))
        sp_text = " ".join(sp_processed)
        test_data_lemma.append(sp_text)
    return pd.Series(test_data_lemma)

# Defining the Streamlit application
def main():
    st.title('SpyEye-An Email Monitoring System to Detect Spam')
    st.write('Enter some input data to get a prediction:')
    input_email = st.text_area('Input data:')
    
    submit_button = st.button('Submit')

    if submit_button:
        if not input_email:
            st.warning('Please enter some input data.')
        else:
            test_pred = pd.Series([input_email])
            test_pred = test_pred.apply(lambda x: ' '.join([word for word in x.split() if word not in set(stopwords.words('english'))]))
            test_pred = test_pred.str.replace('\W', ' ', regex=True)
            test_pred = test_pred.str.lower()
            test_pred = lemma(test_pred)

            # Transform the input email using the vectorizer
            test_data_transformed = vectorizer.transform(test_pred)
            test_data_transformed_1 = vectorizer_1.transform(test_pred)

            # Create dataframe to store results
            results = {
                'Classifier': ['Naive Bayes', 'Random Forest', 'LSTM'],
                'Classification': [None, None, None],
                'Spam Probability': [None, None, None],
                'Ham Probability': [None, None, None]
            }

            # Make predictions
            pred_proba = model.predict_proba(test_data_transformed)[0]
            pred_label = model.predict(test_data_transformed)[0]

            pred_proba_1 = model_1.predict_proba(test_data_transformed)[0]
            pred_label_1 = model_1.predict(test_data_transformed)[0]

            txts = tok.texts_to_sequences(test_pred)
            txts = pad_sequences(txts, maxlen=max_len)
            prediction = model_2.predict(txts)[0][0]

            results['Classification'][0] = pred_label
            results['Spam Probability'][0] = pred_proba[1].round(3)
            results['Ham Probability'][0] = pred_proba[0].round(3)

            results['Classification'][1] = pred_label_1
            results['Spam Probability'][1] = pred_proba_1[1].round(3)
            results['Ham Probability'][1] = pred_proba_1[0].round(3)

            results['Classification'][2] = 'spam' if prediction > 0.5 else 'ham'
            results['Spam Probability'][2] = prediction.round(3)
            results['Ham Probability'][2] = (1 - prediction).round(3)

            st.table(pd.DataFrame(results))

if __name__ == '__main__':
    main()
