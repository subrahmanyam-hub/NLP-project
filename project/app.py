#=====>LOAD LIBRARIES<=====#

# Basic
import streamlit as st
import numpy as np
import re

# CSS Design
from load_css import local_css
local_css(r"C:\Users\subra\Downloads\project\style.css")

# For NLP & Preprocessing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from scipy.sparse import csr_matrix

# To load saved models
import joblib
import xgboost

#Deep Learning Libraries - Pytorch BERT
import torch
import torch.hub
import torch.nn as nn
from transformers import BertTokenizerFast as BertTokenizer, BertModel
from transformers import logging
logging.set_verbosity_warning()
# specify GPU or CPU
device = torch.device("cpu")

#======>BASIC UI<======#

st.title('Abusive Email Classifier')

st.write("""
### Let your serene mind be more productive
and far away from someone spoiling it for you.
""")

model_name = st.sidebar.selectbox(
    'Select Model',
    ('Machine Learning', 'Ensemble Learning', 'Deep Learning')
)

if model_name=='Machine Learning':
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression', 'LinearSVC', 'Multinomial Naive Bayes', 'Random Forest','XGBoost','Perceptron','Support Vector Machine')
)
    st.markdown(f"# {model_name} : {classifier_name}\n"
             "These baseline machine learning approaches are not very apt for NLP problems. They yield poor outputs as semantics is not taken into consideration.")

elif model_name=='Ensemble Learning':
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ("Voting Classifier",)
)
    st.markdown(f"# {model_name} : {classifier_name}\n"
             "Ensemble results are average since it is a collection of baseline models. The overal outcome adds a lot of generalization which is good. We recommed trying out in the deep learning model.")

else :
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('BERT',)
)
    st.markdown(f"# {model_name} : {classifier_name}\n"
             "We get best results using Deep-Learning techniques like BERT and LSTM. Though LSTM makes errors, it is better than the baseline ML approaches. Semantics is taken into consideration which inturn, yields good predictions. BERT performs better since it was trained by Google on a humongous dataset. To save up space on Heroku, we ended up deploying BERT over LSTM in this app as Tensorflow took around 430Mb space.")

user_input = st.text_area("Enter content to check for abuse", "")



#======>LOAD PREBUILT<======#



#For ML models
@st.cache(allow_output_mutation=True)
def load_ml(model):
    if model == 'Logistic Regression':
        return joblib.load('deployment/ml_models/1gsLR.sav')
    elif model == 'LinearSVC':
        return joblib.load('deployment/ml_models/2gsLSVC.sav')
    elif model == 'Multinomial Naive Bayes':
        return joblib.load('deployment/ml_models/3gsMNB.sav')
    elif model == 'Random Forest':
        return joblib.load('deployment/ml_models/4gsRFC.sav')
    elif model == 'XGBoost':
        return joblib.load('deployment/ml_models/5gsXGB.sav')
    elif model == 'Perceptron':
        return joblib.load('deployment/ml_models/6gsPPT.sav')
    elif model == 'Support Vector Machine':
        return joblib.load('deployment/ml_models/7gsSVMC.sav')
    elif model == 'Voting Classifier':
        return joblib.load('deployment/ml_models/8Ensemble.sav')



#BERT Architecture
class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

#======>FUNCTION DEFINITIONS<======#

#Function to clean i/p data:
def cleantext(text):
    text = re.sub(r"\n", " ", text) #remove next "\n"
    text = re.sub(r"[\d-]", "", text) #remove all digits
    text = re.sub(r'[^A-Za-z0-9]+', " ", text) #remove all special charcters
    text = text.lower()
    return text



#Function to get sentiment scores
def sentiscore(text):
    sentialz = SentimentIntensityAnalyzer()
    analysis = sentialz.polarity_scores(text)
    return analysis["compound"]


#Function to predict for ML and Ensemble
def predictor_ml(text,model):
    cv = CountVectorizer()
    X_count = cv.fit_transform([text])
    tfidf_transformer = TfidfTransformer()
    X_tfid = tfidf_transformer.fit_transform(X_count)
    X = csr_matrix((X_tfid.data, X_tfid.indices, X_tfid.indptr), shape=(X_tfid.shape[0], 10000))
    return model.predict(X)


#Function to preprocess and predict for BERT
@st.cache(allow_output_mutation=True)
def predictor_bert(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load the model
    bert = BertModel.from_pretrained('bert-base-uncased')
    for param in bert.parameters():
        param.requires_grad = False
    model_bert = BERT_Arch(bert)
    model_bert = model_bert.to(device)
    path = r'C:\Users\subra\Downloads\project\saved_weights1.pt'
    model_bert.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    max_seq_len = 35
    # tokenize and encode sequences in the test set1
    tokens_test = tokenizer.batch_encode_plus(
        [text],
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    # for test
    test_seq1 = torch.tensor(tokens_test['input_ids'])
    test_mask1 = torch.tensor(tokens_test['attention_mask'])
    # get predictions for test data
    with torch.no_grad():
        preds1 = model_bert(test_seq1.to(device), test_mask1.to(device))
        preds1 = preds1.detach().cpu().numpy()
        preds1 = np.argmax(preds1, axis=1)
        return preds1[0]


#Function to display output
def out(a):
    if a == 0:
        t1 = "<div> <span class='highlight blue'><span class='bold'>Non Abusive</span> </span></div>"
        st.markdown(t1,unsafe_allow_html=True)
    else:
        t2 = "<div> <span class='highlight red'><span class='bold'>Abusive</span> </span></div>"
        st.markdown(t2,unsafe_allow_html=True)


if st.button("Check for Abuse"):
    y = cleantext(user_input)
    st.write(f"## Sentiment Score {sentiscore(y)}")
    if model_name == 'Machine Learning' or model_name == 'Ensemble Learning':
        with st.spinner("Predicting..."):
            o = predictor_ml(y, load_ml(classifier_name))
        out(o)
    else:
        with st.spinner("Predicting..."):
            o = predictor_bert(y)
        out(o)



