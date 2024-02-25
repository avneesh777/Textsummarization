import streamlit as st 
import os


# NLP Pkgs
from textblob import TextBlob 
import spacy
from gensim.summarization import summarize

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function for Sumy Summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

# Function to Analyse Tokens and Lemma
def text_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
    return allData

# Function For Extracting Entities
def entity_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)
    tokens = [ token.text for token in docx]
    entities = [(entity.text,entity.label_)for entity in docx.ents]
    allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
    return allData

def main():
    """ NLP Based App with Streamlit """
    
    # Title
    st.title("Text Alchemy with Streamlit")
    st.subheader("NLP Magic On the Go!")
    st.markdown("""
        #### Description
        + This is a Natural Language Processing(NLP) Based App for NLP tasks like
        Tokenization, NER, Sentiment, Summarization
    """)

    # Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize Your Text")
        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Entity Extraction
    if st.checkbox("Show Named Entities"):
        st.subheader("Analyze Your Text")
        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Extract"):
            entity_result = entity_analyzer(message)
            st.json(entity_result)

    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Analyse Your Text")
        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Analyze Sentiment"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize Your Text")
        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Summarize Text"):
            summarization_option = st.selectbox("Choose Summarizer", ['summary.h5', 'final_model.h5'])
            if summarization_option == 'sumy':
                st.text("Using summary.h5 Summarizer ..")
                summary_result = sumy_summarizer(message)
            else:
                st.text("Using final_model.h5 Summarizer ..")
                summary_result = summarize(message)
            st.success(summary_result)

    # Sidebar
    st.sidebar.subheader("About App")
    st.sidebar.text("Text Alchemy App with Streamlit")

    st.sidebar.subheader("By")
    st.sidebar.text("Srihari Thyagarajan")
    st.sidebar.text("Avneesh Tilwani")
    st.sidebar.text("Neil Shah")


if __name__ == '__main__':
    main()
