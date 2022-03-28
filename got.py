import streamlit as st

from haystack.utils import clean_wiki_text, convert_files_to_dicts, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_document_store():
    document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat")

    # Let's first get some files that we want to use
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
    # Convert files to dicts
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    # Now, let's write the dicts containing documents to our DB.
    document_store.write_documents(dicts)

    st.write("I got doc store")
    return document_store

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_retriever():
    document_store = get_document_store()
    st.write("I got document_store in retreiver")
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
    st.write("I got retriever")
    return retriever

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_reader():
    st.write("I am inside reader")
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    #reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)
    st.write("I got reader")
    return reader

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Question Answering Webapp")
st.text("What would you like to know today?")

with st.spinner ('Loading Model into Memory....'):
    retriever = get_retriever()
    generator = get_reader()
    pipe = ExtractiveQAPipeline(reader, retriever)  

text = st.text_input('Enter your questions here....')
if text:
    st.write("Response:")
    with st.spinner('Searching for answers....'):
        st.write("I'm in spinner")
        prediction = pipe.run(query=text, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})
        st.write('answer: {}'.format(prediction['answers'][0].answer))
    st.write("")
