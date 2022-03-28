import streamlit as st

# from haystack.utils import clean_wiki_text, convert_files_to_dicts, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def get_document_store():
#     document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat")

#     # Let's first get some files that we want to use
#     doc_dir = "data/article_txt_got"
#     s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
#     fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
#     # Convert files to dicts
#     dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
#     # Now, let's write the dicts containing documents to our DB.
#     document_store.write_documents(dicts)

#     st.write("I got doc store")
#     return document_store

#from haystack.nodes import TfidfRetriever

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_retriever():
    document_store = FAISSDocumentStore.load("haystack_got_faiss_1")
    st.write("I got document_store in retreiver")
    #retriever = TfidfRetriever(document_store=document_store)
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
    st.write("I got retriever")
    return retriever

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_reader():
    st.write("I am inside reader")
    #reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=0)
    st.write("I got reader")
    return reader

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Question Answering Webapp")
st.text("What would you like to know today?")

with st.spinner ('Loading Model into Memory....'):
    retriever = get_retriever()
    reader = get_reader()
    pipe = ExtractiveQAPipeline(reader, retriever)  

text = st.text_input('Enter your questions here....')
if text:
    st.write("Response:")
    with st.spinner('Searching for answers....'):
        st.write("I'm in spinner")
        prediction = pipe.run(query=text, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 2}})
        st.write('answer: {}'.format(prediction['answers'][0].answer))
        st.write('answer: {}'.format(prediction['answers'][1].answer))
    st.write("")


# import streamlit as st
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import DensePassageRetriever
# from haystack.nodes import FARMReader, TransformersReader
# from haystack.pipelines import ExtractiveQAPipeline

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def get_retriever():
#     document_store = FAISSDocumentStore.load("haystack_got_faiss_1")
#     st.write("I got document_store")
#     retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
#     st.write("I got retriever")
#     return retriever

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def get_reader():
#     st.write("I am inside reader")
#     reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
#     st.write("I got reader")
#     return reader

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Question Answering Webapp")
# st.text("What would you like to know today?")

# with st.spinner ('Loading Model into Memory....'):
#     retriever = get_retriever()
#     generator = get_reader()
#     pipe = ExtractiveQAPipeline(reader, retriever)  

# text = st.text_input('Enter your questions here....')
# if text:
#     st.write("Response:")
#     with st.spinner('Searching for answers....'):
#         st.write("I'm in spinner")
#         prediction = pipe.run(query=text, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})
#         st.write('answer: {}'.format(prediction['answers'][0].answer))
#     st.write("")



    
    
# import streamlit as st
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import DensePassageRetriever
# from haystack.nodes import Seq2SeqGenerator
# from haystack.pipelines import GenerativeQAPipeline

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def get_retriever():
#     document_store = FAISSDocumentStore.load("haystack_got_faiss_1")
#     st.write("I got document_store")
#     retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
#     st.write("I got retriever")
#     return retriever

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def get_generator():
#     st.write("I am inside generator")
#     #generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
#     generator = Seq2SeqGenerator(model_name_or_path="yjernite/bart_eli5")
#     st.write("I got generator")
#     return generator

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Question Answering Webapp")
# st.text("What would you like to know today?")

# with st.spinner ('Loading Model into Memory....'):
#     retriever = get_retriever()
#     generator = get_generator()
#     pipe = GenerativeQAPipeline(generator, retriever)  

# text = st.text_input('Enter your questions here....')
# if text:
#     st.write("Response:")
#     with st.spinner('Searching for answers....'):
#         st.write("I'm in spinner")
#         prediction = pipe.run(query=text, params={"Retriever": {"top_k": 1}})
# #        st.write('answer: {}'.format(prediction[0]))
# #        st.write('title: {}'.format(prediction[1]))
# #        st.write('paragraph: {}'.format(prediction[2]))
#     st.write("")    


# #from haystack.utils import convert_files_to_dicts, fetch_archive_from_http, clean_wiki_text
# import streamlit as st
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import DensePassageRetriever

# from haystack.utils import print_documents
# from haystack.pipelines import DocumentSearchPipeline

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def get_retriever():
#     st.write("I am in get_retreiver")
#     document_store = FAISSDocumentStore.load("haystack_got_faiss_1")
# #     document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat")
# #     doc_dir = "data/article_txt_got"
# #     s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
# #     fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
# #     dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
# #     document_store.write_documents(dicts)
#     st.write("I got document_store")

#     retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
#     p_retrieval = DocumentSearchPipeline(retriever)
#     st.write("I got p_retrieval")
#     return p_retrieval

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Question Answering Webapp")
# st.text("What would you like to know today?")
    
# text = st.text_input('Enter your questions here....')
# if text:
#     st.write("Response:")
#     with st.spinner('Searching for answers....'):
#         st.write("I am about to call get_retreiver")
#         res = get_retriever().run(query=text, params={"Retriever": {"top_k": 1}})
#         st.write("I got res")
#         st.write('answer: {}'.format(print_documents(res, max_text_len=512)))
#     st.write("")    
    

    
# import streamlit as st
# header = st.container()
# dataset = st.container()
# features = st.container()
# model_training = st.container()
# with header:
#     st.title('Welcome to our startup advisory!')
#     st.text("On this page we will answer questions regarding your startups based on what we learnt from YC's Michael and Dalton")
# with dataset:
#     st.header("YC's YouTube videos")
#     st.text("I found these videos very informative")
# with features:
#     st.header("The features I created")
# with model_training:
#     st.header("Time to train the model!")
#     st.text("Here we'll play with hyperparamets")

# import streamlit as st

# #from haystack.utils import clean_wiki_text, convert_files_to_dicts, fetch_archive_from_http, print_answers
# from haystack.nodes import FARMReader, TransformersReader
# from haystack.document_stores import FAISSDocumentStore
# from haystack.nodes import DensePassageRetriever
# from haystack.nodes import FARMReader, TransformersReader
# from haystack.pipelines import ExtractiveQAPipeline

# # @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# # def get_retriever():
# document_store = FAISSDocumentStore.load("haystack_got_faiss_0322")
# st.write("I got document_store in retreiver")
# retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki", passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",)
# st.write("I got retriever")
# #     return retriever

# # @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# # def get_reader():
# st.write("I am inside reader")
# #reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
# reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)
# st.write("I got reader")
# #     return reader

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Question Answering Webapp")
# st.text("What would you like to know today?")

# with st.spinner ('Loading Model into Memory....'):
# retriever = get_retriever()
# generator = get_reader()
# pipe = ExtractiveQAPipeline(reader, retriever)  

# text = st.text_input('Enter your questions here....')
# if text:
# st.write("Response:")
# with st.spinner('Searching for answers....'):
#     st.write("I'm in spinner")
#     prediction = pipe.run(query=text, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 2}})
#     st.write('answer: {}'.format(prediction['answers'][0].answer))
#     st.write('answer: {}'.format(prediction['answers'][1].answer))
# st.write("")
