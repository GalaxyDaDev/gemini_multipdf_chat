import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

import PyPDF2
import fitz  # PyMuPDF
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'NONE'

# Ensure the environment variable is set
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("The environment variable 'GOOGLE_API_KEY' is not set.")
else:
    genai.configure(api_key=google_api_key)

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PyPDF2.PdfFileReader(pdf)
        for page_num in range(reader.getNumPages()):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    return splitter.split_text(text)

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

def extract_topics(text, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    topics = []
    for topic in lda.components_:
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topics.append(" ".join(topic_words))
    return topics

def extract_important_terms(text, num_terms=10):
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(num_terms)
    return [word for word, _ in common_words]

def summarize_text(text):
    prompt = f"Summarize the following text in bullet points:\n\n{text}"
    summary = genai.generate(prompt)
    return summary

def generate_questions(text):
    prompt = f"Generate questions from the following text:\n\n{text}"
    questions = genai.generate(prompt)
    return questions

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                with ThreadPoolExecutor() as executor:
                    raw_text = executor.submit(get_pdf_text, pdf_docs).result()
                    text_chunks = executor.submit(get_text_chunks, raw_text).result()
                    executor.submit(get_vector_store, text_chunks).result()
                topics = extract_topics(raw_text)
                important_terms = extract_important_terms(raw_text)
                summary = summarize_text(raw_text)
                questions = generate_questions(raw_text)
                st.success("Done")
                st.session_state.topics = topics
                st.session_state.terms = important_terms
                st.session_state.summary = summary
                st.session_state.questions = questions

    st.title("Chat with PDF files using Gemini🤖")

    if "summary" in st.session_state:
        st.write("Summary:")
        st.write(st.session_state.summary)

    if "topics" in st.session_state:
        st.write("Important Topics:")
        for topic in st.session_state.topics:
            st.write(f"- {topic}")

    if "terms" in st.session_state:
        st.write("Important Terms:")
        for term in st.session_state.terms:
            st.write(f"- {term}")

    if "questions" in st.session_state:
        st.write("Generated Questions:")
        for question in st.session_state.questions:
            st.write(f"- {question}")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
