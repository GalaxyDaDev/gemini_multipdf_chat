import os
import re
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

import PyPDF2
import fitz  # PyMuPDF
import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Load environment variables
load_dotenv()

# Initialize models and pipelines
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")

def search_duckduckgo(topic):
    url = f"https://api.duckduckgo.com/?q={topic}&format=json&no_html=1"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get('RelatedTopics', [])
    if data:
        return data[0].get('FirstURL', ''), data[0].get('Text', '')
    return '', ''

def fetch_web_page(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_text_from_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.stripped_strings
    return ' '.join(texts)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PyPDF2.PdfReader(pdf)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ""
    return text

def get_text_chunks(text, chunk_size=2000, chunk_overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def get_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings, chunks

def get_conversational_chain():
    def answer_question(question, embeddings, chunks):
        question_embedding = embedding_model.encode([question])
        similarities = [np.dot(question_embedding, e) for e in embeddings]
        best_idx = np.argmax(similarities)
        return chunks[best_idx]
    return answer_question

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question, embeddings, chunks):
    answer_chain = get_conversational_chain()
    response = answer_chain(user_question, embeddings, chunks)
    return response

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
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def generate_questions(text):
    input_text = "generate questions: " + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True)
    outputs = model.generate(input_ids, max_length=256, num_return_sequences=5, num_beams=5)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def main():
    st.set_page_config(page_title="OpenAI PDF Chatbot", page_icon="🤖")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        topic = st.text_input("Enter the topic name for search")
        if st.button("Submit & Process"):
            if pdf_docs and topic:
                with st.spinner("Processing..."):
                    with ThreadPoolExecutor() as executor:
                        raw_text = executor.submit(get_pdf_text, pdf_docs).result()
                        text_chunks = executor.submit(get_text_chunks, raw_text).result()
                        embeddings, chunks = get_vector_store(text_chunks)

                    url, snippet = search_duckduckgo(topic)
                    if url:
                        web_page_content = fetch_web_page(url)
                        extracted_text = extract_text_from_html(web_page_content)
                    else:
                        extracted_text = snippet

                    if not extracted_text:
                        st.error("No relevant content found.")
                        return

                    topics = extract_topics(extracted_text)
                    important_terms = extract_important_terms(extracted_text)
                    summary = summarize_text(extracted_text)
                    questions = generate_questions(extracted_text)
                    st.success("Done")
                    st.session_state.topics = topics
                    st.session_state.terms = important_terms
                    st.session_state.summary = summary
                    st.session_state.questions = questions
            else:
                st.error("Please upload at least one PDF file and enter a topic name.")

    st.title("Chat with PDF files using OpenAI🤖")

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
                response = user_input(prompt, embeddings, chunks)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
