import os
import re
import requests
from collections import Counter
from dotenv import load_dotenv

import PyPDF2
import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Initialize models and pipelines
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")

def search_web(topic):
    """Searches the web for a given topic and returns titles, links, and snippets."""
    search_url = f"https://www.bing.com/search?q={topic}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for item in soup.find_all('li', {'class': 'b_algo'}):
        title = item.find('h2').text
        link = item.find('a')['href']
        snippet = item.find('p').text if item.find('p') else ""
        links.append((title, link, snippet))
    return links

def fetch_article_content(url):
    """Fetches the text content from a web article."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        return ""

def get_pdf_text(pdf_docs):
    """Extracts text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        reader = PyPDF2.PdfReader(pdf)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ""
    return text

def get_text_chunks(text, chunk_size=2000, chunk_overlap=300):
    """Splits text into chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def get_vector_store(chunks):
    """Generates embeddings for text chunks."""
    embeddings = embedding_model.encode(chunks)
    return embeddings, chunks

def get_conversational_chain():
    """Returns a function to answer questions based on embeddings."""
    def answer_question(question, embeddings, chunks):
        question_embedding = embedding_model.encode([question])
        similarities = [question_embedding @ e for e in embeddings]
        best_idx = similarities.index(max(similarities))
        return chunks[best_idx]
    return answer_question

def clear_chat_history():
    """Clears the chat history."""
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question, embeddings, chunks):
    """Handles user input and returns an appropriate response."""
    answer_chain = get_conversational_chain()
    response = answer_chain(user_question, embeddings, chunks)
    return response

def extract_topics(text, num_topics=5):
    """Extracts topics from text using LDA."""
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
    """Extracts important terms from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(num_terms)
    return [word for word, _ in common_words]

def summarize_text(text):
    """Summarizes text."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def generate_questions(text):
    """Generates questions from text."""
    input_text = "generate questions: " + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True)
    outputs = model.generate(input_ids, max_length=256, num_return_sequences=5, num_beams=5)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Advanced PDF Chatbot", page_icon="🤖")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        topic = st.text_input("Enter the topic name for search")
        if st.button("Submit & Process"):
            if pdf_docs and topic:
                with st.spinner("Processing..."):
                    # Process PDF files
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    embeddings, chunks = get_vector_store(text_chunks)

                    # Search the web for additional information
                    links = search_web(topic)
                    extracted_texts = []
                    for title, link, snippet in links:
                        text = fetch_article_content(link)
                        if text:
                            extracted_texts.append(text)

                    if not extracted_texts:
                        st.error("No relevant content found.")
                        return

                    combined_text = " ".join(extracted_texts)
                    topics = extract_topics(combined_text)
                    important_terms = extract_important_terms(combined_text)
                    summary = summarize_text(combined_text)
                    questions = generate_questions(combined_text)
                    st.success("Done")
                    st.session_state.topics = topics
                    st.session_state.terms = important_terms
                    st.session_state.summary = summary
                    st.session_state.questions = questions
            else:
                st.error("Please upload at least one PDF file and enter a topic name.")

    st.title("Chat with PDF files using Advanced NLP🤖")

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
