# kg_rag_app.py
import os
import streamlit as st
import requests
import validators
from bs4 import BeautifulSoup
import networkx as nx
from urllib.parse import urlparse
import pickle
from youtube_transcript_api import YouTubeTranscriptApi
from pyvis.network import Network
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from kreuzberg import extract_file  # From search result [1]

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SAVE_FILE = "app_state.pkl"

class WebKnowledgeGraphAgent:
    def __init__(self):
        self.graph = nx.Graph()
        self.processed_urls = set()
        self.content_store = []

    def process_url(self, url):
        if url in self.processed_urls:
            return
            
        try:
            if "youtube.com" in url or "youtu.be" in url:
                content = self.process_youtube(url)
            elif url.lower().endswith('.pdf'):
                content = self.process_pdf(url)
            else:
                content = self.scrape_website(url)
                
            if content:
                self.content_store.append({"url": url, "content": content})
                self.processed_urls.add(url)
                
        except Exception as e:
            st.error(f"Error processing {url}: {str(e)}")

    def scrape_website(self, url):
        try:
            response = requests.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find(['main', 'article']) or soup
            return ' '.join(main_content.stripped_strings)[:4000]
        except Exception as e:
            st.error(f"Scraping error: {e}")
            return None

    def process_pdf(self, file_path):
        try:
            result = extract_file(file_path)
            return result.content
        except Exception as e:
            st.error(f"PDF processing error: {e}")
            return None

    def process_youtube(self, url):
        try:
            video_id = url.split("v=")[-1].split("&")[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([t['text'] for t in transcript])
        except Exception as e:
            st.error(f"YouTube transcript error: {e}")
            return None

class KGRAGApp:
    def __init__(self):
        self.agent = WebKnowledgeGraphAgent()
        self.vectorstore = None
        self.load_data()

    def process_urls(self, urls):
        valid_urls = [u for u in urls if validators.url(u)]
        invalid_urls = set(urls) - set(valid_urls)
        
        with st.status("Processing..."):
            for url in valid_urls:
                self.agent.process_url(url)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            texts = [f"{doc['url']}\n{doc['content']}" for doc in self.agent.content_store]
            self.vectorstore = FAISS.from_texts(texts, embeddings)
            
            self.save_data()
            
            if invalid_urls:
                st.warning(f"Skipped invalid URLs: {', '.join(invalid_urls)}")

    def visualize_graph(self):
        net = Network(height="600px", width="100%", notebook=False)
        for node in self.agent.graph.nodes(data=True):
            net.add_node(node[0], label=node[1].get('name', node[0]))
        for edge in self.agent.graph.edges(data=True):
            net.add_edge(edge[0], edge[1], title=edge[2].get('label', ''))
        net.save_graph("graph.html")
        with open("graph.html") as f:
            st.components.v1.html(f.read(), height=800)

    def rag_qa(self, question):
        if not self.vectorstore:
            st.error("Process some content first!")
            return
            
        prompt_template = """Answer using this context:
        {context}
        
        Question: {question}
        Cite sources using [URL] notation."""
        
        qa_chain = RetrievalQA.from_chain_type(
            ChatGroq(temperature=0, model_name="mixtral-8x7b-32768"),
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
        )
        
        result = qa_chain.invoke({"query": question})
        st.markdown(f"**Answer:** {result['result']}")
        
        sources = {doc.metadata['source'] for doc in result.get('source_documents', [])}
        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                st.markdown(f"- {src}")

    def save_data(self):
        with open(SAVE_FILE, "wb") as f:
            pickle.dump({
                "graph": self.agent.graph,
                "content_store": self.agent.content_store,
                "processed_urls": self.agent.processed_urls
            }, f)

    def load_data(self):
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "rb") as f:
                data = pickle.load(f)
                self.agent.graph = data["graph"]
                self.agent.content_store = data["content_store"]
                self.agent.processed_urls = data["processed_urls"]

def main():
    st.set_page_config(page_title="Smart RAG Explorer", layout="wide")
    app = KGRAGApp()
    
    st.title("🧠 Smart Content Analyzer with Groq")
    
    with st.sidebar:
        st.header("⚙️ Input Sources")
        urls = st.text_area("Enter URLs (YouTube, PDFs, websites)", height=150).split("\n")
        pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process Content"):
            all_urls = [u.strip() for u in urls if u.strip()]
            all_urls += [f.name for f in pdf_files]  # Handle uploaded PDFs
            app.process_urls(all_urls)
            
        if st.button("Clear Session"):
            if os.path.exists(SAVE_FILE):
                os.remove(SAVE_FILE)
            st.session_state.clear()
            st.rerun()
    
    tab1, tab2 = st.tabs(["📊 Knowledge Graph", "❓ QA System"])
    
    with tab1:
        if app.agent.processed_urls:
            st.subheader(f"Processed {len(app.agent.processed_urls)} sources")
            app.visualize_graph()
        else:
            st.info("Add content to build the knowledge graph")
    
    with tab2:
        question = st.text_input("Ask about the processed content")
        if question:
            app.rag_qa(question)

if __name__ == "__main__":
    main()
