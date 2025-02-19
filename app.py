# kg_rag_app.py
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import networkx as nx
from urllib.parse import urlparse
from autogen import AssistantAgent, UserProxyAgent
import json
import difflib
from datetime import datetime
from pyvis.network import Network
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Groq Configuration
config_list = [{
    "model": "mixtral-8x7b-32768",
    "api_key": os.environ.get("GROQ_API_KEY", "your-key-here"),
    "base_url": "https://api.groq.com/openai/v1",
    "api_type": "open_ai"
}]

class WebKnowledgeGraphAgent:
    def __init__(self):
        self.graph = nx.Graph()
        self.analyzer = AssistantAgent(
            name="analyzer",
            system_message="""Analyze web content and return JSON with:
- "concepts": [{"id", "name", "type", "time?", "parent?"}]
- "relationships": [{"source", "target", "label"}]
Example:
{
  "concepts": [
    {"id": "1", "name": "AI", "type": "concept"},
    {"id": "2", "name": "ML", "type": "concept", "parent": "1"}
  ],
  "relationships": [
    {"source": "1", "target": "2", "label": "subfield"}
  ]
}""",
            llm_config={"config_list": config_list, "temperature": 0.1}
        )
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
        self.concept_similarity_threshold = 0.75

    def scrape_website(self, url):
        try:
            response = requests.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find(['main', 'article']) or soup
            return ' '.join(main_content.stripped_strings)[:4000]
        except Exception as e:
            st.error(f"Scraping error: {e}")
            return None

    def process_websites(self, urls):
        for url in urls:
            content = self.scrape_website(url)
            if content:
                data = self.analyze_content(content)
                self.add_to_graph(url, data)
        return self.graph

    def analyze_content(self, content):
        try:
            response = self.user_proxy.initiate_chat(
                self.analyzer,
                message=f"Analyze this content:\n{content}",
                summary_method="last_msg"
            )
            return json.loads(response.summary)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return {"concepts": [], "relationships": []}

    def add_to_graph(self, url, data):
        domain = urlparse(url).netloc
        for concept in data.get("concepts", []):
            self.graph.add_node(
                concept["id"],
                name=concept["name"],
                domain=domain,
                type=concept.get("type", "concept")
            )
        for rel in data.get("relationships", []):
            self.graph.add_edge(
                rel["source"],
                rel["target"],
                label=rel.get("label", "related")
            )

class KGRAGApp:
    def __init__(self):
        self.agent = WebKnowledgeGraphAgent()
        self.vectorstore = None
        self.scraped_content = []

    def process_urls(self, urls):
        with st.spinner("Building knowledge graph..."):
            self.agent.process_websites(urls)
            self.scraped_content = [
                {"url": url, "content": self.agent.scrape_website(url)} 
                for url in urls
            ]
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            texts = [f"{doc['url']}\n{doc['content']}" for doc in self.scraped_content]
            self.vectorstore = FAISS.from_texts(texts, embeddings)

    def visualize_graph(self):
        net = Network(height="600px", width="100%")
        for node, data in self.agent.graph.nodes(data=True):
            net.add_node(node, label=data['name'], title=data['domain'])
        for edge in self.agent.graph.edges(data=True):
            net.add_edge(edge[0], edge[1], title=edge[2].get('label', ''))
        net.save_graph("graph.html")
        with open("graph.html") as f:
            st.components.v1.html(f.read(), height=800)

    def ask_question(self, question):
        if not self.vectorstore:
            st.error("Process URLs first!")
            return
            
        qa_chain = RetrievalQA.from_chain_type(
            ChatGroq(temperature=0, model_name="mixtral-8x7b-32768"),
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PromptTemplate.from_template(
                "Answer using this context:\n{context}\n\nQuestion: {question}"
            )}
        )
        result = qa_chain.invoke({"query": question})
        st.markdown(f"**Answer:** {result['result']}")

def main():
    st.set_page_config(page_title="KG+RAG App", layout="wide")
    st.title("Knowledge Graph & RAG Explorer")
    
    app = KGRAGApp()
    
    with st.sidebar:
        st.header("Configuration")
        urls = st.text_area("Enter URLs (one per line)", height=150)
        if st.button("Process URLs"):
            app.process_urls(urls.split("\n"))
    
    tab1, tab2 = st.tabs(["Knowledge Graph", "QA System"])
    
    with tab1:
        if app.agent.graph.nodes:
            app.visualize_graph()
        else:
            st.info("Add URLs to build the knowledge graph")
    
    with tab2:
        question = st.text_input("Ask a question about the content")
        if question:
            app.ask_question(question)

if __name__ == "__main__":
    main()
