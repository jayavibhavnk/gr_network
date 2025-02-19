# kg_rag_app.py
import os
import streamlit as st
from web_knowledge_graph_agent import WebKnowledgeGraphAgent
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pyvis.network import Network
from typing import Dict, List

class KGRAGApp:
    def __init__(self):
        self.agent = WebKnowledgeGraphAgent()
        self.vectorstore = None
        self.scraped_content = []
        
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please set OPENAI_API_KEY environment variable")
            st.stop()

    def process_urls(self, urls: List[str]):
        with st.status("Processing URLs..."):
            self.agent.process_websites(urls)
            
            # Store content for RAG
            self.scraped_content = [
                {"url": url, "content": self.agent.scrape_website(url)} 
                for url in urls
            ]
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            texts = [f"URL: {doc['url']}\nContent: {doc['content']}" for doc in self.scraped_content]
            self.vectorstore = FAISS.from_texts(texts, embeddings)
            
            st.session_state.processed = True

    def visualize_graph(self):
        net = Network(height="600px", width="100%", notebook=False)
        
        for node, data in self.agent.graph.nodes(data=True):
            net.add_node(node, label=data['name'], title=data.get('domain', ''),
                        group=data.get('type', 'concept'))
            
        for edge in self.agent.graph.edges(data=True):
            net.add_edge(edge[0], edge[1], title=edge[2].get('label', ''))
        
        net.save_graph("knowledge_graph.html")
        with open("knowledge_graph.html", "r") as f:
            html = f.read()
        
        st.components.v1.html(html, height=800)

    def rag_qa(self, question: str):
        if not self.vectorstore:
            st.error("Process some URLs first!")
            return
            
        prompt_template = """Use the following context to answer the question.
        Cite sources using [URL] notation. If unsure, say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        Answer:"""
        
        qa_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        result = qa.invoke({"query": question})
        st.markdown(f"**Answer:** {result['result']}")
        
        if result['source_documents']:
            st.markdown("**Sources:**")
            for doc in result['source_documents']:
                url = doc.metadata.get('source', 'Unknown URL')
                st.markdown(f"- {url}")

def main():
    st.set_page_config(page_title="Knowledge Graph Explorer", layout="wide")
    app = KGRAGApp()
    
    st.title("Knowledge Graph Explorer with RAG")
    
    tab1, tab2, tab3 = st.tabs(["Process URLs", "Explore Graph", "QA System"])
    
    with tab1:
        st.header("Process Websites")
        urls = st.text_area("Enter URLs (one per line)", height=100).split("\n")
        if st.button("Build Knowledge Graph"):
            app.process_urls([url.strip() for url in urls if url.strip()])
            
        if 'processed' in st.session_state:
            st.success(f"Graph built with {app.agent.graph.number_of_nodes()} nodes and "
                      f"{app.agent.graph.number_of_edges()} relationships")
    
    with tab2:
        st.header("Explore Knowledge Graph")
        if 'processed' in st.session_state:
            app.visualize_graph()
            
            st.subheader("Graph Query")
            col1, col2 = st.columns(2)
            
            with col1:
                node_query = st.text_input("Search nodes")
                if node_query:
                    nodes = [
                        (n, d) for n, d in app.agent.graph.nodes(data=True)
                        if node_query.lower() in d['name'].lower()
                    ]
                    if nodes:
                        st.markdown("**Matching Nodes:**")
                        for node in nodes:
                            st.markdown(f"- {node[1]['name']} ({node[0]})")
            
            with col2:
                rel_query = st.selectbox("Relationship types", 
                                        ["All", "hierarchy", "similar_concept", "temporal_relation"])
                if rel_query != "All":
                    edges = [
                        (u, v, d) for u, v, d in app.agent.graph.edges(data=True)
                        if d['label'] == rel_query
                    ]
                    st.markdown(f"**{len(edges)} {rel_query} relationships**")
        else:
            st.info("Process some URLs first!")
    
    with tab3:
        st.header("Question Answering System")
        question = st.text_input("Ask about the content")
        if question:
            app.rag_qa(question)

if __name__ == "__main__":
    main()
