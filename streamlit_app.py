import streamlit as st
import requests
import xml.etree.ElementTree as ET
from transformers import pipeline
import networkx as nx
from pyvis.network import Network

@st.cache(allow_output_mutation=True)
def load_pipeline_summarizer():
    # Initialize the BART summarization pipeline
    summarizer = pipeline("summarization", "facebook/bart-large-cnn")
    return summarizer

summarizer = load_pipeline_summarizer()

def fetch_papers(subtopic):
    """Fetch papers from the arXiv API based on a subtopic."""
    url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': f'all:{subtopic}',
        'start': 0,
        'max_results': 10
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        papers.append((title, summary))
    return papers

def summarize_text(text):
    """Generate a summary for a general audience using BART."""
    summary_text = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary_text[0]['summary_text']

def build_interactive_network(papers):
    """Build an interactive network graph using pyvis."""
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.force_atlas_2based()

    for title, _ in papers:
        words = title.split()
        for word in words:
            net.add_node(word, title=word)

    for title, _ in papers:
        words = title.split()
        for i in range(len(words) - 1):
            net.add_edge(words[i], words[i+1])

    # Save and read the network to integrate it with Streamlit
    path = "tmp/arxiv_network.html"
    net.show(path)
    return path

st.title('arXiv Paper Explorer')

# User input for subtopic
subtopic = st.text_input('Enter a subtopic to search:', 'machine learning')

if st.button('Fetch Papers'):
    papers = fetch_papers(subtopic)
    if papers:
        st.write(f"Found {len(papers)} papers on '{subtopic}'.")

        # Display paper titles and summaries
        for title, summary in papers:
            with st.expander(title):
                summary_response = summarize_text(summary)
                st.write(summary_response)

        # Build and display the interactive network graph
        network_path = build_interactive_network(papers)
        st.components.v1.html(open(network_path, 'r').read(), height=800)
    else:
        st.write("No papers found.")
