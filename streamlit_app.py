import streamlit as st
import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
from transformers import pipeline
import re
from collections import Counter

@st.cache_resource
def load_pipeline_summarizer():
    # Initialize the BART summarization pipeline
    # summarizer = pipeline("summarization", "Falconsai/text_summarization")
    summarizer = pipeline("summarization", "pszemraj/led-base-book-summary")
    return summarizer

summarizer = load_pipeline_summarizer()

@st.cache_resource
def load_sentence_transformer():
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_transformer

model = load_sentence_transformer()

def fetch_papers(subtopic):
    """Fetch papers from the arXiv API based on a subtopic."""
    url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': f'all:{subtopic}',
        'start': 0,
        'max_results': 5
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        papers.append((title, summary))
    return papers

def find_important_word(title, summary):
    """Find the most important word from the title based on the abstract."""
    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    summary_words = re.findall(r'\b\w+\b', summary.lower())
    common_words = title_words.intersection(summary_words)
    if common_words:
        # Select the most frequent common word in the summary
        summary_word_count = Counter(summary_words)
        most_important = sorted(common_words, key=lambda x: summary_word_count[x], reverse=True)[0]
        return most_important
    return title.split()[0]  # Fallback to the first word of the title if no common word is found


def summarize_abstract(abstract):
    """Generate a summary for an abstract using a pretrained model."""
    summary_text = summarizer(abstract, max_length=130, min_length=30, do_sample=False)
    return summary_text[0]['summary_text']


def calculate_similarity(papers):
    """Calculate similarity between paper abstracts using embeddings."""
    abstracts = [summary for _, summary in papers]
    embeddings = model.encode(abstracts)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def build_interactive_network(papers, similarity_matrix, threshold=0.3):
    """Build an interactive network graph based on abstract similarity."""
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.force_atlas_2based()

    for i, (title, _, important_word) in enumerate(papers):
        net.add_node(i, label=important_word, title=title)

    # Add edges based on similarity score
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if similarity_matrix[i][j] > threshold:
                net.add_edge(i, j, value=similarity_matrix[i][j])

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

        # Calculate similarities and build the network graph
        similarity_matrix = calculate_similarity(papers)
        network_path = build_interactive_network(papers, similarity_matrix)
        st.components.v1.html(open(network_path, 'r').read(), height=800)

        # Display paper titles and summaries
        for title, summary, _ in papers:
            with st.expander(title):
                summary_response = summarize_abstract(summary)
                st.write(summary_response)
    else:
        st.write("No papers found.")