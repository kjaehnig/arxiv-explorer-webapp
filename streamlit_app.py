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
import nltk


stopwords_not_downloaded = False

try:
    stop_words = set(nltk.corpus.stopwords.words("english"))
except:
    stopwords_not_downloaded = True

if stopwords_not_downloaded:
    st.write("downloading NLTK stopwords...")
    nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words("english"))

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


with st.sidebar:
    st.header("Control Panel")
    # Sampling number input
    max_results = st.slider('N (Max Results)', min_value=5, max_value=50, value=5)
    if max_results > 10:
        st.warning("Setting N above 10 may slow down the app.")

    thresh_value = st.slider('Threshold', min_value=0.01, max_value=0.9, value=0.25)


def fetch_papers(subtopic, max_results=5):
    """Fetch papers from the arXiv API based on a subtopic."""
    url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': f'all:{subtopic}',
        'start': 0,
        'max_results': max_results
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
        important_word = find_important_word(title, summary)
        papers.append((title, summary, important_word))  # Append tuple with all necessary elements
    return papers

def find_important_word(title, summary):
    """Find the most important word from the title based on the abstract."""
    title_words = set(re.findall(r'\b\w+\b', title.lower())) - stop_words
    summary_words = re.findall(r'\b\w+\b', summary.lower())
    summary_words = [w for w in summary_words if w not in stop_words]

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


def calculate_cosine_similarity(papers):
    """Calculate similarity between paper abstracts using embeddings."""
    abstracts = [summary for _, summary, _ in papers]
    embeddings = model.encode(abstracts)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def jaccard_similarity(set1, set2):
    """Calculate Jaccard Similarity between two sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def calculate_similarity(papers):
    """Calculate similarity between paper abstracts using Jaccard similarity."""
    # Convert abstracts to sets of words excluding stopwords
    abstract_sets = []
    for _, summary, _ in papers:
        words = set(re.findall(r'\b\w+\b', summary.lower()))
        filtered_words = {word for word in words if word not in stop_words}
        abstract_sets.append(filtered_words)

    # Calculate Jaccard similarity for each pair of abstract sets
    num_papers = len(abstract_sets)
    similarity_matrix = np.zeros((num_papers, num_papers))

    for i in range(num_papers):
        for j in range(i + 1, num_papers):
            similarity = jaccard_similarity(abstract_sets[i], abstract_sets[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Mirror the similarity

    return similarity_matrix



def build_interactive_network(papers, similarity_matrix, threshold=thresh_value):
    """Build an interactive network graph based on abstract similarity."""
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.force_atlas_2based()

    for i, (title, _, important_word) in enumerate(papers):
        net.add_node(i, label=important_word, title=title)

    # Add edges based on similarity score
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if similarity_matrix[i][j] > threshold:
                net.add_edge(i, j, value=float(similarity_matrix[i][j]))

    path = "arxiv_network.html"
    net.save_graph(path)
    net.show(path)
    return path

st.title('arXiv Paper Explorer')

# User input for subtopic
subtopic = st.text_input('Enter a subtopic to search:', 'machine learning')

if st.button('Fetch Papers'):
    papers = fetch_papers(subtopic, max_results)
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