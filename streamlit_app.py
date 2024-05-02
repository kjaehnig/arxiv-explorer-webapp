import streamlit as st
import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import networkx as nx
from transformers import pipeline
import re
from collections import Counter, defaultdict, deque
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
    max_results = st.slider('Max Arxiv Queries (MAQ)', min_value=5, max_value=30, value=5)
    if max_results > 10:
        st.warning("Setting MAQ above 10 may slow down the app.")

    thresh_value = st.slider('Similarity Threshold', min_value=0.01, max_value=0.99, value=0.25)

    print_out_paper_summaries = st.sidebar.checkbox('Get summaries with Sentence-Transformer?', value=False)
    if print_out_paper_summaries:
        st.warning("This is currently slow. May crash with MAQ > 20.")

    # Label for the group of checkboxes
    st.subheader('Network Layout')

    # Check if either checkbox is already selected (preserves state across runs)
    group_color = st.session_state.get('group_color', False)
    mst = st.session_state.get('mst', False)

    # Conditional logic to disable checkboxes based on the state of the other
    if group_color:
        mst_chkbox = st.sidebar.checkbox('MST', value=mst, disabled=True, key='mst')
        group_color_chkbox = st.sidebar.checkbox('Group Color', value=group_color, key='group_color')
    elif mst:
        group_color_chkbox = st.sidebar.checkbox('Group Color', value=group_color, disabled=True, key='group_color')
        mst_chkbox = st.checkbox('MST', value=mst, key='mst')
    else:
        group_color_chkbox = st.sidebar.checkbox('Group Color', value=group_color, key='group_color')
        mst_chkbox = st.sidebar.checkbox('MST', value=mst, key='mst')



    # Display the current state of checkboxes (for demonstration)
    st.write('Group Color:', group_color)
    st.write('MST:', mst)


def calculate_category_groups_dfs(papers):
    from collections import defaultdict
    import itertools

    # Create a dictionary to hold the category connections
    category_connections = defaultdict(set)

    # Map each paper to a set of its categories
    paper_categories = [set(categories) for _, _, _, categories in papers]

    # Compare each paper's categories with every other paper's categories
    for i, categories_i in enumerate(paper_categories):
        for j, categories_j in enumerate(paper_categories):
            if i != j:
                common_categories = categories_i.intersection(categories_j)
                if common_categories:
                    category_connections[i].update(common_categories)

    # Generate a group number based on connectivity
    visited = set()
    group_id = 0
    paper_group = {}

    # Simple DFS to assign groups based on connected category components
    def dfs(paper_index, group_id):
        stack = [paper_index]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                paper_group[node] = group_id
                for neighbour in category_connections[node]:
                    if neighbour not in visited:
                        stack.append(neighbour)

    for paper_index in range(len(papers)):
        if paper_index not in visited:
            dfs(paper_index, group_id)
            group_id += 1

    return paper_group

def calculate_category_groups_bfs(papers):
    # Map each paper to a set of its categories
    paper_categories = [set(categories) for _, _, _, categories in papers]

    # Create an adjacency list based on category overlap
    adjacency_list = defaultdict(list)
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            overlap = len(paper_categories[i].intersection(paper_categories[j]))
            if overlap > 0:  # There is some overlap
                adjacency_list[i].append((j, overlap))
                adjacency_list[j].append((i, overlap))

    # Use BFS to assign groups based on connectivity
    visited = set()
    group_id = 0
    paper_group = {}
    overlap_weights = {}

    def bfs(start):
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                paper_group[node] = group_id
                for neighbour, weight in adjacency_list[node]:
                    if neighbour not in visited:
                        queue.append(neighbour)
                        overlap_weights[(node, neighbour)] = weight

    for paper_index in range(len(papers)):
        if paper_index not in visited:
            bfs(paper_index)
            group_id += 1

    return paper_group, overlap_weights

def fetch_papers(subtopic, max_results=5):
    """Fetch papers from the arXiv API based on a subtopic and retrieve their fields."""
    url = 'http://export.arxiv.org/api/query'

    # Construct the query to search across all categories ('cat:*') and include the subtopic in the title/abstract
    # query = f'all:{subtopic} AND cat:*'
    params = {
        'search_query': f"all: {subtopic}",
        'start': 0,
        'max_results': max_results
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.content)
    papers = []

    ns = {'atom': 'http://www.w3.org/2005/Atom'}  # Namespace dictionary
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        all_categories = [category.get('term') for category in entry.findall('atom:category', ns)]
        primary_category = next((category.get('term') for category in entry.findall('atom:category', ns)
                                 if category.get('primary') == 'true'), all_categories[0] if all_categories else 'Uncategorized')

        papers.append((title, summary, primary_category, all_categories))

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
    abstracts = [summary for _, summary, _, _ in papers]
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
    for _, summary, _, _ in papers:
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


def build_interactive_network(papers, similarity_matrix, threshold=0.25):
    """Build an interactive network graph based on abstract similarity, labeling nodes with unique categories."""
    net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.set_options("""
    var options = {
      nodes: {
        font: {
          size: 24,  // Set font size to 24
        }
      }
    }
    """)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.05)

    if group_color_chkbox:
        # Calculate group identifiers based on category overlap
        paper_group = calculate_category_groups_dfs(papers)

        # Set to keep track of already used primary categories
        used_categories = set()

        # Add nodes with primary categories, ensuring uniqueness where possible
        for i, (title, _, primary_category, _) in enumerate(papers):
            if primary_category in used_categories:
                label = f"{primary_category} ({i})"
            else:
                used_categories.add(primary_category)
                label = primary_category

            net.add_node(i, label=label, title=title, group=paper_group[i])

        # Add edges based on similarity score
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                if similarity_matrix[i][j] > threshold:
                    net.add_edge(i, j, value=float(similarity_matrix[i][j]))
    if mst_chkbox:
        # Calculate group identifiers and overlap weights
        paper_group, overlap_weights = calculate_category_groups_bfs(papers)

        # Build a graph to calculate MST
        G = nx.Graph()
        for (i, j), weight in overlap_weights.items():
            G.add_edge(i, j, weight=-weight)  # Negative weight for maximum overlap (MST inverts to minimum)

        mst = nx.minimum_spanning_tree(G, weight='weight')  # Calculate MST

        # Add nodes and edges from MST to pyvis network
        for node in mst.nodes:
            title, _, primary_category, _ = papers[node]
            net.add_node(node, label=primary_category, title=title, group=paper_group[node])

        for i, j in mst.edges:
            net.add_edge(i, j, value=float(overlap_weights[(i, j)]))

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
        similarity_matrix = calculate_cosine_similarity(papers)
        network_path = build_interactive_network(papers, similarity_matrix)
        st.components.v1.html(open(network_path, 'r').read(), height=800)

        if print_out_paper_summaries:
            # Display paper titles and summaries
            for title, summary, _, cat in papers:
                with st.expander(title + f"(found in {cat}"):
                    summary_response = summarize_abstract(summary)
                    st.write(summary_response)
    else:
        st.write("No papers found.")