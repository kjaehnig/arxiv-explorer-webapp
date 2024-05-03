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
import spacy



# all arxiv taxonomic categories
arxiv_categories = {
    # Physics
    'astro-ph': 'Astrophysics',
    'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
    'astro-ph.EP': 'Earth and Planetary Astrophysics',
    'astro-ph.GA': 'Astrophysics of Galaxies',
    'astro-ph.HE': 'High Energy Astrophysical Phenomena',
    'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
    'astro-ph.SR': 'Solar and Stellar Astrophysics',
    'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
    'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
    'cond-mat.mtrl-sci': 'Materials Science',
    'cond-mat.other': 'Other Condensed Matter',
    'cond-mat.quant-gas': 'Quantum Gases',
    'cond-mat.soft': 'Soft Condensed Matter',
    'cond-mat.stat-mech': 'Statistical Mechanics',
    'cond-mat.str-el': 'Strongly Correlated Electrons',
    'cond-mat.supr-con': 'Superconductivity',
    'gr-qc': 'General Relativity and Quantum Cosmology',
    'hep-ex': 'High Energy Physics - Experiment',
    'hep-lat': 'High Energy Physics - Lattice',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-th': 'High Energy Physics - Theory',
    'math-ph': 'Mathematical Physics',
    'nlin.AO': 'Adaptation and Self-Organizing Systems',
    'nlin.CG': 'Cellular Automata and Lattice Gases',
    'nlin.CD': 'Chaotic Dynamics',
    'nlin.SI': 'Exactly Solvable and Integrable Systems',
    'nlin.PS': 'Pattern Formation and Solitons',
    'nucl-ex': 'Nuclear Experiment',
    'nucl-th': 'Nuclear Theory',
    'physics.acc-ph': 'Accelerator Physics',
    'physics.ao-ph': 'Atmospheric and Oceanic Physics',
    'physics.atom-ph': 'Atomic Physics',
    'physics.atm-clus': 'Atomic and Molecular Clusters',
    'physics.bio-ph': 'Biological Physics',
    'physics.chem-ph': 'Chemical Physics',
    'physics.class-ph': 'Classical Physics',
    'physics.comp-ph': 'Computational Physics',
    'physics.data-an': 'Data Analysis, Statistics and Probability',
    'physics.flu-dyn': 'Fluid Dynamics',
    'physics.gen-ph': 'General Physics',
    'physics.geo-ph': 'Geophysics',
    'physics.hist-ph': 'History and Philosophy of Physics',
    'physics.ins-det': 'Instrumentation and Detectors',
    'physics.med-ph': 'Medical Physics',
    'physics.optics': 'Optics',
    'physics.ed-ph': 'Physics Education',
    'physics.soc-ph': 'Physics and Society',
    'physics.plasm-ph': 'Plasma Physics',
    'physics.pop-ph': 'Popular Physics',
    'physics.space-ph': 'Space Physics',
    'quant-ph': 'Quantum Physics',

    # Mathematics
    'math.AC': 'Commutative Algebra',
    'math.AG': 'Algebraic Geometry',
    'math.AP': 'Analysis of PDEs',
    'math.AT': 'Algebraic Topology',
    'math.CA': 'Classical Analysis and ODEs',
    'math.CO': 'Combinatorics',
    'math.CT': 'Category Theory',
    'math.CV': 'Complex Variables',
    'math.DG': 'Differential Geometry',
    'math.DS': 'Dynamical Systems',
    'math.FA': 'Functional Analysis',
    'math.GM': 'General Mathematics',
    'math.GN': 'General Topology',
    'math.GR': 'Group Theory',
    'math.GT': 'Geometric Topology',
    'math.HO': 'History and Overview',
    'math.IT': 'Information Theory',
    'math.KT': 'K-Theory and Homology',
    'math.LO': 'Logic',
    'math.MP': 'Mathematical Physics',
    'math.MG': 'Metric Geometry',
    'math.NT': 'Number Theory',
    'math.NA': 'Numerical Analysis',
    'math.OA': 'Operator Algebras',
    'math.OC': 'Optimization and Control',
    'math.PR': 'Probability',
    'math.QA': 'Quantum Algebra',
    'math.RT': 'Representation Theory',
    'math.RA': 'Rings and Algebras',
    'math.SP': 'Spectral Theory',
    'math.ST': 'Statistics Theory',
    'math.SG': 'Symplectic Geometry',

    # Computer Science
    'cs.AI': 'Artificial Intelligence',
    'cs.AR': 'Hardware Architecture',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.CL': 'Computation and Language',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.DB': 'Databases',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LG': 'Machine Learning',
    'cs.LO': 'Logic in Computer Science',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.MS': 'Mathematical Software',
    'cs.NA': 'Numerical Analysis',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.OH': 'Other Computer Science',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SC': 'Symbolic Computation',
    'cs.SD': 'Sound',
    'cs.SE': 'Software Engineering',
    'cs.SI': 'Social and Information Networks',
    'cs.SY': 'Systems and Control',
    'eess.AS': 'Audio and Speech Processing',
    'eess.IV': 'Image and Video Processing',
    'eess.SP': 'Signal Processing',

    # Other categories
    'econ.EM': 'Econometrics',
    'q-bio': 'Quantitative Biology',
    'q-fin': 'Quantitative Finance',
    'stat.AP': 'Applications',
    'stat.CO': 'Computation',
    'stat.ML': 'Machine Learning',
    'stat.ME': 'Methodology',
    'stat.OT': 'Other Statistics',
    'stat.TH': 'Theory',
}

color_palette = [
    "#E6194B",  # Bright Red
    "#3CB44B",  # Bright Green
    "#FFE119",  # Bright Yellow
    "#4363D8",  # Bright Blue
    "#F58231",  # Bright Orange
    "#911EB4",  # Light Purple
    "#42D4F4",  # Bright Cyan
    "#F032E6",  # Bright Magenta
    "#BFEF45",  # Light Lime
    "#FABEBE",  # Soft Pink
    "#469990",  # Desaturated Teal
    "#DCBEFF",  # Light Lavender
    "#9A6324",  # Ochre
    "#FFFAC8",  # Light Beige
    "#800000",  # Cranberry
    "#AAFFC3",  # Pale Mint
    "#808000",  # Olive
    "#FFD8B1",  # Peach
    "#FFFEB7",  # Light Yellow
    "#A9A9A9",  # Bright Grey
    "#FFD700",  # Gold
    "#7FFFD4",  # Aquamarine
    "#AA6E28",  # Bronze
    "#FFC0CB",  # Pink
    "#008080",  # Teal
    "#0000FF",  # Pure Blue
    "#FA8072",  # Salmon
    "#32CD32",  # Lime Green
    "#00008B",  # Dark Blue
    "#800080",  # Purple
]

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

@st.cache_resource
def load_nlp_model():
    # Load a pre-trained NLP model
    nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_nlp_model()

def summarize_title_with_ner(titles):
    summaries = []
    for title in titles:
        doc = nlp(title)
        # Extract entities and some other potentially important nouns
        keywords = [token.text for token in doc if token.ent_type_ or token.pos_ in ['NOUN', 'PROPN']]
        summaries.append(" ".join(keywords[:4]))  # Keep only the first 4
    return summaries

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
    group_color = st.session_state.get('group_color', True)
    mst = st.session_state.get('mst', False)

    # Conditional logic to disable checkboxes based on the state of the other
    if group_color:
        group_color_chkbox = st.sidebar.checkbox('Group Color', value=group_color, key='group_color')
        mst_chkbox = st.sidebar.checkbox('MST', value=mst, disabled=True, key='mst')
    elif mst:
        group_color_chkbox = st.sidebar.checkbox('Group Color', value=group_color, disabled=True, key='group_color')
        mst_chkbox = st.checkbox('MST', value=mst, key='mst')
    else:
        group_color_chkbox = st.sidebar.checkbox('Group Color', value=group_color, key='group_color')
        mst_chkbox = st.sidebar.checkbox('MST', value=mst, key='mst')

    # show_legend = st.sidebar.checkbox("Display Graph Legend",
    #                                   value=False,
    #                                   disabled=True if mst_chkbox else False)

    # Display the current state of checkboxes (for demonstration)
    st.write('Group Color:', group_color)
    st.write('MST:', mst)


# def calculate_category_groups_dfs(papers):
#     from collections import defaultdict
#     import itertools
#
#     # Create a dictionary to hold the category connections
#     category_connections = defaultdict(set)
#
#     # Map each paper to a set of its categories
#     paper_categories = [set(categories) for _, _, _, categories in papers]
#
#     # Compare each paper's categories with every other paper's categories
#     for i, categories_i in enumerate(paper_categories):
#         for j, categories_j in enumerate(paper_categories):
#             if i != j:
#                 common_categories = categories_i.intersection(categories_j)
#                 if common_categories:
#                     category_connections[i].update(common_categories)
#
#     # Generate a group number based on connectivity
#     visited = set()
#     group_id = 0
#     paper_group = {}
#
#     # Simple DFS to assign groups based on connected category components
#     def dfs(paper_index, group_id):
#         stack = [paper_index]
#         while stack:
#             node = stack.pop()
#             if node not in visited:
#                 visited.add(node)
#                 paper_group[node] = group_id
#                 for neighbour in category_connections[node]:
#                     if neighbour not in visited:
#                         stack.append(neighbour)
#
#     for paper_index in range(len(papers)):
#         if paper_index not in visited:
#             dfs(paper_index, group_id)
#             group_id += 1
#
#     return paper_group

# def calculate_category_groups_dfs(papers):
#     from collections import defaultdict
#
#     # Function to parse categories into main and sub-subject components
#     def parse_categories(categories):
#         parsed_categories = set()
#         for cat in categories:
#             if '-' in cat:
#                 main_cat, sub_cat = cat.split('-', 1)
#                 full_name = arxiv_categories.get(cat, cat)
#                 main_subject = full_name.split(' - ')[0] if ' - ' in full_name else full_name
#                 parsed_categories.add((main_cat, sub_cat, main_subject))
#             else:
#                 parsed_categories.add((cat, '', arxiv_categories.get(cat, cat)))
#         return parsed_categories
#
#     # Create a dictionary to hold the category connections
#     category_connections = defaultdict(set)
#
#     # Map each paper to a set of its parsed categories
#     paper_categories = [parse_categories(categories) for _, _, _, categories in papers]
#
#     # Compare each paper's categories with every other paper's categories
#     for i, categories_i in enumerate(paper_categories):
#         for j, categories_j in enumerate(paper_categories):
#             if i != j:
#                 common_categories = categories_i.intersection(categories_j)
#                 if common_categories:
#                     category_connections[i].update(common_categories)
#
#     # Generate a group number based on connectivity
#     visited = set()
#     group_id = 0
#     paper_group = {}
#
#     # Simple DFS to assign groups based on connected category components
#     def dfs(paper_index, group_id):
#         stack = [paper_index]
#         while stack:
#             node = stack.pop()
#             if node not in visited:
#                 visited.add(node)
#                 paper_group[node] = group_id
#                 for neighbour in category_connections[node]:
#                     if neighbour not in visited:
#                         stack.append(neighbour)
#
#     for paper_index in range(len(papers)):
#         if paper_index not in visited:
#             dfs(paper_index, group_id)
#             group_id += 1
#
#     return paper_group

def calculate_category_groups_dfs(papers):
    from collections import defaultdict

    # Assume categories are parsed to extract both main and sub-subject components
    def parse_categories(categories):
        parsed = set()
        for cat in categories:
            main_cat, sub_cat = cat.split('-', 1) if '-' in cat else (cat, '')
            full_name = arxiv_categories.get(cat, "Other")
            parsed.add((main_cat, sub_cat, full_name))
        return parsed

    # Create a dictionary to hold the category connections
    category_connections = defaultdict(set)
    paper_categories = [parse_categories(categories) for _, _, _, categories in papers]

    for i, categories_i in enumerate(paper_categories):
        for j, categories_j in enumerate(paper_categories):
            if i != j:
                common_categories = categories_i.intersection(categories_j)
                if common_categories:
                    category_connections[i].update(common_categories)

    visited = set()
    group_id = 0
    paper_group = {}

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
    """
    Build an interactive network graph based on abstract similarity,
    labeling nodes with unique categories.
    """
    net = Network(height="700px",
                  width="100%",
                  bgcolor="#222222",
                  font_color="white",
                  notebook=True)
    net.force_atlas_2based(gravity=-45,
                           central_gravity=0.02,
                           spring_length=100,
                           spring_strength=0.05)

    # dict for used group colors
    # group_colors = {}

    # Set to keep track of already used primary categories
    used_categories = set()
    group_details = {}

    unique_labels = []
    # Add nodes with primary categories, ensuring uniqueness where possible
    for i, (title, _, primary_category, _) in enumerate(papers):
        if primary_category in used_categories:
            label = f"{primary_category} ({i})"
            unique_labels.append(label)
        else:
            used_categories.add(primary_category)
            label = primary_category
            unique_labels.append(label)

    if group_color_chkbox:
        # Calculate group identifiers based on category overlap
        paper_group = calculate_category_groups_dfs(papers)

        # for i, (title, _, primary_category, _) in enumerate(papers):
        #     net.add_node(i, label=unique_labels[i], title=title, group=paper_group[i])

        # Assign unique colors and create unique labels based on group and category info
        # Initialize group details
        for index, (title, _, primary_category, categories) in enumerate(papers):
            group = paper_group[index]
            group_label = f"{primary_category.split('-')[0]}-{arxiv_categories.get(primary_category, 'Other')}"
            # primary_category = categories[0] if categories else "Unknown"
            title_important_words = ' '.join([wr for wr in title.split() if wr not in stop_words])

            if group not in group_details:
                group_details[group] = {
                    'category': primary_category.split('.')[0],
                    'papers': [],
                    'color': color_palette[len(group_details) % len(color_palette)],
                    'group_label':group_label,
                    'title': title_important_words
                }
            group_details[group]['papers'].append(title)

            net.add_node(index, label=primary_category, title=title, color=group_details[group]['color'])

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
        for i, node in enumerate(mst.nodes):
            title, _, primary_category, categories = papers[node]
            group = paper_group[node]
            group_label = f"{primary_category.split('-')[0]}-{arxiv_categories.get(primary_category, 'Other')}"
            # primary_category = categories[0] if categories else "Unknown"
            title_important_words = ' '.join([wr for wr in title.split() if wr not in stop_words])

            if group not in group_details:
                group_details[group] = {
                    'category': primary_category.split('.')[0],
                    'papers': [],
                    'color': color_palette[len(group_details) % len(color_palette)],
                    'group_label':group_label,
                    'title': title_important_words
                }
            group_details[group]['papers'].append(title)
            net.add_node(node, label=title_important_words, title=title, group=paper_group[node], color=group_details[group]['color'])

        for i, j in mst.edges:
            net.add_edge(i, j, value=float(overlap_weights[(i, j)]))

    path = "arxiv_network.html"
    net.save_graph(path)
    net.show(path)
    return path, group_details


# Function to display the legend with interactive buttons
def display_interactive_legend(group_details):
    cols_per_row = 3  # Define the number of columns in the grid

    # Iterate over the groups and create buttons with corresponding expanders
    for idx, (group_label, details) in enumerate(group_details.items()):
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row)  # Create a new row of columns

        # Get the correct column for the current item
        col = cols[idx % cols_per_row]

        # Define a unique key for each button based on its index
        button_key = f"button_{idx}"
        # Button CSS to set the background color and style
        button_style = f"background-color: {details['color']}; color: white; border: none; border-radius: 5px; width: 100%;"
        button_html = f"<style>.{button_key} {{ {button_style} }}</style>"

        with col:
            st.markdown(button_html, unsafe_allow_html=True)
            # Render the button and check if it has been pressed
            if st.button(f"Group-{idx}", key=button_key, help=f"Show papers for {group_label}"):
                # Display an expander with the list of papers in this group
                with st.expander(f"Papers in Group-{idx}"):
                    for paper in details['papers']:
                        st.write(paper)

def display_groups_with_expanders(group_details):
    for group_label, details in group_details.items():
        # Create unique keys for button and expander based on group label
        button_key = f"button_{group_label}"
        expander_key = f"expander_{group_label}"

        # Define a button with custom color and full width
        button_style = f"background-color: {details['color']}; color: white; border-radius: 8px; border: none; width: 100%;"
        button_html = f"<button style='{button_style}' onclick='document.getElementById(\"{expander_key}\").style.display = \"block\";'>{group_label}</button>"

        # Render the button
        st.markdown(button_html, unsafe_allow_html=True)

        # Initially hidden expander for showing the papers
        expander_html = f"<div id='{expander_key}' style='display: none;'>"
        with st.container():
            if st.session_state.get(button_key, False):
                expander = st.expander(f"Papers in {group_label}", expanded=True)
                with expander:
                    for paper in details['papers']:
                        st.write(paper)
            else:
                expander = st.expander(f"Papers in {group_label}")
                with expander:
                    for paper in details['papers']:
                        st.write(paper)
            st.session_state[button_key] = expander.expanded

# Display the interactive legend
# display_groups_with_expanders(group_details)

st.title('arXiv Paper Explorer')

# User input for subtopic
subtopic = st.text_input('Enter a subtopic to search:', 'machine learning')

if st.button('Fetch Papers'):
    papers = fetch_papers(subtopic, max_results)
    if papers:
        st.write(f"Found {len(papers)} papers on '{subtopic}'.")

        # Calculate similarities and build the network graph
        similarity_matrix = calculate_cosine_similarity(papers)
        network_path, group_details = build_interactive_network(papers, similarity_matrix)
        HtmlFile = open(network_path, 'r', encoding='utf-8')
        st.components.v1.html(HtmlFile.read(), height=700)

        # if show_legend:
        #     display_groups_with_expanders(group_details)

        if print_out_paper_summaries:
            # Display paper titles and summaries
            for title, summary, _, cat in papers:
                with st.expander(title + f"(found in {cat}"):
                    summary_response = summarize_abstract(summary)
                    st.write(summary_response)
    else:
        st.write("No papers found.")