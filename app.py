"""
RAG Knowledge Base
 
A simple Retrieval-Augmented Generation (RAG) app built with
Streamlit, LangChain, and ChromaDB. No API keys needed!
 
"""
 
import streamlit as st
import numpy as np
 
st.set_page_config(
    page_title="Macedonian Cuisine",
    page_icon="𓌉◯𓇋",
    layout="wide",
)
 
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background-color: #f7f5f5;
    border: 1px solid #991212;
    padding: 12px;
    border-radius: 12px;
}
div[data-testid="stMetric"] label {
    color: #991212 !important;
    font-weight: 600;
}
div[data-testid="stMetricValue"] > div {
    color: #1a1a1a !important;
}
</style>
""", unsafe_allow_html=True)
# ──────────────────────────────────────────────────────────────────────
# DOCUMENTS — Macedonian Cuisine
# ──────────────────────────────────────────────────────────────────────

DOCUMENTS = [

    """Macedonian cuisine is a Balkan cuisine shaped by Ottoman, Byzantine, and
Mediterranean influences. It is based on fresh vegetables, beans, dairy
products, bread, grilled meats, and aromatic herbs. Seasonal ingredients
are key: fresh vegetables dominate spring and summer, while pickled and
preserved foods are used in winter. Food is strongly connected to
hospitality, family meals, and social gatherings.""",

    """Tavče Gravče is the national dish of North Macedonia. It is a baked bean
dish made with white beans, onions, peppers, oil, and mint, cooked in a
traditional clay pot called a tavče. The pot gives the dish a creamy inside
and a slightly crispy top. It is served as a main dish or with grilled meat
and is popular during Orthodox fasting because it can be made without meat
or dairy.""",

    """Ajvar is a roasted red pepper spread and one of the most recognizable
foods in Macedonia. Peppers are roasted, peeled, ground, and slowly cooked
with oil and salt. Some versions include eggplant, garlic, or chili. Ajvar
is eaten with bread, cheese, or grilled meats and is a key part of meze.
Families often prepare it together in autumn and store it in jars for
winter.""",

    """Kebapi are small grilled rolls of minced meat, made from beef, pork, or
a mix of both. They are cooked over charcoal and served with lepinja bread,
raw onion, ajvar, and white cheese. Kebapi are found in grill restaurants,
street food stands, and home gatherings. They are one of the most popular
everyday grilled dishes in Macedonia.""",

    """Pastrmajlija is a regional specialty from Shtip in eastern Macedonia.
It is an oval flatbread topped with pieces of salted or dried pork called
pastrma, baked in a hot oven until the bread is crispy and the meat is
rich and aromatic. Pastrmajlija is eaten hot and is considered one of
Macedonia's best-known regional dishes.""",

    """Shopska Salata is one of the most common salads in Macedonian cuisine.
It is made from tomatoes, cucumbers, onions, and grated white brined
cheese called sirenje, dressed with oil. Stuffed vegetables are also
popular: polneti piperki (stuffed peppers) are filled with rice, minced
meat, herbs, and spices, then baked in tomato sauce. These dishes are
everyday staples of Macedonian home cooking.""",

    """Macedonian dairy products appear in many everyday meals. Sirenje is a
soft white brined cheese, salty and tangy, eaten with bread, salads, and
grilled meats. Kashkaval is a hard yellow cheese that can be served cold,
melted, or fried. Kiselo mleko is a sour milk similar to yogurt and is
eaten with bread, burek, soups, and beans.""",

    """Bread and pastries are central to Macedonian food culture. Pogača is a
round bread prepared for everyday meals and celebrations. Burek is thin
filo pastry filled with cheese, minced meat, or spinach, usually eaten
for breakfast with kiselo mleko. Zelnik is similar but filled with
greens, leek, egg, and cheese. Macedonian sweets include baklava, tulumbi,
halva, and sutlija, which is rice pudding made from rice, milk, and sugar.""",

    """North Macedonia produces wine and rakija. The main grape variety is
Vranec, a dark full bodied red wine grown in the Vardar region.
International varieties such as Cabernet Sauvignon, Merlot, and
Chardonnay are also produced. Rakija is a traditional fruit brandy made
from grapes, plums, or quince. It is served before and after meals and
during celebrations. Sharing rakija is a gesture of hospitality and
friendship.""",

    """Macedonian cuisine varies by region. The west, near Lake Ohrid and Lake
Prespa, is known for freshwater fish such as Ohrid trout. Eastern
Macedonia is known for rice, peppers, tomatoes, and pastrmajlija. Skopje
blends Macedonian, Turkish, Albanian, and Roma traditions, especially in
the old Čaršija bazaar. Meze is shared small plates of cheese, meats,
ajvar, and pickles, and Orthodox fasting dishes made from beans,
lentils, and vegetables are important parts of the food culture.""",

]



# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────
 
@st.cache_resource(show_spinner="Loading embedding model...")

class SimpleVectorStore:
    def __init__(self, chunks, matrix, vectorizer):
        self.chunks = chunks
        self.matrix = matrix
        self.vectorizer = vectorizer

    def similarity_search_with_score(self, query, k=3):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_k = np.argsort(scores)[::-1][:k]

        class Doc:
            def __init__(self, content):
                self.page_content = content

        return [(Doc(self.chunks[i]), 1 - float(scores[i]))
                for i in top_k]

@st.cache_resource(show_spinner="Building vector database...")
def build_vector_store(_documents: tuple):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)

    return SimpleVectorStore(chunks, matrix, vectorizer), chunks

# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 𓌉◯𓇋 Macedonian Cuisine")
st.sidebar.caption("Semantic search app")
st.sidebar.markdown("---")
 
from streamlit_option_menu import option_menu
 
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Search", "Examples", "Gallery", "Explore Chunks", "About"],
        icons=["house", "search", "lightbulb", "images", "boxes", "info-circle"],
        default_index=0,
        styles={
            "nav-link-selected": {"background-color": "#991212"},
        },
    )
 
page = selected
 
 
 
# ──────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────
if page == "Home":

    # ── Hero banner ────────────────────────────────────────────────
    import base64, os

    def img_to_b64(path):
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    b64 = img_to_b64("image.jpg")
    img_src = f"data:image/jpeg;base64,{b64}" if b64 else ""

    st.markdown(f"""
        <div style="
            position: relative;
            width: 100%;
            height: 420px;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 36px;
        ">
            <img src="{img_src}" style="
                width: 100%; height: 100%;
                object-fit: cover;
                display: block;
                filter: brightness(0.45);
            "/>
            <div style="
                position: absolute;
                top: 0; left: 0;
                width: 100%; height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                padding: 20px;
            ">
                <p style="
                    color: #f0a0a0;
                    font-size: 0.9rem;
                    letter-spacing: 4px;
                    text-transform: uppercase;
                    margin-bottom: 12px;
                ">A culinary journey through the Balkans</p>
                <h1 style="
                    color: #ffffff;
                    font-size: 3rem;
                    font-weight: 800;
                    margin: 0 0 12px 0;
                    line-height: 1.15;
                    text-shadow: 0 2px 12px rgba(0,0,0,0.5);
                ">Македонска Кујна</h1>
                <p style="
                    color: #e8e8e8;
                    font-size: 1.15rem;
                    max-width: 560px;
                    margin: 0;
                    line-height: 1.6;
                ">Explore centuries of tradition — from clay-pot beans<br>to roasted pepper relish and homemade brandy.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Pull quote ─────────────────────────────────────────────────
    st.markdown("""
        <div style="
            border-left: 5px solid #991212;
            background-color: #f7f5f5;
            border-radius: 0 12px 12px 0;
            padding: 18px 24px;
            margin-bottom: 36px;
        ">
            <p style="
                font-size: 1.2rem;
                font-style: italic;
                color: #3a3a3a;
                margin: 0 0 6px 0;
                line-height: 1.6;
            ">"Offering food and drink to a visitor is considered a basic
            expression of respect and friendship."</p>
            <p style="
                font-size: 0.82rem;
                color: #991212;
                font-weight: 600;
                margin: 0;
                letter-spacing: 1px;
                text-transform: uppercase;
            ">— Macedonian Hospitality Tradition</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Feature cards ──────────────────────────────────────────────
    st.markdown("### What's inside")

    features = [
        ("🫙", "Iconic Dishes",      "Tavče Gravče, Ajvar, Pastrmajlija — the pillars of Macedonian cooking."),
        ("🗺️", "Regional Flavours",  "From Ohrid's freshwater trout to Shtip's wood-fired pastrmajlija."),
        ("🧀", "Dairy & Bread",      "Sirenje, kashkaval, burek, zelnik, baklava — staples of every Macedonian table."),
        ("🍷", "Wine & Rakija",      "Vranec reds and fruit brandy rooted in centuries of Balkan viticulture."),
        ("🌿", "Fasting & Meze",     "Plant-based dishes shaped by Orthodoxy and shared small plates for every table."),
        ("🥗", "Salads & Vegetables","Shopska Salata, stuffed peppers, and seasonal produce central to everyday meals."),
    ]

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
                <div style="
                    border: 1px solid #991212;
                    border-top: 4px solid #991212;
                    border-radius: 12px;
                    padding: 20px 18px;
                    margin-bottom: 18px;
                    background-color: #f7f5f5;
                    min-height: 130px;
                ">
                    <div style="font-size: 1.8rem; margin-bottom: 8px;">{icon}</div>
                    <p style="
                        margin: 0 0 6px 0;
                        font-size: 0.95rem;
                        font-weight: 700;
                        color: #5e0909;
                    ">{title}</p>
                    <p style="
                        margin: 0;
                        font-size: 0.83rem;
                        color: #444;
                        line-height: 1.5;
                    ">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    # ── CTA strip ──────────────────────────────────────────────────
    st.markdown(f"""
        <div style="
            background-color: #991212;
            border-radius: 12px;
            padding: 24px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 8px;
        ">
            <div>
                <p style="color: #fff; font-size: 1.1rem; font-weight: 700; margin: 0 0 4px 0;">
                    Ready to explore?
                </p>
                <p style="color: #f0c0c0; font-size: 0.88rem; margin: 0;">
                    Knowledge base contains <strong style="color:#fff">{len(DOCUMENTS)} documents</strong>
                    — ask anything about Macedonian cuisine.
                </p>
            </div>
            <div style="
                background: #fff;
                color: #991212;
                font-weight: 700;
                padding: 10px 22px;
                border-radius: 8px;
                font-size: 0.9rem;
                white-space: nowrap;
            ">Open Search →</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.caption("Built with Streamlit · LangChain · ChromaDB · TF-IDF (scikit-learn)")
 
 
 
# ──────────────────────────────────────────────────────────────────────
# SEARCH PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Search":

    st.markdown("""
        <div style="
            background-color: #991212;
            border-radius: 12px;
            padding: 28px 32px;
            margin-bottom: 28px;
        ">
            <p style="color:#f0a0a0; font-size:0.85rem; letter-spacing:3px;
                      text-transform:uppercase; margin-bottom:8px;">
                Semantic search
            </p>
            <h1 style="color:#fff; margin:0 0 8px 0; font-size:2rem; font-weight:800;">
                ⌕ Search Macedonian Cuisine
            </h1>
            <p style="color:#f0c0c0; margin:0; font-size:0.95rem;">
                Ask anything — the app finds the most relevant passages by meaning, not just keywords.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        div[data-testid="stTextInput"] input {
            border: 1px solid #991212 !important;
            border-radius: 8px !important;
            background-color: #f7f5f5 !important;
        }
        div[data-testid="stTextInput"] input:focus {
            box-shadow: 0 0 0 2px #99121233 !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background-color: #991212 !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] > div:first-child {
            background-color: #991212 !important;
        }
        div[data-testid="stMetric"] {
            background-color: #f7f5f5 !important;
            border: 1px solid #991212 !important;
            border-top: 4px solid #991212 !important;
            border-radius: 10px !important;
            padding: 16px !important;
        }
        div[data-testid="stMetric"] label {
            color: #991212 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1a1a1a !important;
            font-size: 2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hint cards
    hints = [
        "What is tavče gravče?",
        "What are the most popular grilled meat dishes?",
        "How does Macedonian cuisine differ by region?",
        "What wines are produced in North Macedonia?",
    ]
    hint_html = "".join([
        f"""<span style="
            display:inline-block;
            background:#f7f5f5;
            border:1px solid #991212;
            border-radius:20px;
            padding:5px 14px;
            margin:4px;
            font-size:0.82rem;
            color:#5e0909;
            font-weight:500;
        ">{h}</span>"""
        for h in hints
    ])
    st.markdown(f"""
        <div style="margin-bottom:20px;">
            <p style="font-size:0.8rem; color:#888; margin-bottom:8px;
                      text-transform:uppercase; letter-spacing:2px;">
                Example questions
            </p>
            {hint_html}
        </div>
    """, unsafe_allow_html=True)

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Your question",
                              placeholder="e.g. What are the most popular grilled meat dishes?")
    with col2:
        num_results = st.slider("Results", 1, 3, 3)

    if query:
        with st.spinner("Searching..."):
            results = vector_store.similarity_search_with_score(query, k=num_results)

        st.markdown(f"""
            <p style="font-size:0.8rem; color:#991212; font-weight:700;
                      text-transform:uppercase; letter-spacing:2px; margin:20px 0 12px 0;">
                Top {len(results)} results
            </p>
        """, unsafe_allow_html=True)

        for i, (doc, score) in enumerate(results, 1):
            similarity = max(0, 1 - score)
            st.markdown(f"""
                <div style="
                    border: 1px solid #991212;
                    border-left: 5px solid #991212;
                    border-radius: 10px;
                    padding: 16px 20px;
                    margin-bottom: 12px;
                    background-color: #f7f5f5;
                ">
                    <div style="display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:10px;">
                        <span style="color:#991212; font-weight:700;
                                     font-size:0.85rem;">Result {i}</span>
                        <span style="background:#991212; color:#fff;
                                     padding:3px 10px; border-radius:20px;
                                     font-size:0.78rem; font-weight:600;">
                            {similarity:.0%} match
                        </span>
                    </div>
                    <p style="margin:0; color:#333; font-size:0.9rem;
                              line-height:1.6;">{doc.page_content}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Documents", len(DOCUMENTS))
    stat2.metric("Chunks", len(chunks))
    stat3.metric("Query length", len(query) if query else 0)
    st.caption("Powered by TF-IDF (scikit-learn) embeddings + ChromaDB") 
 
# 
 
 
# ──────────────────────────────────────────────────────────────────────
# GALLERY PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Gallery":
    import base64, os

    def img_to_b64(path):
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # ── Hero banner ────────────────────────────────────────────────
    st.markdown("""
        <div style="
            background-color: #991212;
            border-radius: 12px;
            padding: 28px 32px;
            margin-bottom: 28px;
        ">
            <p style="color:#f0a0a0; font-size:1rem; letter-spacing:3px;
                      text-transform:uppercase; margin-bottom:8px;">
                Visual tour
            </p>
            <h1 style="color:#fff; margin:0 0 8px 0; font-size:2rem; font-weight:900;">
                Food Gallery
            </h1>
            <p style="color:#f0c0c0; margin:0; font-size:0.95rem;">
                A journey through the colours, textures, and flavours of Macedonian cuisine.
            </p>
        </div>
    """, unsafe_allow_html=True)

    gallery_items = [
        ("ajvar.jpg",          "Ajvar",         "Condiment", "Roasted red pepper relish, slow cooked with oil and salt."),
        ("pastrmajlija.jpg",   "Pastrmajlija",  "Specialty", "Oval baked dough topped with cured pork, a specialty of Shtip."),
        ("rakija.jpg",         "Rakija",         "Beverage",  "Traditional fruit brandy, the cornerstone of Macedonian hospitality."),
        ("shopska_salata.jpg", "Shopska Salata", "Salad",     "The iconic salad of tomato, cucumber, and grated white cheese."),
        ("tavche_gravche.jpg", "Tavče Gravče",   "Main dish", "Baked white beans in an earthenware pot."),
        ("zelnik.jpg",         "Zelnik",         "Pastry",    "Flaky filo pastry filled with greens, leek, and cheese."),
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, (filename, title, tag, caption_text) in enumerate(gallery_items):
        b64 = img_to_b64(filename)
        img_src = f"data:image/jpeg;base64,{b64}" if b64 else ""

        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #991212;
                    border-radius: 12px;
                    overflow: hidden;
                    margin-bottom: 20px;
                    background-color: #f7f5f5;
                    box-shadow: 0 2px 8px rgba(153,18,18,0.08);
                ">
                    <img src="{img_src}"
                         style="width:100%; height:250px;
                                object-fit:cover; display:block;" />
                    <div style="padding: 14px 16px 16px 16px;">
                        <div style="display:flex; align-items:center;
                                    justify-content:space-between; margin-bottom:6px;">
                            <p style="margin:0; font-size:1rem; font-weight:700;
                                      color:#5e0909;">{title}</p>
                            <span style="background:#991212; color:#fff;
                                         padding:2px 10px; border-radius:20px;
                                         font-size:0.72rem; font-weight:600;
                                         white-space:nowrap;">{tag}</span>
                        </div>
                        <p style="margin:0; font-size:0.83rem; color:#444;
                                  line-height:1.5;">{caption_text}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("All dishes are traditional to North Macedonian cuisine.")
 
 
 
# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Explore Chunks":
    import altair as alt
    import pandas as pd

    # ── Hero banner ────────────────────────────────────────────────
    st.markdown("""
        <div style="
            background-color: #991212;
            border-radius: 12px;
            padding: 28px 32px;
            margin-bottom: 28px;
        ">
            <p style="color:#f0a0a0; font-size:0.85rem; letter-spacing:3px;
                      text-transform:uppercase; margin-bottom:8px;">
                Under the hood
            </p>
            <h1 style="color:#fff; margin:0 0 8px 0; font-size:2rem; font-weight:800;">
                Explore Chunks
            </h1>
            <p style="color:#f0c0c0; margin:0; font-size:0.95rem;">
                See how documents are sliced into passages that power semantic search.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        div[data-testid="stMetric"] {
            background-color: #f7f5f5 !important;
            border: 1px solid #991212 !important;
            border-top: 4px solid #991212 !important;
            border-radius: 10px !important;
            padding: 16px !important;
        }
        div[data-testid="stMetric"] label {
            color: #991212 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #1a1a1a !important;
            font-size: 2rem !important;
        }
        div[data-testid="stTextInput"] input {
            border: 1px solid #991212 !important;
            border-radius: 8px !important;
            background-color: #f7f5f5 !important;
        }
        div[data-testid="stTextInput"] input:focus {
            box-shadow: 0 0 0 2px #99121233 !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #991212 !important;
            border-left: 5px solid #991212 !important;
            border-radius: 10px !important;
            background-color: #f7f5f5 !important;
            margin-bottom: 8px !important;
        }
        div[data-testid="stExpander"] summary {
            color: #991212 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpander"] summary:hover {
            color: #5e0909 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))
    lengths = [len(c) for c in chunks]

    # ── Stat cards ─────────────────────────────────────────────────
    st.markdown("""
        <p style="font-size:0.8rem; color:#991212; font-weight:700;
                  text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;">
            Corpus statistics
        </p>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total chunks",    len(chunks))
    s2.metric("Avg chunk size",  f"{int(np.mean(lengths))} chars")
    s3.metric("Min chunk size",  f"{min(lengths)} chars")
    s4.metric("Max chunk size",  f"{max(lengths)} chars")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart ──────────────────────────────────────────────────────
    st.markdown("""
        <p style="font-size:0.8rem; color:#991212; font-weight:700;
                  text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;">
            Chunk length distribution
        </p>
    """, unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "chunk": range(1, len(lengths) + 1),
        "chars": lengths
    })
    st.altair_chart(
        alt.Chart(chart_df).mark_bar(color="#991212", opacity=0.85).encode(
            x=alt.X("chunk:O", axis=None, title="Chunks"),
            y=alt.Y("chars:Q", title="Characters"),
            tooltip=[
                alt.Tooltip("chunk:O", title="Chunk"),
                alt.Tooltip("chars:Q", title="Length (chars)")
            ],
        ).properties(height=260).configure_axis(
            labelColor="#444",
            titleColor="#991212",
        ).configure_view(strokeOpacity=0),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter & expanders ─────────────────────────────────────────
    st.markdown("""
        <p style="font-size:0.8rem; color:#991212; font-weight:700;
                  text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;">
            Browse chunks
            </p>
             """, unsafe_allow_html=True)

    keyword = st.text_input(
        "Filter by keyword",
        placeholder="e.g. ajvar"
    )

    filtered_chunks = chunks
    if keyword:
        filtered_chunks = [c for c in chunks if keyword.lower() in c.lower()]

    st.markdown(f"""
        <div style="
            background-color: #f7f5f5;
            border: 1px solid #991212;
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 14px;
            display: inline-block;
            ">
            <span style="color:#991212; font-weight:700;">
                {len(filtered_chunks)}
              </span>
            
        </div>
    """, unsafe_allow_html=True)

    for i, chunk in enumerate(filtered_chunks, 1):
        with st.expander(f"Chunk {i}  ·  {len(chunk)} chars"):
            st.text(chunk)

    st.markdown("---")
    st.caption("Powered by  TF-IDF (scikit-learn) embeddings + ChromaDB")

# ──────────────────────────────────────────────────────────────────────
# ABOUT PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "About":

    st.markdown("""
        <div style="
            background-color: #991212;
            border-radius: 12px;
            padding: 28px 32px;
            margin-bottom: 28px;
        ">
            <p style="color:#f0a0a0; font-size:0.85rem; letter-spacing:3px;
                      text-transform:uppercase; margin-bottom:8px;">
                Under the hood
            </p>
            <h1 style="color:#fff; margin:0 0 8px 0; font-size:2rem; font-weight:800;">
                About This App
            </h1>
            <p style="color:#f0c0c0; margin:0; font-size:0.95rem;">
                A semantic search tool built on Macedonian culinary knowledge.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # How it works — numbered steps
    st.markdown("""
        <p style="font-size:0.8rem; color:#991212; font-weight:700;
                  text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;">
            How it works
        </p>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Chunk",   "Documents are split into small overlapping passages using RecursiveCharacterTextSplitter."),
        ("02", "Embed",   "Each chunk is converted into a vector of numbers by the all-MiniLM-L6-v2 model."),
        ("03", "Store",   "Vectors are indexed in ChromaDB, a local in-memory vector database."),
        ("04", "Search",  "Your query is embedded and compared to every chunk by cosine similarity."),
        ("05", "Return",  "The closest matching chunks are ranked and returned as results."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
            <div style="
                display:flex;
                align-items:flex-start;
                gap:18px;
                border: 1px solid #991212;
                border-left: 5px solid #991212;
                border-radius: 10px;
                padding: 16px 20px;
                margin-bottom: 10px;
                background-color: #f7f5f5;
            ">
                <span style="
                    font-size:1.4rem;
                    font-weight:900;
                    color:#e8c8c8;
                    min-width:36px;
                    line-height:1.2;
                ">{num}</span>
                <div>
                    <p style="margin:0 0 4px 0; font-weight:700;
                              color:#5e0909; font-size:0.95rem;">{title}</p>
                    <p style="margin:0; color:#444;
                              font-size:0.88rem; line-height:1.5;">{desc}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech stack cards
    st.markdown("""
        <p style="font-size:0.8rem; color:#991212; font-weight:700;
                  text-transform:uppercase; letter-spacing:2px; margin-bottom:14px;">
            Technical setup
        </p>
    """, unsafe_allow_html=True)

    stack = [
        ("🧠", "Embedding model", "TF-IDF (scikit-learn)"),
        ("🗄️", "Vector database",  "ChromaDB"),
        ("✂️", "Chunking method",  "RecursiveCharacterTextSplitter"),
        ("📐", "Chunk size",       "400 chars / 50 overlap"),
        ("🖥️", "Framework",        "Streamlit + LangChain"),
        ("📚", "Documents",        f"{len(DOCUMENTS)} knowledge base entries"),
    ]

    tc1, tc2, tc3 = st.columns(3)
    tcols = [tc1, tc2, tc3]
    for i, (icon, label, value) in enumerate(stack):
        with tcols[i % 3]:
            st.markdown(f"""
                <div style="
                    border: 1px solid #991212;
                    border-top: 4px solid #991212;
                    border-radius: 10px;
                    padding: 16px;
                    margin-bottom: 14px;
                    background-color: #f7f5f5;
                    text-align: center;
                ">
                    <div style="font-size:1.6rem; margin-bottom:6px;">{icon}</div>
                    <p style="margin:0 0 4px 0; font-size:0.75rem; color:#991212;
                              font-weight:700; text-transform:uppercase;
                              letter-spacing:1px;">{label}</p>
                    <p style="margin:0; font-size:0.95rem;
                              color:#1a1a1a; font-weight:600;">{value}</p>
                </div>
            """, unsafe_allow_html=True)

    st.caption("Built with Streamlit · LangChain · ChromaDB · TF-IDF (scikit-learn)")
 
    
# ──────────────────────────────────────────────────────────────────────
# EXAMPLES PAGE
# ──────────────────────────────────────────────────────────────────────
 
elif page == "Examples":
    st.markdown("""
        <div style="
            background-color: #991212;
            border-radius: 12px;
            padding: 28px 32px;
            margin-bottom: 28px;
        ">
            <p style="color:#f0a0a0; font-size:0.85rem; letter-spacing:3px;
                      text-transform:uppercase; margin-bottom:8px;">
                Try it yourself
            </p>
            <h1 style="color:#fff; margin:0 0 8px 0; font-size:2rem; font-weight:800;">
                Example Queries
            </h1>
            <p style="color:#f0c0c0; margin:0; font-size:0.95rem;">
                Click any question below to instantly search the knowledge base.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* Style the example buttons */
        div[data-testid="stButton"] button {
            background-color: #f7f5f5 !important;
            border: 1px solid #991212 !important;
            border-left: 5px solid #991212 !important;
            border-radius: 10px !important;
            color: #5e0909 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 14px 20px !important;
            width: 100% !important;
            text-align: left !important;
            margin-bottom: 4px !important;
            transition: background-color 0.2s !important;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #f0e8e8 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    example_queries = [
     "What is Tavče Gravče?",
     "How is ajvar made and served?",
     "What are kebapi served with?",
     "What is Shopska Salata made of?",
     "What dairy products are common in Macedonian cuisine?",
     "What is Vranec wine?",
     "What pastries and sweets are popular in Macedonia?",
     "How does Macedonian cuisine differ by region?",]

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    for q in example_queries:
        if st.button(q):
            results = vector_store.similarity_search_with_score(q, k=3)

            st.markdown(f"""
                <div style="
                    background-color: #f7f5f5;
                    border-left: 5px solid #991212;
                    border-radius: 0 12px 12px 0;
                    padding: 14px 20px;
                    margin: 16px 0 8px 0;
                ">
                    <p style="margin:0; font-size:0.8rem; color:#991212;
                              font-weight:700; text-transform:uppercase; letter-spacing:2px;">
                        Searching for
                    </p>
                    <p style="margin:4px 0 0 0; font-size:1rem;
                              color:#3a3a3a; font-style:italic;">"{q}"</p>
                </div>
            """, unsafe_allow_html=True)

            for i, (doc, score) in enumerate(results, 1):
                similarity = max(0, 1 - score)
                st.markdown(f"""
                    <div style="
                        border: 1px solid #991212;
                        border-left: 5px solid #991212;
                        border-radius: 10px;
                        padding: 16px 20px;
                        margin-bottom: 12px;
                        background-color: #f7f5f5;
                    ">
                        <div style="display:flex; justify-content:space-between;
                                    align-items:center; margin-bottom:10px;">
                            <span style="color:#991212; font-weight:700;
                                         font-size:0.85rem;">Result {i}</span>
                            <span style="background:#991212; color:#fff;
                                         padding:3px 10px; border-radius:20px;
                                         font-size:0.78rem; font-weight:600;">
                                {similarity:.0%} match
                            </span>
                        </div>
                        <p style="margin:0; color:#333; font-size:0.9rem;
                                  line-height:1.6;">{doc.page_content}</p>
                    </div>
                """, unsafe_allow_html=True)