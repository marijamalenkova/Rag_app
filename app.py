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

    """Macedonian cuisine is one of the richest and most distinctive food 
traditions in the Balkans. Shaped by centuries of Ottoman, Byzantine, 
and Mediterranean influence, as well as by the country's varied geography 
and climate, it offers a deeply satisfying combination of flavors, 
techniques, and ingredients. The cuisine is largely based on fresh 
vegetables, slow-cooked meats, dairy products, legumes, and aromatic 
herbs.

A defining characteristic of Macedonian food is its emphasis on 
freshness and seasonality. Meals are built around whatever is available 
locally and in season, with preservation techniques such as drying, 
pickling, and fermenting playing an important role in extending the 
harvest throughout the year. The result is a cuisine that changes with 
the seasons and remains closely connected to the agricultural landscape.

Meals in Macedonia are rarely rushed. Eating together is a social 
activity, and hospitality is expressed through generous portions, 
numerous dishes, and a warm welcome to guests. Offering food and drink 
to a visitor is considered a basic expression of respect and friendship, 
and refusing hospitality can be seen as impolite. This culture of 
generosity shapes every aspect of how food is prepared and shared.""",


    """Tavce gravce is widely regarded as the national dish of North 
Macedonia. It is a hearty baked bean dish prepared with dried white 
beans, onions, peppers, and a blend of spices including dried red pepper 
and mint. The beans are first cooked and then finished in a traditional 
earthenware pot called a tavce, which gives the dish both its name and 
its characteristic flavor. The slow baking in a clay vessel creates a 
slightly crispy top layer while keeping the inside creamy and rich.

The dish is often served as a main course on its own or alongside grilled 
meats. It is also a common choice for fasting periods in the Orthodox 
Christian calendar, as it contains no meat and is naturally filling and 
nutritious. Different families and regions have their own variations of 
the recipe, with adjustments in the type of pepper used, the amount of 
fat, and the choice of additional vegetables.

Tavce gravce is present at family gatherings, celebrations, and everyday 
tables alike. It represents the broader value placed on simple, 
ingredient-driven cooking in Macedonian food culture, where patience in 
preparation and quality of raw materials matter more than elaborate 
technique.""",


    """Ajvar is a roasted red pepper relish that is among the most iconic 
condiments in Macedonian cuisine. It is made primarily from fleshy red 
peppers, which are first roasted over an open flame or in a wood-fired 
oven, then peeled, ground, and cooked down slowly with oil and salt. 
Some versions include roasted eggplant, which adds depth and a slightly 
smoky character. Garlic, vinegar, and chili peppers are used in varying 
amounts depending on regional and family traditions.

The preparation of ajvar is one of the most significant annual food 
events in Macedonia. In late summer and early autumn, when peppers are 
at peak ripeness and available in abundance, families gather to make 
large quantities of ajvar together. The process involves roasting 
hundreds of peppers, peeling them by hand, and stirring the mixture 
over heat for several hours until it reaches the right consistency. 
This collective effort reinforces family bonds and community ties.

Once cooled and jarred, ajvar is stored for use throughout the winter 
months. It is served as a spread on bread, as a side dish, as a topping 
for grilled meats, and as a flavor base for cooked dishes. Ajvar is 
so deeply embedded in Macedonian food culture that it is considered 
a symbol of the country's culinary identity.""",


    """Macedonian grilled meats are central to the country's food culture. 
The most common forms include kebapi, which are small elongated rolls 
of seasoned minced meat, typically made from beef, pork, or a mixture 
of the two, and cooked directly on a charcoal grill. Kebapi are usually 
served in a flatbread called lepinja, accompanied by raw onion, ajvar, 
and fresh cheese. They are found at dedicated grilling restaurants, 
at street food stalls, and at home gatherings across the country.

Pastrmajlija is another beloved meat dish, originating from the city 
of Shtip in eastern Macedonia. It consists of pieces of cured pork 
meat arranged on an oval leavened dough base and baked in a wood-fired 
oven until the edges of the dough are crispy and the meat is tender 
and aromatic. Pastrmajlija is often considered a regional specialty 
unique to Macedonia, though it has become popular throughout the country.

Sis kebab refers to cubes or pieces of marinated meat skewered and 
grilled over charcoal. Lamb, chicken, and pork are all used. Grilling 
is closely associated with festive occasions, summer gatherings, and 
weekend meals in Macedonia, and the preparation of a proper charcoal 
fire is itself considered a skill worth taking seriously.""",


    """Fresh vegetables occupy a central role in Macedonian cuisine. The 
country benefits from a continental and Mediterranean climate that 
supports abundant production of tomatoes, peppers, cucumbers, eggplant, 
zucchini, onions, garlic, and leafy greens. Vegetable dishes are eaten 
throughout the year, with seasonal variation determining what is served 
at any given time.

Shopska salata is the most widely recognized salad in Macedonian and 
regional Balkan cuisine. It is made from diced tomatoes, cucumbers, and 
raw onions, topped with finely grated white brined cheese, and dressed 
with oil and occasionally vinegar. The combination of the crisp raw 
vegetables and the salty cheese has made shopska salata a standard 
accompaniment to nearly every meal, from simple everyday lunches to 
festive spreads.

Stuffed vegetables, known locally as polneti piperki or polneta tikvicka, 
are popular dishes prepared by filling hollowed peppers or zucchini with 
a mixture of seasoned rice, minced meat, and onion, then baking them 
slowly in tomato sauce or stock. These dishes are often prepared in 
large batches and eaten over several days, as the flavors deepen on 
reheating. Seasonal soups made from fresh vegetables are also a 
significant part of the Macedonian table, particularly during spring 
and summer.""",


    """Macedonian dairy products are diverse and deeply integrated into 
the local diet. The most important is sirenje, a white brined cheese 
made from cow's, sheep's, or goat's milk. Sirenje is soft and crumbly 
with a salty, tangy flavor, and it is used in almost every context: 
crumbled over salads, served alongside grilled meats, baked into 
pastries, eaten with bread at breakfast, or combined with eggs. It is 
one of the few products found on the table at virtually every meal.

Kashkaval is a hard yellow cheese made primarily from sheep's milk or 
a blend of sheep and cow milk. It is pressed, aged, and develops a 
rich, buttery flavor that intensifies with time. Kashkaval is eaten 
on its own as part of a cold appetizer spread, melted over dishes, 
or fried in breadcrumbs as a hot starter. The sheep-milk varieties 
from the Bitola and Ohrid regions are considered particularly fine.

Kiselo mleko, a thick soured milk similar to yogurt but with a looser 
texture and sharper tang, is consumed daily in many Macedonian 
households. It is eaten with bread, alongside savory dishes, stirred 
into soups, or used as the base of drinks. Full-fat kiselo mleko from 
sheep or buffalo milk is especially prized for its richness. Dairy 
farming has a long history in the mountainous regions of Macedonia, 
and these products reflect that pastoral heritage.""",


    """Bread holds an important symbolic and practical place in Macedonian 
food culture. The most traditional form is pogaca, a round leavened 
flatbread baked in a wood-fired oven or under a clay bell. Pogaca 
can be plain or enriched with oil, cheese, or herbs. It is prepared 
for everyday meals, but also for celebrations and ritual occasions, 
where it may be decorated or made according to specific family recipes 
passed down through generations.

Burek is one of the most popular baked goods throughout North Macedonia. 
It is made from thin sheets of filo pastry layered with fillings such 
as white cheese, spinach, minced meat, or pumpkin, then baked until 
golden and flaky. Burek is commonly eaten for breakfast with kiselo 
mleko or yogurt, and bakeries producing fresh burek throughout the 
day are a feature of every Macedonian town. The cheese and meat 
varieties are the most traditional, though regional and seasonal 
variations exist.

Zelnik is a traditional pastry similar to burek, made with filo dough 
and filled with young greens, leek, or spinach mixed with egg and 
cheese. It is particularly common in spring when fresh greens are 
available. Like burek, zelnik is baked in large round trays and cut 
into portions for serving. Pastries and breads of this type form 
an important bridge between the home kitchen and everyday public 
eating culture in Macedonia.""",


    """Macedonian sweets draw heavily from Ottoman pastry traditions 
while incorporating local ingredients and preferences. Baklava is 
perhaps the most internationally recognized sweet in this tradition, 
made from many thin layers of filo pastry filled with chopped walnuts 
or other nuts, baked until crisp, and then soaked in a sweet syrup 
flavored with lemon or rose water. Macedonian baklava tends to use 
walnuts as the primary filling and is typically cut into diamond shapes.

Tulumbi are deep-fried dough pastries shaped like small ridged 
cylinders, soaked in a thick sweet syrup immediately after frying. 
They are commonly prepared during religious holidays and celebrations 
and are sold at sweet shops alongside other syrup-soaked pastries. 
The crispy exterior and sweet interior make them a popular treat 
across generations.

Halva, made from ground sesame paste or from semolina cooked with 
oil and sugar, is another widely consumed sweet with deep roots in 
Ottoman food culture. Homemade semolina halva is prepared in many 
Macedonian households as a simple dessert or as an offering during 
commemorative occasions. Sutlija, a rice pudding made with milk, 
sugar, and rice, baked slowly until a golden crust forms on top, 
is a beloved everyday dessert found in homes and small restaurants 
across the country.""",


    """North Macedonia has a long winemaking tradition, with evidence of 
viticulture dating back several thousand years. The country today 
produces wines from both international varieties such as Cabernet 
Sauvignon, Merlot, Chardonnay, and Sauvignon Blanc, and from the 
indigenous grape variety Vranec, which is the most widely planted 
and most important local variety.

Vranec produces a deeply colored, full-bodied red wine with robust 
tannins, dark fruit notes, and good acidity. It grows well in the 
warm continental climate of the Vardar wine region, which covers the 
central part of the country. Wines made from Vranec range from 
everyday table wines to structured premium bottles aged in oak. The 
variety is unique to the southern Balkans and defines the character 
of Macedonian red wine production.

Rakija is the traditional distilled spirit throughout the Balkans 
and holds a central place in Macedonian hospitality customs. It is 
typically made from grapes, plums, or quince, and is consumed as 
a welcome drink, at celebrations, after meals, and as a remedy for 
various ailments according to folk tradition. Homemade rakija is 
common, and sharing a glass is considered an important social 
gesture. The quality and strength of rakija varies widely depending 
on the producer and the fruit used.""",


    """Macedonian cuisine shows significant variation across different 
regions of the country, reflecting differences in climate, geography, 
ethnic traditions, and historical influences. The western regions, 
including Ohrid, Struga, and the mountainous areas near Albania, have 
a strong tradition of freshwater fish dishes based on species found in 
Lake Ohrid and Lake Prespa. Ohrid trout, which is a protected endemic 
species, is a celebrated local ingredient, and grilled or baked fish 
is a defining part of the diet in these areas.

In eastern Macedonia, particularly around Shtip, Kochani, and Strumica, 
the climate is warmer and produces excellent vegetables and rice. 
Kochansko rice is grown in the Kochani valley and is considered among 
the finest domestic rice varieties, used widely in traditional rice 
dishes, stuffed vegetables, and pilaf. The eastern part of the country 
also has a strong tradition of pepper growing, and many of the country's 
best ajvar comes from this region.

The capital Skopje reflects the urban diversity of Macedonian cuisine, 
where traditional dishes exist alongside influences from Turkish, 
Albanian, and Roma food cultures. The old bazaar area of Skopje, known 
as Carsija, has been a center of food and commerce for centuries, and 
the concentration of traditional eateries, sweet shops, and spice 
vendors there offers one of the best introductions to the range of 
flavors in Macedonian and regional Balkan cuisine.""",


    """Meze culture is an important aspect of Macedonian hospitality and 
dining. Meze refers to a selection of small dishes, both hot and cold, 
served at the beginning of a meal or as the main basis of a shared 
spread. A typical meze selection might include sliced sirenje, 
kashkaval, cured meats, olives, roasted peppers, ajvar, fresh 
tomatoes with oil, pickled vegetables, and bread. The variety of 
dishes allows everyone at the table to taste a little of everything.

Turshija refers to pickled vegetables and is an essential component 
of the Macedonian table, particularly in autumn and winter when 
fresh produce is less available. Cabbages, cucumbers, mixed vegetables, 
and peppers are pickled in brine or vinegar and served as a sharp, 
acidic counterpoint to rich grilled meats or heavy stews. Making 
turshija at home is a common autumn tradition, often done alongside 
the ajvar-making process.

Pindjur is a cooked condiment similar in character to ajvar but made 
from a mixture of roasted peppers, tomatoes, eggplant, and garlic, 
cooked down together until thick and concentrated. It has a more 
complex, slightly sweeter flavor than pure pepper ajvar and is used 
in similar ways: as a bread spread, a side dish, or a flavoring 
ingredient. Like ajvar, pindjur is made in large quantities in autumn 
and preserved in jars for the winter months.""",


    """The food calendar in North Macedonia is shaped by both agricultural 
seasons and religious observance. The Orthodox Christian calendar 
includes a substantial number of fasting periods throughout the year, 
during which meat, dairy products, and eggs are avoided. This has 
historically encouraged the development of a wide range of plant-based 
dishes that are satisfying, flavorful, and filling without using 
animal products.

Lenten and fasting dishes include preparations based on beans, lentils, 
chickpeas, dried mushrooms, rice, and seasonal vegetables. Many of 
these recipes are ancient and have been passed down without significant 
change over generations. The coincidence of fasting with winter months 
also means that preserved vegetables, legumes, and dried herbs are 
the main ingredients in many traditional fasting preparations.

Spring and summer bring a transformation to the Macedonian kitchen, 
as fresh herbs, young vegetables, and seasonal produce return to 
the market. Lamb is the most traditional meat associated with the 
Easter celebration, often roasted whole on a spit outdoors. Autumn 
is the season of preserving, when families invest significant time 
in making ajvar, pindjur, turshija, and other stored foods that will 
sustain the household through the colder months. Understanding this 
seasonal and religious food calendar is essential to understanding 
the full depth of Macedonian culinary culture."""

]



# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────
 
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
 
@st.cache_resource(show_spinner="Building vector database...")
def build_vector_store(_documents: tuple):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
 
    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))
 
    embeddings = load_embedding_model()
 
    # --- Store in ChromaDB ---
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="knowledge_base",
    )
    return vector_store, chunks
 
 
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
        ("🧀", "Dairy & Bread",      "Sirenje, kashkaval, burek, zelnik — staples of every Macedonian table."),
        ("🍷", "Wine & Rakija",      "Vranec reds and fruit brandy rooted in centuries of Balkan viticulture."),
        ("🌿", "Fasting Traditions", "Plant-based dishes shaped by the Orthodox Christian food calendar."),
        ("🥗", "Meze Culture",       "Small shared plates that turn every meal into a communal celebration."),
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
    st.caption("Built with Streamlit · LangChain · ChromaDB · all-MiniLM-L6-v2")
 
 
 
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
        "How is ajvar traditionally made?",
        "What dairy products are used in Macedonian cooking?",
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
        num_results = st.slider("Results", 1, 10, 3)

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
    st.caption("Powered by all-MiniLM-L6-v2 embeddings + ChromaDB") 
 
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
        ("ajvar.jpg",          "Ajvar",         "Condiment", "Roasted red pepper relish, slow-cooked with oil and salt."),
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
            <span style="color:#444; font-size:0.9rem;">
                chunk{"s" if len(filtered_chunks) != 1 else ""} 
                {"matching" if keyword else "total"}
                {f" — <em>{keyword}</em>" if keyword else ""}
            </span>
        </div>
    """, unsafe_allow_html=True)

    for i, chunk in enumerate(filtered_chunks, 1):
        with st.expander(f"Chunk {i}  ·  {len(chunk)} chars"):
            st.text(chunk)

    st.markdown("---")
    st.caption("Powered by all-MiniLM-L6-v2 embeddings + ChromaDB")

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
        ("🧠", "Embedding model", "all-MiniLM-L6-v2"),
        ("🗄️", "Vector database",  "ChromaDB"),
        ("✂️", "Chunking method",  "RecursiveCharacterTextSplitter"),
        ("📐", "Chunk size",       "500 chars / 50 overlap"),
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

    st.caption("Built with Streamlit · LangChain · ChromaDB · all-MiniLM-L6-v2")
 
    
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
        "What is the national dish of North Macedonia?",
        "How is ajvar made and when is it prepared?",
        "What cheeses are traditional in Macedonian cuisine?",
        "What grilled meat dishes are popular in Macedonia?",
        "How does the Orthodox calendar influence Macedonian food?",
    ]

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