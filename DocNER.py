import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import spacy_streamlit as spt
from spacy import displacy
import fitz  
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()


st.set_page_config(
    page_title="DocNER",
    page_icon="📄",
    layout="centered"
)


working_dir = os.path.dirname(os.path.abspath(__file__))

# Load NER models
model_name = "blaze999/Medical-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
medical_ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
nlp_spacy = spacy.load("en_core_web_sm")

# RAG functions
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Background styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/021/599/588/small/abstract-white-and-gray-overlap-circles-background-3d-paper-circle-banner-with-drop-shadows-minimal-simple-design-for-presentation-flyer-brochure-website-book-etc-vector.jpg");
    background-size: cover;
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.5);
}
.css-1d391kg {
    color: white !important;
}
.css-1v0mbdj {
    color: white !important;
}
.css-1d391kg > div {
    background-color: #4CAF50;
    color: white;
}
.css-1d391kg > div:hover {
    background-color: #3e8e41;
}
.fixed-title {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    # background-color: rgba(200, 200, 200, 0.8);  /* Set the background color to a grey shade */
    color: grey;
    text-align: center;
    padding: 10px;
    z-index: 1000;
}
</style>
"""

# Entity types
entity_types = {
    "🧑 PERSON": "People, including fictional.",
    "🌍 NORP": "Nationalities or religious or political groups.",
    "🏢 FAC": "Buildings, airports, highways, bridges, etc.",
    "🏢 ORG": "Companies, agencies, institutions, etc.",
    "🌍 GPE": "Countries, cities, states.",
    "📍 LOC": "Non-GPE locations, mountain ranges, bodies of water.",
    "📦 PRODUCT": "Objects, vehicles, foods, etc. (Not services.)",
    "🏟️ EVENT": "Named hurricanes, battles, wars, sports events, etc.",
    "🎨 WORK_OF_ART": "Titles of books, songs, etc.",
    "📜 LAW": "Named documents made into laws.",
    "🗣️ LANGUAGE": "Any named language.",
    "📅 DATE": "Absolute or relative dates or periods.",
    "⏰ TIME": "Times smaller than a day.",
    "💯 PERCENT": "Percentage, including '%'.",
    "💰 MONEY": "Monetary values, including unit.",
    "📏 QUANTITY": "Measurements, as of weight or distance.",
    "🔢 ORDINAL": "\"First\", \"Second\", etc.",
    "🔢 CARDINAL": "Numerals that do not fall under another type."
}

def main():
    # st.markdown(page_bg_img, unsafe_allow_html=True)
    # st.markdown('<h1 style="color: grey;">DocNER - NER & Document Chat Assistant</h1>', unsafe_allow_html=True)
    
    # st.sidebar.markdown('<h1 style="color: white;">Menu</h1>', unsafe_allow_html=True)
    # menu = ['Tokenize', 'NER', 'Upload File']
    # choice = st.sidebar.selectbox('', menu)

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<div class="fixed-title"><h1 style="color: grey;">DocNER : Document Chat Assistant with NER</h1></div>', unsafe_allow_html=True)
    
    # Add some margin to avoid overlap with the fixed title
    st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)

    st.sidebar.markdown('<h1 style="color: white;">Menu</h1>', unsafe_allow_html=True)
    menu = ['Tokenize', 'NER', 'Upload File']
    choice = st.sidebar.selectbox('', menu)

    # Notation section in sidebar
    st.sidebar.markdown('<h3 style="color: white;">Notations</h3>', unsafe_allow_html=True)
    with st.sidebar.expander("Entity Notations"):
        st.markdown("<p style='color:white; font-weight:bold;'>Entity Types:</p>", unsafe_allow_html=True)
        for symbol, description in entity_types.items():
            st.markdown(f"<p style='color:white;'><b>{symbol}:</b> {description}</p>", unsafe_allow_html=True)

    # Initialize session state for RAG
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if choice == 'NER':
        st.markdown('<h2 style="color: white;">Named Entity Recognition</h2>', unsafe_allow_html=True)
        text = st.text_area("", "", placeholder="Enter text here...")

        if st.button("Submit"):
            if text:
                doc_spacy = nlp_spacy(text)
                spacy_output = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc_spacy.ents]
                medical_output = medical_ner(text)
                formatted_medical_output = [(entity['word'], entity['entity_group'], entity['start'], entity['end']) 
                                         for entity in medical_output]
                
                combined_output = spacy_output + formatted_medical_output
                processed_entities = []
                for entity, label, start, end in combined_output:
                    if label == "O" or not entity.strip():
                        continue
                    processed_entities.append({"start": start, "end": end, "label": label})

                doc_dict = {
                    "text": text,
                    "ents": [{"start": ent["start"], "end": ent["end"], "label": ent["label"]} 
                            for ent in processed_entities],
                    "title": None
                }
                html = displacy.render(doc_dict, style="ent", manual=True)
                st.markdown(html, unsafe_allow_html=True)

    elif choice == 'Tokenize':
        st.markdown('<h2 style="color: white;">Word Tokenization</h2>', unsafe_allow_html=True)
        raw_text = st.text_area("", "", placeholder="Enter text here...")

        if st.button("Tokenize"):
            docs = nlp_spacy(raw_text)
            if raw_text.strip():
                spt.visualize_tokens(docs)

    elif choice == 'Upload File':
        st.subheader("Upload a Text or PDF File")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

        if uploaded_file:
            # Create tabs for NER and RAG
            tab1, tab2 = st.tabs(["Named Entity Recognition", "Chat with Document"])

            with tab1:
                if st.button("Process for NER"):
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    else:
                        text = uploaded_file.read().decode("utf-8")

                    st.write("File Contents:")
                    st.text(text)

                    if text:
                        doc_spacy = nlp_spacy(text)
                        spacy_output = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                                      for ent in doc_spacy.ents]
                        medical_output = medical_ner(text)
                        formatted_medical_output = [(entity['word'], entity['entity_group'], 
                                                   entity['start'], entity['end']) 
                                                  for entity in medical_output]
                        
                        combined_output = spacy_output + formatted_medical_output
                        processed_entities = []
                        for entity, label, start, end in combined_output:
                            if label == "O" or not entity.strip():
                                continue
                            processed_entities.append({"start": start, "end": end, "label": label})

                        doc_dict = {
                            "text": text,
                            "ents": [{"start": ent["start"], "end": ent["end"], "label": ent["label"]} 
                                    for ent in processed_entities],
                            "title": None
                        }
                        html = displacy.render(doc_dict, style="ent", manual=True)
                        st.markdown(html, unsafe_allow_html=True)

            with tab2:
                if uploaded_file.type == "application/pdf":
                    # Create a temporary file path for RAG processing
                    file_path = os.path.join(working_dir, uploaded_file.name)
                    try:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        if "vectorstore" not in st.session_state:
                            st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

                        if "conversation_chain" not in st.session_state:
                            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

                        # Display chat interface
                        for message in st.session_state.chat_history:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

                        user_input = st.chat_input("Ask about the document...")

                        if user_input:
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            
                            with st.chat_message("user"):
                                st.markdown(user_input)

                            with st.chat_message("assistant"):
                                response = st.session_state.conversation_chain({"question": user_input})
                                assistant_response = response["answer"]
                                st.markdown(assistant_response)
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": assistant_response})

                    finally:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                else:
                    st.warning("RAG feature is only available for PDF files.")

if __name__ == '__main__':
    main()