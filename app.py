import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Banking KYC Assistant",
    page_icon="🏦"
)

st.title("🏦 Banking KYC Assistant")
st.write("Ask questions from KYC/AML policy documents")

# ==========================
# LOAD PDF
# ==========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        text += page.get_text()

    st.success("PDF Loaded Successfully")

    # ==========================
    # CHUNKING
    # ==========================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # ==========================
    # EMBEDDINGS
    # ==========================

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    # ==========================
    # FAISS
    # ==========================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    # ==========================
    # LOAD FLAN-T5
    # ==========================

    @st.cache_resource
    def load_llm():

        tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-base"
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base"
        )

        return tokenizer, model

    tokenizer, model = load_llm()

    # ==========================
    # SEARCH FUNCTION
    # ==========================

    def search_document(
        question,
        top_k=3
    ):

        query_embedding = embedding_model.encode(
            [question]
        ).astype("float32")

        distances, indices = index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx in indices[0]:
            results.append(
                chunks[idx]
            )

        return results

    # ==========================
    # RAG
    # ==========================

    def rag_answer(question):

        retrieved_chunks = search_document(
            question
        )

        context = "\n\n".join(
            retrieved_chunks
        )

        prompt = f"""
        Answer only using the context.

        Context:
        {context}

        Question:
        {question}
        """

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer

    # ==========================
    # CHAT UI
    # ==========================

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("Get Answer"):

        answer = rag_answer(
            question
        )

        st.subheader("Answer")

        st.write(answer)
