import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="RAGOps Developer UI",
    page_icon="🔎",
    layout="wide"
)


st.title("RAGOps Developer UI")
st.write("Upload documents and test semantic search.")


st.header("1. Upload document")

uploaded_file = st.file_uploader(
    "Choose a TXT or Markdown file",
    type=["txt", "md"]
)

if uploaded_file is not None:
    if st.button("Upload and process"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(f"{API_BASE_URL}/upload", files=files)

        if response.status_code == 200:
            st.success("File uploaded and processed successfully.")
            st.json(response.json())
        else:
            st.error("Upload failed.")
            st.text(response.text)


st.divider()


st.header("2. Semantic search")

query = st.text_input("Ask a question about your documents")

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=3
)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        payload = {
            "query": query,
            "top_k": top_k
        }

        response = requests.post(f"{API_BASE_URL}/search", json=payload)

        if response.status_code == 200:
            data = response.json()

            st.subheader("Results")

            for index, result in enumerate(data["results"], start=1):
                score = result["score"]
                source_file = result["source_file"]
                chunk_id = result["chunk_id"]
                text = result["text"]

                with st.expander(
                    f"Result {index} | Score: {score:.4f} | Source: {source_file} | Chunk: {chunk_id}",
                    expanded=index == 1
                ):
                    st.write(text)
        else:
            st.error("Search failed.")
            st.text(response.text)