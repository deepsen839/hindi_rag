import streamlit as st
from rag.retriever import HindiRetriever
from rag.generator import HindiGroundedGenerator, select_citations

st.set_page_config(page_title='MegaMindz Hindi RAG', page_icon='🇮🇳', layout='wide')
st.title('🇮🇳 Cross-Lingual Hindi RAG Engine')
st.caption('English/Hindi questions · Hindi Devanagari evidence · ChromaDB · PDF page + section + chunk citations')

@st.cache_resource(show_spinner='Loading BGE-M3 and ChromaDB...')
def load_retriever():
    return HindiRetriever(use_reranker=False)

@st.cache_resource(show_spinner='Loading Qwen2.5 on CPU...')
def load_generator():
    return HindiGroundedGenerator()

with st.sidebar:
    st.header('Retrieval settings')
    top_k = st.slider('Evidence passages', 1, 10, 5)
    hybrid = st.checkbox('Hindi BM25 + dense hybrid retrieval', value=True)
    generate = st.checkbox('Generate Hindi answer with Qwen', value=True)
    st.caption('English questions use multilingual BGE-M3 dense retrieval; BM25 is fused only for Devanagari queries.')

question = st.text_input('Ask in English or Hindi', placeholder='Where did Kalam study aeronautical engineering? / कलाम ने वैमानिकी कहाँ पढ़ी?')
if st.button('Search and answer', type='primary', disabled=not question.strip()):
    try:
        with st.spinner('Retrieving Hindi evidence from ChromaDB...'):
            hits = load_retriever().search(question, top_k=top_k, hybrid=hybrid)
        if not hits:
            st.warning('No evidence retrieved. Try rephrasing the question.')
        else:
            if generate:
                with st.spinner('Generating grounded Hindi answer...'):
                    answer = load_generator().answer(question, hits)
                st.subheader('हिंदी उत्तर')
                st.markdown(answer)
                st.subheader('स्वचालित स्रोत-संदर्भ')
                for citation in select_citations(question, answer, hits):
                    st.caption(f"PDF पृष्ठ {citation['page']} · अनुभाग: {citation['section']} · chunk_id: {citation['chunk_id']} · score: {citation['score']}")
            st.subheader('Retrieved Hindi evidence')
            for hit in hits:
                with st.expander(f"PDF page {hit['page']} · {hit.get('section','दस्तावेज़')} · score {hit['score']:.4f}"):
                    st.write(hit['text'])
                    st.caption(f"Chunk ID: {hit['chunk_id']} · type: {hit.get('source_type','paragraph')}")
    except Exception as exc:
        st.exception(exc)
