import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------- Utility Functions -------------------

def get_text_from_file(file):
    file_extension = file.name.split('.')[-1].lower()
    text = ""
    if file_extension == 'pdf':
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension == 'txt':
        text = str(file.read(), "utf-8")
    elif file_extension == 'docx':
        document = Document(file)
        for para in document.paragraphs:
            text += para.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not in the context, say: "Answer is not available in the context."
    Do not make up answers.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("📌 **Reply:**", response["output_text"])

# ------------------- Summarization -------------------


def get_summary_chain():
    prompt_template = """
    Summarize the following text in a clear and concise way:

    Text:
    {text}

    Summary:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = LLMChain(llm=model, prompt=prompt)  # CHANGED TO LLMChain
    return chain

def summarize_text(paragraph):
    chain = get_summary_chain()
    response = chain.run(text=paragraph)  # UPDATED TO USE .run()
    return response


# ------------------- Career Counseling -------------------
def career_counseling_response(query):
    prompt_template = """
    You are a helpful and professional career counselor. Answer the following query with actionable advice and clarity.

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt)  # CHANGED
    return chain.run(question=query)


# ------------------- Personal & Emotional Support -------------------

def personal_support_response(query):
    prompt_template = """
    You are a friendly, understanding, and supportive AI assistant. A user is asking a personal or emotional question.
    Respond with empathy, encouragement, and practical advice.

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt)  # CHANGED
    return chain.run(question=query)

def Entertainment(query):
    prompt_template = """
    You are a friendly, understanding, and supportive AI assistant. A user is asking any entertainment related question like about songs, films, web series, TV shows, web shows, short films, recommendations.
    Also give recommendations about films/shows related to a particular topic or genre the user wants.
    Respond with entertaining, encouraging, and practical advice.

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt) 
    return chain.run(question=query)  # ✅ Now it matches
# ------------------- MCQ & Notes Generator -------------------

def generate_mcq_and_notes(text):
    prompt_template = """
    You're a helpful educational assistant. From the text below, do the following:

    1. Generate 3 multiple choice questions (MCQs) with 4 options each and indicate the correct answer.
    2. Provide short notes summarizing the key points.

    Text:
    {text}

    Output:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain.run(text=text)

def main():
    st.set_page_config("💡 Gemini AI Assistant", layout="wide")
    st.title("🌟 Gemini-Powered AI Assistant")
    st.markdown("Unlock knowledge, insight, and support — all in one place.")

    # Initialize chat history in session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    tab1, tab2, tab3, tab4, tab5= st.tabs([
        "📁 Ask from Documents", 
        "✍️ Summarizer", 
        "💼 Career & Personal", 
        "🎬 Entertainment",
        " 🧑‍🎓 MCQ & Notes"
    ])

    # Tab 1: Ask from File
    with tab1:
        st.subheader("📂 Ask Questions from Uploaded Files")
        uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        process_btn = st.button("📄 Process Files")
        if process_btn:
            if uploaded_files:
                with st.spinner("Processing files..."):
                    raw_text = ""
                    for file in uploaded_files:
                        raw_text += get_text_from_file(file) + "\n"
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ Files processed and ready.")
            else:
                st.warning("⚠️ Upload at least one file.")

        file_query = st.text_input("🔎 Ask something from the uploaded files")
        if st.button("Ask"):
            if file_query.strip():
                with st.spinner("Searching through documents..."):
                    user_input(file_query)
                    st.session_state.chat_history.append(("User", file_query))
            else:
                st.warning("⚠️ Please enter a question.")

    # Tab 2: Summarizer
    with tab2:
        st.subheader("📝 Summarize Text")
        paragraph = st.text_area("✏️ Enter text to summarize", height=200)
        if st.button("📚 Generate Summary"):
            if paragraph.strip():
                with st.spinner("Summarizing..."):
                    summary = summarize_text(paragraph)
                    st.success("✅ Summary:")
                    st.markdown(f"> {summary}")
                    st.session_state.chat_history.append(("User", paragraph))
                    st.session_state.chat_history.append(("Bot", summary))
            else:
                st.warning("⚠️ Please enter text.")

    # Tab 3: Career + Personal
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎓 Career Counseling")
            c_query = st.text_input("👔 Ask about career, jobs, education", key="career")
            if st.button("🎯 Get Career Advice"):
                if c_query.strip():
                    with st.spinner("Advising..."):
                        ans = career_counseling_response(c_query)
                        st.success("✅ Career Advice:")
                        st.markdown(f"> {ans}")
                        st.session_state.chat_history.append(("User", c_query))
                        st.session_state.chat_history.append(("Bot", ans))
                else:
                    st.warning("⚠️ Please enter a career question.")

        with col2:
            st.subheader("💖 Personal & Emotional Support")
            p_query = st.text_input("💬 Share your feelings or struggles", key="personal")
            if st.button("💡 Get Support"):
                if p_query.strip():
                    with st.spinner("Caring..."):
                        reply = personal_support_response(p_query)
                        st.success("🫶 Supportive Advice:")
                        st.markdown(f"> {reply}")
                        st.session_state.chat_history.append(("User", p_query))
                        st.session_state.chat_history.append(("Bot", reply))
                else:
                    st.warning("⚠️ Share something for support.")

    # Tab 4: Entertainment
    with tab4:
        st.subheader("🎥 Entertainment Corner")
        e_query = st.text_input("🎶 Ask about movies, shows, music, etc.", key="entertainment")
        if st.button("🎬 Get Recommendations"):
            if e_query.strip():
                with st.spinner("Fetching recommendations..."):
                    rec = Entertainment(e_query)
                    st.success("🍿 Entertainment Suggestion:")
                    st.markdown(f"> {rec}")
                    st.session_state.chat_history.append(("User", e_query))
                    st.session_state.chat_history.append(("Bot", rec))
            else:
                st.warning("⚠️ Ask an entertainment-related question.")
        

    with tab5:
        st.subheader("🧾 Generate MCQs and Short Notes")
        mcq_input = st.text_area("📘 Enter a topic or paste some study material", height=200)
        if st.button("🚀 Generate MCQs & Notes"):
            if mcq_input.strip():
                with st.spinner("Generating learning material..."):
                    result = generate_mcq_and_notes(mcq_input)
                    st.success("🎯 AI Generated Content:")
                    st.markdown(result)
                    st.session_state.chat_history.append(("User", mcq_input))
                    st.session_state.chat_history.append(("Bot", result))
            else:
                st.warning("⚠️ Please enter some content.")

# ------------------- Run App -------------------
if __name__ == "__main__":
    main()


