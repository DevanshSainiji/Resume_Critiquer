import streamlit as st
import PyPDF2

#IO Used for handling streams/data in memory like files without creating actual files on disk.

#Common uses:

#reading uploaded files
#converting bytes/text
#in-memory file operations
import io


#OS Used to interact with the operating system.

#Common uses:

#file paths
#environment variables
#folders/files handling
import os

from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

#to load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#to give name to our page/tab
st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃",layout="centered")
st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

# Initialize session state variables
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "analysis_content" not in st.session_state:
    st.session_state.analysis_content = None
if "ats_score" not in st.session_state:
    st.session_state.ats_score = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)",type=["pdf","txt"])
job_role= st.text_input("Enter the job role you are targetting (optional)")

analyze = st.button("Analyze Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        # .read() is reading the pdf file, .BytesIO is converting it to bytes
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("File does not have any content...")
            st.stop()

        # client = OpenAI(api_key=OPENAI_API_KEY)
        client = Groq(api_key=GROQ_API_KEY)

        # Pre-check: Verify if the document is a resume
        validation_prompt = f"""
        Analyze the following text snippet and determine if it represents a resume, CV, or professional work profile.
        Answer with EXACTLY "YES" or "NO". Do not include any other text, explanation, or punctuation.

        Text snippet:
        {file_content[:1500]}
        """
        
        validation_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise classifier that responds only with YES or NO."},
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        is_resume = validation_response.choices[0].message.content.strip().upper()
        if "NO" in is_resume:
            st.error("The uploaded file does not appear to be a resume. Please upload a valid resume.")
            st.stop()

        prompt = f"""Please analyze this resume and provide constructive feedback.
        First, calculate an ATS compatibility score (0-100) for the resume against the target job role: '{job_role if job_role else 'general job applications'}'.
        Format the first line of your output EXACTLY as: [ATS Score: <score>] (e.g., [ATS Score: 85]).

        Focus on the following aspects for feedback:
        1. Content clarity and impact
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else 'general job applications'}
        
        Resume content:
        {file_content}
        
        Please provide your analysis in a clear, structured format with specific recommendations."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":"You are an expert resume reviewer with years of experience in HR and recruitment."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        import re
        analysis_content = response.choices[0].message.content
        ats_match = re.search(r'\[ATS Score:\s*(\d+)\]', analysis_content)
        
        if ats_match:
            st.session_state.ats_score = int(ats_match.group(1))
            st.session_state.analysis_content = analysis_content.replace(ats_match.group(0), "").strip()
        else:
            st.session_state.ats_score = None
            st.session_state.analysis_content = analysis_content
            
        st.session_state.file_content = file_content
        st.session_state.messages = [] # Reset chat history for new resume
    except Exception as e:
        st.error(f"An error occured: {str(e)}")

# Render analysis results and chat section if they exist in session state
if st.session_state.analysis_content is not None:
    st.write("---")
    
    # Display the critique results
    if st.session_state.ats_score is not None:
        st.markdown("### ATS Analysis Results")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(label="ATS Score", value=f"{st.session_state.ats_score}/100")
        with col2:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # subtle spacer
            st.progress(st.session_state.ats_score / 100.0)
            
        st.markdown(st.session_state.analysis_content)
    else:
        st.markdown("### Analysis Results")
        st.markdown(st.session_state.analysis_content)

    st.write("---")
    st.markdown("### Chat with AI about your Resume")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize last audio ID in session state to prevent repeat processing
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None

    user_prompt = None

    # Render voice recording button
    col1, col2 = st.columns([1, 4])
    with col1:
        st.write("🎙️ Voice Command:")
    with col2:
        audio = mic_recorder(
            start_prompt="Record Command",
            stop_prompt="Stop Recording",
            just_once=True,
            key="voice_recorder"
        )

    if audio and audio["id"] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio["id"]
        try:
            client = Groq(api_key=GROQ_API_KEY)
            with st.spinner("Transcribing voice command..."):
                audio_file = io.BytesIO(audio['bytes'])
                audio_file.name = "audio.wav"
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo"
                )
                user_prompt = transcription.text.strip()
                if user_prompt:
                    st.success(f"Transcribed: \"{user_prompt}\"")
        except Exception as e:
            st.error(f"Failed to transcribe audio: {e}")

    # Chat input
    chat_input_val = st.chat_input("Ask follow-up questions or discuss your resume...")
    if chat_input_val:
        user_prompt = chat_input_val.strip()

    if user_prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_prompt)
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Generate response from AI
        try:
            client = Groq(api_key=GROQ_API_KEY)
            
            # Construct conversation history for context
            system_message = {
                "role": "system",
                "content": f"""You are an expert resume reviewer and career coach.
                You are discussing a resume with the user.
                Below is the content of their resume and the initial critique you provided.
                Use this context to answer their questions.
                
                CRITICAL GUARDRAIL:
                You must ONLY answer queries and discuss topics that are directly related to the user's resume, their career, target job roles, professional experience, job applications, or the initial critique. 
                If the user asks any question or makes a request that is NOT related to these topics (e.g., general knowledge, greetings that turn into off-topic discussion, programming concepts not in the resume, writing general essays, solving math problems, or general chat), you must politely decline to answer, explaining that your only purpose is to help them with their resume, career, and job applications.
                
                Resume content:
                {st.session_state.file_content}
                
                Initial Critique:
                {st.session_state.analysis_content}
                """
            }
            
            messages = [system_message]
            messages.extend(st.session_state.messages)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                assistant_response = response.choices[0].message.content
                message_placeholder.markdown(assistant_response)
                
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")