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

#to load environment variables
load_dotenv()
#to give name to our page/tab
st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃",layout="centered")
st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
            ats_score = int(ats_match.group(1))
            display_content = analysis_content.replace(ats_match.group(0), "").strip()
            
            st.markdown("### ATS Analysis Results")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(label="ATS Score", value=f"{ats_score}/100")
            with col2:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # subtle spacer
                st.progress(ats_score / 100.0)
                
            st.markdown(display_content)
        else:
            st.markdown("### Analysis Results")
            st.markdown(analysis_content)
    except Exception as e:
        st.error(f"An error occured: {str(e)}")