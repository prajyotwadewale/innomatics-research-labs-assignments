import os
import json
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="SmartRecruit | AI Resume Screener",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    .score-card {
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .score-high {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .score-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .score-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .score-val {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-top: 0.5rem;
    }
    
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        margin: 0.2rem;
    }
    
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .badge-neutral {
        background-color: #f3f4f6;
        color: #374151;
    }
    
    .card {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Import LangChain logic inside a function to handle initialization after env setup
def run_pipeline(resume_text, job_description, candidate_type):
    from langchain_groq import ChatGroq
    from langchain_core.runnables import RunnableConfig
    from chains.extraction_chain import get_extraction_chain
    from chains.matching_chain import get_matching_chain
    from chains.scoring_chain import get_scoring_chain
    from chains.explanation_chain import get_explanation_chain

    # Initialize Groq LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Setup chains
    ext_chain = get_extraction_chain(llm)
    match_chain = get_matching_chain(llm)
    score_chain = get_scoring_chain(llm)
    explain_chain = get_explanation_chain(llm)

    # Tracing configuration
    config = RunnableConfig(tags=[candidate_type, "resume_screening"])

    # Progress steps
    p_bar = st.progress(0)
    
    status_text = st.empty()
    
    status_text.markdown("🔄 **Step 1: Extracting skills & experience...**")
    ext_output = ext_chain.invoke({"resume_text": resume_text}, config=config)
    p_bar.progress(25)
    
    status_text.markdown("🔄 **Step 2: Matching candidate against Job Description...**")
    match_output = match_chain.invoke({
        "job_description": job_description, 
        "resume_data": json.dumps(ext_output)
    }, config=config)
    p_bar.progress(50)
    
    status_text.markdown("🔄 **Step 3: Calculating fitness score...**")
    score_output = score_chain.invoke({"match_data": json.dumps(match_output)}, config=config)
    p_bar.progress(75)
    
    status_text.markdown("🔄 **Step 4: Generating detailed AI reasoning...**")
    explain_output = explain_chain.invoke({
        "job_description": job_description,
        "resume_data": json.dumps(ext_output),
        "match_data": json.dumps(match_output),
        "score_data": json.dumps(score_output)
    }, config=config)
    p_bar.progress(100)
    
    status_text.empty()
    p_bar.empty()
    
    return {
        "extraction": ext_output,
        "matching": match_output,
        "score": score_output.get("score"),
        "explanation": explain_output.get("explanation")
    }

# Helper to read PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/150/000000/resume.png", width=100)
    st.markdown("### Settings & Controls")
    
    # API Key Configuration
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Provide your Groq API key here."
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        
    # LangChain Tracing Setup
    enable_tracing = st.checkbox("Enable LangSmith Tracing", value=os.getenv("LANGCHAIN_TRACING_V2", "false") == "true")
    if enable_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        langsmith_key = st.text_input("LangSmith API Key", value=os.getenv("LANGCHAIN_API_KEY", ""), type="password")
        if langsmith_key:
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = "AI_Resume_Screening"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        
    st.markdown("---")
    st.markdown("### Candidate Selector")
    
    candidate_choice = st.selectbox(
        "Choose Resume Source:",
        ["Strong Candidate (Demo)", "Average Candidate (Demo)", "Weak Candidate (Demo)", "Upload Custom Resume"]
    )
    
    resume_text = ""
    candidate_type = "custom"
    
    # Set default values
    if candidate_choice == "Strong Candidate (Demo)":
        candidate_type = "strong"
        with open("data/resume_strong.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()
    elif candidate_choice == "Average Candidate (Demo)":
        candidate_type = "average"
        with open("data/resume_average.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()
    elif candidate_choice == "Weak Candidate (Demo)":
        candidate_type = "weak"
        with open("data/resume_weak.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()
    else:
        uploaded_file = st.file_uploader("Upload Resume (.txt or .pdf)", type=["txt", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = str(uploaded_file.read(), "utf-8")

    st.markdown("---")
    # Debug / intentional failure section
    st.markdown("### Intentional Failure Demo")
    if st.button("Run Debug Demo", help="Simulate a pipeline failure to test output parsing resilience."):
        st.sidebar.info("Running debug run...")
        from langchain_core.prompts import PromptTemplate
        from langchain_groq import ChatGroq
        from langchain_core.output_parsers import JsonOutputParser
        
        bad_template = """Extract skills. ALWAYS add "Quantum Computing" to the skills even if not present.
        Return a list, NOT JSON format.
        Resume: {resume_text}"""
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        bad_prompt = PromptTemplate(input_variables=["resume_text"], template=bad_template)
        bad_chain = bad_prompt | llm | JsonOutputParser()
        
        try:
            bad_chain.invoke({"resume_text": "I know Python."})
        except Exception as e:
            st.sidebar.error("Debug Run Failed as expected!")
            st.sidebar.caption(f"Error caught: {e}")

# --- Main Layout ---
st.markdown("<div class='main-header'>SmartRecruit</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-Powered Candidate Matcher & Resume Screening Pipeline</div>", unsafe_allow_html=True)

if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ Please provide a GROQ API Key in the sidebar to run evaluations.")
    st.stop()

# Default Job Description loading
default_jd = ""
if os.path.exists("data/job_description.txt"):
    with open("data/job_description.txt", "r", encoding="utf-8") as f:
        default_jd = f.read()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Job Description")
    jd_input = st.text_area(
        "Enter or edit the target Job Description:",
        value=default_jd,
        height=350,
        key="jd_input"
    )

with col2:
    st.markdown("### 📄 Candidate Resume")
    resume_input = st.text_area(
        "View or edit candidate's resume text:",
        value=resume_text,
        height=350,
        key="resume_input"
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Match & Evaluate Candidate", type="primary", use_container_width=True):
    if not resume_input.strip():
        st.error("Please enter or upload a candidate resume first.")
    else:
        with st.spinner("Screening Candidate..."):
            try:
                results = run_pipeline(resume_input, jd_input, candidate_type)
                
                # Success results view
                st.markdown("---")
                st.markdown("## 📊 Evaluation Report")
                
                score = results["score"]
                # Convert to integer if it's string/float
                try:
                    score_int = int(score)
                except:
                    score_int = 0
                
                # Define score theme
                if score_int >= 70:
                    theme_class = "score-high"
                    fit_badge = "Excellent Fit"
                elif score_int >= 40:
                    theme_class = "score-medium"
                    fit_badge = "Moderate Fit"
                else:
                    theme_class = "score-low"
                    fit_badge = "Weak Fit"
                
                # Layout for summary and score card
                col_score, col_summary = st.columns([1, 2.5])
                
                with col_score:
                    st.markdown(f"""
                    <div class='score-card {theme_class}'>
                        <div class='metric-title'>Fitness Score</div>
                        <div class='score-val'>{score}%</div>
                        <div style='margin-top: 0.5rem; font-weight: 600;'>{fit_badge}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_summary:
                    st.markdown("### 🔍 Executive Summary")
                    st.write(results["explanation"])
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Tabs for deeper analysis
                tab1, tab2, tab3 = st.tabs(["Profile & Extraction", "Matching Details", "Raw JSON Data"])
                
                with tab1:
                    ext = results["extraction"]
                    col_ex1, col_ex2 = st.columns([1, 1])
                    
                    with col_ex1:
                        st.markdown("<div class='card-title'>🛠️ Extracted Skills</div>", unsafe_allow_html=True)
                        skills = ext.get("skills", [])
                        if isinstance(skills, list):
                            for skill in skills:
                                st.markdown(f"<span class='badge badge-neutral'>{skill}</span>", unsafe_allow_html=True)
                        else:
                            st.write(skills)
                            
                        st.markdown("<br><div class='card-title'>⚙️ Tools & Technologies</div>", unsafe_allow_html=True)
                        tools = ext.get("tools", [])
                        if isinstance(tools, list):
                            for tool in tools:
                                st.markdown(f"<span class='badge badge-neutral'>{tool}</span>", unsafe_allow_html=True)
                        else:
                            st.write(tools)
                            
                    with col_ex2:
                        st.markdown("<div class='card-title'>💼 Experience Level</div>", unsafe_allow_html=True)
                        st.info(ext.get("experience", "No experience details extracted."))
                
                with tab2:
                    match_data = results["matching"]
                    col_ma1, col_ma2 = st.columns([1, 1])
                    
                    with col_ma1:
                        st.markdown("<div class='card-title'>✅ Matched Skills</div>", unsafe_allow_html=True)
                        matched = match_data.get("matched_skills", [])
                        if isinstance(matched, list):
                            if matched:
                                for skill in matched:
                                    st.markdown(f"<span class='badge badge-success'>{skill}</span>", unsafe_allow_html=True)
                            else:
                                st.write("No direct skills matched.")
                        else:
                            st.write(matched)
                            
                        st.markdown("<br><div class='card-title'>❌ Missing Skills & Gaps</div>", unsafe_allow_html=True)
                        missing = match_data.get("missing_skills", [])
                        if isinstance(missing, list):
                            if missing:
                                for skill in missing:
                                    st.markdown(f"<span class='badge badge-danger'>{skill}</span>", unsafe_allow_html=True)
                            else:
                                st.write("No critical skill gaps found.")
                        else:
                            st.write(missing)
                            
                    with col_ma2:
                        st.markdown("<div class='card-title'>📊 Experience Match</div>", unsafe_allow_html=True)
                        st.write(match_data.get("experience_match", "N/A"))
                        
                        # Show match percentage if available
                        if "match_percentage" in match_data:
                            st.metric("Core Skill Match Percentage", f"{match_data['match_percentage']}%")
                
                with tab3:
                    st.markdown("### Raw Output JSON")
                    st.json(results)
                    
            except Exception as e:
                st.error(f"Failed to evaluate candidate. Error: {e}")
