import streamlit as st
import pandas as pd
import numpy as np
import re
import PyPDF2
import docx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cream + brown theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #5D4037;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #FFF8E1, #D7CCC8);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #A1887F;
    }
    .score-card {
        background: linear-gradient(135deg, #FFF8E1, #D7CCC8);
        padding: 2rem;
        border-radius: 15px;
        color: #5D4037;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #A1887F;
    }
    .metric-card {
        background: #FFF8E1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #8D6E63;
        margin: 0.5rem 0;
        color: #5D4037;
        border: 1px solid #D7CCC8;
    }
    .skill-card {
        background: #FFF8E1;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #D7CCC8;
        margin: 0.5rem 0;
        color: #5D4037;
    }
    .course-card {
        background: linear-gradient(135deg, #FFF8E1, #D7CCC8);
        padding: 1.5rem;
        border-radius: 10px;
        color: #5D4037;
        margin: 0.5rem 0;
        border: 1px solid #A1887F;
    }
    .section-header {
        background: linear-gradient(135deg, #FFF8E1, #D7CCC8);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #A1887F;
        color: #5D4037;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from different file types
def extract_text_from_file(uploaded_file):
    """Extract text from PDF or Word document"""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                   "application/msword"]:
            doc = docx.Document(uploaded_file)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        else:
            return f"Error: Unsupported file format. Please use PDF or Word document."
        
        return text
    except Exception as e:
        return f"Error reading file: {e}"

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = ' '.join(text.split())
    return text

def extract_skills(text):
    """Extract skills from text"""
    technical_skills = [
        'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin',
        'html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring',
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server',
        'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'git',
        'machine learning', 'deep learning', 'ai', 'data analysis', 'data science',
        'tableau', 'power bi', 'excel', 'tensorflow', 'pytorch', 'scikit learn',
        'agile', 'scrum', 'devops', 'rest api', 'graphql', 'microservices',
        'nosql', 'big data', 'hadoop', 'spark', 'nlp', 'computer vision',
        'opencv', 'cnn', 'faster r-cnn', 'hitl', 'html', 'css'
    ]
    
    soft_skills = [
        'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
        'time management', 'adaptability', 'creativity', 'work ethic', 'interpersonal skills',
        'project management', 'presentation', 'negotiation', 'decision making', 'collaboration'
    ]
    
    found_tech_skills = []
    found_soft_skills = []
    text_lower = text.lower()
    
    for skill in technical_skills:
        skill_pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(skill_pattern, text_lower):
            found_tech_skills.append(skill)
    
    for skill in soft_skills:
        skill_pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(skill_pattern, text_lower):
            found_soft_skills.append(skill)
    
    return found_tech_skills, found_soft_skills

def extract_experience(text):
    """Extract experience from text"""
    experience_patterns = [
        r'(\d+)\+? years?', r'(\d+)\+? yrs', r'year.*experience', 
        r'experience.*year', r'(\d+).*year', r'(\d+).*yr'
    ]
    
    experience = 0
    for pattern in experience_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if isinstance(match, str) and match.isdigit():
                exp = int(match)
                if exp > experience:
                    experience = exp
    
    return experience

def calculate_ats_score(resume_text, job_description=None):
    """Calculate ATS score between resume and job description"""
    resume_processed = preprocess_text(resume_text)
    
    if job_description:
        # Calculate score against job description
        jd_processed = preprocess_text(job_description)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        ats_score = round(similarity * 100, 2)
    else:
        # Calculate base ATS score (resume quality without JD)
        # This is a simplified version that checks resume structure and content
        word_count = len(resume_processed.split())
        skill_count = len(extract_skills(resume_text)[0])
        exp_years = extract_experience(resume_text)
        
        # Score based on resume quality metrics
        ats_score = min(100, (word_count / 500 * 30) + (skill_count / 15 * 40) + (exp_years * 6))
        ats_score = round(ats_score, 2)
    
    return ats_score

def analyze_eligibility(resume_text, job_description=None):
    """Analyze eligibility based on resume and optional job description"""
    ats_score = calculate_ats_score(resume_text, job_description)
    
    resume_skills, resume_soft_skills = extract_skills(resume_text)
    resume_experience = extract_experience(resume_text)
    
    if job_description:
        jd_skills, jd_soft_skills = extract_skills(job_description)
        jd_experience = extract_experience(job_description)
        
        # Calculate skills match
        tech_skills_match = len(set(resume_skills) & set(jd_skills))
        tech_skills_required = len(jd_skills)
        
        if tech_skills_required > 0:
            skills_match_percentage = (tech_skills_match / tech_skills_required) * 100
        else:
            skills_match_percentage = 100
        
        # Calculate experience match
        if jd_experience > 0:
            if resume_experience >= jd_experience:
                experience_match = 100
            else:
                experience_match = (resume_experience / jd_experience) * 100
        else:
            experience_match = 100
        
        # Overall eligibility score (weighted average)
        eligibility_score = (ats_score * 0.4) + (skills_match_percentage * 0.4) + (experience_match * 0.2)
        
        return ats_score, eligibility_score, skills_match_percentage, experience_match, jd_skills, jd_experience
    else:
        # Without JD, provide general resume quality assessment
        skills_match_percentage = 100  # Not applicable without JD
        experience_match = 100  # Not applicable without JD
        
        # Base eligibility score on resume quality
        word_count = len(resume_text.split())
        skill_count = len(resume_skills)
        
        eligibility_score = min(100, (word_count / 500 * 40) + (skill_count / 15 * 40) + (resume_experience * 4))
        eligibility_score = round(eligibility_score, 2)
        
        return ats_score, eligibility_score, skills_match_percentage, experience_match, [], 0

def generate_recommendations(resume_text, job_description=None):
    """Generate improvement recommendations"""
    recommendations = []
    
    resume_skills, resume_soft_skills = extract_skills(resume_text)
    resume_experience = extract_experience(resume_text)
    
    if job_description:
        jd_skills, jd_soft_skills = extract_skills(job_description)
        jd_experience = extract_experience(job_description)
        
        # Skills recommendations
        missing_tech_skills = set(jd_skills) - set(resume_skills)
        missing_soft_skills = set(jd_soft_skills) - set(resume_soft_skills)
        
        if missing_tech_skills:
            recommendations.append(f"Develop these technical skills: {', '.join(missing_tech_skills)}")
        if missing_soft_skills:
            recommendations.append(f"Develop these soft skills: {', '.join(missing_soft_skills)}")
        
        # Experience recommendations
        if resume_experience < jd_experience:
            recommendations.append(f"Gain more experience. The job requires {jd_experience}+ years, but you have {resume_experience} years.")
    else:
        # General recommendations without JD
        if len(resume_skills) < 10:
            recommendations.append("Consider adding more technical skills to your resume.")
        if resume_experience < 2:
            recommendations.append("Highlight any projects or internships to compensate for limited work experience.")
        recommendations.append("Consider adding a skills section to make your technical abilities more visible.")
    
    # General recommendations for all cases
    recommendations.append("Tailor your resume to include relevant keywords.")
    recommendations.append("Quantify your achievements with specific numbers and metrics.")
    recommendations.append("Use bullet points to make your resume more readable.")
    
    if job_description:
        missing_skills = list(set(jd_skills) - set(resume_skills)) if job_description else []
    else:
        missing_skills = []
    
    return recommendations, missing_skills

# Course database
COURSE_DATABASE = {
    'python': [
        {'platform': 'Coursera', 'name': 'Python for Everybody', 'cost': 'Free', 'duration': '3 months', 
         'url': 'https://www.coursera.org/specializations/python', 'rating': 4.8},
        {'platform': 'YouTube', 'name': 'Python Tutorial for Beginners', 'cost': 'Free', 'duration': '6 hours', 
         'url': 'https://www.youtube.com/watch?v=_uQrJ0TkZlc', 'rating': 4.9}
    ],
    'machine learning': [
        {'platform': 'Coursera', 'name': 'Machine Learning by Andrew Ng', 'cost': 'Free', 'duration': '3 months', 
         'url': 'https://www.coursera.org/learn/machine-learning', 'rating': 4.9},
        {'platform': 'YouTube', 'name': 'Machine Learning Tutorial', 'cost': 'Free', 'duration': '10 hours', 
         'url': 'https://www.youtube.com/watch?v=NWONeJKn6kc', 'rating': 4.8}
    ],
    'data analysis': [
        {'platform': 'Coursera', 'name': 'Google Data Analytics Professional Certificate', 'cost': 'Free', 'duration': '6 months', 
         'url': 'https://www.coursera.org/professional-certificates/google-data-analytics', 'rating': 4.8}
    ],
    'sql': [
        {'platform': 'Coursera', 'name': 'SQL for Data Science', 'cost': 'Free', 'duration': '1 month', 
         'url': 'https://www.coursera.org/learn/sql-for-data-science', 'rating': 4.7}
    ],
    'tensorflow': [
        {'platform': 'Coursera', 'name': 'TensorFlow Developer Professional Certificate', 'cost': 'Free', 'duration': '4 months', 
         'url': 'https://www.coursera.org/professional-certificates/tensorflow-in-practice', 'rating': 4.8}
    ],
    'opencv': [
        {'platform': 'Udemy', 'name': 'OpenCV Python for Beginners', 'cost': '₹455', 'duration': '10 hours', 
         'url': 'https://www.udemy.com/course/opencv-python-for-beginners/', 'rating': 4.6}
    ],
    'tableau': [
        {'platform': 'Coursera', 'name': 'Data Visualization with Tableau', 'cost': 'Free', 'duration': '2 months', 
         'url': 'https://www.coursera.org/specializations/data-visualization', 'rating': 4.7}
    ],
    'power bi': [
        {'platform': 'Coursera', 'name': 'Microsoft Power BI Data Analyst Professional Certificate', 'cost': 'Free', 'duration': '3 months', 
         'url': 'https://www.coursera.org/professional-certificates/microsoft-power-bi-data-analyst', 'rating': 4.7}
    ],
    'nlp': [
        {'platform': 'Coursera', 'name': 'Natural Language Processing Specialization', 'cost': 'Free', 'duration': '4 months', 
         'url': 'https://www.coursera.org/specializations/natural-language-processing', 'rating': 4.8}
    ],
    'cnn': [
        {'platform': 'Coursera', 'name': 'Convolutional Neural Networks', 'cost': 'Free', 'duration': '1 month', 
         'url': 'https://www.coursera.org/learn/convolutional-neural-networks', 'rating': 4.8}
    ]
}

def suggest_courses(missing_skills):
    """Suggest courses for missing skills"""
    course_suggestions = {}
    
    for skill in missing_skills:
        skill_lower = skill.lower()
        if skill_lower in COURSE_DATABASE:
            course_suggestions[skill] = COURSE_DATABASE[skill_lower]
        else:
            course_suggestions[skill] = [
                {'platform': 'Coursera', 'name': f'{skill.title()} Specialization', 'cost': 'Free/Paid', 
                 'duration': '2-6 months', 'url': 'https://www.coursera.org/search?query=' + skill.replace(' ', '%20'), 'rating': 'N/A'},
                {'platform': 'Udemy', 'name': f'Complete {skill.title()} Course', 'cost': '₹455', 
                 'duration': '10-30 hours', 'url': 'https://www.udemy.com/courses/search/?q=' + skill.replace(' ', '%20'), 'rating': 'N/A'}
            ]
    
    return course_suggestions

def create_gauge_chart(score, title="ATS Score"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24, 'color': '#5D4037'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': "#5D4037"},
            'bar': {'color': "#8D6E63"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#D7CCC8",
            'steps': [
                {'range': [0, 50], 'color': '#FFCDD2'},
                {'range': [50, 80], 'color': '#FFF8E1'},
                {'range': [80, 100], 'color': '#C8E6C9'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='#FFF8E1',
        font={'color': "#5D4037", 'family': "Arial"}
    )
    return fig

def get_score_bar(score):
    bars = int(score / 5)
    return "█" * bars + "░" * (20 - bars)

def main():
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'jd_text' not in st.session_state:
        st.session_state.jd_text = ""
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 RESUME ANALYZER PRO</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #5D4037; font-size: 18px;">Complete ATS Resume Analysis System with Optional Job Description Matching</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #FFF8E1, #D7CCC8); padding: 20px; border-radius: 10px; border: 2px solid #A1887F;">
        <h2 style="color: #5D4037; margin: 0;">📊 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📄 Resume Analysis", "📚 Course Suggestions"])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📄 Resume Analysis":
        show_analysis_page()
    elif page == "📚 Course Suggestions":
        show_course_suggestions_page()

def show_home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="section-header">
            <h2>🚀 Welcome to Resume Analyzer Pro</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        This powerful tool helps you optimize your resume for Applicant Tracking Systems (ATS) 
        and land your dream job!
        
        ### ✨ Key Features:
        
        🔍 **ATS Score Analysis** - Get instant feedback on resume matching  
        📊 **Detailed Breakdown** - Understand strengths and improvements  
        🎯 **Skill Gap Analysis** - Identify missing skills  
        📚 **Personalized Learning** - Get course recommendations  
        🎨 **Beautiful Visualizations** - Interactive charts and progress bars
        
        ### 🎯 How It Works:
        1. Upload your resume (PDF or DOCX)
        2. Paste a job description (optional)
        3. Get instant ATS score and detailed analysis
        4. Receive personalized course suggestions
        
        Start by navigating to **Resume Analysis**!
        """)
    
    with col2:
        st.markdown("""
        <div class="score-card">
            <h3>🚀 Get Started</h3>
            <p>Click on <b>Resume Analysis</b> to begin your resume optimization journey!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Why ATS Matters?</h4>
            <p>75% of resumes are rejected by ATS before reaching human recruiters.</p>
        </div>
        """, unsafe_allow_html=True)

def show_analysis_page():
    st.markdown("""
    <div class="section-header">
        <h2>📄 Resume & Job Description Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Upload Your Resume")
        resume_file = st.file_uploader("Choose PDF or DOCX file", type=['pdf', 'docx'], key="resume_upload")
        
        if resume_file is not None:
            with st.spinner("Extracting text from resume..."):
                resume_text = extract_text_from_file(resume_file)
                
                if "Error" not in resume_text:
                    st.session_state.resume_text = resume_text
                    st.success("✅ Resume uploaded successfully!")
                    
                    with st.expander("View Extracted Resume Text"):
                        st.text_area("Resume Content", resume_text, height=200, label_visibility="collapsed")
                else:
                    st.error(resume_text)
    
    with col2:
        st.markdown("### 📝 Job Description (Optional)")
        jd_text = st.text_area("Paste job description:", height=200, 
                              placeholder="Paste the job description here for targeted analysis...",
                              key="jd_input")
        st.session_state.jd_text = jd_text
    
    if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        if not st.session_state.resume_text:
            st.error("❌ Please upload a resume first!")
            return
        
        with st.spinner("🤖 Analyzing your resume... This may take a few seconds"):
            has_jd = bool(st.session_state.jd_text.strip())
            
            if has_jd:
                ats_score, eligibility_score, skills_match, experience_match, jd_skills, jd_experience = analyze_eligibility(
                    st.session_state.resume_text, st.session_state.jd_text)
                recommendations, missing_skills = generate_recommendations(
                    st.session_state.resume_text, st.session_state.jd_text)
            else:
                ats_score, eligibility_score, skills_match, experience_match, jd_skills, jd_experience = analyze_eligibility(
                    st.session_state.resume_text)
                recommendations, missing_skills = generate_recommendations(
                    st.session_state.resume_text)
            
            resume_skills, resume_soft_skills = extract_skills(st.session_state.resume_text)
            resume_experience = extract_experience(st.session_state.resume_text)
            
            st.session_state.update({
                'analysis_done': True,
                'ats_score': ats_score,
                'eligibility_score': eligibility_score,
                'skills_match': skills_match,
                'experience_match': experience_match,
                'resume_skills': resume_skills,
                'resume_soft_skills': resume_soft_skills,
                'jd_skills': jd_skills,
                'jd_experience': jd_experience,
                'resume_experience': resume_experience,
                'missing_skills': missing_skills,
                'recommendations': recommendations,
                'has_jd': has_jd
            })
    
    if st.session_state.get('analysis_done', False):
        display_analysis_results()

def display_analysis_results():
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h2>📊 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Main score cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ats_score = st.session_state.get('ats_score', 0)
        st.markdown(f"""
        <div class="score-card">
            <h3>🎯 ATS Score</h3>
            <h1>{ats_score:.1f}%</h1>
            <p>{get_score_bar(ats_score)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        eligibility_score = st.session_state.get('eligibility_score', 0)
        st.markdown(f"""
        <div class="score-card">
            <h3>📈 Eligibility Score</h3>
            <h1>{eligibility_score:.1f}%</h1>
            <p>{get_score_bar(eligibility_score)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.get('has_jd', False):
            skills_match = st.session_state.get('skills_match', 0)
            st.markdown(f"""
            <div class="score-card">
                <h3>🔧 Skills Match</h3>
                <h1>{skills_match:.1f}%</h1>
                <p>{get_score_bar(skills_match)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            experience = st.session_state.get('resume_experience', 0)
            st.markdown(f"""
            <div class="score-card">
                <h3>💼 Experience</h3>
                <h1>{experience} yrs</h1>
                <p>Years of experience</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Status and gauge chart
    col1, col2 = st.columns([1, 2])
    
    with col1:
        eligibility_score = st.session_state.get('eligibility_score', 0)
        fig = create_gauge_chart(eligibility_score, "Overall Score")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if eligibility_score >= 80:
            st.success("🎉 EXCELLENT - Strong resume quality! Your resume is well-optimized for ATS.")
        elif eligibility_score >= 60:
            st.warning("📋 GOOD - Some improvements could help boost your score further.")
        else:
            st.error("💡 NEEDS IMPROVEMENT - Significant improvements needed. Check recommendations below.")
        
        # Additional metrics if JD is provided
        if st.session_state.get('has_jd', False):
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                exp_match = st.session_state.get('experience_match', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💼 Experience Match</h4>
                    <h2>{exp_match:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col2:
                skills_found = len(st.session_state.get('resume_skills', []))
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🔧 Skills Found</h4>
                    <h2>{skills_found}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    # Skills Analysis
    st.markdown("### 🎯 Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Your Technical Skills")
        resume_skills = st.session_state.get('resume_skills', [])
        if resume_skills:
            for skill in resume_skills:
                st.markdown(f'<div class="skill-card">🔹 {skill.title()}</div>', unsafe_allow_html=True)
        else:
            st.info("No technical skills detected in your resume.")
    
    with col2:
        if st.session_state.get('has_jd', False):
            st.markdown("#### 🔍 Required vs Your Skills")
            jd_skills = st.session_state.get('jd_skills', [])
            resume_skills_set = set(st.session_state.get('resume_skills', []))
            
            if jd_skills:
                for skill in jd_skills:
                    status = "✅" if skill in resume_skills_set else "❌"
                    st.markdown(f'<div class="skill-card">{status} {skill.title()}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific skills mentioned in the job description.")
        else:
            st.markdown("#### 💡 Soft Skills Detected")
            soft_skills = st.session_state.get('resume_soft_skills', [])
            if soft_skills:
                for skill in soft_skills:
                    st.markdown(f'<div class="skill-card">🌟 {skill.title()}</div>', unsafe_allow_html=True)
            else:
                st.info("Consider adding more soft skills to your resume.")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    recommendations = st.session_state.get('recommendations', [])
    
    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="metric-card">
            <h4>📌 Suggestion {i}</h4>
            <p>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Missing skills section
    if st.session_state.get('has_jd', False) and st.session_state.get('missing_skills'):
        missing_skills = st.session_state.get('missing_skills', [])
        st.markdown("### 🎯 Skills to Develop")
        
        st.warning(f"**{len(missing_skills)} skills missing** from your resume that are mentioned in the job description:")
        
        for skill in missing_skills:
            st.markdown(f"🔸 **{skill.title()}**")

def show_course_suggestions_page():
    st.markdown("""
    <div class="section-header">
        <h2>📚 Personalized Course Suggestions</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get('analysis_done', False):
        st.warning("👆 Please complete resume analysis first! Go to the 'Resume Analysis' page.")
        return
    
    missing_skills = st.session_state.get('missing_skills', [])
    
    if not missing_skills and st.session_state.get('has_jd', False):
        st.success("🎉 Great! You have all the required skills mentioned in the job description!")
        st.info("💡 Here are some general course suggestions to enhance your skills further:")
        missing_skills = ['python', 'machine learning', 'aws', 'data analysis']
    elif not st.session_state.get('has_jd', False):
        st.info("💡 No job description provided. Showing general skill enhancement courses.")
        missing_skills = ['python', 'machine learning', 'aws', 'sql', 'docker']
    else:
        st.warning(f"🔍 **{len(missing_skills)} skills missing** from your resume. Here are course suggestions:")
    
    course_suggestions = suggest_courses(missing_skills[:6])  # Limit to 6 skills
    
    for skill in list(course_suggestions.keys())[:6]:
        st.markdown(f"#### 🔧 {skill.title()}")
        
        courses = course_suggestions.get(skill, [])
        for j, course in enumerate(courses, 1):
            with st.container():
                st.markdown(f"""
                <div class="course-card">
                    <h4>🎯 {course['name']}</h4>
                    <p><strong>Platform:</strong> {course['platform']}</p>
                    <p><strong>Cost:</strong> {course['cost']}</p>
                    <p><strong>Duration:</strong> {course['duration']}</p>
                    <p><strong>Rating:</strong> {course['rating']}/5.0</p>
                    <p><strong>URL:</strong> <a href="{course['url']}" target="_blank">Enroll Now</a></p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()