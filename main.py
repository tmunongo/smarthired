
import hashlib
import os
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import chromadb
import streamlit as st
import yaml
from chromadb.config import Settings
from google import genai
from google.genai import types

# =============================================================================
# MACHINE LEARNING CONCEPTS DEMONSTRATION
# =============================================================================

@dataclass
class JobExperience:
    """
    Data structure representing work experience
    This demonstrates structured data handling in ML pipelines
    """
    title: str
    company: str
    start_date: date
    end_date: Optional[date]
    achievements: List[str]
    technologies: List[str] = None
    
    def experience_months(self) -> int:
        """Calculate experience duration - useful for ML feature engineering"""
        end = self.end_date or date.today()
        return (end.year - self.start_date.year) * 12 + (end.month - self.start_date.month)

# =============================================================================
# FILE MANAGEMENT UTILITIES
# =============================================================================

def load_file_with_fallback(filename: str, fallback_content: str = "") -> str:
    """Load file content with fallback to default content"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Create file with fallback content
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(fallback_content)
            return fallback_content
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return fallback_content

def save_file(filename: str, content: str) -> bool:
    """Save content to file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        st.error(f"Error saving {filename}: {str(e)}")
        return False

# =============================================================================
# ABSTRACTION LAYER FOR LLM PROVIDERS
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

class GeminiProvider(LLMProvider):
    """Google Gemini implementation with improved error handling"""
    
    def __init__(self, api_key: str):
        try:
            self.client = genai.Client(api_key=api_key)
            self.model = 'gemini-2.5-flash' 
            self.embedding_model = "text-embedding-004"  # Updated model
            # Test the connection
            test_response = self.client.models.embed_content(
                model=self.embedding_model,
                contents="test"
            )
            st.success("✅ Gemini API connected successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with better error handling and batching"""
        try:
            embeddings = []
            for text in texts:
                if not text.strip():  # Skip empty texts
                    continue
                    
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type='retrieval_document',
                    )
                )
                # result.embeddings is a list of ContentEmbedding objects; extract .values
                embeddings.append([emb.values for emb in result.embeddings][0])
            
            st.success(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            st.error(f"❌ Error generating embeddings: {str(e)}")
            st.error(f"Full traceback: {traceback.format_exc()}")
            return []

# =============================================================================
# RAG SYSTEM IMPLEMENTATION - FIXED VERSION
# =============================================================================

class ResumeRAGSystem:
    """Fixed RAG system with proper state management"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False  # Use in-memory for simplicity
        ))
        self.collection = None
        self.experiences = []
        self.system_prompts = {}
    
    def load_system_prompts(self, prompts_file: str = "system_prompts.md"):
        """Load system prompts from markdown file"""
        default_prompts = """# System Prompts for RAG Resume Builder

## RESUME_GENERATION
You are an expert resume writer and career coach. Based on the job description and the candidate's relevant achievements, create tailored resume bullet points.

Create 5-7 bullet points that best match the job requirements:
- Use action verbs and quantify results where possible
- Highlight relevant technologies and skills mentioned in the job description
- Make each bullet point concise but impactful
- Rewrite achievements to emphasize aspects most relevant to this role
- Start each point with a strong action verb
- Include metrics and numbers when available

## COVER_LETTER_GENERATION
Write a professional cover letter for a software engineer position.

Requirements:
- Professional tone, 3-4 paragraphs
- Connect candidate's experience to job requirements
- Show enthusiasm for the role and company
- Include specific technologies mentioned in job description
- Keep it concise but compelling
- Address the hiring manager professionally

## CAREER_ANALYSIS
As a senior engineering manager and career coach, analyze the software engineer's career progression.

Provide analysis on:
- Experience level appropriateness for years in field
- Technology breadth and depth assessment
- Career progression and growth trajectory
- Areas for improvement and next steps
- Market competitiveness

Provide honest, constructive feedback in a supportive tone.
"""
        
        content = load_file_with_fallback(prompts_file, default_prompts)
        
        # Parse the markdown file to extract prompts
        sections = content.split('## ')
        for section in sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            prompt_name = lines[0].strip()
            prompt_content = '\n'.join(lines[1:]).strip()
            self.system_prompts[prompt_name] = prompt_content
        
        st.success(f"✅ Loaded {len(self.system_prompts)} system prompts")
    
    def load_resume_data(self, yaml_content: str) -> List[JobExperience]:
        """Fixed data loading with better error handling"""
        try:
            if not yaml_content.strip():
                st.error("❌ Resume data is empty")
                return []
            
            data = yaml.safe_load(yaml_content)
            if not data:
                st.error("❌ Failed to parse YAML data")
                return []
            
            if 'experience' not in data:
                st.error("❌ No 'experience' section found in YAML data")
                return []
            
            experiences = []
            
            for i, job in enumerate(data.get('experience', [])):
                try:
                    # Validate required fields
                    required_fields = ['title', 'company', 'start_date']
                    for field in required_fields:
                        if field not in job:
                            st.error(f"❌ Missing required field '{field}' in job {i+1}")
                            continue
                    
                    start_date = datetime.strptime(job['start_date'], '%Y-%m-%d').date()
                    end_date = None
                    if job.get('end_date') and job['end_date'].lower() != 'present':
                        end_date = datetime.strptime(job['end_date'], '%Y-%m-%d').date()
                    
                    exp = JobExperience(
                        title=job['title'],
                        company=job['company'],
                        start_date=start_date,
                        end_date=end_date,
                        achievements=job.get('achievements', []),
                        technologies=job.get('technologies', [])
                    )
                    experiences.append(exp)
                    
                except Exception as e:
                    st.error(f"❌ Error processing job {i+1}: {str(e)}")
                    continue
            
            self.experiences = experiences  # Store in instance
            st.success(f"✅ Loaded {len(experiences)} job experiences")
            
            # Debug info
            total_achievements = sum(len(exp.achievements) for exp in experiences)
            st.info(f"📊 Total achievements to index: {total_achievements}")
            
            return experiences
            
        except yaml.YAMLError as e:
            st.error(f"❌ YAML parsing error: {str(e)}")
            return []
        except Exception as e:
            st.error(f"❌ Error parsing resume data: {str(e)}")
            st.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def create_vector_index(self, experiences: List[JobExperience]):
        """Fixed vector indexing with better error handling"""
        try:
            # Create or recreate collection
            collection_name = "resume_achievements"
            try:
                self.client.delete_collection(collection_name)
            except:
                pass  # Collection might not exist
            
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Prepare documents for indexing
            documents = []
            metadatas = []
            ids = []
            
            for i, exp in enumerate(experiences):
                if not exp.achievements:
                    st.warning(f"⚠️ No achievements found for {exp.title} at {exp.company}")
                    continue
                    
                for j, achievement in enumerate(exp.achievements):
                    if not achievement.strip():
                        continue
                        
                    # Create rich context for each achievement
                    doc_text = f"""
Role: {exp.title} at {exp.company}
Duration: {exp.experience_months()} months
Technologies: {', '.join(exp.technologies or ['N/A'])}
Achievement: {achievement}
"""
                    
                    documents.append(doc_text.strip())
                    metadatas.append({
                        'job_title': exp.title,
                        'company': exp.company,
                        'experience_months': exp.experience_months(),
                        'technologies': ', '.join(exp.technologies or []),
                        'achievement': achievement
                    })
                    ids.append(f"exp_{i}_achievement_{j}")
            
            if not documents:
                st.error("❌ No documents to index!")
                return False
            
            st.info(f"📝 Preparing to index {len(documents)} achievements...")
            
            # Get embeddings
            embeddings = self.llm.get_embeddings(documents)
            if not embeddings:
                st.error("❌ Failed to generate embeddings")
                return False
            
            if len(embeddings) != len(documents):
                st.error(f"❌ Embedding count mismatch: {len(embeddings)} vs {len(documents)}")
                return False
            
            # Add to vector database
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            st.success(f"✅ Successfully indexed {len(documents)} achievements in vector database")
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating vector index: {str(e)}")
            st.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def retrieve_relevant_achievements(self, job_description: str, n_results: int = 8) -> List[Dict]:
        """Fixed retrieval with better error handling"""
        if not self.collection:
            st.error("❌ No vector index found. Please load and index your resume data first.")
            return []
        
        if not job_description.strip():
            st.error("❌ Job description is empty")
            return []
        
        try:
            # Get embedding for job description
            job_embeddings = self.llm.get_embeddings([job_description])
            if not job_embeddings:
                st.error("❌ Failed to generate embedding for job description")
                return []
            
            # Check collection size
            collection_count = self.collection.count()
            st.info(f"📊 Searching through {collection_count} indexed achievements...")
            
            if collection_count == 0:
                st.error("❌ Vector database is empty!")
                return []
            
            # Semantic search in vector space
            results = self.collection.query(
                query_embeddings=job_embeddings,
                n_results=min(n_results, collection_count)
            )
            
            if not results['documents'][0]:
                st.error("❌ No results returned from vector search")
                return []
            
            relevant_achievements = []
            for i in range(len(results['documents'][0])):
                relevant_achievements.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            st.success(f"✅ Found {len(relevant_achievements)} relevant achievements")
            return relevant_achievements
            
        except Exception as e:
            st.error(f"❌ Error retrieving achievements: {str(e)}")
            st.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def generate_tailored_resume(self, job_description: str, relevant_achievements: List[Dict]) -> str:
        """Generate resume using system prompts"""
        context = "\n\n".join([
            f"Achievement {i+1}: {ach['document']}" 
            for i, ach in enumerate(relevant_achievements)
        ])
        
        system_prompt = self.system_prompts.get('RESUME_GENERATION', '')
        
        prompt = f"""
{system_prompt}

JOB DESCRIPTION:
{job_description}

RELEVANT CANDIDATE ACHIEVEMENTS:
{context}

OUTPUT FORMAT:
• [Bullet point 1]
• [Bullet point 2]
...

TAILORED RESUME BULLET POINTS:
"""
        
        return self.llm.generate_text(prompt, max_tokens=800)
    
    def generate_cover_letter(self, job_description: str, company_name: str, role_title: str) -> str:
        """Generate cover letter using system prompts"""
        if not self.experiences:
            return "No experience data available for cover letter generation."
        
        # Get career summary
        total_months = sum(exp.experience_months() for exp in self.experiences)
        total_years = total_months / 12
        
        all_technologies = []
        for exp in self.experiences:
            if exp.technologies:
                all_technologies.extend(exp.technologies)
        unique_techs = list(set(all_technologies))
        
        system_prompt = self.system_prompts.get('COVER_LETTER_GENERATION', '')
        
        prompt = f"""
{system_prompt}

POSITION: {role_title} at {company_name}

CANDIDATE BACKGROUND:
- {total_years:.1f} years of software engineering experience
- Specializes in web-based SaaS platforms
- Technologies: {', '.join(unique_techs[:10])}

JOB DESCRIPTION:
{job_description}

COVER LETTER:
"""
        
        return self.llm.generate_text(prompt, max_tokens=600)
    
    def analyze_career_progression(self) -> str:
        """Analyze career using system prompts"""
        if not self.experiences:
            return "❌ No experience data available for analysis."
        
        total_experience = sum(exp.experience_months() for exp in self.experiences) / 12
        
        # Technology analysis
        all_techs = []
        for exp in self.experiences:
            if exp.technologies:
                all_techs.extend(exp.technologies)
        unique_techs = list(set(all_techs))
        
        # Experience summary
        exp_summary = []
        for exp in self.experiences:
            exp_summary.append({
                'title': exp.title,
                'company': exp.company,
                'duration_months': exp.experience_months(),
                'achievements_count': len(exp.achievements)
            })
        
        system_prompt = self.system_prompts.get('CAREER_ANALYSIS', '')
        
        prompt = f"""
{system_prompt}

CAREER DATA:
Total Experience: {total_experience:.1f} years
Technologies: {', '.join(unique_techs)}
Experience Summary: {exp_summary}

CAREER ANALYSIS:
"""
        
        return self.llm.generate_text(prompt, max_tokens=800)

# =============================================================================
# STREAMLIT UI - ENHANCED VERSION
# =============================================================================

def main():
    st.set_page_config(
        page_title="RAG Resume Builder",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🤖 RAG-Powered Resume Builder")
    st.markdown("*Learn Machine Learning by building a practical RAG application*")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password")
        
        if not api_key:
            st.warning("Please enter your Gemini API key to continue")
            st.markdown("[Get your API key here](https://aistudio.google.com/app/apikey)")
            return
        
        # Initialize RAG system
        try:
            if st.session_state.rag_system is None:
                with st.spinner("Initializing RAG system..."):
                    llm_provider = GeminiProvider(api_key)
                    st.session_state.rag_system = ResumeRAGSystem(llm_provider)
                    st.session_state.rag_system.load_system_prompts()
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Your Resume Data")
        
        # Load from file with default content
        default_yaml = """experience:
  - title: "Senior Software Engineer"
    company: "TechCorp"
    start_date: "2022-01-15"
    end_date: "present"
    technologies: ["Python", "React", "PostgreSQL", "AWS", "Docker"]
    achievements:
      - "Led development of microservices architecture serving 100K+ users daily"
      - "Reduced API response time by 40% through database optimization and caching"
      - "Mentored 3 junior developers and established comprehensive code review process"
      - "Implemented CI/CD pipeline reducing deployment time from 2 hours to 15 minutes"
      - "Designed and built real-time analytics dashboard using React and WebSocket"

  - title: "Software Engineer"
    company: "StartupXYZ"
    start_date: "2021-06-01"
    end_date: "2022-01-10"
    technologies: ["JavaScript", "Node.js", "MongoDB", "Docker", "Redis"]
    achievements:
      - "Built real-time chat feature using WebSocket, increasing user engagement by 25%"
      - "Developed RESTful APIs handling 10K+ requests per minute with 99.9% uptime"
      - "Created automated testing suite achieving 85% code coverage"
      - "Optimized database queries resulting in 30% faster page load times"
      - "Integrated payment processing system handling $50K+ monthly transactions"

  - title: "Junior Software Developer"
    company: "DevShop Inc"
    start_date: "2020-01-01"
    end_date: "2021-05-30"
    technologies: ["HTML", "CSS", "JavaScript", "PHP", "MySQL"]
    achievements:
      - "Developed responsive web applications for 5+ client projects"
      - "Fixed 100+ bugs and implemented 20+ new features in legacy codebase"
      - "Collaborated with designers to implement pixel-perfect UI components"
      - "Participated in agile development process and daily standups"
"""
        
        # Load existing file or create with default
        yaml_content = load_file_with_fallback("work.yaml", default_yaml)
        
        # Text area for editing
        edited_yaml = st.text_area(
            "Resume Data (work.yaml)",
            value=yaml_content,
            height=400,
            help="Edit your work experience data. Changes will be saved to work.yaml"
        )
        
        # Save button
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("💾 Save to File"):
                if save_file("work.yaml", edited_yaml):
                    st.success("✅ Saved to work.yaml")
        
        with col1b:
            if st.button("📊 Load & Index Data", type="primary"):
                if st.session_state.rag_system:
                    with st.spinner("Loading and indexing resume data..."):
                        experiences = st.session_state.rag_system.load_resume_data(edited_yaml)
                        if experiences:
                            success = st.session_state.rag_system.create_vector_index(experiences)
                            st.session_state.data_loaded = success
                        else:
                            st.session_state.data_loaded = False
    
    with col2:
        st.header("🎯 Job Application")
        
        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the job description here...",
            help="Paste the full job description to get the best matches"
        )
        
        col2a, col2b = st.columns(2)
        with col2a:
            company_name = st.text_input("Company Name", placeholder="e.g., Google")
        with col2b:
            role_title = st.text_input("Role Title", placeholder="e.g., Senior Software Engineer")
    
    # Status indicators
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        if st.session_state.rag_system:
            st.success("✅ RAG System Ready")
        else:
            st.error("❌ RAG System Not Ready")
    
    with status_col2:
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded & Indexed")
        else:
            st.error("❌ Data Not Loaded")
    
    with status_col3:
        if job_description:
            st.success("✅ Job Description Ready")
        else:
            st.error("❌ No Job Description")
    
    # Generate outputs
    if st.button("🚀 Generate Tailored Resume & Cover Letter", type="primary"):
        if not job_description:
            st.error("❌ Please provide a job description")
            return
        
        if not st.session_state.data_loaded:
            st.error("❌ Please load your resume data first")
            return
        
        if not st.session_state.rag_system:
            st.error("❌ RAG system not initialized")
            return
        
        with st.spinner("🤖 RAG system working... Retrieving relevant achievements and generating content..."):
            
            # Step 1: Retrieve relevant achievements
            st.write("**Step 1: Semantic Retrieval** 🔍")
            relevant_achievements = st.session_state.rag_system.retrieve_relevant_achievements(job_description)
            
            if relevant_achievements:
                with st.expander("🔍 Retrieved Relevant Achievements (Click to see RAG in action!)", expanded=True):
                    for i, ach in enumerate(relevant_achievements):
                        similarity_score = 1 - ach['distance']
                        st.write(f"**Match {i+1}** (Similarity: {similarity_score:.3f})")
                        st.write(f"📍 {ach['metadata']['job_title']} at {ach['metadata']['company']}")
                        st.write(f"💡 {ach['metadata']['achievement']}")
                        st.write("---")
                
                # Step 2: Generate tailored content
                st.write("**Step 2: Content Generation** ✨")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("📄 Tailored Resume Bullet Points")
                    resume_bullets = st.session_state.rag_system.generate_tailored_resume(job_description, relevant_achievements)
                    st.write(resume_bullets)
                
                with col4:
                    st.subheader("💌 Cover Letter")
                    if company_name and role_title:
                        cover_letter = st.session_state.rag_system.generate_cover_letter(job_description, company_name, role_title)
                        st.write(cover_letter)
                    else:
                        st.info("Please provide company name and role title for cover letter generation")
            
            else:
                st.error("❌ No relevant achievements found. Check the debug info above.")
    
    # Career Analysis Section
    st.header("📈 Career Progression Analysis")
    if st.button("🎯 Analyze My Career Progress"):
        if not st.session_state.data_loaded:
            st.error("❌ Please load your resume data first")
            return
        
        if not st.session_state.rag_system:
            st.error("❌ RAG system not initialized")
            return
        
        with st.spinner("Analyzing your career progression..."):
            analysis = st.session_state.rag_system.analyze_career_progression()
            st.write(analysis)
    
    # Debug section
    with st.expander("🔧 Debug Information"):
        if st.session_state.rag_system and st.session_state.data_loaded:
            st.write(f"**Experiences loaded:** {len(st.session_state.rag_system.experiences)}")
            if st.session_state.rag_system.collection:
                count = st.session_state.rag_system.collection.count()
                st.write(f"**Vector database size:** {count} documents")
            else:
                st.write("**Vector database:** Not initialized")
        else:
            st.write("No debug information available - load data first")

if __name__ == "__main__":
    main()
