# RAG Resume Builder - Setup Guide

## Installation

### 1. Create a virtual environment

```bash
uv sync
```

If you don't have `uv` installed: [Link](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Install dependencies

```bash
pip install streamlit google-generativeai chromadb pyyaml
```

### 3. Get Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key for use in the application

### 4. Run the application

```bash
uv run streamlit run main.py
```

## Resume Data Format (YAML)

Your resume data should follow this structure:

```yaml
experience:
  - title: "Senior Software Engineer"
    company: "TechCorp"
    start_date: "2022-01-15"
    end_date: "present" # or specific date like "2023-12-31"
    technologies: ["Python", "React", "PostgreSQL", "AWS", "Docker"]
    achievements:
      - "Led development of microservices architecture serving 100K+ users"
      - "Reduced API response time by 40% through database optimization"
      - "Mentored 3 junior developers and established code review process"
      - "Implemented CI/CD pipeline reducing deployment time by 60%"

  - title: "Software Engineer"
    company: "StartupXYZ"
    start_date: "2021-06-01"
    end_date: "2022-01-10"
    technologies: ["JavaScript", "Node.js", "MongoDB", "Docker"]
    achievements:
      - "Built real-time chat feature using WebSocket, increasing user engagement 25%"
      - "Developed RESTful APIs handling 10K+ requests per minute"
      - "Created automated testing suite achieving 85% code coverage"
      - "Optimized database queries resulting in 30% faster page load times"
```

## Key Features

### 🤖 RAG Pipeline

- **Document Indexing**: Converts achievements to embeddings
- **Semantic Retrieval**: Finds relevant experiences based on job description
- **Context-Aware Generation**: Creates tailored content using retrieved context

### 🎯 Practical Applications

- **Resume Tailoring**: Generate bullet points specific to each job application
- **Cover Letter Writing**: Create personalized cover letters
- **Career Analysis**: Get feedback on your progression and market competitiveness

### 🎓 Learning Opportunities

- **Vector Embeddings**: See how text becomes numerical representations
- **Similarity Search**: Understand semantic matching beyond keywords
- **Prompt Engineering**: Learn to craft effective LLM prompts
- **Software Architecture**: Observe clean code patterns and abstractions

## Next Steps for Learning

### Phase 1: Understand the Basics

1. Run the application and observe how RAG retrieval works
2. Experiment with different job descriptions
3. Check the "Retrieved Relevant Achievements" section to see semantic matching

### Phase 2: Explore the Code

1. Study the `ResumeRAGSystem` class to understand the RAG pipeline
2. Look at how embeddings are generated and stored
3. Examine the prompt engineering in generation functions

### Phase 3: Extend and Experiment

1. Add more sophisticated retrieval (hybrid search, re-ranking)
2. Implement different LLM providers (OpenAI, Anthropic)
3. Add evaluation metrics for generated content quality
4. Create a feedback loop to improve results

### Phase 4: Production Considerations

1. Add caching for embeddings
2. Implement user authentication
3. Add batch processing for multiple job applications
4. Create an API version with FastAPI

## Machine Learning Concepts Covered

- **Embeddings**: Text-to-vector conversion for semantic understanding
- **Vector Databases**: Efficient storage and retrieval of high-dimensional data
- **Similarity Metrics**: Cosine similarity for semantic matching
- **Information Retrieval**: Ranking and selecting relevant documents
- **Natural Language Generation**: Using LLMs for content creation
- **Feature Engineering**: Extracting useful information from raw data
- **Data Pipeline Design**: From ingestion to inference
