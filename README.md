# 🤖 SmartHire - LLM-Powered Job Search

> An intelligent job search assistant powered by Large Language Models that matches candidates with personalized job opportunities and generates custom cover letters.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://github.com/langchain-ai/langchain)
[![Chainlit](https://img.shields.io/badge/Chainlit-2.8-purple.svg)](https://github.com/Chainlit/chainlit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

SmartHire is an advanced job search platform that leverages the power of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to provide personalized job recommendations. The system analyzes your resume, understands your skills and experience, and matches you with relevant job opportunities from a curated database.

### ✨ Key Features

- **🔍 Intelligent Job Matching**: Semantic search powered by vector embeddings to find jobs that truly match your profile
- **📄 Resume Analysis**: Automatic extraction and summarization of skills, experience, and qualifications from PDF resumes
- **💬 Conversational Interface**: Natural language chat interface to discuss job preferences and requirements
- **✍️ Cover Letter Generation**: AI-powered custom cover letter creation tailored to specific job descriptions
- **🎯 Three Assistant Modes**:
  - **Vanilla ChatGPT**: General-purpose conversational assistant
  - **Jobs Finder**: Resume-aware job search with personalized recommendations
  - **Jobs Agent**: Advanced agent with tools for job search and cover letter generation

## 🛠️ Tech Stack

- **LLM Providers**: OpenAI GPT / Google Gemini
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **UI**: Chainlit
- **PDF Processing**: PyPDF2
- **Data Processing**: Pandas

## 📦 Project Structure

```
SmartHire-LLM-Powered-Job-Search/
├── backend/
│   ├── models/
│   │   ├── chatgpt_clone.py          # Basic chat assistant
│   │   ├── resume_summarizer_chain.py # Resume analysis
│   │   ├── jobs_finder.py             # Job search assistant
│   │   └── jobs_finder_agent.py       # Agent with tools
│   ├── app.py                         # Chainlit application
│   ├── config.py                      # Configuration management
│   ├── etl.py                         # ETL pipeline for job data
│   ├── retriever.py                   # Vector search retriever
│   ├── llm_factory.py                 # LLM provider factory
│   └── utils.py                       # Utility functions
├── dataset/
│   └── jobs.csv                       # Job listings database
├── tests/                             # Test suite
├── requirements.txt                   # Python dependencies
└── env.example                        # Environment variables template
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- pip (Python package manager)
- An API key from either:
  - [OpenAI](https://platform.openai.com/) (paid)
  - [Google AI Studio](https://aistudio.google.com/) (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Brayan010702/SmartHire-LLM-Powered-Job-Search.git
   cd SmartHire-LLM-Powered-Job-Search
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```bash
   # For Google Gemini (Free)
   LLM_PROVIDER="gemini"
   GEMINI_LLM_MODEL="gemini-2.0-flash-exp"
   GOOGLE_API_KEY="your-google-api-key-here"
   LANGCHAIN_VERBOSE=true

   # OR for OpenAI
   # LLM_PROVIDER="openai"
   # OPENAI_LLM_MODEL="gpt-4o-mini"
   # OPENAI_API_KEY="your-openai-api-key-here"
   # LANGCHAIN_VERBOSE=true
   ```

5. **Run the ETL pipeline** (first time only)

   This creates the vector database from the job listings:
   ```bash
   python -m backend.etl
   ```

   This will process 100 job listings and create a ChromaDB vector store (~5-10 seconds).

6. **Launch the application**
   ```bash
   python -m chainlit run -w backend/app.py
   ```

   The app will be available at **http://localhost:8000**

## 💡 Usage

### 1. Select an Assistant Profile

When you first open the app, you'll be prompted to choose one of three assistant modes:

- **Vanilla ChatGPT**: A general-purpose conversational AI without job search capabilities
- **Jobs Finder Assistant**: Upload your resume and get personalized job recommendations
- **Jobs Agent**: Advanced mode with automatic cover letter generation

### 2. Upload Your Resume (for Jobs Finder & Jobs Agent)

- Click the upload button and select your PDF resume
- The system will automatically extract and summarize your skills and experience

### 3. Start Searching

Example queries:
- "I'm looking for remote software engineering positions"
- "Find me senior data scientist roles in New York"
- "What jobs match my background in machine learning?"

### 4. Generate Cover Letters (Jobs Agent only)

- Ask the agent to write a cover letter for a specific job
- Example: "Write a cover letter for the Software Engineer position at Google"

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests -v

# Run specific test file
python -m pytest tests/backend/test_etl.py -v

# Run with coverage
python -m pytest tests --cov=backend
```

## 🔧 Configuration

### Changing LLM Provider

Edit your `.env` file:

```bash
# Switch to OpenAI
LLM_PROVIDER="openai"
OPENAI_LLM_MODEL="gpt-4o-mini"
OPENAI_API_KEY="your-key-here"

# Switch to Gemini
LLM_PROVIDER="gemini"
GEMINI_LLM_MODEL="gemini-2.0-flash-exp"
GOOGLE_API_KEY="your-key-here"
```

### Customizing the Job Database

To use your own job listings:

1. Replace or update `dataset/jobs.csv` with your data
2. Ensure it has these columns: `description`, `Employment type`, `Seniority level`, `company`, `location`, `post_url`, `title`
3. Re-run the ETL pipeline: `python -m backend.etl`

## 📊 How It Works

### Architecture

```
User Query → Resume Summarizer → Vector Search → LLM → Personalized Response
                                        ↓
                                  ChromaDB
                                (Job Embeddings)
```

1. **Resume Processing**: PDF is parsed and key information is extracted
2. **Query Enhancement**: User query is combined with resume summary
3. **Semantic Search**: Vector similarity search finds relevant jobs
4. **LLM Generation**: Language model generates personalized recommendations
5. **Cover Letter**: (Agent mode) Custom letters based on job description + resume

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built as part of the Anyone AI LLMs Specialization program
- Powered by [LangChain](https://github.com/langchain-ai/langchain)
- UI built with [Chainlit](https://github.com/Chainlit/chainlit)
- Vector search by [ChromaDB](https://www.trychroma.com/)

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

**Made with ❤️ by the SmartHire Team**
