# 🤖 LLM Response Evaluation System

A Streamlit application for evaluating LLM responses using RAG (Retrieval-Augmented Generation) with Milvus vector database.

## Requirements

- **Python Version**: 3.10
- **Milvus**: Vector database instance (local or remote)
- **API Keys**: LLM provider (OpenAI/Groq/etc.) + Claude API key
- **Optional**: Poppler-utils (for OCR functionality with scanned PDFs)

## Docker Setup (Recommended)

The easiest way to run this application is using Docker:

### Quick Start with Docker

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM-Response-Evaluation
   ```

2. **Clean up any existing containers** (if you have containers from previous runs):
   ```bash
   docker-compose down
   ```

3. **Build and start all services**:
   ```bash
   docker-compose up -d --build
   ```

   This will start:
   - **Milvus standalone** (vector database) on port 19530
   - **etcd** (Milvus dependency)
   - **MinIO** (object storage) on ports 9000 and 9001
   - **Streamlit application** on port 8501

4. **Access the application**:
   - Open your browser to: http://localhost:8501
   - Milvus is available at localhost:19530

5. **View logs**:
   ```bash
   docker-compose logs -f streamlit
   ```

6. **Stop all services**:
   ```bash
   docker-compose down
   ```

### Docker Features

- ✅ **Poppler installed** - OCR functionality works out of the box
- ✅ **All dependencies** - Python packages and system libraries pre-installed
- ✅ **Health checks** - Automatic monitoring of service status
- ✅ **Volume persistence** - Milvus data persists across restarts
- ✅ **Development mode** - Mount your code for live updates

### Docker Configuration

The Docker setup includes:
- **Base image**: Python 3.10 slim
- **System packages**: Poppler-utils (for OCR), curl (for health checks)
- **Port**: 8501 (Streamlit web interface)
- **Health check**: Automatic restart on failure
- **Dependencies**: All Python packages from requirements.txt

### Manual Installation (Non-Docker)

If you prefer to run without Docker:

1. **Install Python 3.10** and required dependencies
2. **Install Poppler**:
   - **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`
3. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Milvus** (either Docker or standalone):
   ```bash
   docker-compose up etcd minio standalone -d
   ```
5. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```

## Configuration

**Important**: This app does NOT use environment variables or .env files. All configuration is entered through **UI input fields** in the Streamlit web interface (Configuration tab). You will manually type/paste your API keys and settings into text boxes, dropdowns, and sliders.

### 1. Configuration Tab
- **Milvus**: Host, port, username, password
  - **Docker**: Automatically configured to use `standalone` as host
  - **Local**: Use `localhost` as host
- **LLM Provider**: Choose provider (OpenAI, Groq, Together AI, Google, Mistral) and enter API key
- **Claude**: API key for evaluation (required)
- **Embedding Model**: Select provider and model
- **RAG Settings**: Chunk size, overlap, number of chunks

### 2. RAG Documents Tab
- Use default collection or create new one
- Upload PDF, DOCX, TXT, or MD files

### 3. Evaluation Data Tab
- Download sample Excel template
- Upload Excel file with columns: `Question`, `Ground truth`, `Chat history` (optional)

### 4. Custom Metrics Tab
- Add custom evaluation metrics (optional)

### 5. Run Evaluation Tab
- Select number of questions
- Click "Start Evaluation"

### 6. Results Tab
- View detailed evaluation scores
- Export results to Excel

## Features

- **Multi-Provider Support**: OpenAI, Groq, Together AI, Google Gemini, Mistral
- **Embedding Options**: OpenAI, HuggingFace, SentenceTransformers, Ollama
- **Evaluation Metrics**: Answer relevancy, correctness, hallucination detection, and more
- **Custom Metrics**: Add domain-specific evaluation criteria
- **Document Processing**: PDF, DOCX, TXT, MD support
- **OCR Support**: Extract text from scanned PDFs using vision models
- **Excel Export**: Export detailed evaluation reports

## OCR Configuration (Optional)

To use OCR for scanned PDFs:

1. **Select Models Tab**: Go to "👁️ OCR Models" section
2. **Configure OCR Model**:
   - Model Name: e.g., `qwen/Qwen2-VL-7B-Instruct` or any vision-based model
   - API Key: Your DeepInfra or provider API key
   - Provider: DeepInfra or OpenAI-compatible API
3. **Use OCR**: When uploading PDFs in RAG Documents tab, check "Use OCR for scanned PDFs"
4. **Automatic Detection**: The system automatically detects scanned PDFs (no extractable text) and uses OCR

**Recommended OCR Models**:
- `qwen/Qwen2-VL-7B-Instruct` (via DeepInfra)
- `microsoft/unilm-dit-layoutlm` (via DeepInfra)
- OpenAI GPT-4 Vision (via OpenAI)

## API Keys Required

Get your API keys from:
- **OpenAI**: https://platform.openai.com/api-keys
- **Claude**: https://console.anthropic.com/
- **Groq**: https://console.groq.com/keys
- **Together AI**: https://api.together.xyz/settings/api-keys
- **Google**: https://makersuite.google.com/app/apikey
- **Mistral**: https://console.mistral.ai/api-keys/
- **DeepInfra**: https://deepinfra.com/dash/api_keys (for OCR models)


## Troubleshooting

### Docker Issues

**Container name already in use:**
```bash
docker-compose down
docker-compose up -d --build
```

**Port already in use:**
- Stop the service using port 8501: `netstat -ano | findstr :8501` (Windows) or `lsof -i :8501` (Linux/Mac)
- Or change the port in docker-compose.yml

**Build failures:**
- Ensure you have sufficient disk space
- Clear Docker cache: `docker system prune -a`
- Check Docker logs: `docker-compose logs streamlit`

**Milvus connection issues:**
- Wait for all services to be healthy: `docker-compose ps`
- Check Milvus logs: `docker-compose logs standalone`
- Verify Milvus is reachable: `curl http://localhost:19530/healthz`
- **Docker**: Ensure Milvus host is set to `standalone` (auto-detected in Configuration tab)
- **Local**: Ensure Milvus host is set to `localhost`
- Restart Streamlit container after configuration changes: `docker restart rag-eval-streamlit`

### OCR Issues

**Poppler not found (only for manual/non-Docker installs):**
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases and add to PATH
- macOS: `brew install poppler`
- Linux: `sudo apt-get install poppler-utils`
- Docker: Poppler is pre-installed, no action needed

**OCR extraction fails:**
- Verify your OCR model supports vision/image inputs
- Check API key is valid for your provider
- Ensure the PDF is a scanned document (not text-based)

### General Issues

**Session state reset:**
- Streamlit sessions reset on page refresh in some configurations
- Use the "Save Configuration" button to persist settings

**Memory issues:**
- Reduce chunk size or number of chunks to retrieve
- Use smaller embedding models
- Process fewer documents at once

## Notes

- Different embedding models require separate Milvus collections due to dimension differences
- Start with 1-5 questions to test your setup before running full evaluations
- OpenAI embeddings require API key; HuggingFace/SentenceTransformers are free and run locally
- Use Docker for the easiest setup with all dependencies pre-configured

