# Deployment Guide

This guide provides instructions for deploying the Human Rights Legal Assistant RAG System in various environments.

## 🚀 Deployment Options

### 1. Local Development
- **Use Case**: Development, testing, small-scale usage
- **Requirements**: Python 3.8+, 8GB RAM recommended
- **Setup Time**: 10 minutes

### 2. Cloud Deployment (Docker)
- **Use Case**: Production, scalable deployment
- **Requirements**: Docker, cloud platform account
- **Setup Time**: 30 minutes

### 3. Streamlit Cloud
- **Use Case**: Quick demo deployment
- **Requirements**: GitHub repository, Streamlit account
- **Setup Time**: 15 minutes

## 🔧 Local Deployment

### Prerequisites
```bash
python --version  # Python 3.8+
pip --version     # Latest pip
git --version     # Git for cloning
```

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/human-rights-rag-system.git
cd human-rights-rag-system

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run application
streamlit run app.py
```

## 🐳 Docker Deployment

### Dockerfile
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/chroma_db

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  rag-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENV=${PINECONE_ENV}
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Add Redis for caching
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### Deploy with Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☁️ Cloud Platform Deployment

### AWS Deployment (ECS)

1. **Create ECR Repository**:
```bash
aws ecr create-repository --repository-name human-rights-rag
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
```

2. **Build and Push Image**:
```bash
docker build -t human-rights-rag .
docker tag human-rights-rag:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/human-rights-rag:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/human-rights-rag:latest
```

3. **Create ECS Service**:
```json
{
  "family": "human-rights-rag-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "rag-app",
      "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/human-rights-rag:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "OPENAI_API_KEY", "value": "your-key"},
        {"name": "GROQ_API_KEY", "value": "your-key"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/human-rights-rag",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform (Cloud Run)

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT/human-rights-rag
gcloud run deploy human-rights-rag \
  --image gcr.io/YOUR_PROJECT/human-rights-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key,GROQ_API_KEY=your-key
```

### Azure Container Instances

```bash
# Create resource group
az group create --name human-rights-rag-rg --location eastus

# Deploy container
az container create \
  --resource-group human-rights-rag-rg \
  --name human-rights-rag \
  --image YOUR_REGISTRY/human-rights-rag:latest \
  --dns-name-label human-rights-rag \
  --ports 8501 \
  --environment-variables OPENAI_API_KEY=your-key GROQ_API_KEY=your-key
```

## 🌐 Streamlit Cloud Deployment

1. **Push to GitHub**:
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub repository
- Set branch to `main`
- Set main file path to `app.py`
- Add secrets in the Streamlit Cloud dashboard

3. **Configure Secrets**:
In Streamlit Cloud settings, add:
```toml
[secrets]
OPENAI_API_KEY = "your-openai-key"
GROQ_API_KEY = "your-groq-key"
PINECONE_API_KEY = "your-pinecone-key"
PINECONE_ENV = "your-pinecone-env"
```

## 🔧 Production Configuration

### Environment Variables
```bash
# Required
export OPENAI_API_KEY="your-key"
export GROQ_API_KEY="your-key"

# Optional but recommended
export PINECONE_API_KEY="your-key"
export PINECONE_ENV="us-west1-gcp"
export LANGSMITH_API_KEY="your-key"

# Performance tuning
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
```

### Security Considerations
- Use environment variables for API keys
- Enable HTTPS in production
- Implement rate limiting
- Monitor API usage and costs
- Regular security updates

### Performance Optimization
- Use Redis for caching
- Implement connection pooling
- Monitor memory usage
- Set up log aggregation
- Configure auto-scaling

## 📊 Monitoring

### Health Checks
```python
# Add to app.py for health monitoring
import streamlit as st

def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}
```

### Logging Configuration
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**:
   - Increase container memory
   - Optimize model loading
   - Use model quantization

2. **API Rate Limits**:
   - Implement exponential backoff
   - Use multiple API keys
   - Add request caching

3. **Database Connection**:
   - Check network connectivity
   - Verify credentials
   - Monitor database health

### Support
- Check application logs
- Monitor resource usage
- Test API connectivity
- Verify environment variables

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers
- Implement session affinity
- Share state via external storage

### Vertical Scaling
- Monitor CPU/memory usage
- Optimize model loading
- Use faster storage (SSD)

### Database Scaling
- ChromaDB: Consider sharding
- Pinecone: Use multiple indexes
- Implement read replicas
