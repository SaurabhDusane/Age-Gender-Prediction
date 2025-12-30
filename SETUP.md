# Setup Guide

Complete setup instructions for the Demographic Analysis System.

## Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU support)
- PostgreSQL 15+
- Redis 7+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (optional, for containerized deployment)

## Installation

### 1. Clone and Setup Python Environment

```bash
cd age-gender-prediction
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Models

```bash
python scripts/download_models.py
```

**Manual Downloads Required:**
- DEX age model: [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- FairFace models: [GitHub](https://github.com/dchen236/FairFace)
- ArcFace embeddings: [InsightFace](https://github.com/deepinsight/insightface)

Place downloaded models in `models/` directory.

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your configuration:
- Database credentials
- API settings
- Model paths
- Ethical AI preferences

### 4. Setup Database

**Option A: Docker**
```bash
docker-compose up -d postgres redis
```

**Option B: Manual**
Install PostgreSQL and Redis, then:

```bash
# In PostgreSQL
CREATE DATABASE demographics;
CREATE EXTENSION vector;
```

```bash
# Initialize schema
python scripts/init_database.py
```

### 5. Verify Installation

```bash
# Test pipeline
python scripts/test_pipeline.py

# Run tests
pytest tests/

# Start API server
python main.py api
```

Visit `http://localhost:8000/docs` for API documentation.

### 6. Setup Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` for dashboard.

## Docker Deployment

### Full Stack with GPU

```bash
docker-compose up --build
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- API Backend (port 8000)
- Frontend Dashboard (port 3000)

### GPU Requirements

Ensure NVIDIA Docker runtime is installed:
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Usage Examples

### Command Line

```bash
# Analyze image
python main.py image -i photo.jpg -o result.jpg

# Process video
python main.py video -i video.mp4 -o output.mp4

# Webcam analysis
python main.py webcam

# Start API server
python main.py api --host 0.0.0.0 --port 8000
```

### Python API

```python
from src.core.pipeline import DemographicPipeline
import cv2

pipeline = DemographicPipeline(device='cuda')

frame = cv2.imread('image.jpg')
result = pipeline.process(frame)

for face in result.faces:
    print(f"Age: {face.age}, Gender: {face.gender}, Emotion: {face.emotion}")
```

### REST API

```bash
# Analyze image
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@image.jpg" \
  -F "save_to_db=true"

# Get statistics
curl "http://localhost:8000/api/v1/faces/statistics"

# Search faces
curl "http://localhost:8000/api/v1/faces/search?age_min=25&age_max=35&gender=male"
```

## Performance Optimization

### TensorRT Optimization (NVIDIA GPUs)

```python
# Convert models to TensorRT
# See notebooks/03_performance_optimization.ipynb
```

### Multi-Processing

Enable multi-processing in `config/config.yaml`:
```yaml
performance:
  threading:
    num_threads: 4
    use_multiprocessing: true
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use smaller models
- Enable mixed precision training

### Slow FPS
- Disable temporal smoothing
- Reduce quality assessment checks
- Use ONNX/TensorRT optimization

### Database Connection Issues
- Check PostgreSQL is running
- Verify credentials in `.env`
- Ensure pgvector extension is installed

### Model Loading Errors
- Verify models are in `models/` directory
- Check model file permissions
- Try downloading models again

## Development

### Run in Development Mode

```bash
# Backend with auto-reload
python main.py api --reload

# Frontend with hot reload
cd frontend && npm run dev
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Adding New Models

1. Create model class in `src/models/`
2. Add to pipeline in `src/core/pipeline.py`
3. Update config in `config/config.yaml`
4. Add tests in `tests/`

## Security Considerations

- Change default passwords in `.env`
- Enable HTTPS in production
- Implement rate limiting
- Review audit logs regularly
- Follow GDPR/CCPA compliance guidelines

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs
- Email: support@example.com

## License

MIT License - See LICENSE file for details.
