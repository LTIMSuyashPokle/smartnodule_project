# SMARTNODULE MODULE 4 - QUICK START SCRIPT
# Automated setup and launch script

#!/bin/bash

echo "SmartNodule Module 4 - Quick Start Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Step 1: Check prerequisites
print_header "1. Checking Prerequisites"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    print_error "pip is not installed!"
    exit 1
fi

print_status "Python 3 and pip found ✅"

# Step 2: Create project directory structure
print_header "2. Creating Project Structure"

# Create main directories
directories=(
    "api"
    "mlops" 
    "active_learning"
    "monitoring"
    "deployment"
    "logs"
    "mlflow-artifacts"
    "case_retrieval"
    "report_templates"
    "training_data"
    "explanations"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_status "Created directory: $dir"
done

# Step 3: Check for model file
print_header "3. Checking Model File"

MODEL_FILE="smartnodule_memory_optimized_best.pth"
if [ -f "$MODEL_FILE" ]; then
    print_status "Model file found: $MODEL_FILE ✅"
else
    print_warning "Model file not found: $MODEL_FILE"
    print_warning "Please ensure your trained model is named '$MODEL_FILE' and placed in the project root"
fi

# Step 4: Create requirements.txt if it doesn't exist
print_header "4. Setting Up Dependencies"

if [ ! -f "requirements.txt" ]; then
    print_status "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Core ML/AI
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Web frameworks
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
requests>=2.31.0

# Database and storage
faiss-cpu>=1.7.4
sqlalchemy>=2.0.0

# MLOps and monitoring
mlflow>=2.8.0
plotly>=5.17.0
psutil>=5.9.0

# Medical imaging
pydicom>=2.4.0
scipy>=1.11.0
scikit-image>=0.21.0

# Report generation
reportlab>=4.0.0

# Configuration and utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-multipart>=0.0.6
EOF
    print_status "requirements.txt created"
else
    print_status "requirements.txt already exists"
fi

# Step 5: Create .env file if it doesn't exist  
print_header "5. Setting Up Configuration"

if [ ! -f ".env" ]; then
    print_status "Creating .env configuration..."
    #cat > .env << 'EOF'
# API Configuration
API_HOST=localhost
API_PORT=8000
API_WORKERS=1
API_SECRET_KEY=smartnodule-secret-key-change-this

# Database Configuration
DATABASE_URL=sqlite:///./smartnodule.db
CASE_RETRIEVAL_DB=case_retrieval.db

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT=./mlflow-artifacts
MLFLOW_EXPERIMENT_NAME=smartnodule_production

# Model Configuration
MODEL_PATH=smartnodule_memory_optimized_best.pth
MODEL_VERSION=2.0.0
CONFIDENCE_THRESHOLD=0.8

# Monitoring Configuration
METRICS_DB=realtime_metrics.db
ALERTS_DB=alerts.db
AUDIT_LOG_DB=audit_log.db
USAGE_ANALYTICS_DB=usage_analytics.db

# Cache Configuration
CACHE_DB=cache_system.db
CACHE_MAX_MEMORY=1000

# FAISS Configuration
FAISS_INDEX_PATH=case_retrieval/case_retrieval_index.faiss
CASE_METADATA_PATH=case_retrieval/case_metadata.csv
FEATURE_EMBEDDINGS_PATH=case_retrieval/feature_embeddings.npy

# Production Settings
DEBUG=False
LOG_LEVEL=INFO
MAX_REQUEST_SIZE=50MB
REQUEST_TIMEOUT=300
EOF
    print_status ".env configuration created"
else
    print_status ".env configuration already exists"
fi


# Step 6: Create virtual environment
print_header "6. Setting Up Virtual Environment"

if [ ! -d "smartnodule_env" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv smartnodule_env
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source smartnodule_env/bin/activate
print_status "Virtual environment activated"



# Step 7: Install dependencies
print_header "7. Installing Dependencies"
print_status "Installing Python packages (this may take a few minutes)..."

pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_status "Dependencies installed successfully ✅"
else
    print_error "Failed to install dependencies"
    exit 1
fi


# Step 8: Create launcher scripts
print_header "8. Creating Launcher Scripts"


# Create MLflow launcher
#cat > start_mlflow.sh << 'EOF'
#!/bin/bash
echo "Starting MLflow Server..."
#source smartnodule_env/bin/activate
source venv/bin/activate
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
EOF
chmod +x start_mlflow.sh
print_status "Created start_mlflow.sh"


# Create API launcher
#cat > start_api.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting FastAPI Server..."
#source smartnodule_env/bin/activate
source venv/bin/activate
python -c "
try:
    from api.main import app
    import uvicorn
    print('✅ API components loaded successfully')
    uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=True)
except ImportError as e:
    print(f'⚠️ API components not fully implemented yet: {e}')
    print('Running in minimal mode...')
    # Create minimal API server for testing
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title='SmartNodule API - Minimal Mode')
    
    @app.get('/health')
    def health_check():
        return {'status': 'healthy', 'mode': 'minimal'}
        
    @app.get('/status') 
    def status():
        return {'message': 'SmartNodule API running in minimal mode'}
    
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
"
EOF
chmod +x start_api.sh
print_status "Created start_api.sh"


# Create Streamlit launcher
#cat > start_streamlit.sh << 'EOF'
#!/bin/bash
echo "🎨 Starting Streamlit UI..."
#source smartnodule_env/bin/activate
source venv/bin/activate
# Check if enhanced app exists, otherwise use current app
if [ -f "Enhanced_SmartNodule_App_v2.py" ]; then
    echo "Using enhanced Streamlit app..."
    streamlit run Enhanced_SmartNodule_App_v2.py --server.port 8501
elif [ -f "app.py" ]; then
    echo "Using current Streamlit app..."
    streamlit run app.py --server.port 8501
else
    echo "❌ No Streamlit app found!"
    echo "Please ensure app.py or Enhanced_SmartNodule_App_v2.py exists"
    exit 1
fi
EOF
chmod +x start_streamlit.sh
print_status "Created start_streamlit.sh"


# Create complete launcher
#cat > start_all.sh << 'EOF'
#!/bin/bash
echo "Starting Complete SmartNodule System"
echo "======================================="

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️ Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Check ports
echo "Checking ports..."
check_port 5000 || (echo "MLflow port 5000 in use"; exit 1)
check_port 8000 || (echo "API port 8000 in use"; exit 1) 
check_port 8501 || (echo "Streamlit port 8501 in use"; exit 1)

echo "All ports available ✅"

# Start MLflow in background
echo "Starting MLflow Server..."
./start_mlflow.sh > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!
sleep 5

# Start API in background
echo "Starting API Server..." 
./start_api.sh > logs/api.log 2>&1 &
API_PID=$!
sleep 5

# Start Streamlit
echo "Starting Streamlit UI..."
./start_streamlit.sh

# Cleanup on exit
trap 'echo "Stopping services..."; kill $MLFLOW_PID $API_PID' EXIT
EOF
chmod +x start_all.sh
print_status "Created start_all.sh"



# Step 9: Create README
print_header "9. Creating Documentation"

#cat > README.md << 'EOF'
# SmartNodule Module 4 - Production System

## Quick Start

1. **Start all services**:
   ```bash
   ./start_all.sh
   ```

2. **Or start services individually**:
   ```bash
   # Terminal 1: MLflow
   ./start_mlflow.sh
   
   # Terminal 2: API
   ./start_api.sh
   
   # Terminal 3: Streamlit  
   ./start_streamlit.sh
   ```

3. **Access the system**:
   - Streamlit UI: http://localhost:8501
   - FastAPI Docs: http://localhost:8000/docs
   - MLflow UI: http://localhost:5000

## System Requirements

- **Model File**: `smartnodule_memory_optimized_best.pth` (place in root directory)
- **Python**: 3.8+ with virtual environment
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space

## Features

- 🤖 AI-powered nodule detection
- 📊 Real-time monitoring
- 🔬 MLOps pipeline with MLflow
- 👨‍⚕️ Expert annotation interface
- 📄 Professional report generation
- 🏥 System health monitoring
- 🚨 Intelligent alerting
- 📈 Performance analytics

## Troubleshooting

- Check logs in `logs/` directory
- Ensure all ports are available (5000, 8000, 8501)
- Verify model file exists and has correct permissions
- Check virtual environment is activated

## Architecture

```
SmartNodule Module 4
├── FastAPI Backend (Port 8000)
├── Streamlit Frontend (Port 8501) 
├── MLflow Tracking (Port 5000)
├── SQLite Databases
└── File Storage System
```
EOF
print_status "Created README.md"

# Step 10: Final checks and summary
print_header "10. Final Setup Summary"

echo ""
echo "🎉 SmartNodule Module 4 Setup Complete!"
echo "======================================"
echo ""
echo "📁 Project Structure:"
ls -la
echo ""

print_status "✅ Directory structure created"
print_status "✅ Virtual environment setup"
print_status "✅ Dependencies installed"
print_status "✅ Configuration files created"
print_status "✅ Launcher scripts created"
print_status "✅ Documentation created"

echo ""
echo "🚨 IMPORTANT: Before starting the system:"
echo "1. Ensure your model file 'smartnodule_memory_optimized_best.pth' is in the root directory"
echo "2. Replace app.py with the enhanced version if available"
echo "3. Implement the Module 4 components in their respective directories"
echo ""

echo "🚀 To start the complete system:"
echo "   ./start_all.sh"
echo ""

echo "📖 For individual services:"
echo "   ./start_mlflow.sh     (MLflow Server)"
echo "   ./start_api.sh        (FastAPI Backend)" 
echo "   ./start_streamlit.sh  (Streamlit UI)"
echo ""

echo "🌐 System URLs:"
echo "   Streamlit UI:  http://localhost:8501"
echo "   API Docs:      http://localhost:8000/docs"
echo "   MLflow UI:     http://localhost:5000"
echo ""

print_status "Setup completed successfully! 🎊"

# Create a simple system check script
#cat > check_system.sh << 'EOF'
#!/bin/bash
echo "🔍 SmartNodule System Health Check"
echo "================================="

# Check Python
python3 --version
echo "Python: ✅"

# Check virtual environment
if [ -d "smartnodule_env" ]; then
    echo "Virtual Environment: ✅"
else
    echo "Virtual Environment: ❌"
fi

# Check model file
if [ -f "smartnodule_memory_optimized_best.pth" ]; then
    echo "Model File: ✅"
    ls -lh smartnodule_memory_optimized_best.pth
else
    echo "Model File: ❌ (Place your model file in the root directory)"
fi

# Check key files
files=("app.py" ".env" "requirements.txt")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "$file: ✅"
    else
        echo "$file: ❌"
    fi
done

# Check directories
directories=("api" "mlops" "monitoring" "active_learning" "deployment")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "$dir/: ✅"
    else
        echo "$dir/: ❌"
    fi
done

echo ""
echo "System check complete!"
EOF
chmod +x check_system.sh
print_status "Created check_system.sh for system validation"

echo ""
print_warning "Run './check_system.sh' to validate your system setup"
echo ""