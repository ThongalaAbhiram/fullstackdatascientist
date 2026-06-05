# Startup Validation Platform - Backend Integration Guide

## ✅ Integration Status

**All backend modules successfully integrated and tested.**

- **Test Result**: ✅ **100% PASS** (14/14 tests)
- **Integration Level**: Complete
- **Status**: Production-ready for local development

---

## 📋 Integrated Components

### 1. **FastAPI Main Application** (`backend/app/main.py`)
- Central entry point combining all modules
- CORS middleware enabled for development
- Routes: `/`, `/api/*`, `/competitors/*`, optionally `/ann/*`

### 2. **API Module** (`backend/app/api/`)
- **CRUD Operations**: Create, Read, Update, Delete startups
- **Health Checks**: System status endpoint
- **Startup Store**: In-memory data store with seed data (688 startups)
- **Endpoints**:
  - `GET /api/health` - Health check
  - `GET /api/startups` - List with filters (status, category, country, pagination)
  - `GET /api/startups/{id}` - Get single startup
  - `POST /api/startups` - Create startup
  - `PUT /api/startups/{id}` - Update startup
  - `DELETE /api/startups/{id}` - Delete startup

### 3. **Prediction Engine** (`backend/app/api/store.py`)
- **Algorithm**: Heuristic-based scoring (no external ML model required)
- **Features**: Status, founding year, categories, founders, investors, location, description
- **Endpoint**: `POST /api/predict/success`
- **Output**: Success probability, confidence level, contributing factors

### 4. **Competitor Analysis** (`backend/app/competitor_analysis/`)
- **Algorithm**: KMeans clustering (8 clusters) + cosine similarity
- **Training**: Cluster model pre-trained and saved
- **Endpoints**:
  - `GET /competitors?company=<name>&top_n=5` - Find competitors for dataset company
  - `POST /competitors/by-features` - Find competitors for new/unknown startup
  - `GET /competitors/clusters/summary` - View cluster statistics

### 5. **ANN Model** (`backend/app/models/ANN_Model/`) [Optional]
- Optional integration point (requires TensorFlow + trained model)
- Currently not mounted in main app (requires model artifacts)
- Can be integrated when `/models/ANN_Model/best_model.keras` is available

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install Python 3.8+
# Conda/venv recommended

# Install dependencies
pip install -r backend/requirements.txt
```

### Run the Backend
```bash
# Set PYTHONPATH and run main app
set PYTHONPATH=C:\path\to\backend\Startup-validation\backend
python -m uvicorn app.main:app --reload --port 8000
```

Or directly:
```bash
cd backend/Startup-validation
python backend/app/main.py
```

**API will be available at**: `http://localhost:8000`

**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

---

## 📊 Test Coverage

### Comprehensive Integration Tests
Run all tests together:
```bash
set PYTHONPATH=C:\path\to\backend\Startup-validation\backend
python backend/tests/comprehensive_integration_test.py
```

**Test Categories**:

#### Tier 1: System Health (2/2 ✓)
- Root endpoint
- Health check with startup count

#### Tier 2: CRUD Operations (5/5 ✓)
- List startups (paginated)
- Filter by status, country
- Create startup
- Get startup by ID
- Update startup
- Delete startup

#### Tier 3: Prediction (2/2 ✓)
- High-confidence prediction (well-funded, operating)
- Low-confidence prediction (inactive/dead startup)

#### Tier 4: Competitor Analysis (3/3 ✓)
- Find competitors by company name (from dataset)
- Find competitors by features (new startup)
- Cluster summary statistics

#### Tier 5: Cleanup (1/1 ✓)
- Delete created startup

**Total**: **14/14 tests passing** ✅

---

## 📁 File Structure

```
backend/
├── requirements.txt              # All runtime dependencies
├── app/
│   ├── main.py                  # ⭐ Central FastAPI app (integrated)
│   ├── api/
│   │   ├── routes.py            # API endpoints
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── store.py             # Startup store + heuristic prediction
│   ├── competitor_analysis/
│   │   ├── clustering.py        # KMeans clustering trainer
│   │   ├── similarity_engine.py  # Cosine similarity ranking
│   │   ├── competitor_endpoint.py # Competitor API router
│   │   └── models/
│   │       └── cluster_model.pkl # Pre-trained cluster model
│   ├── models/
│   │   └── ANN_Model/           # (Optional) Neural network
│   └── features/
│       └── success_prediction/  # (Optional) ML features
├── tests/
│   ├── integration_test_client.py
│   └── comprehensive_integration_test.py ⭐ (Run this!)
└── database/
    └── data/
        ├── raw/
        │   └── Startups.csv     # Dataset (688 companies)
        └── cleaned/
            └── Startups_cleaned.csv
```

---

## 🔧 Key Configuration

### `backend/requirements.txt`
```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

### `backend/app/main.py`
```python
# Combines:
from .api.routes import router as api_router
from .competitor_analysis.competitor_endpoint import router as competitor_router

# CORS enabled for development
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Routes mounted
app.include_router(api_router)                                   # /api/*
app.include_router(competitor_router, prefix="/competitors")     # /competitors/*
```

---

## 🧪 Example API Calls

### 1. Health Check
```bash
curl http://localhost:8000/api/health
```
**Response**:
```json
{
  "status": "ok",
  "message": "Backend API is ready",
  "total_startups": 688
}
```

### 2. Create Startup
```bash
curl -X POST http://localhost:8000/api/startups \
  -H "Content-Type: application/json" \
  -d '{
    "company": "NewStartup AI",
    "status": "Operating",
    "year_founded": 2023,
    "description": "AI platform",
    "categories": ["AI", "SaaS"],
    "founders": ["Alice"],
    "investors": ["VC Fund"],
    "funding_rounds": [],
    "city": "SF",
    "state": "CA",
    "country": "USA"
  }'
```

### 3. Predict Success
```bash
curl -X POST http://localhost:8000/api/predict/success \
  -H "Content-Type: application/json" \
  -d '{
    "company": "TestCo",
    "status": "Operating",
    "year_founded": 2020,
    "categories": ["SaaS"],
    "founders": ["Alice", "Bob"],
    "investors": ["Acme VC"],
    "country": "USA"
  }'
```
**Response**:
```json
{
  "company": "TestCo",
  "predicted_success": true,
  "probability": 0.71,
  "confidence": "medium",
  "factors": [
    "Operating status is a positive signal",
    "Recent founding year supports momentum",
    "Category coverage contributes 0.04 to the score",
    ...
  ],
  "model_name": "heuristic-ann-placeholder"
}
```

### 4. Find Competitors
```bash
curl http://localhost:8000/competitors?company=Google&top_n=5
```

### 5. Find Competitors by Features
```bash
curl -X POST http://localhost:8000/competitors/by-features \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "NewTechCorp",
    "categories": "AI, Analytics",
    "year_founded": 2022,
    "total_funding": 5000000,
    "headquarters_city": "San Francisco",
    "headquarters_country": "United States",
    "top_n": 5
  }'
```

---

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'app'
**Fix**: Set `PYTHONPATH` to point to the `backend` directory
```bash
set PYTHONPATH=C:\startup validation platform\backend\Startup-validation\backend
```

### Cluster model not found
**Fix**: Pre-trained model is already saved at:
```
backend/app/competitor_analysis/models/cluster_model.pkl
```
If missing, retrain:
```bash
python -m app.competitor_analysis.clustering --data database/data/raw/Startups.csv
```

### Port 8000 already in use
**Fix**: Use a different port:
```bash
python -m uvicorn app.main:app --reload --port 8001
```

---

## 🔐 Security Notes (Development)

- CORS is **permissive** (`allow_origins=["*"]`) for development only
- In production, restrict to specific origins
- No authentication/authorization in this setup

---

## 📈 Next Steps

1. **Frontend Integration**: Connect React frontend to these API endpoints
2. **ANN Model**: Train and integrate TensorFlow model when ready
3. **Database**: Replace in-memory store with persistent DB (PostgreSQL, MongoDB)
4. **Authentication**: Add JWT/OAuth for production
5. **Monitoring**: Add logging, metrics, error tracking

---

## 📞 Support

All components tested and integrated. Backend is ready for:
- ✅ Local development
- ✅ Frontend integration testing
- ✅ API specification review
- ✅ Deployment planning
