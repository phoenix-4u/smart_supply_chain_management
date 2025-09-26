# Health First Agentic AI Supply Chain Management System

## Overview
This project implements an advanced agentic AI system for Health First organization's supply chain management. The system consists of multiple AI agents working collaboratively to monitor market intelligence, predict demand, optimize inventory, and coordinate with suppliers in the healthcare distribution industry.

## Architecture

### Core Agents
1. **Market Intelligence Agent** - Monitors external market signals and trends
2. **Demand Prediction Agent** - Forecasts demand using ML models and market intelligence
3. **Inventory Optimization Agent** - Optimizes stock levels based on demand predictions
4. **Supplier Coordination Agent** - Manages supplier relationships and procurement

### Technology Stack
- **Backend**: Python with FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite (development), PostgreSQL (production)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Real-time Processing**: asyncio, websockets
- **APIs**: RESTful APIs with FastAPI
- **Monitoring**: Custom dashboard with real-time metrics

## Project Structure
```
health-first-ai-supply-chain/
├── README.md
├── requirements.txt
├── backend/
│   ├── main.py                 # FastAPI main application
│   ├── config/
│   │   └── settings.py         # Configuration settings
│   ├── agents/
│   │   ├── market_intelligence_agent.py
│   │   ├── demand_prediction_agent.py
│   │   ├── inventory_optimization_agent.py
│   │   └── supplier_coordination_agent.py
│   ├── models/
│   │   ├── data_models.py      # Pydantic data models
│   │   └── database.py         # Database models and connections
│   ├── utils/
│   │   ├── data_processing.py  # Data processing utilities
│   │   └── ml_utils.py         # Machine learning utilities
│   └── data/
│       ├── sample_data.json    # Sample data for testing
│       └── historical_data.csv # Historical sales data
├── frontend/
│   └── dashboard.py            # Streamlit dashboard
├── tests/
│   ├── test_agents.py          # Agent testing
│   └── test_api.py             # API testing
└── docs/
    └── api_documentation.md    # API documentation
```

## Features

### Market Intelligence Agent
- **Social Media Monitoring**: Track healthcare-related mentions, sentiment, and trends
- **Competitor Tracking**: Monitor price changes, product launches, and market activities
- **Economic Intelligence**: Track macroeconomic indicators affecting healthcare
- **Weather/Disaster Monitoring**: Monitor events impacting healthcare supply chains
- **Healthcare Utilization Trends**: Track hospital admissions, disease outbreaks
- **Pricing Intelligence**: Monitor competitor pricing and market rates

### Demand Prediction Agent  
- **Multi-source Data Integration**: Combines market intelligence with historical data
- **Dynamic Forecasting**: Continuously refines demand predictions
- **Machine Learning Models**: Uses ensemble methods for improved accuracy
- **Real-time Adjustments**: Responds to market intelligence triggers
- **Regional Granularity**: Provides location-specific demand forecasts

### Inventory Optimization Agent
- **Dynamic Stock Management**: Optimizes inventory levels based on demand forecasts
- **Safety Stock Calculation**: Maintains appropriate buffer levels
- **Reorder Point Optimization**: Determines optimal reorder triggers
- **Multi-location Management**: Handles inventory across multiple distribution centers

### Supplier Coordination Agent
- **Automated Procurement**: Generates purchase orders based on inventory needs
- **Supplier Performance Tracking**: Monitors delivery times and quality metrics
- **Risk Assessment**: Evaluates supplier reliability and diversification needs
- **Contract Management**: Tracks contract terms and renewal dates

## Installation and Setup

### Prerequisites
- Python 3.9+
- pip package manager
- Git

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/health-first/ai-supply-chain.git
   cd ai-supply-chain
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**:
   ```bash
   python backend/utils/init_db.py
   ```

6. **Load sample data**:
   ```bash
   python backend/utils/load_sample_data.py
   ```

## Running the Application

### Backend API Server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API will be available at: http://localhost:8000

### Streamlit Dashboard
```bash
cd frontend
streamlit run dashboard.py
```
Dashboard will be available at: http://localhost:8501

## API Endpoints

### Market Intelligence
- `GET /api/market-intelligence/social-media` - Get social media metrics
- `GET /api/market-intelligence/competitors` - Get competitor tracking data
- `GET /api/market-intelligence/alerts` - Get active alerts and triggers

### Demand Prediction
- `GET /api/demand/forecast` - Get demand forecasts
- `POST /api/demand/trigger` - Trigger forecast recalculation
- `GET /api/demand/accuracy` - Get prediction accuracy metrics

### Inventory Optimization
- `GET /api/inventory/levels` - Get current inventory levels
- `GET /api/inventory/recommendations` - Get optimization recommendations
- `POST /api/inventory/adjust` - Adjust inventory parameters

### Supplier Coordination
- `GET /api/suppliers/performance` - Get supplier performance metrics
- `GET /api/suppliers/orders` - Get purchase order status
- `POST /api/suppliers/order` - Create new purchase order

## Dashboard Features

### Overview Page
- Real-time KPI metrics
- Agent status indicators
- Recent alerts and notifications
- Executive summary charts

### Market Intelligence Dashboard
- Social media sentiment analysis
- Competitor price tracking
- Economic indicator trends
- Weather/disaster impact maps

### Demand Forecasting Dashboard
- Demand trend charts
- Forecast accuracy metrics
- Regional demand heatmaps
- Product category breakdowns

### Inventory Management Dashboard
- Stock level visualizations
- Reorder recommendations
- Supplier performance scorecards
- Cost analysis charts

## Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///./health_first.db

# External APIs (configure as needed)
SOCIAL_MEDIA_API_KEY=your_key_here
WEATHER_API_KEY=your_key_here
ECONOMIC_DATA_API_KEY=your_key_here

# Agent Configuration
MARKET_INTELLIGENCE_UPDATE_INTERVAL=300
DEMAND_FORECAST_UPDATE_INTERVAL=600
INVENTORY_CHECK_INTERVAL=900
```

## Testing

### Run Unit Tests
```bash
pytest tests/
```

### Run Integration Tests
```bash
pytest tests/integration/
```

### API Testing
```bash
# Test API endpoints
python tests/test_api.py
```

## Deployment

### Docker Deployment
```bash
# Build containers
docker-compose build

# Run services
docker-compose up -d
```

### Production Considerations
- Use PostgreSQL for production database
- Implement proper logging and monitoring
- Set up SSL/TLS certificates
- Configure load balancing for high availability
- Implement backup and disaster recovery

## Monitoring and Alerts

### System Health Monitoring
- Agent status monitoring
- API response time tracking
- Database connection health
- Memory and CPU usage monitoring

### Business Alerts
- Demand forecast accuracy drops
- Inventory levels reach critical thresholds
- Supplier performance degradation
- Market intelligence triggers activated

## Data Sources

### Internal Data
- Historical sales data
- Inventory levels
- Supplier contracts
- Customer demand patterns

### External Data Sources
- Social media APIs (Twitter, LinkedIn)
- Competitor pricing databases
- Economic indicator feeds (BLS, Fed)
- Weather and disaster APIs (NOAA)
- Healthcare utilization data (CDC, CMS)

## Security

### Authentication & Authorization
- JWT token-based authentication
- Role-based access control
- API key management
- Secure credential storage

### Data Protection
- Encryption at rest and in transit
- PII data anonymization
- Audit logging
- Compliance with healthcare regulations

## Contributing

### Development Guidelines
1. Follow PEP 8 style guidelines
2. Write comprehensive unit tests
3. Update documentation for new features
4. Use semantic versioning for releases

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request with description

## License
This project is proprietary to Health First Organization.

## Support

### Documentation
- API Documentation: `/docs` endpoint when running the API
- Architecture Documentation: `docs/architecture.md`
- Deployment Guide: `docs/deployment.md`

### Contact
- Project Team: ai-supply-chain@healthfirst.com
- Technical Support: tech-support@healthfirst.com

## Changelog

### Version 1.0.0
- Initial release with core agent functionality
- Streamlit dashboard implementation
- Basic market intelligence monitoring
- Demand forecasting with ML models
- Inventory optimization algorithms
- Supplier coordination workflows

---