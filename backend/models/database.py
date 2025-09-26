#File: backend/models/database.py"

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import json
from typing import Optional, List, Dict, Any
import asyncio
import aiosqlite
from config.settings import settings

Base = declarative_base()

# Database Models
class MarketIntelligenceRecord(Base):
    """Market intelligence data storage"""
    __tablename__ = "market_intelligence"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    source_type = Column(String(50), nullable=False)  # social_media, competitor, economic, etc.
    data = Column(JSON, nullable=False)
    processed = Column(Boolean, default=False)
    alerts_generated = Column(Integer, default=0)

    def __repr__(self):
        return f"<MarketIntelligence(id={self.id}, source={self.source_type}, timestamp={self.timestamp})>"

class DemandForecastRecord(Base):
    """Demand forecast storage"""
    __tablename__ = "demand_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    forecast_date = Column(DateTime, nullable=False)
    product_category = Column(String(50), nullable=False)
    region = Column(String(50), nullable=False)
    predicted_demand = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=False)
    confidence_upper = Column(Float, nullable=False)
    model_version = Column(String(20), nullable=False)
    accuracy_score = Column(Float, nullable=True)
    factors = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<DemandForecast(product={self.product_category}, region={self.region}, demand={self.predicted_demand})>"

class InventoryRecord(Base):
    """Inventory tracking"""
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(50), nullable=False, unique=True)
    product_name = Column(String(200), nullable=False)
    product_category = Column(String(50), nullable=False)
    location = Column(String(100), nullable=False)
    current_stock = Column(Integer, nullable=False, default=0)
    safety_stock_level = Column(Integer, nullable=False, default=0)
    reorder_point = Column(Integer, nullable=False, default=0)
    max_stock_level = Column(Integer, nullable=False)
    unit_cost = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=func.now())

    # Relationships
    transactions = relationship("InventoryTransaction", back_populates="inventory_item")

    def __repr__(self):
        return f"<Inventory(product={self.product_name}, stock={self.current_stock}, location={self.location})>"

class InventoryTransaction(Base):
    """Inventory transaction history"""
    __tablename__ = "inventory_transactions"

    id = Column(Integer, primary_key=True, index=True)
    inventory_id = Column(Integer, ForeignKey("inventory.id"), nullable=False)
    transaction_type = Column(String(20), nullable=False)  # 'in', 'out', 'adjustment'
    quantity = Column(Integer, nullable=False)
    unit_cost = Column(Float, nullable=True)
    reference_id = Column(String(100), nullable=True)  # PO number, sale ID, etc.
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=func.now())

    # Relationships
    inventory_item = relationship("InventoryRecord", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction(type={self.transaction_type}, qty={self.quantity}, timestamp={self.timestamp})>"

class SupplierRecord(Base):
    """Supplier information"""
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(String(50), nullable=False, unique=True)
    supplier_name = Column(String(200), nullable=False)
    contact_info = Column(JSON, nullable=False)
    product_categories = Column(JSON, nullable=False)
    regions_served = Column(JSON, nullable=False)
    contract_start_date = Column(DateTime, nullable=False)
    contract_end_date = Column(DateTime, nullable=False)
    performance_metrics = Column(JSON, nullable=True)
    risk_level = Column(String(20), default='low')
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    purchase_orders = relationship("PurchaseOrderRecord", back_populates="supplier")

    def __repr__(self):
        return f"<Supplier(name={self.supplier_name}, risk={self.risk_level})>"

class PurchaseOrderRecord(Base):
    """Purchase order tracking"""
    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(50), nullable=False, unique=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    order_date = Column(DateTime, nullable=False)
    expected_delivery_date = Column(DateTime, nullable=False)
    actual_delivery_date = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False, default='pending')
    items = Column(JSON, nullable=False)
    total_amount = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    supplier = relationship("SupplierRecord", back_populates="purchase_orders")

    def __repr__(self):
        return f"<PurchaseOrder(id={self.order_id}, supplier={self.supplier_id}, status={self.status})>"

class AlertRecord(Base):
    """System alerts and notifications"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(50), nullable=False, unique=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    source = Column(String(50), nullable=False)
    alert_metadata = Column(JSON, nullable=True)
    acknowledged = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Alert(type={self.alert_type}, severity={self.severity}, resolved={self.resolved})>"

class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)
    agent_name = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Metric(name={self.metric_name}, value={self.metric_value}, agent={self.agent_name})>"

# Database Connection Management
class DatabaseManager:
    """Database connection and session management"""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.database_url = settings.DATABASE_URL

    def initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.engine = create_engine(
                self.database_url,
                echo=settings.DATABASE_ECHO
            )

            # Create all tables
            Base.metadata.create_all(bind=self.engine)

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            print("✅ Database initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Database initialization failed: {str(e)}")
            return False

    def get_session(self) -> Session:
        """Get database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize_database() first.")
        return self.SessionLocal()

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()

# Global database manager instance
db_manager = DatabaseManager()

# Database operations
class DatabaseOperations:
    """Database operations for the application"""

    @staticmethod
    def save_market_intelligence(source_type: str, data: Dict[str, Any], session: Session):
        """Save market intelligence data"""
        record = MarketIntelligenceRecord(
            source_type=source_type,
            data=data
        )
        session.add(record)
        session.commit()
        return record

    @staticmethod
    def save_demand_forecast(
        forecast_date: datetime,
        product_category: str,
        region: str,
        predicted_demand: float,
        confidence_lower: float,
        confidence_upper: float,
        model_version: str,
        factors: List[str],
        session: Session
    ):
        """Save demand forecast"""
        record = DemandForecastRecord(
            forecast_date=forecast_date,
            product_category=product_category,
            region=region,
            predicted_demand=predicted_demand,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            model_version=model_version,
            factors=factors
        )
        session.add(record)
        session.commit()
        return record

    @staticmethod
    def update_inventory(
        product_id: str,
        current_stock: int,
        session: Session
    ):
        """Update inventory levels"""
        inventory = session.query(InventoryRecord).filter_by(product_id=product_id).first()
        if inventory:
            inventory.current_stock = current_stock
            inventory.last_updated = datetime.now()
            session.commit()
        return inventory

    @staticmethod
    def create_inventory_transaction(
        inventory_id: int,
        transaction_type: str,
        quantity: int,
        unit_cost: Optional[float],
        reference_id: Optional[str],
        notes: Optional[str],
        session: Session
    ):
        """Create inventory transaction record"""
        transaction = InventoryTransaction(
            inventory_id=inventory_id,
            transaction_type=transaction_type,
            quantity=quantity,
            unit_cost=unit_cost,
            reference_id=reference_id,
            notes=notes
        )
        session.add(transaction)
        session.commit()
        return transaction

    @staticmethod
    def save_alert(
        alert_id: str,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        source: str,
        metadata: Dict[str, Any],
        session: Session
    ):
        """Save system alert"""
        alert = AlertRecord(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            source=source,
            alert_metadata=metadata
        )
        session.add(alert)
        session.commit()
        return alert

    @staticmethod
    def get_recent_forecasts(
        product_category: Optional[str] = None,
        region: Optional[str] = None,
        days_back: int = 7,
        session: Session = None
    ) -> List[DemandForecastRecord]:
        """Get recent demand forecasts"""
        query = session.query(DemandForecastRecord)

        if product_category:
            query = query.filter_by(product_category=product_category)
        if region:
            query = query.filter_by(region=region)

        cutoff_date = datetime.now() - timedelta(days=days_back)
        query = query.filter(DemandForecastRecord.created_at >= cutoff_date)

        return query.order_by(DemandForecastRecord.created_at.desc()).all()

    @staticmethod
    def get_inventory_below_safety_stock(session: Session) -> List[InventoryRecord]:
        """Get inventory items below safety stock levels"""
        return session.query(InventoryRecord).filter(
            InventoryRecord.current_stock < InventoryRecord.safety_stock_level
        ).all()

    @staticmethod
    def get_active_alerts(session: Session) -> List[AlertRecord]:
        """Get unresolved alerts"""
        return session.query(AlertRecord).filter_by(resolved=False).order_by(
            AlertRecord.created_at.desc()
        ).all()

    @staticmethod
    def save_system_metric(
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        agent_name: str,
        session: Session
    ):
        """Save system performance metric"""
        metric = SystemMetrics(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            agent_name=agent_name
        )
        session.add(metric)
        session.commit()
        return metric

# Sample data functions for testing
def create_sample_data(session: Session):
    """Create sample data for testing"""

    # Sample suppliers
    suppliers_data = [
        {
            "supplier_id": "SUPP001",
            "supplier_name": "MediSupply Corp",
            "contact_info": {"email": "contact@medisupply.com", "phone": "555-0101"},
            "product_categories": ["pharmaceuticals", "medical_supplies"],
            "regions_served": ["northeast", "southeast"],
            "contract_start_date": datetime(2024, 1, 1),
            "contract_end_date": datetime(2025, 12, 31),
            "performance_metrics": {"on_time_delivery": 0.95, "quality_score": 4.2}
        },
        {
            "supplier_id": "SUPP002", 
            "supplier_name": "HealthTech Solutions",
            "contact_info": {"email": "sales@healthtech.com", "phone": "555-0202"},
            "product_categories": ["ppe", "diagnostic_equipment"],
            "regions_served": ["midwest", "west_coast"],
            "contract_start_date": datetime(2024, 3, 1),
            "contract_end_date": datetime(2026, 2, 28),
            "performance_metrics": {"on_time_delivery": 0.89, "quality_score": 4.5}
        }
    ]

    for supplier_data in suppliers_data:
        supplier = SupplierRecord(**supplier_data)
        session.add(supplier)

    # Sample inventory items
    inventory_data = [
        {
            "product_id": "PROD001",
            "product_name": "N95 Respirator Masks",
            "product_category": "ppe",
            "location": "Warehouse A",
            "current_stock": 5000,
            "safety_stock_level": 2000,
            "reorder_point": 3000,
            "max_stock_level": 10000,
            "unit_cost": 2.50
        },
        {
            "product_id": "PROD002",
            "product_name": "Surgical Gloves (Box)",
            "product_category": "ppe",
            "location": "Warehouse B",
            "current_stock": 1500,
            "safety_stock_level": 1000,
            "reorder_point": 1200,
            "max_stock_level": 5000,
            "unit_cost": 15.00
        },
        {
            "product_id": "PROD003",
            "product_name": "IV Fluid Bags",
            "product_category": "medical_supplies",
            "location": "Warehouse A",
            "current_stock": 800,
            "safety_stock_level": 500,
            "reorder_point": 600,
            "max_stock_level": 2000,
            "unit_cost": 8.75
        }
    ]

    for item_data in inventory_data:
        inventory_item = InventoryRecord(**item_data)
        session.add(inventory_item)

    session.commit()
    print("✅ Sample data created successfully")

# Initialize database function
def init_database():
    """Initialize database and create sample data"""
    success = db_manager.initialize_database()
    if success:
        session = db_manager.get_session()
        try:
            create_sample_data(session)
        except Exception as e:
            print(f"Warning: Sample data creation failed: {str(e)}")
        finally:
            session.close()
    return success

if __name__ == "__main__":
    init_database()
