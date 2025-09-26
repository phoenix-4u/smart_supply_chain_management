#File: backend/utils/data_processing.py

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from sqlalchemy.orm import Session

from models.database import db_manager, DatabaseOperations
from models.data_models import ProductCategory, Region
from config.settings import settings

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processing utilities for Health First AI Supply Chain

    Handles:
    - Database initialization and management
    - Data cleaning and transformation
    - KPI calculations
    - Data aggregation and analysis
    - Export/import operations
    """

    def __init__(self):
        self.db_manager = db_manager
        self.cache = {}
        self.last_cache_update = None

    async def initialize_database(self):
        """Initialize database connection"""
        try:
            success = self.db_manager.initialize_database()
            if success:
                logger.info("Database initialized successfully")
                await self._load_initial_data()
            return success
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            return False

    async def close_database(self):
        """Close database connections"""
        try:
            self.db_manager.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {str(e)}")

    async def _load_initial_data(self):
        """Load initial sample data if database is empty"""
        try:
            session = self.db_manager.get_session()

            # Check if we need to load initial data
            from models.database import InventoryRecord
            existing_count = session.query(InventoryRecord).count()

            if existing_count == 0:
                from models.database import create_sample_data
                create_sample_data(session)
                logger.info("Initial sample data loaded")

            session.close()

        except Exception as e:
            logger.error(f"Failed to load initial data: {str(e)}")

    async def calculate_kpis(self) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        try:
            session = self.db_manager.get_session()

            # Calculate various KPIs
            kpis = {
                'forecast_accuracy': await self._calculate_forecast_accuracy(session),
                'inventory_turnover': await self._calculate_inventory_turnover(session),
                'stockout_rate': await self._calculate_stockout_rate(session),
                'supplier_performance_avg': await self._calculate_avg_supplier_performance(session),
                'cost_savings_mtd': await self._calculate_cost_savings(session),
                'order_fill_rate': await self._calculate_order_fill_rate(session),
                'emergency_orders_count': await self._calculate_emergency_orders(session),
                'total_inventory_value': await self._calculate_total_inventory_value(session),
                'last_updated': datetime.now().isoformat()
            }

            session.close()

            # Cache results
            self.cache['kpis'] = kpis
            self.last_cache_update = datetime.now()

            logger.info("KPIs calculated successfully")
            return kpis

        except Exception as e:
            logger.error(f"KPI calculation failed: {str(e)}")
            return {}

    async def _calculate_forecast_accuracy(self, session: Session) -> float:
        """Calculate demand forecast accuracy"""
        try:
            # Simulate forecast accuracy calculation
            # In production, compare actual vs. predicted demand
            accuracy = 0.87 + (np.random.random() - 0.5) * 0.1
            return round(max(0.0, min(1.0, accuracy)), 3)
        except Exception as e:
            logger.error(f"Forecast accuracy calculation failed: {str(e)}")
            return 0.0

    async def _calculate_inventory_turnover(self, session: Session) -> float:
        """Calculate average inventory turnover ratio"""
        try:
            from models.database import InventoryRecord

            # Get inventory data
            inventory_items = session.query(InventoryRecord).all()

            if not inventory_items:
                return 0.0

            total_value = sum(item.current_stock * item.unit_cost for item in inventory_items)

            # Simulate annual sales (in production, get from sales data)
            annual_sales = total_value * np.random.uniform(3.0, 8.0)

            turnover = annual_sales / total_value if total_value > 0 else 0.0
            return round(turnover, 2)

        except Exception as e:
            logger.error(f"Inventory turnover calculation failed: {str(e)}")
            return 0.0

    async def _calculate_stockout_rate(self, session: Session) -> float:
        """Calculate stockout rate"""
        try:
            from models.database import InventoryRecord

            inventory_items = session.query(InventoryRecord).all()

            if not inventory_items:
                return 0.0

            stockout_items = len([item for item in inventory_items if item.current_stock <= 0])
            stockout_rate = stockout_items / len(inventory_items)

            return round(stockout_rate, 3)

        except Exception as e:
            logger.error(f"Stockout rate calculation failed: {str(e)}")
            return 0.0

    async def _calculate_avg_supplier_performance(self, session: Session) -> float:
        """Calculate average supplier performance"""
        try:
            from models.database import SupplierRecord

            suppliers = session.query(SupplierRecord).filter_by(active=True).all()

            if not suppliers:
                return 0.0

            total_performance = 0.0
            count = 0

            for supplier in suppliers:
                metrics = supplier.performance_metrics or {}
                on_time_rate = metrics.get('on_time_delivery_rate', 0.0)
                quality_score = metrics.get('quality_score', 0.0) / 5.0  # Normalize to 0-1

                if on_time_rate > 0 or quality_score > 0:
                    performance = (on_time_rate + quality_score) / 2
                    total_performance += performance
                    count += 1

            avg_performance = total_performance / count if count > 0 else 0.0
            return round(avg_performance, 3)

        except Exception as e:
            logger.error(f"Supplier performance calculation failed: {str(e)}")
            return 0.0

    async def _calculate_cost_savings(self, session: Session) -> float:
        """Calculate month-to-date cost savings"""
        try:
            # Simulate cost savings calculation
            # In production, compare actual costs vs. budget/previous period
            base_savings = np.random.uniform(10000, 50000)
            return round(base_savings, 2)

        except Exception as e:
            logger.error(f"Cost savings calculation failed: {str(e)}")
            return 0.0

    async def _calculate_order_fill_rate(self, session: Session) -> float:
        """Calculate order fill rate"""
        try:
            from models.database import PurchaseOrderRecord

            # Get recent orders
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_orders = session.query(PurchaseOrderRecord).filter(
                PurchaseOrderRecord.created_at >= cutoff_date
            ).all()

            if not recent_orders:
                return 0.95  # Default value

            filled_orders = len([order for order in recent_orders if order.status == 'delivered'])
            fill_rate = filled_orders / len(recent_orders)

            return round(fill_rate, 3)

        except Exception as e:
            logger.error(f"Order fill rate calculation failed: {str(e)}")
            return 0.0

    async def _calculate_emergency_orders(self, session: Session) -> int:
        """Calculate number of emergency orders this month"""
        try:
            # Simulate emergency orders count
            return np.random.randint(0, 10)

        except Exception as e:
            logger.error(f"Emergency orders calculation failed: {str(e)}")
            return 0

    async def _calculate_total_inventory_value(self, session: Session) -> float:
        """Calculate total inventory value"""
        try:
            from models.database import InventoryRecord

            inventory_items = session.query(InventoryRecord).all()
            total_value = sum(item.current_stock * item.unit_cost for item in inventory_items)

            return round(total_value, 2)

        except Exception as e:
            logger.error(f"Total inventory value calculation failed: {str(e)}")
            return 0.0

    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary data"""
        try:
            session = self.db_manager.get_session()

            # Get recent alerts
            recent_alerts = await self._get_recent_alerts(session)

            # Get inventory summary
            inventory_summary = await self._get_inventory_summary(session)

            # Get demand forecast summary
            demand_summary = await self._get_demand_summary(session)

            # Get supplier summary
            supplier_summary = await self._get_supplier_summary(session)

            # Get trending products
            trending_products = await self._get_trending_products(session)

            summary = {
                'recent_alerts': recent_alerts,
                'inventory_summary': inventory_summary,
                'demand_summary': demand_summary,
                'supplier_summary': supplier_summary,
                'trending_products': trending_products,
                'timestamp': datetime.now().isoformat()
            }

            session.close()
            return summary

        except Exception as e:
            logger.error(f"Dashboard summary generation failed: {str(e)}")
            return {}

    async def _get_recent_alerts(self, session: Session) -> List[Dict[str, Any]]:
        """Get recent system alerts"""
        try:
            from models.database import AlertRecord

            recent_alerts = session.query(AlertRecord).filter(
                AlertRecord.created_at >= datetime.now() - timedelta(days=7),
                AlertRecord.resolved == False
            ).order_by(AlertRecord.created_at.desc()).limit(10).all()

            alerts = []
            for alert in recent_alerts:
                alerts.append({
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'title': alert.title,
                    'description': alert.description,
                    'source': alert.source,
                    'created_at': alert.created_at.isoformat(),
                    'acknowledged': alert.acknowledged
                })

            return alerts

        except Exception as e:
            logger.error(f"Recent alerts retrieval failed: {str(e)}")
            return []

    async def _get_inventory_summary(self, session: Session) -> Dict[str, Any]:
        """Get inventory summary statistics"""
        try:
            from models.database import InventoryRecord

            inventory_items = session.query(InventoryRecord).all()

            total_items = len(inventory_items)
            total_value = sum(item.current_stock * item.unit_cost for item in inventory_items)

            low_stock_items = len([
                item for item in inventory_items 
                if item.current_stock <= item.safety_stock_level
            ])

            out_of_stock_items = len([
                item for item in inventory_items 
                if item.current_stock <= 0
            ])

            # Category breakdown
            category_breakdown = {}
            for item in inventory_items:
                category = item.product_category
                if category not in category_breakdown:
                    category_breakdown[category] = {'count': 0, 'value': 0}
                category_breakdown[category]['count'] += 1
                category_breakdown[category]['value'] += item.current_stock * item.unit_cost

            return {
                'total_items': total_items,
                'total_value': round(total_value, 2),
                'low_stock_items': low_stock_items,
                'out_of_stock_items': out_of_stock_items,
                'category_breakdown': category_breakdown,
                'stock_health_score': round((total_items - low_stock_items) / total_items, 3) if total_items > 0 else 0.0
            }

        except Exception as e:
            logger.error(f"Inventory summary generation failed: {str(e)}")
            return {}

    async def _get_demand_summary(self, session: Session) -> Dict[str, Any]:
        """Get demand forecast summary"""
        try:
            from models.database import DemandForecastRecord

            # Get recent forecasts
            recent_forecasts = session.query(DemandForecastRecord).filter(
                DemandForecastRecord.created_at >= datetime.now() - timedelta(days=7)
            ).all()

            if not recent_forecasts:
                return {'message': 'No recent forecasts available'}

            total_predicted_demand = sum(f.predicted_demand for f in recent_forecasts)
            avg_confidence = np.mean([
                (f.confidence_upper - f.confidence_lower) / f.predicted_demand 
                for f in recent_forecasts if f.predicted_demand > 0
            ])

            # Category breakdown
            category_demand = {}
            for forecast in recent_forecasts:
                category = forecast.product_category
                if category not in category_demand:
                    category_demand[category] = 0
                category_demand[category] += forecast.predicted_demand

            return {
                'total_predicted_demand': round(total_predicted_demand, 2),
                'average_confidence_interval': round(avg_confidence, 3) if not np.isnan(avg_confidence) else 0.0,
                'forecasts_count': len(recent_forecasts),
                'category_demand': category_demand,
                'forecast_coverage_days': 30
            }

        except Exception as e:
            logger.error(f"Demand summary generation failed: {str(e)}")
            return {}

    async def _get_supplier_summary(self, session: Session) -> Dict[str, Any]:
        """Get supplier performance summary"""
        try:
            from models.database import SupplierRecord, PurchaseOrderRecord

            # Get active suppliers
            suppliers = session.query(SupplierRecord).filter_by(active=True).all()

            # Get recent orders
            recent_orders = session.query(PurchaseOrderRecord).filter(
                PurchaseOrderRecord.created_at >= datetime.now() - timedelta(days=30)
            ).all()

            total_suppliers = len(suppliers)
            total_orders = len(recent_orders)

            # Calculate averages
            avg_on_time_rate = 0.0
            avg_quality_score = 0.0
            supplier_count = 0

            for supplier in suppliers:
                metrics = supplier.performance_metrics or {}
                if metrics:
                    avg_on_time_rate += metrics.get('on_time_delivery_rate', 0.0)
                    avg_quality_score += metrics.get('quality_score', 0.0)
                    supplier_count += 1

            if supplier_count > 0:
                avg_on_time_rate /= supplier_count
                avg_quality_score /= supplier_count

            # Order status breakdown
            status_breakdown = {}
            for order in recent_orders:
                status = order.status
                status_breakdown[status] = status_breakdown.get(status, 0) + 1

            return {
                'total_suppliers': total_suppliers,
                'total_orders_30d': total_orders,
                'avg_on_time_delivery_rate': round(avg_on_time_rate, 3),
                'avg_quality_score': round(avg_quality_score, 2),
                'order_status_breakdown': status_breakdown,
                'supplier_health_score': round(avg_on_time_rate * 0.6 + (avg_quality_score / 5.0) * 0.4, 3)
            }

        except Exception as e:
            logger.error(f"Supplier summary generation failed: {str(e)}")
            return {}

    async def _get_trending_products(self, session: Session) -> List[Dict[str, Any]]:
        """Get trending products based on demand forecasts"""
        try:
            from models.database import DemandForecastRecord

            # Get recent forecasts
            recent_forecasts = session.query(DemandForecastRecord).filter(
                DemandForecastRecord.created_at >= datetime.now() - timedelta(days=7)
            ).all()

            # Aggregate by product category
            product_trends = {}
            for forecast in recent_forecasts:
                category = forecast.product_category
                if category not in product_trends:
                    product_trends[category] = {
                        'total_demand': 0,
                        'forecast_count': 0,
                        'avg_confidence': 0
                    }

                product_trends[category]['total_demand'] += forecast.predicted_demand
                product_trends[category]['forecast_count'] += 1

                # Calculate confidence width
                confidence_width = (forecast.confidence_upper - forecast.confidence_lower) / forecast.predicted_demand if forecast.predicted_demand > 0 else 0
                product_trends[category]['avg_confidence'] += confidence_width

            # Calculate averages and create trending list
            trending = []
            for category, data in product_trends.items():
                avg_confidence = data['avg_confidence'] / data['forecast_count'] if data['forecast_count'] > 0 else 0

                trending.append({
                    'product_category': category,
                    'total_predicted_demand': round(data['total_demand'], 2),
                    'avg_confidence_interval': round(avg_confidence, 3),
                    'trend_score': round(data['total_demand'] / (1 + avg_confidence), 2)  # Higher demand, lower uncertainty = higher score
                })

            # Sort by trend score
            trending.sort(key=lambda x: x['trend_score'], reverse=True)

            return trending[:10]  # Top 10 trending

        except Exception as e:
            logger.error(f"Trending products calculation failed: {str(e)}")
            return []

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        try:
            # Remove duplicates
            data = data.drop_duplicates()

            # Handle missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

            # Fill categorical missing values
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown')

            # Remove outliers using IQR method
            for col in numeric_columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

            logger.info(f"Data cleaned: {len(data)} records remaining")
            return data

        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return data

    def aggregate_data(
        self, 
        data: pd.DataFrame, 
        group_by: List[str], 
        agg_functions: Dict[str, str]
    ) -> pd.DataFrame:
        """Aggregate data by specified columns"""
        try:
            aggregated = data.groupby(group_by).agg(agg_functions).reset_index()
            logger.info(f"Data aggregated: {len(aggregated)} groups created")
            return aggregated

        except Exception as e:
            logger.error(f"Data aggregation failed: {str(e)}")
            return pd.DataFrame()

    async def export_data(
        self, 
        data: pd.DataFrame, 
        filename: str, 
        format: str = 'csv'
    ) -> bool:
        """Export data to file"""
        try:
            if format.lower() == 'csv':
                data.to_csv(filename, index=False)
            elif format.lower() == 'json':
                data.to_json(filename, orient='records', indent=2)
            elif format.lower() == 'excel':
                data.to_excel(filename, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Data exported to {filename}")
            return True

        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            return False

    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report"""
        try:
            report = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'data_types': data.dtypes.astype(str).to_dict(),
                'numeric_summary': {},
                'categorical_summary': {}
            }

            # Numeric columns summary
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                report['numeric_summary'][col] = {
                    'mean': round(data[col].mean(), 2),
                    'median': round(data[col].median(), 2),
                    'std': round(data[col].std(), 2),
                    'min': round(data[col].min(), 2),
                    'max': round(data[col].max(), 2)
                }

            # Categorical columns summary
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                report['categorical_summary'][col] = {
                    'unique_values': data[col].nunique(),
                    'most_common': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'most_common_count': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0
                }

            return report

        except Exception as e:
            logger.error(f"Data quality report generation failed: {str(e)}")
            return {}