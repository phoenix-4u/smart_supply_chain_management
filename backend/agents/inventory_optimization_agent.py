#File: backend/agents/inventory_optimization_agent.py

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import uuid

from models.data_models import (
    InventoryStatus, InventoryItem, InventoryRecommendation, 
    ProductCategory, AlertSeverity
)
from models.database import DatabaseOperations, db_manager, InventoryRecord
from config.settings import settings

logger = logging.getLogger(__name__)

class InventoryOptimizationAgent:
    """
    Inventory Optimization Agent for Health First Supply Chain

    Responsibilities:
    - Optimize inventory levels based on demand forecasts
    - Calculate safety stock and reorder points
    - Generate inventory recommendations
    - Monitor stock levels and alert on critical situations
    """

    def __init__(self):
        self.agent_name = "Inventory Optimization Agent"
        self.current_inventory = {}
        self.recommendations = []
        self.optimization_parameters = {
            'safety_stock_multiplier': settings.SAFETY_STOCK_MULTIPLIER,
            'reorder_point_buffer': settings.REORDER_POINT_BUFFER,
            'max_inventory_multiplier': settings.MAX_INVENTORY_LEVEL_MULTIPLIER
        }
        self.demand_forecasts = {}

    async def optimize_inventory(self):
        """Main optimization cycle for inventory management"""
        try:
            logger.info("Starting inventory optimization cycle")

            # Fetch current inventory levels
            await self._fetch_current_inventory()

            # Fetch demand forecasts
            await self._fetch_demand_forecasts()

            # Calculate optimal inventory levels
            optimization_results = await self._calculate_optimal_levels()

            # Generate recommendations
            recommendations = await self._generate_recommendations(optimization_results)

            # Update inventory parameters
            await self._update_inventory_parameters(optimization_results)

            # Check for critical alerts
            await self._check_critical_alerts()

            self.recommendations = recommendations

            logger.info(f"Inventory optimization completed: {len(recommendations)} recommendations generated")

        except Exception as e:
            logger.error(f"Inventory optimization failed: {str(e)}")
            raise

    async def _fetch_current_inventory(self):
        """Fetch current inventory levels from database"""
        try:
            session = db_manager.get_session()
            inventory_records = session.query(InventoryRecord).all()

            self.current_inventory = {}
            for record in inventory_records:
                key = f"{record.product_category}_{record.location}"
                self.current_inventory[key] = {
                    'product_id': record.product_id,
                    'product_name': record.product_name,
                    'product_category': record.product_category,
                    'location': record.location,
                    'current_stock': record.current_stock,
                    'safety_stock_level': record.safety_stock_level,
                    'reorder_point': record.reorder_point,
                    'max_stock_level': record.max_stock_level,
                    'unit_cost': record.unit_cost,
                    'last_updated': record.last_updated
                }

            session.close()
            logger.info(f"Fetched inventory for {len(self.current_inventory)} items")

        except Exception as e:
            logger.error(f"Failed to fetch current inventory: {str(e)}")
            self.current_inventory = {}

    async def _fetch_demand_forecasts(self):
        """Fetch demand forecasts from demand prediction agent"""
        try:
            # In production, this would fetch from the Demand Prediction Agent
            # For now, simulate demand forecasts

            self.demand_forecasts = {}

            for category in ProductCategory:
                for location in ["Warehouse A", "Warehouse B", "Warehouse C"]:
                    key = f"{category.value}_{location}"

                    # Generate simulated demand forecast
                    base_demand = self._get_base_demand(category)
                    daily_demand = base_demand * random.uniform(0.8, 1.2)

                    # Add variability and trend
                    demand_std = daily_demand * 0.2
                    trend_factor = random.uniform(0.95, 1.05)

                    self.demand_forecasts[key] = {
                        'daily_demand_mean': daily_demand,
                        'daily_demand_std': demand_std,
                        'trend_factor': trend_factor,
                        'lead_time_days': random.randint(3, 14),
                        'service_level': 0.95  # Target service level
                    }

            logger.info(f"Fetched demand forecasts for {len(self.demand_forecasts)} items")

        except Exception as e:
            logger.error(f"Failed to fetch demand forecasts: {str(e)}")
            self.demand_forecasts = {}

    def _get_base_demand(self, category: ProductCategory) -> float:
        """Get base daily demand for product category"""
        base_demands = {
            ProductCategory.PHARMACEUTICALS: 100,
            ProductCategory.MEDICAL_SUPPLIES: 80,
            ProductCategory.PPE: 150,
            ProductCategory.SURGICAL_INSTRUMENTS: 20,
            ProductCategory.DIAGNOSTIC_EQUIPMENT: 15,
            ProductCategory.HOSPITAL_EQUIPMENT: 10,
            ProductCategory.HOME_HEALTHCARE: 30,
            ProductCategory.DIGITAL_HEALTH: 5
        }
        return base_demands.get(category, 50)

    async def _calculate_optimal_levels(self) -> Dict[str, Dict[str, Any]]:
        """Calculate optimal inventory levels for all items"""
        optimization_results = {}

        try:
            for item_key, inventory_data in self.current_inventory.items():
                if item_key not in self.demand_forecasts:
                    continue

                forecast_data = self.demand_forecasts[item_key]

                # Calculate optimal parameters
                optimal_params = self._calculate_item_optimization(inventory_data, forecast_data)
                optimization_results[item_key] = optimal_params

            logger.info(f"Calculated optimization for {len(optimization_results)} items")
            return optimization_results

        except Exception as e:
            logger.error(f"Optimization calculation failed: {str(e)}")
            return {}

    def _calculate_item_optimization(
        self, 
        inventory_data: Dict[str, Any], 
        forecast_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate optimization parameters for a single item"""

        try:
            # Extract parameters
            daily_demand = forecast_data['daily_demand_mean']
            demand_std = forecast_data['daily_demand_std']
            lead_time = forecast_data['lead_time_days']
            service_level = forecast_data['service_level']

            current_stock = inventory_data['current_stock']
            unit_cost = inventory_data['unit_cost']

            # Calculate safety stock using demand uncertainty and lead time
            # Safety stock = Z-score * sqrt(lead_time) * demand_std
            z_score = self._get_z_score(service_level)
            safety_stock = z_score * np.sqrt(lead_time) * demand_std * self.optimization_parameters['safety_stock_multiplier']
            safety_stock = max(0, int(safety_stock))

            # Calculate reorder point
            # Reorder point = (daily_demand * lead_time) + safety_stock + buffer
            average_lead_time_demand = daily_demand * lead_time
            reorder_point = average_lead_time_demand + safety_stock
            reorder_point *= (1 + self.optimization_parameters['reorder_point_buffer'])
            reorder_point = max(0, int(reorder_point))

            # Calculate economic order quantity (EOQ)
            # Simplified EOQ calculation
            annual_demand = daily_demand * 365
            ordering_cost = 50  # Estimated ordering cost
            holding_cost_rate = 0.2  # 20% of unit cost per year
            holding_cost = unit_cost * holding_cost_rate

            if holding_cost > 0:
                eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                eoq = max(1, int(eoq))
            else:
                eoq = int(daily_demand * 30)  # 30 days supply

            # Calculate maximum stock level
            max_stock_level = reorder_point + eoq
            max_stock_level *= self.optimization_parameters['max_inventory_multiplier']
            max_stock_level = int(max_stock_level)

            # Calculate optimal order quantity if needed
            stock_needed = max(0, reorder_point - current_stock)
            optimal_order_qty = max(0, min(eoq, max_stock_level - current_stock))

            # Calculate inventory metrics
            days_of_supply = current_stock / daily_demand if daily_demand > 0 else 0
            turnover_rate = annual_demand / max(current_stock, 1)

            # Determine urgency
            urgency = self._determine_urgency(current_stock, reorder_point, safety_stock)

            return {
                'current_stock': current_stock,
                'optimal_safety_stock': safety_stock,
                'optimal_reorder_point': reorder_point,
                'optimal_max_stock': max_stock_level,
                'economic_order_quantity': eoq,
                'recommended_order_qty': optimal_order_qty,
                'days_of_supply': days_of_supply,
                'turnover_rate': turnover_rate,
                'urgency': urgency,
                'daily_demand': daily_demand,
                'lead_time': lead_time,
                'service_level': service_level
            }

        except Exception as e:
            logger.error(f"Item optimization calculation failed: {str(e)}")
            return {}

    def _get_z_score(self, service_level: float) -> float:
        """Get z-score for given service level"""
        # Mapping of service levels to z-scores
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.97: 1.88,
            0.99: 2.33,
            0.995: 2.58
        }

        # Find closest service level
        closest_level = min(z_scores.keys(), key=lambda x: abs(x - service_level))
        return z_scores[closest_level]

    def _determine_urgency(self, current_stock: int, reorder_point: int, safety_stock: int) -> AlertSeverity:
        """Determine urgency level based on stock levels"""

        if current_stock <= safety_stock * 0.5:
            return AlertSeverity.CRITICAL
        elif current_stock <= safety_stock:
            return AlertSeverity.HIGH
        elif current_stock <= reorder_point:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    async def _generate_recommendations(self, optimization_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate inventory recommendations based on optimization results"""

        recommendations = []

        try:
            for item_key, optimization in optimization_results.items():
                inventory_data = self.current_inventory[item_key]

                # Check if action is needed
                recommended_order_qty = optimization.get('recommended_order_qty', 0)
                urgency = optimization.get('urgency', AlertSeverity.LOW)

                if recommended_order_qty > 0 or urgency in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:

                    # Calculate cost impact
                    unit_cost = inventory_data['unit_cost']
                    cost_impact = recommended_order_qty * unit_cost

                    # Generate reason
                    reason = self._generate_recommendation_reason(optimization, inventory_data)

                    # Estimate stockout date
                    stockout_date = None
                    if optimization.get('daily_demand', 0) > 0:
                        days_until_stockout = inventory_data['current_stock'] / optimization['daily_demand']
                        if days_until_stockout <= 30:  # Only if within 30 days
                            stockout_date = datetime.now() + timedelta(days=days_until_stockout)

                    recommendation = {
                        'recommendation_id': str(uuid.uuid4()),
                        'product_id': inventory_data['product_id'],
                        'product_name': inventory_data['product_name'],
                        'location': inventory_data['location'],
                        'current_stock': inventory_data['current_stock'],
                        'recommended_order_quantity': recommended_order_qty,
                        'urgency': urgency,
                        'reason': reason,
                        'cost_impact': cost_impact,
                        'expected_stockout_date': stockout_date,
                        'optimization_details': optimization,
                        'created_at': datetime.now()
                    }

                    recommendations.append(recommendation)

            # Sort by urgency and cost impact
            recommendations.sort(key=lambda x: (
                ['low', 'medium', 'high', 'critical'].index(x['urgency'].value),
                -x['cost_impact']
            ))

            logger.info(f"Generated {len(recommendations)} inventory recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return []

    def _generate_recommendation_reason(
        self, 
        optimization: Dict[str, Any], 
        inventory_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable reason for recommendation"""

        current_stock = inventory_data['current_stock']
        reorder_point = optimization.get('optimal_reorder_point', 0)
        safety_stock = optimization.get('optimal_safety_stock', 0)
        days_of_supply = optimization.get('days_of_supply', 0)

        if current_stock <= safety_stock * 0.5:
            return f"Critical: Stock below 50% of safety level ({current_stock} units, {days_of_supply:.1f} days supply)"
        elif current_stock <= safety_stock:
            return f"Stock below safety level ({current_stock} units, {days_of_supply:.1f} days supply)"
        elif current_stock <= reorder_point:
            return f"Stock below reorder point ({current_stock} units, {days_of_supply:.1f} days supply)"
        else:
            return "Optimization-based reorder recommendation"

    async def _update_inventory_parameters(self, optimization_results: Dict[str, Dict[str, Any]]):
        """Update inventory parameters in database"""

        try:
            session = db_manager.get_session()

            for item_key, optimization in optimization_results.items():
                inventory_data = self.current_inventory[item_key]
                product_id = inventory_data['product_id']

                # Update inventory record with optimized parameters
                inventory_record = session.query(InventoryRecord).filter_by(product_id=product_id).first()

                if inventory_record:
                    inventory_record.safety_stock_level = optimization.get('optimal_safety_stock', inventory_record.safety_stock_level)
                    inventory_record.reorder_point = optimization.get('optimal_reorder_point', inventory_record.reorder_point)
                    inventory_record.max_stock_level = optimization.get('optimal_max_stock', inventory_record.max_stock_level)
                    inventory_record.last_updated = datetime.now()

            session.commit()
            session.close()

            logger.info(f"Updated inventory parameters for {len(optimization_results)} items")

        except Exception as e:
            logger.error(f"Failed to update inventory parameters: {str(e)}")

    async def _check_critical_alerts(self):
        """Check for critical inventory situations and generate alerts"""

        try:
            critical_items = []

            for item_key, inventory_data in self.current_inventory.items():
                current_stock = inventory_data['current_stock']
                safety_stock = inventory_data['safety_stock_level']

                # Check for critical conditions
                if current_stock <= 0:
                    critical_items.append({
                        'type': 'stockout',
                        'severity': 'critical',
                        'item': inventory_data,
                        'message': f"STOCKOUT: {inventory_data['product_name']} at {inventory_data['location']}"
                    })
                elif current_stock <= safety_stock * 0.5:
                    critical_items.append({
                        'type': 'critical_low',
                        'severity': 'critical',
                        'item': inventory_data,
                        'message': f"CRITICAL LOW: {inventory_data['product_name']} below 50% of safety stock"
                    })

            # Store critical alerts in database
            if critical_items:
                session = db_manager.get_session()

                for alert in critical_items:
                    DatabaseOperations.save_alert(
                        alert_id=str(uuid.uuid4()),
                        alert_type=alert['type'],
                        severity=alert['severity'],
                        title=alert['message'],
                        description=f"Product: {alert['item']['product_name']}, Location: {alert['item']['location']}, Stock: {alert['item']['current_stock']}",
                        source="inventory_optimization",
                        metadata=alert['item'],
                        session=session
                    )

                session.close()
                logger.warning(f"Generated {len(critical_items)} critical inventory alerts")

        except Exception as e:
            logger.error(f"Critical alert check failed: {str(e)}")

    # Public API methods
    async def get_inventory_levels(
        self, 
        product_category: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[InventoryItem]:
        """Get current inventory levels"""

        try:
            inventory_items = []

            for item_key, inventory_data in self.current_inventory.items():
                # Apply filters
                if product_category and inventory_data['product_category'] != product_category:
                    continue
                if location and inventory_data['location'] != location:
                    continue

                # Calculate additional metrics
                forecast_key = f"{inventory_data['product_category']}_{inventory_data['location']}"
                forecast_data = self.demand_forecasts.get(forecast_key, {})
                daily_demand = forecast_data.get('daily_demand_mean', 0)
                lead_time = forecast_data.get('lead_time_days', 7)

                total_value = inventory_data['current_stock'] * inventory_data['unit_cost']

                inventory_item = InventoryItem(
                    product_id=inventory_data['product_id'],
                    product_name=inventory_data['product_name'],
                    product_category=ProductCategory(inventory_data['product_category']),
                    location=inventory_data['location'],
                    current_stock=inventory_data['current_stock'],
                    safety_stock_level=inventory_data['safety_stock_level'],
                    reorder_point=inventory_data['reorder_point'],
                    max_stock_level=inventory_data['max_stock_level'],
                    average_daily_usage=daily_demand,
                    lead_time_days=lead_time,
                    unit_cost=inventory_data['unit_cost'],
                    total_value=total_value
                )

                inventory_items.append(inventory_item)

            return inventory_items

        except Exception as e:
            logger.error(f"Failed to get inventory levels: {str(e)}")
            return []

    async def get_recommendations(self) -> Dict[str, Any]:
        """Get current inventory recommendations"""

        try:
            # Convert recommendations to response format
            recommendation_items = []

            for rec in self.recommendations:
                recommendation = InventoryRecommendation(
                    product_id=rec['product_id'],
                    current_stock=rec['current_stock'],
                    recommended_order_quantity=rec['recommended_order_quantity'],
                    urgency=rec['urgency'],
                    reason=rec['reason'],
                    cost_impact=rec['cost_impact'],
                    expected_stockout_date=rec.get('expected_stockout_date')
                )
                recommendation_items.append(recommendation)

            # Calculate summary statistics
            total_inventory_value = sum(
                item['current_stock'] * item['unit_cost'] 
                for item in self.current_inventory.values()
            )

            items_below_safety = len([
                item for item in self.current_inventory.values()
                if item['current_stock'] <= item['safety_stock_level']
            ])

            items_requiring_reorder = len([
                item for item in self.current_inventory.values()
                if item['current_stock'] <= item['reorder_point']
            ])

            return {
                'recommendations': recommendation_items,
                'summary': {
                    'total_items': len(self.current_inventory),
                    'total_inventory_value': total_inventory_value,
                    'items_below_safety_stock': items_below_safety,
                    'items_requiring_reorder': items_requiring_reorder,
                    'recommendations_count': len(recommendation_items)
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to get recommendations: {str(e)}")
            return {'recommendations': [], 'summary': {}}

    async def adjust_parameters(self, adjustment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust inventory optimization parameters"""

        try:
            # Update optimization parameters
            if 'safety_stock_multiplier' in adjustment_data:
                self.optimization_parameters['safety_stock_multiplier'] = adjustment_data['safety_stock_multiplier']

            if 'reorder_point_buffer' in adjustment_data:
                self.optimization_parameters['reorder_point_buffer'] = adjustment_data['reorder_point_buffer']

            if 'max_inventory_multiplier' in adjustment_data:
                self.optimization_parameters['max_inventory_multiplier'] = adjustment_data['max_inventory_multiplier']

            # Trigger re-optimization with new parameters
            await self.optimize_inventory()

            return {
                'status': 'success',
                'message': 'Inventory parameters updated and re-optimization completed',
                'updated_parameters': self.optimization_parameters,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Parameter adjustment failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now()
            }