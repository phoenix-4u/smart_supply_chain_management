#File: backend/agents/supplier_coordination_agent.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import uuid
import json

from models.data_models import (
    SupplierPerformance, Supplier, PurchaseOrder, SupplierMetrics, 
    ProductCategory, Region, OrderStatus, AlertSeverity
)
from models.database import (
    DatabaseOperations, db_manager, SupplierRecord, PurchaseOrderRecord
)
from config.settings import settings

logger = logging.getLogger(__name__)

class SupplierCoordinationAgent:
    """
    Supplier Coordination Agent for Health First Supply Chain

    Responsibilities:
    - Manage supplier relationships and performance
    - Generate and track purchase orders
    - Monitor supplier delivery performance
    - Coordinate procurement based on inventory needs
    - Assess supplier risks and diversification
    """

    def __init__(self):
        self.agent_name = "Supplier Coordination Agent"
        self.suppliers = {}
        self.purchase_orders = {}
        self.performance_metrics = {}
        self.risk_assessments = {}
        self.procurement_recommendations = []

    async def coordinate_suppliers(self):
        """Main coordination cycle for supplier management"""
        try:
            logger.info("Starting supplier coordination cycle")

            # Fetch supplier data and performance
            await self._fetch_supplier_data()

            # Update supplier performance metrics
            await self._update_supplier_performance()

            # Assess supplier risks
            await self._assess_supplier_risks()

            # Check for procurement needs
            await self._check_procurement_needs()

            # Generate procurement recommendations
            await self._generate_procurement_recommendations()

            # Process pending purchase orders
            await self._process_purchase_orders()

            # Generate supplier alerts
            await self._generate_supplier_alerts()

            logger.info("Supplier coordination cycle completed successfully")

        except Exception as e:
            logger.error(f"Supplier coordination failed: {str(e)}")
            raise

    async def _fetch_supplier_data(self):
        """Fetch supplier information from database"""
        try:
            session = db_manager.get_session()
            supplier_records = session.query(SupplierRecord).filter_by(active=True).all()

            self.suppliers = {}
            for record in supplier_records:
                supplier_data = {
                    'supplier_id': record.supplier_id,
                    'supplier_name': record.supplier_name,
                    'contact_info': record.contact_info,
                    'product_categories': record.product_categories,
                    'regions_served': record.regions_served,
                    'contract_start_date': record.contract_start_date,
                    'contract_end_date': record.contract_end_date,
                    'performance_metrics': record.performance_metrics or {},
                    'risk_level': record.risk_level,
                    'active': record.active
                }
                self.suppliers[record.supplier_id] = supplier_data

            session.close()
            logger.info(f"Fetched data for {len(self.suppliers)} active suppliers")

        except Exception as e:
            logger.error(f"Failed to fetch supplier data: {str(e)}")
            self.suppliers = {}

    async def _update_supplier_performance(self):
        """Update supplier performance metrics based on recent orders"""
        try:
            session = db_manager.get_session()

            for supplier_id, supplier_data in self.suppliers.items():
                # Get recent orders for this supplier
                recent_orders = session.query(PurchaseOrderRecord).filter(
                    PurchaseOrderRecord.supplier_id == supplier_id,
                    PurchaseOrderRecord.created_at >= datetime.now() - timedelta(days=90)
                ).all()

                if recent_orders:
                    metrics = self._calculate_supplier_metrics(recent_orders)
                    self.performance_metrics[supplier_id] = metrics

                    # Update supplier record
                    supplier_record = session.query(SupplierRecord).filter_by(supplier_id=supplier_id).first()
                    if supplier_record:
                        supplier_record.performance_metrics = metrics
                else:
                    # Generate simulated metrics for demonstration
                    self.performance_metrics[supplier_id] = self._generate_simulated_metrics()

            session.commit()
            session.close()

            logger.info(f"Updated performance metrics for {len(self.performance_metrics)} suppliers")

        except Exception as e:
            logger.error(f"Failed to update supplier performance: {str(e)}")

    def _calculate_supplier_metrics(self, orders: List[PurchaseOrderRecord]) -> Dict[str, float]:
        """Calculate performance metrics from order history"""

        if not orders:
            return self._generate_simulated_metrics()

        # Calculate on-time delivery rate
        delivered_orders = [o for o in orders if o.status == OrderStatus.DELIVERED.value and o.actual_delivery_date]
        on_time_deliveries = 0

        for order in delivered_orders:
            if order.actual_delivery_date <= order.expected_delivery_date:
                on_time_deliveries += 1

        on_time_delivery_rate = on_time_deliveries / len(delivered_orders) if delivered_orders else 0.0

        # Calculate average lead time
        lead_times = []
        for order in delivered_orders:
            lead_time = (order.actual_delivery_date - order.order_date).days
            lead_times.append(lead_time)

        average_lead_time = sum(lead_times) / len(lead_times) if lead_times else 7.0

        # Simulate other metrics (in production, calculate from actual data)
        quality_score = random.uniform(3.5, 5.0)
        cost_competitiveness = random.uniform(0.7, 0.95)
        reliability_score = on_time_delivery_rate * random.uniform(0.9, 1.1)
        communication_score = random.uniform(3.8, 5.0)

        return {
            'on_time_delivery_rate': round(on_time_delivery_rate, 3),
            'quality_score': round(quality_score, 2),
            'average_lead_time_days': round(average_lead_time, 1),
            'cost_competitiveness': round(cost_competitiveness, 3),
            'reliability_score': round(min(1.0, reliability_score), 3),
            'communication_score': round(communication_score, 2),
            'last_updated': datetime.now().isoformat()
        }

    def _generate_simulated_metrics(self) -> Dict[str, float]:
        """Generate simulated supplier metrics for demonstration"""
        return {
            'on_time_delivery_rate': round(random.uniform(0.75, 0.98), 3),
            'quality_score': round(random.uniform(3.5, 5.0), 2),
            'average_lead_time_days': round(random.uniform(3.0, 14.0), 1),
            'cost_competitiveness': round(random.uniform(0.7, 0.95), 3),
            'reliability_score': round(random.uniform(0.8, 0.98), 3),
            'communication_score': round(random.uniform(3.8, 5.0), 2),
            'last_updated': datetime.now().isoformat()
        }

    async def _assess_supplier_risks(self):
        """Assess risks for each supplier"""
        try:
            for supplier_id, supplier_data in self.suppliers.items():
                metrics = self.performance_metrics.get(supplier_id, {})

                risk_factors = []
                risk_score = 0.0

                # Performance-based risk factors
                on_time_rate = metrics.get('on_time_delivery_rate', 0.0)
                if on_time_rate < 0.90:
                    risk_factors.append(f"Low on-time delivery rate: {on_time_rate:.1%}")
                    risk_score += 0.3

                quality_score = metrics.get('quality_score', 0.0)
                if quality_score < 4.0:
                    risk_factors.append(f"Low quality score: {quality_score}/5")
                    risk_score += 0.2

                lead_time = metrics.get('average_lead_time_days', 0.0)
                if lead_time > 10:
                    risk_factors.append(f"Long lead time: {lead_time} days")
                    risk_score += 0.1

                # Contract-based risk factors
                contract_end = supplier_data['contract_end_date']
                if isinstance(contract_end, str):
                    contract_end = datetime.fromisoformat(contract_end)

                days_to_expiry = (contract_end - datetime.now()).days
                if days_to_expiry < 90:
                    risk_factors.append(f"Contract expiring in {days_to_expiry} days")
                    risk_score += 0.4

                # Geographic concentration risk
                regions_served = supplier_data.get('regions_served', [])
                if len(regions_served) == 1:
                    risk_factors.append("Geographic concentration - single region")
                    risk_score += 0.2

                # Product concentration risk
                product_categories = supplier_data.get('product_categories', [])
                if len(product_categories) == 1:
                    risk_factors.append("Product concentration - single category")
                    risk_score += 0.1

                # Determine risk level
                if risk_score >= 0.7:
                    risk_level = AlertSeverity.CRITICAL
                elif risk_score >= 0.5:
                    risk_level = AlertSeverity.HIGH
                elif risk_score >= 0.3:
                    risk_level = AlertSeverity.MEDIUM
                else:
                    risk_level = AlertSeverity.LOW

                self.risk_assessments[supplier_id] = {
                    'risk_score': round(risk_score, 2),
                    'risk_level': risk_level,
                    'risk_factors': risk_factors,
                    'assessment_date': datetime.now().isoformat()
                }

            logger.info(f"Completed risk assessment for {len(self.risk_assessments)} suppliers")

        except Exception as e:
            logger.error(f"Supplier risk assessment failed: {str(e)}")

    async def _check_procurement_needs(self):
        """Check for procurement needs based on inventory recommendations"""
        try:
            # In production, this would interface with the Inventory Optimization Agent
            # For now, simulate procurement needs

            procurement_needs = []

            # Generate simulated procurement needs
            for category in list(ProductCategory)[:4]:  # Limit for demonstration
                if random.random() < 0.6:  # 60% chance of need
                    need = {
                        'product_category': category.value,
                        'quantity_needed': random.randint(100, 1000),
                        'urgency': random.choice(list(AlertSeverity)),
                        'target_delivery_date': datetime.now() + timedelta(days=random.randint(3, 21)),
                        'budget_limit': random.randint(5000, 50000),
                        'preferred_regions': random.sample(list(Region), k=random.randint(1, 3))
                    }
                    procurement_needs.append(need)

            self.procurement_needs = procurement_needs
            logger.info(f"Identified {len(procurement_needs)} procurement needs")

        except Exception as e:
            logger.error(f"Failed to check procurement needs: {str(e)}")
            self.procurement_needs = []

    async def _generate_procurement_recommendations(self):
        """Generate procurement recommendations based on needs and supplier capabilities"""
        try:
            recommendations = []

            for need in getattr(self, 'procurement_needs', []):
                # Find suitable suppliers
                suitable_suppliers = self._find_suitable_suppliers(need)

                if suitable_suppliers:
                    # Rank suppliers based on performance and cost
                    ranked_suppliers = self._rank_suppliers(suitable_suppliers, need)

                    recommendation = {
                        'recommendation_id': str(uuid.uuid4()),
                        'product_category': need['product_category'],
                        'quantity_needed': need['quantity_needed'],
                        'urgency': need['urgency'],
                        'recommended_suppliers': ranked_suppliers[:3],  # Top 3 suppliers
                        'estimated_cost': need['budget_limit'] * random.uniform(0.7, 0.95),
                        'recommended_delivery_date': need['target_delivery_date'],
                        'reasoning': self._generate_recommendation_reasoning(ranked_suppliers[0], need),
                        'created_at': datetime.now()
                    }

                    recommendations.append(recommendation)

            self.procurement_recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} procurement recommendations")

        except Exception as e:
            logger.error(f"Failed to generate procurement recommendations: {str(e)}")
            self.procurement_recommendations = []

    def _find_suitable_suppliers(self, need: Dict[str, Any]) -> List[str]:
        """Find suppliers that can fulfill the procurement need"""

        suitable_suppliers = []

        for supplier_id, supplier_data in self.suppliers.items():
            # Check if supplier serves the required product category
            if need['product_category'] in supplier_data.get('product_categories', []):
                # Check if supplier serves required regions
                need_regions = [r.value for r in need.get('preferred_regions', [])]
                supplier_regions = supplier_data.get('regions_served', [])

                if any(region in supplier_regions for region in need_regions):
                    suitable_suppliers.append(supplier_id)

        return suitable_suppliers

    def _rank_suppliers(self, supplier_ids: List[str], need: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank suppliers based on performance, cost, and suitability"""

        ranked_suppliers = []

        for supplier_id in supplier_ids:
            supplier_data = self.suppliers[supplier_id]
            metrics = self.performance_metrics.get(supplier_id, {})
            risk_assessment = self.risk_assessments.get(supplier_id, {})

            # Calculate ranking score
            score = 0.0

            # Performance factors (40% weight)
            on_time_rate = metrics.get('on_time_delivery_rate', 0.0)
            quality_score = metrics.get('quality_score', 0.0) / 5.0
            reliability = metrics.get('reliability_score', 0.0)
            performance_score = (on_time_rate + quality_score + reliability) / 3
            score += performance_score * 0.4

            # Cost competitiveness (30% weight)
            cost_competitiveness = metrics.get('cost_competitiveness', 0.0)
            score += cost_competitiveness * 0.3

            # Risk factor (20% weight)
            risk_score = risk_assessment.get('risk_score', 0.5)
            risk_factor = max(0, 1.0 - risk_score)  # Invert risk score
            score += risk_factor * 0.2

            # Lead time factor (10% weight)
            lead_time = metrics.get('average_lead_time_days', 7.0)
            lead_time_score = max(0, 1.0 - (lead_time / 21.0))  # Normalize to 21 days max
            score += lead_time_score * 0.1

            ranked_supplier = {
                'supplier_id': supplier_id,
                'supplier_name': supplier_data['supplier_name'],
                'ranking_score': round(score, 3),
                'performance_metrics': metrics,
                'risk_level': risk_assessment.get('risk_level', AlertSeverity.LOW),
                'estimated_lead_time': metrics.get('average_lead_time_days', 7)
            }

            ranked_suppliers.append(ranked_supplier)

        # Sort by ranking score (descending)
        ranked_suppliers.sort(key=lambda x: x['ranking_score'], reverse=True)

        return ranked_suppliers

    def _generate_recommendation_reasoning(self, top_supplier: Dict[str, Any], need: Dict[str, Any]) -> str:
        """Generate reasoning for supplier recommendation"""

        reasons = []

        # Performance-based reasons
        on_time_rate = top_supplier['performance_metrics'].get('on_time_delivery_rate', 0.0)
        if on_time_rate > 0.95:
            reasons.append(f"Excellent on-time delivery rate ({on_time_rate:.1%})")

        quality_score = top_supplier['performance_metrics'].get('quality_score', 0.0)
        if quality_score > 4.5:
            reasons.append(f"High quality score ({quality_score}/5)")

        # Risk-based reasons
        if top_supplier['risk_level'] == AlertSeverity.LOW:
            reasons.append("Low risk profile")

        # Lead time reasons
        lead_time = top_supplier['estimated_lead_time']
        if lead_time <= 7:
            reasons.append(f"Short lead time ({lead_time} days)")

        # Urgency matching
        if need['urgency'] in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] and lead_time <= 5:
            reasons.append("Can meet urgent delivery requirements")

        if not reasons:
            reasons.append("Best overall ranking among available suppliers")

        return "; ".join(reasons)

    async def _process_purchase_orders(self):
        """Process and update purchase order statuses"""
        try:
            session = db_manager.get_session()

            # Get active purchase orders
            active_orders = session.query(PurchaseOrderRecord).filter(
                PurchaseOrderRecord.status.in_(['pending', 'approved', 'in_transit'])
            ).all()

            for order in active_orders:
                # Simulate order status updates
                if order.status == 'pending':
                    # 30% chance to approve pending orders
                    if random.random() < 0.3:
                        order.status = 'approved'
                        logger.info(f"Purchase order {order.order_id} approved")

                elif order.status == 'approved':
                    # 20% chance to ship approved orders
                    if random.random() < 0.2:
                        order.status = 'in_transit'
                        logger.info(f"Purchase order {order.order_id} shipped")

                elif order.status == 'in_transit':
                    # Check if expected delivery date has passed
                    if datetime.now() >= order.expected_delivery_date:
                        # 80% chance of on-time delivery
                        if random.random() < 0.8:
                            order.status = 'delivered'
                            order.actual_delivery_date = datetime.now()
                            logger.info(f"Purchase order {order.order_id} delivered")
                        # else: order remains in transit (delayed)

            session.commit()
            session.close()

            logger.info(f"Processed {len(active_orders)} purchase orders")

        except Exception as e:
            logger.error(f"Purchase order processing failed: {str(e)}")

    async def _generate_supplier_alerts(self):
        """Generate alerts for supplier-related issues"""
        try:
            alerts = []

            # Performance alerts
            for supplier_id, metrics in self.performance_metrics.items():
                supplier_name = self.suppliers[supplier_id]['supplier_name']

                # On-time delivery alerts
                on_time_rate = metrics.get('on_time_delivery_rate', 0.0)
                if on_time_rate < settings.SUPPLIER_PERFORMANCE_THRESHOLD:
                    alerts.append({
                        'type': 'poor_performance',
                        'severity': 'medium',
                        'supplier_id': supplier_id,
                        'message': f"{supplier_name} on-time delivery rate below threshold ({on_time_rate:.1%})"
                    })

                # Quality alerts
                quality_score = metrics.get('quality_score', 0.0)
                if quality_score < settings.QUALITY_SCORE_THRESHOLD:
                    alerts.append({
                        'type': 'quality_issue',
                        'severity': 'medium',
                        'supplier_id': supplier_id,
                        'message': f"{supplier_name} quality score below threshold ({quality_score}/5)"
                    })

            # Risk alerts
            for supplier_id, risk_data in self.risk_assessments.items():
                if risk_data['risk_level'] in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                    supplier_name = self.suppliers[supplier_id]['supplier_name']
                    alerts.append({
                        'type': 'supplier_risk',
                        'severity': risk_data['risk_level'].value,
                        'supplier_id': supplier_id,
                        'message': f"{supplier_name} identified as {risk_data['risk_level'].value} risk",
                        'risk_factors': risk_data['risk_factors']
                    })

            # Contract expiration alerts
            for supplier_id, supplier_data in self.suppliers.items():
                contract_end = supplier_data['contract_end_date']
                if isinstance(contract_end, str):
                    contract_end = datetime.fromisoformat(contract_end)

                days_to_expiry = (contract_end - datetime.now()).days
                if days_to_expiry <= 30:
                    alerts.append({
                        'type': 'contract_expiration',
                        'severity': 'high' if days_to_expiry <= 7 else 'medium',
                        'supplier_id': supplier_id,
                        'message': f"{supplier_data['supplier_name']} contract expires in {days_to_expiry} days"
                    })

            # Store alerts in database
            if alerts:
                session = db_manager.get_session()

                for alert in alerts:
                    # Convert list descriptions to strings
                    description = alert.get('risk_factors', '')
                    if isinstance(description, list):
                        description = '\n'.join(description)

                    DatabaseOperations.save_alert(
                        alert_id=str(uuid.uuid4()),
                        alert_type=alert['type'],
                        severity=alert['severity'],
                        title=alert['message'],
                        description=description,
                        source="supplier_coordination",
                        metadata=alert,
                        session=session
                    )

                session.close()
                logger.info(f"Generated {len(alerts)} supplier alerts")

        except Exception as e:
            logger.error(f"Supplier alert generation failed: {str(e)}")

    # Public API methods
    async def get_supplier_performance(self) -> List[Supplier]:
        """Get supplier performance data"""
        try:
            suppliers = []

            for supplier_id, supplier_data in self.suppliers.items():
                metrics_data = self.performance_metrics.get(supplier_id, {})
                risk_data = self.risk_assessments.get(supplier_id, {})

                # Create supplier metrics object
                metrics = SupplierMetrics(
                    on_time_delivery_rate=metrics_data.get('on_time_delivery_rate', 0.0),
                    quality_score=metrics_data.get('quality_score', 0.0),
                    average_lead_time_days=metrics_data.get('average_lead_time_days', 0.0),
                    cost_competitiveness=metrics_data.get('cost_competitiveness', 0.0),
                    reliability_score=metrics_data.get('reliability_score', 0.0),
                    communication_score=metrics_data.get('communication_score', 0.0)
                )

                # Create supplier object
                supplier = Supplier(
                    supplier_id=supplier_id,
                    supplier_name=supplier_data['supplier_name'],
                    contact_info=supplier_data['contact_info'],
                    product_categories=[ProductCategory(cat) for cat in supplier_data.get('product_categories', [])],
                    regions_served=[Region(reg) for reg in supplier_data.get('regions_served', [])],
                    contract_start_date=supplier_data['contract_start_date'],
                    contract_end_date=supplier_data['contract_end_date'],
                    metrics=metrics,
                    risk_level=risk_data.get('risk_level', AlertSeverity.LOW)
                )

                suppliers.append(supplier)

            return suppliers

        except Exception as e:
            logger.error(f"Failed to get supplier performance: {str(e)}")
            return []

    async def get_purchase_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get purchase orders with optional status filter"""
        try:
            session = db_manager.get_session()

            query = session.query(PurchaseOrderRecord)
            if status:
                query = query.filter_by(status=status)

            orders = query.order_by(PurchaseOrderRecord.created_at.desc()).limit(50).all()

            purchase_orders = []
            for order in orders:
                # Get supplier name
                supplier = session.query(SupplierRecord).filter_by(id=order.supplier_id).first()
                supplier_name = supplier.supplier_name if supplier else "Unknown Supplier"

                order_data = {
                    'order_id': order.order_id,
                    'supplier_id': order.supplier_id,
                    'supplier_name': supplier_name,
                    'order_date': order.order_date.isoformat(),
                    'expected_delivery_date': order.expected_delivery_date.isoformat(),
                    'actual_delivery_date': order.actual_delivery_date.isoformat() if order.actual_delivery_date else None,
                    'status': order.status,
                    'items': order.items,
                    'total_amount': order.total_amount,
                    'notes': order.notes
                }
                purchase_orders.append(order_data)

            session.close()
            return purchase_orders

        except Exception as e:
            logger.error(f"Failed to get purchase orders: {str(e)}")
            return []

    async def create_purchase_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new purchase order"""
        try:
            session = db_manager.get_session()

            # Generate order ID
            order_id = f"PO{datetime.now().strftime('%Y%m%d')}{random.randint(1000, 9999)}"

            # Create purchase order record
            purchase_order = PurchaseOrderRecord(
                order_id=order_id,
                supplier_id=order_data['supplier_id'],
                order_date=datetime.now(),
                expected_delivery_date=datetime.fromisoformat(order_data['expected_delivery_date']),
                status=OrderStatus.PENDING.value,
                items=order_data['items'],
                total_amount=order_data['total_amount'],
                notes=order_data.get('notes', '')
            )

            session.add(purchase_order)
            session.commit()

            result = {
                'status': 'success',
                'order_id': order_id,
                'message': f"Purchase order {order_id} created successfully",
                'order_details': {
                    'order_id': order_id,
                    'supplier_id': order_data['supplier_id'],
                    'total_amount': order_data['total_amount'],
                    'expected_delivery_date': order_data['expected_delivery_date'],
                    'status': OrderStatus.PENDING.value
                },
                'timestamp': datetime.now().isoformat()
            }

            session.close()
            logger.info(f"Created purchase order {order_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to create purchase order: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
