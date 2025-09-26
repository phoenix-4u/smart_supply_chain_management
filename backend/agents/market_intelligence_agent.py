#File: backend/agents/market_intelligence_agent.py

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import requests
from textblob import TextBlob
import uuid

from models.data_models import (
    MarketIntelligenceResponse, SocialMediaMetrics, CompetitorData, 
    WeatherAlert, AlertSeverity, Region
)
from models.database import DatabaseOperations, db_manager
from config.settings import settings

logger = logging.getLogger(__name__)

class MarketIntelligenceAgent:
    """
    Market Intelligence Agent for Health First Supply Chain

    Monitors external market signals including:
    - Social media monitoring
    - Competitor tracking
    - Economic indicators
    - Weather and natural disasters
    - Healthcare utilization trends
    - Pricing intelligence
    """

    def __init__(self):
        self.agent_name = "Market Intelligence Agent"
        self.last_update = None
        self.active_alerts = []
        self.data_cache = {}
        self.competitors = settings.HEALTH_FIRST_COMPETITORS

    async def update_intelligence(self):
        """Main update cycle for market intelligence"""
        try:
            logger.info("Starting market intelligence update cycle")

            # Gather data from all sources
            social_media_data = await self._monitor_social_media()
            competitor_data = await self._track_competitors()
            economic_data = await self._monitor_economic_indicators()
            weather_data = await self._monitor_weather_disasters()
            healthcare_data = await self._monitor_healthcare_utilization()
            pricing_data = await self._monitor_pricing_intelligence()

            # Process triggers and alerts
            alerts = await self._process_market_triggers(
                social_media_data, competitor_data, economic_data, 
                weather_data, healthcare_data, pricing_data
            )

            # Store data in database
            await self._store_intelligence_data({
                "social_media": social_media_data,
                "competitors": competitor_data,
                "economic": economic_data,
                "weather": weather_data,
                "healthcare": healthcare_data,
                "pricing": pricing_data,
                "alerts": alerts
            })

            self.last_update = datetime.now()
            logger.info("Market intelligence update completed successfully")

        except Exception as e:
            logger.error(f"Market intelligence update failed: {str(e)}")
            raise

    async def _monitor_social_media(self) -> Dict[str, Any]:
        """Monitor social media for healthcare-related mentions"""
        try:
            # Simulate social media monitoring (replace with real API calls)
            # In production, integrate with Twitter API, LinkedIn API, etc.

            # Base metrics with some randomization for simulation
            base_mentions = 6000
            mention_variation = random.randint(-500, 1000)
            mentions_7d = base_mentions + mention_variation

            # Calculate percentage change
            previous_mentions = self.data_cache.get('previous_mentions', 5500)
            mentions_change_pct = ((mentions_7d - previous_mentions) / previous_mentions) * 100
            self.data_cache['previous_mentions'] = mentions_7d

            # Sentiment analysis (simulated)
            sentiment_positive = round(random.uniform(0.6, 0.8), 2)
            sentiment_negative = round(random.uniform(0.1, 0.2), 2)
            sentiment_neutral = round(1.0 - sentiment_positive - sentiment_negative, 2)

            # Share of voice
            share_of_voice = {
                "Health First": round(random.uniform(0.35, 0.45), 2),
                "Cardinal Health": round(random.uniform(0.20, 0.30), 2),
                "Cencora": round(random.uniform(0.15, 0.25), 2),
                "Others": 0.0
            }
            share_of_voice["Others"] = round(1.0 - sum([v for k, v in share_of_voice.items() if k != "Others"]), 2)

            # Trending topics
            trending_topics = [
                "#PPEshortage", "#SeasonalAllergies", "#Telehealth", 
                "#FluSeason2024", "#DigitalHealth", "#SupplyChain"
            ]
            random.shuffle(trending_topics)
            trending_topics = trending_topics[:4]

            # Geographical hotspots
            geographical_hotspots = {
                "Midwest": random.randint(80, 150),
                "Southeast": random.randint(60, 120),
                "Northeast": random.randint(40, 100),
                "West Coast": random.randint(50, 110),
                "Southwest": random.randint(30, 90)
            }

            # Influencer mentions
            influencer_mentions = [
                {
                    "influencer": "Dr. Sarah Smith",
                    "followers": 120000,
                    "product_mentioned": "Allergy Medication",
                    "engagement": "High",
                    "sentiment": "Positive"
                },
                {
                    "influencer": "RN Patel",
                    "followers": 50000,
                    "product_mentioned": "Home Care Kits",
                    "engagement": "Medium",
                    "sentiment": "Neutral"
                }
            ]

            social_media_data = {
                "mentions_7d": mentions_7d,
                "mentions_change_pct": mentions_change_pct,
                "sentiment_positive": sentiment_positive,
                "sentiment_negative": sentiment_negative,
                "sentiment_neutral": sentiment_neutral,
                "share_of_voice": share_of_voice,
                "trending_topics": trending_topics,
                "geographical_hotspots": geographical_hotspots,
                "influencer_mentions": influencer_mentions,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Social media monitoring completed: {mentions_7d} mentions")
            return social_media_data

        except Exception as e:
            logger.error(f"Social media monitoring failed: {str(e)}")
            return {}

    async def _track_competitors(self) -> List[Dict[str, Any]]:
        """Track competitor activities and market positioning"""
        try:
            competitor_data = []

            for competitor in self.competitors:
                # Simulate competitor data (replace with real competitor intelligence)
                data = {
                    "competitor_name": competitor,
                    "market_share": round(random.uniform(0.05, 0.30), 3),
                    "price_changes": self._generate_price_changes(),
                    "product_launches": self._generate_product_launches(),
                    "supply_chain_status": random.choice(["normal", "disrupted", "optimized"]),
                    "recent_news": self._generate_competitor_news(competitor),
                    "last_updated": datetime.now().isoformat()
                }
                competitor_data.append(data)

            logger.info(f"Competitor tracking completed for {len(competitor_data)} competitors")
            return competitor_data

        except Exception as e:
            logger.error(f"Competitor tracking failed: {str(e)}")
            return []

    async def _monitor_economic_indicators(self) -> Dict[str, Any]:
        """Monitor macroeconomic and industry indicators"""
        try:
            # Simulate economic data (replace with real economic APIs)
            economic_data = {
                "unemployment_rate": round(random.uniform(3.0, 5.0), 1),
                "consumer_confidence_index": round(random.uniform(90, 110), 1),
                "healthcare_spending_growth": round(random.uniform(2.0, 6.0), 1),
                "inflation_rate": round(random.uniform(2.0, 4.0), 1),
                "gdp_growth": round(random.uniform(1.5, 3.5), 1),
                "healthcare_cpi": round(random.uniform(2.5, 5.0), 1),
                "prescription_drug_utilization": round(random.uniform(-2.0, 8.0), 1),
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Economic indicators monitoring completed")
            return economic_data

        except Exception as e:
            logger.error(f"Economic monitoring failed: {str(e)}")
            return {}

    async def _monitor_weather_disasters(self) -> List[Dict[str, Any]]:
        """Monitor weather events and natural disasters"""
        try:
            # Simulate weather/disaster monitoring (replace with NOAA API, etc.)
            weather_alerts = []

            # Randomly generate weather events
            if random.random() < 0.3:  # 30% chance of weather event
                alert = {
                    "alert_id": str(uuid.uuid4()),
                    "alert_type": random.choice(["hurricane", "flood", "wildfire", "blizzard"]),
                    "severity": random.choice(["low", "medium", "high", "critical"]),
                    "affected_regions": random.sample(list(Region), k=random.randint(1, 3)),
                    "population_affected": random.randint(50000, 2000000),
                    "healthcare_facilities_affected": random.randint(5, 50),
                    "expected_duration_hours": random.randint(12, 168),
                    "demand_impact_forecast": {
                        "ppe": random.uniform(1.5, 3.0),
                        "medical_supplies": random.uniform(1.2, 2.5),
                        "pharmaceuticals": random.uniform(1.1, 1.8)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                weather_alerts.append(alert)

            logger.info(f"Weather monitoring completed: {len(weather_alerts)} active alerts")
            return weather_alerts

        except Exception as e:
            logger.error(f"Weather monitoring failed: {str(e)}")
            return []

    async def _monitor_healthcare_utilization(self) -> Dict[str, Any]:
        """Monitor healthcare utilization and public health trends"""
        try:
            # Simulate healthcare utilization data (replace with CDC/CMS APIs)
            healthcare_data = {
                "hospital_occupancy_rate": round(random.uniform(0.70, 0.95), 3),
                "icu_utilization_rate": round(random.uniform(0.60, 0.90), 3),
                "emergency_room_visits": random.randint(800, 1200),
                "elective_procedures_volume": random.randint(200, 400),
                "flu_incidence_rate": round(random.uniform(5.0, 25.0), 1),
                "covid_cases_weekly": random.randint(1000, 5000),
                "vaccination_rate": round(random.uniform(0.60, 0.85), 3),
                "prescription_fill_rate": round(random.uniform(0.85, 0.95), 3),
                "supply_burn_rate": {
                    "ppe_per_day": random.randint(500, 1500),
                    "iv_fluids_per_day": random.randint(100, 300),
                    "ventilators_in_use": random.randint(50, 150)
                },
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Healthcare utilization monitoring completed")
            return healthcare_data

        except Exception as e:
            logger.error(f"Healthcare utilization monitoring failed: {str(e)}")
            return {}

    async def _monitor_pricing_intelligence(self) -> Dict[str, Any]:
        """Monitor pricing trends and competitive pricing"""
        try:
            # Simulate pricing intelligence (replace with pricing databases)
            pricing_data = {
                "average_price_changes": {
                    "ppe": round(random.uniform(-0.05, 0.10), 3),
                    "pharmaceuticals": round(random.uniform(-0.02, 0.08), 3),
                    "medical_supplies": round(random.uniform(-0.03, 0.06), 3)
                },
                "competitor_pricing_differential": {
                    "Cardinal Health": round(random.uniform(-0.10, 0.10), 3),
                    "Cencora": round(random.uniform(-0.08, 0.12), 3),
                    "Henry Schein": round(random.uniform(-0.06, 0.08), 3)
                },
                "market_price_volatility": round(random.uniform(0.02, 0.15), 3),
                "reimbursement_rate_changes": round(random.uniform(-0.03, 0.05), 3),
                "discount_activity": random.choice(["low", "medium", "high"]),
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Pricing intelligence monitoring completed")
            return pricing_data

        except Exception as e:
            logger.error(f"Pricing intelligence monitoring failed: {str(e)}")
            return {}

    async def _process_market_triggers(self, *data_sources) -> List[Dict[str, Any]]:
        """Process market data and generate actionable triggers"""
        triggers = []

        try:
            social_media_data, competitor_data, economic_data, weather_data, healthcare_data, pricing_data = data_sources

            # Social Media Triggers
            if social_media_data.get("mentions_change_pct", 0) > settings.SOCIAL_MEDIA_SPIKE_THRESHOLD * 100:
                triggers.append({
                    "trigger_id": str(uuid.uuid4()),
                    "type": "social_media_spike",
                    "severity": "high",
                    "description": f"Social media mentions increased by {social_media_data['mentions_change_pct']:.1f}%",
                    "affected_products": ["ppe", "medical_supplies"],
                    "recommended_action": "Increase demand forecast for trending products",
                    "timestamp": datetime.now().isoformat()
                })

            # Competitor Triggers
            for competitor in competitor_data:
                if competitor.get("supply_chain_status") == "disrupted":
                    triggers.append({
                        "trigger_id": str(uuid.uuid4()),
                        "type": "competitor_disruption",
                        "severity": "medium",
                        "description": f"{competitor['competitor_name']} experiencing supply chain disruption",
                        "opportunity": "Potential market share gain",
                        "recommended_action": "Increase inventory and prepare for demand surge",
                        "timestamp": datetime.now().isoformat()
                    })

            # Weather Triggers
            for alert in weather_data:
                if alert.get("severity") in ["high", "critical"]:
                    triggers.append({
                        "trigger_id": str(uuid.uuid4()),
                        "type": "weather_disaster",
                        "severity": alert["severity"],
                        "description": f"{alert['alert_type'].title()} affecting {alert['population_affected']} people",
                        "affected_regions": alert["affected_regions"],
                        "demand_impact": alert["demand_impact_forecast"],
                        "recommended_action": "Implement emergency supply protocols",
                        "timestamp": datetime.now().isoformat()
                    })

            # Healthcare Utilization Triggers
            if healthcare_data.get("hospital_occupancy_rate", 0) > 0.90:
                triggers.append({
                    "trigger_id": str(uuid.uuid4()),
                    "type": "hospital_capacity_strain",
                    "severity": "high",
                    "description": f"Hospital occupancy at {healthcare_data['hospital_occupancy_rate']:.1%}",
                    "affected_products": ["medical_supplies", "pharmaceuticals"],
                    "recommended_action": "Increase emergency supply allocation",
                    "timestamp": datetime.now().isoformat()
                })

            # Economic Triggers
            if economic_data.get("unemployment_rate", 0) > 4.5:
                triggers.append({
                    "trigger_id": str(uuid.uuid4()),
                    "type": "economic_downturn",
                    "severity": "medium",
                    "description": f"Unemployment rate at {economic_data['unemployment_rate']}%",
                    "impact": "Potential reduction in elective procedures",
                    "recommended_action": "Adjust demand forecast for non-essential products",
                    "timestamp": datetime.now().isoformat()
                })

            logger.info(f"Processed {len(triggers)} market triggers")
            self.active_alerts = triggers
            return triggers

        except Exception as e:
            logger.error(f"Market trigger processing failed: {str(e)}")
            return []

    def _generate_price_changes(self) -> List[Dict[str, Any]]:
        """Generate simulated price changes"""
        price_changes = []

        if random.random() < 0.4:  # 40% chance of price change
            price_changes.append({
                "product": random.choice(["N95 masks", "Surgical gloves", "IV fluids", "Antibiotics"]),
                "change_percent": round(random.uniform(-10.0, 15.0), 1),
                "effective_date": datetime.now().isoformat(),
                "reason": random.choice(["Market adjustment", "Supply shortage", "New contract", "Competitive response"])
            })

        return price_changes

    def _generate_product_launches(self) -> List[Dict[str, Any]]:
        """Generate simulated product launches"""
        product_launches = []

        if random.random() < 0.2:  # 20% chance of new product
            product_launches.append({
                "product_name": f"New Digital Health Solution {random.randint(100, 999)}",
                "category": random.choice(["digital_health", "diagnostic_equipment", "home_healthcare"]),
                "launch_date": datetime.now().isoformat(),
                "target_market": random.choice(["Hospitals", "Clinics", "Home Care", "Long-term Care"])
            })

        return product_launches

    def _generate_competitor_news(self, competitor: str) -> List[str]:
        """Generate simulated competitor news"""
        news_templates = [
            f"{competitor} announces expansion into new geographic markets",
            f"{competitor} reports strong quarterly earnings growth",
            f"{competitor} launches new digital supply chain platform",
            f"{competitor} forms strategic partnership with major health system",
            f"{competitor} invests in AI-powered inventory management"
        ]

        # Return 0-2 news items
        num_items = random.randint(0, 2)
        return random.sample(news_templates, min(num_items, len(news_templates)))

    async def _store_intelligence_data(self, data: Dict[str, Any]):
        """Store market intelligence data in database"""
        try:
            session = db_manager.get_session()

            # Store each data source
            for source_type, source_data in data.items():
                if source_data:  # Only store non-empty data
                    DatabaseOperations.save_market_intelligence(
                        source_type=source_type,
                        data=source_data,
                        session=session
                    )

            # Store alerts
            for alert in data.get("alerts", []):
                DatabaseOperations.save_alert(
                    alert_id=alert["trigger_id"],
                    alert_type=alert["type"],
                    severity=alert["severity"],
                    title=alert["description"],
                    description=alert.get("recommended_action", ""),
                    source="market_intelligence",
                    metadata=alert,
                    session=session
                )

            session.close()
            logger.info("Market intelligence data stored successfully")

        except Exception as e:
            logger.error(f"Failed to store market intelligence data: {str(e)}")

    # Public API methods
    async def get_social_media_data(self) -> Dict[str, Any]:
        """Get current social media monitoring data"""
        return await self._monitor_social_media()

    async def get_competitor_data(self) -> List[Dict[str, Any]]:
        """Get current competitor tracking data"""
        return await self._track_competitors()

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active market intelligence alerts"""
        return self.active_alerts

    async def get_summary(self) -> MarketIntelligenceResponse:
        """Get comprehensive market intelligence summary"""
        try:
            # Get fresh data
            social_media_data = await self._monitor_social_media()
            competitor_data = await self._track_competitors()
            economic_data = await self._monitor_economic_indicators()
            weather_data = await self._monitor_weather_disasters()
            healthcare_data = await self._monitor_healthcare_utilization()
            pricing_data = await self._monitor_pricing_intelligence()

            # Create response model
            social_metrics = SocialMediaMetrics(
                mentions_7d=social_media_data.get("mentions_7d", 0),
                mentions_change_pct=social_media_data.get("mentions_change_pct", 0),
                sentiment_positive=social_media_data.get("sentiment_positive", 0),
                sentiment_negative=social_media_data.get("sentiment_negative", 0),
                sentiment_neutral=social_media_data.get("sentiment_neutral", 0),
                share_of_voice=social_media_data.get("share_of_voice", {}),
                trending_topics=social_media_data.get("trending_topics", []),
                geographical_hotspots=social_media_data.get("geographical_hotspots", {}),
                influencer_mentions=social_media_data.get("influencer_mentions", [])
            )

            competitors = [
                CompetitorData(
                    competitor_name=comp["competitor_name"],
                    price_changes=comp.get("price_changes", []),
                    product_launches=comp.get("product_launches", []),
                    market_share=comp.get("market_share", 0),
                    recent_news=comp.get("recent_news", []),
                    supply_chain_status=comp.get("supply_chain_status", "normal")
                )
                for comp in competitor_data
            ]

            weather_alerts = [
                WeatherAlert(
                    alert_id=alert["alert_id"],
                    alert_type=alert["alert_type"],
                    severity=AlertSeverity(alert["severity"]),
                    affected_regions=[Region(r) for r in alert["affected_regions"]],
                    population_affected=alert["population_affected"],
                    healthcare_facilities_affected=alert["healthcare_facilities_affected"],
                    expected_duration_hours=alert["expected_duration_hours"],
                    demand_impact_forecast=alert["demand_impact_forecast"]
                )
                for alert in weather_data
            ]

            economic_data.pop("timestamp", None)

            return MarketIntelligenceResponse(
                social_media=social_metrics,
                competitors=competitors,
                weather_alerts=weather_alerts,
                economic_indicators=economic_data,
                healthcare_utilization=healthcare_data,
                pricing_intelligence=pricing_data,
                active_triggers=self.active_alerts
            )

        except Exception as e:
            logger.error(f"Failed to get market intelligence summary: {str(e)}")
            raise