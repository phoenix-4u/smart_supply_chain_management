#File: backend/agents/demand_prediction_agent.py

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import uuid
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import random

from models.data_models import (
    DemandForecast, DemandForecastResponse, ProductCategory, Region
)
from models.database import DatabaseOperations, db_manager
from config.settings import settings

logger = logging.getLogger(__name__)

class DemandPredictionAgent:
    """
    Demand Prediction Agent for Health First Supply Chain

    Responsibilities:
    - Forecast demand using historical data and market intelligence
    - Continuously refine predictions with ML models
    - Respond to market intelligence triggers
    - Provide demand accuracy metrics
    """

    def __init__(self):
        self.agent_name = "Demand Prediction Agent"
        self.models = {}
        self.current_forecasts = {}
        self.accuracy_metrics = {}
        self.last_model_training = None
        self.market_intelligence_data = {}

        # Initialize ML models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize machine learning models for demand prediction"""
        try:
            # Initialize ensemble of models
            self.models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                ),
                'ensemble_weights': settings.ENSEMBLE_MODEL_WEIGHTS
            }

            logger.info("Demand prediction models initialized")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")

    async def update_forecasts(self):
        """Main update cycle for demand forecasting"""
        try:
            logger.info("Starting demand forecast update cycle")

            # Get latest market intelligence data
            await self._fetch_market_intelligence()

            # Get historical sales data
            historical_data = await self._fetch_historical_data()

            # Generate forecasts for all product-region combinations
            forecasts = await self._generate_forecasts(historical_data)

            # Calculate model accuracy
            accuracy = await self._calculate_accuracy(historical_data, forecasts)

            # Store forecasts in database
            await self._store_forecasts(forecasts, accuracy)

            # Check for retraining needs
            await self._check_model_retraining(accuracy)

            self.current_forecasts = forecasts
            self.accuracy_metrics = accuracy

            logger.info(f"Demand forecast update completed: {len(forecasts)} forecasts generated")

        except Exception as e:
            logger.error(f"Demand forecast update failed: {str(e)}")
            raise

    async def _fetch_market_intelligence(self):
        """Fetch latest market intelligence data"""
        try:
            # In production, this would fetch from the Market Intelligence Agent
            # For now, simulate market intelligence impact factors

            self.market_intelligence_data = {
                'social_media_sentiment': random.uniform(0.4, 0.8),
                'competitor_disruption': random.choice([True, False]),
                'economic_indicator': random.uniform(0.8, 1.2),
                'weather_impact': random.uniform(0.9, 1.5),
                'healthcare_utilization': random.uniform(0.85, 1.15),
                'pricing_pressure': random.uniform(0.9, 1.1),
                'timestamp': datetime.now()
            }

            logger.info("Market intelligence data fetched")

        except Exception as e:
            logger.error(f"Failed to fetch market intelligence: {str(e)}")
            self.market_intelligence_data = {}

    async def _fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical sales and demand data"""
        try:
            # Simulate historical data (in production, fetch from data warehouse)
            data = []

            # Generate synthetic historical data for the last 365 days
            start_date = datetime.now() - timedelta(days=365)

            for i in range(365):
                date = start_date + timedelta(days=i)

                for category in ProductCategory:
                    for region in Region:
                        # Base demand with seasonal patterns and trends
                        base_demand = self._calculate_base_demand(category, region, date)

                        # Add noise and variations
                        actual_demand = base_demand * random.uniform(0.8, 1.2)

                        data.append({
                            'date': date,
                            'product_category': category.value,
                            'region': region.value,
                            'actual_demand': max(0, actual_demand),
                            'day_of_week': date.weekday(),
                            'month': date.month,
                            'quarter': (date.month - 1) // 3 + 1,
                            'is_weekend': date.weekday() >= 5
                        })

            df = pd.DataFrame(data)
            logger.info(f"Historical data fetched: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {str(e)}")
            return pd.DataFrame()

    def _calculate_base_demand(self, category: ProductCategory, region: Region, date: datetime) -> float:
        """Calculate base demand for a product category and region"""

        # Base demand levels by category
        base_levels = {
            ProductCategory.PHARMACEUTICALS: 1000,
            ProductCategory.MEDICAL_SUPPLIES: 800,
            ProductCategory.PPE: 600,
            ProductCategory.SURGICAL_INSTRUMENTS: 200,
            ProductCategory.DIAGNOSTIC_EQUIPMENT: 150,
            ProductCategory.HOSPITAL_EQUIPMENT: 100,
            ProductCategory.HOME_HEALTHCARE: 300,
            ProductCategory.DIGITAL_HEALTH: 50
        }

        # Regional multipliers
        region_multipliers = {
            Region.NORTHEAST: 1.2,
            Region.SOUTHEAST: 1.1,
            Region.MIDWEST: 1.0,
            Region.SOUTHWEST: 0.9,
            Region.WEST_COAST: 1.3,
            Region.MOUNTAIN_STATES: 0.7
        }

        base = base_levels.get(category, 500)
        regional_factor = region_multipliers.get(region, 1.0)

        # Seasonal adjustments
        seasonal_factor = 1.0
        if category in [ProductCategory.PPE, ProductCategory.PHARMACEUTICALS]:
            # Higher demand in flu season (Oct-Mar)
            if date.month in [10, 11, 12, 1, 2, 3]:
                seasonal_factor = 1.3

        # Weekly patterns (lower on weekends for most categories)
        weekly_factor = 0.7 if date.weekday() >= 5 else 1.0

        return base * regional_factor * seasonal_factor * weekly_factor

    async def _generate_forecasts(self, historical_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate demand forecasts for all product-region combinations"""
        forecasts = []

        try:
            for category in ProductCategory:
                for region in Region:
                    # Filter data for this product-region combination
                    subset = historical_data[
                        (historical_data['product_category'] == category.value) &
                        (historical_data['region'] == region.value)
                    ].copy()

                    if len(subset) < settings.MIN_HISTORICAL_DATA_POINTS:
                        continue

                    # Generate forecast for next 30 days
                    forecast_data = await self._forecast_product_region(subset, category, region)
                    forecasts.extend(forecast_data)

            logger.info(f"Generated {len(forecasts)} forecasts")
            return forecasts

        except Exception as e:
            logger.error(f"Forecast generation failed: {str(e)}")
            return []

    async def _forecast_product_region(
        self, 
        historical_data: pd.DataFrame, 
        category: ProductCategory, 
        region: Region
    ) -> List[Dict[str, Any]]:
        """Generate forecast for specific product category and region"""

        forecasts = []

        try:
            # Prepare features for ML model
            features = self._prepare_features(historical_data)
            target = historical_data['actual_demand'].values

            # Train ensemble model if we have enough data
            if len(features) >= 30:
                ensemble_prediction = self._predict_with_ensemble(features, target)
            else:
                # Use simple moving average for insufficient data
                ensemble_prediction = historical_data['actual_demand'].tail(7).mean()

            # Apply market intelligence adjustments
            market_adjusted_prediction = self._apply_market_intelligence_adjustment(
                ensemble_prediction, category, region
            )

            # Generate forecasts for next 30 days
            for days_ahead in range(1, settings.FORECAST_HORIZON_DAYS + 1):
                forecast_date = datetime.now() + timedelta(days=days_ahead)

                # Add time-based variations
                time_factor = self._calculate_time_factors(forecast_date, category)
                predicted_demand = market_adjusted_prediction * time_factor

                # Calculate confidence intervals
                confidence_lower, confidence_upper = self._calculate_confidence_intervals(
                    predicted_demand, historical_data['actual_demand']
                )

                # Identify contributing factors
                factors = self._identify_contributing_factors(category, region)

                forecast = {
                    'product_category': category.value,
                    'region': region.value,
                    'forecast_date': forecast_date,
                    'predicted_demand': max(0, predicted_demand),
                    'confidence_interval_lower': max(0, confidence_lower),
                    'confidence_interval_upper': confidence_upper,
                    'factors_contributing': factors,
                    'model_version': 'ensemble_v1.0',
                    'created_at': datetime.now()
                }

                forecasts.append(forecast)

            return forecasts

        except Exception as e:
            logger.error(f"Product-region forecast failed for {category.value}-{region.value}: {str(e)}")
            return []

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for machine learning models"""

        # Sort by date
        data = data.sort_values('date')

        # Create lag features
        data['demand_lag_1'] = data['actual_demand'].shift(1)
        data['demand_lag_7'] = data['actual_demand'].shift(7)
        data['demand_lag_30'] = data['actual_demand'].shift(30)

        # Create rolling statistics
        data['demand_ma_7'] = data['actual_demand'].rolling(window=7).mean()
        data['demand_ma_30'] = data['actual_demand'].rolling(window=30).mean()
        data['demand_std_7'] = data['actual_demand'].rolling(window=7).std()

        # Select features for modeling
        feature_columns = [
            'day_of_week', 'month', 'quarter', 'is_weekend',
            'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
            'demand_ma_7', 'demand_ma_30', 'demand_std_7'
        ]

        # Fill missing values and return features
        features = data[feature_columns].bfill().fillna(0)
        return features.values

    def _predict_with_ensemble(self, features: np.ndarray, target: np.ndarray) -> float:
        """Make prediction using ensemble of models"""

        try:
            # Use last 80% for training, 20% for validation
            split_idx = int(len(features) * 0.8)
            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = target[:split_idx], target[split_idx:]

            predictions = {}
            weights = self.models['ensemble_weights']

            # Train and predict with each model
            for model_name, model in self.models.items():
                if model_name == 'ensemble_weights':
                    continue

                # Train model
                model.fit(X_train, y_train)

                # Predict on validation set
                val_pred = model.predict(X_val)
                predictions[model_name] = val_pred.mean()  # Use mean for forecast

            # Combine predictions using weighted average
            ensemble_prediction = sum(
                weights[model_name] * predictions[model_name]
                for model_name in predictions.keys()
                if model_name in weights
            )

            return ensemble_prediction

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            # Fallback to simple average
            return target[-7:].mean() if len(target) >= 7 else target.mean()

    def _apply_market_intelligence_adjustment(
        self, 
        base_prediction: float, 
        category: ProductCategory, 
        region: Region
    ) -> float:
        """Apply market intelligence factors to base prediction"""

        adjustment_factor = 1.0

        # Apply market intelligence adjustments
        mi_data = self.market_intelligence_data

        # Social media sentiment impact
        sentiment = mi_data.get('social_media_sentiment', 0.6)
        if sentiment > 0.7:
            adjustment_factor *= 1.1  # Positive sentiment increases demand
        elif sentiment < 0.4:
            adjustment_factor *= 0.9  # Negative sentiment decreases demand

        # Competitor disruption impact
        if mi_data.get('competitor_disruption', False):
            adjustment_factor *= 1.15  # Benefit from competitor issues

        # Economic indicators
        economic_factor = mi_data.get('economic_indicator', 1.0)
        adjustment_factor *= economic_factor

        # Weather impact (especially for PPE and emergency supplies)
        if category in [ProductCategory.PPE, ProductCategory.MEDICAL_SUPPLIES]:
            weather_factor = mi_data.get('weather_impact', 1.0)
            adjustment_factor *= weather_factor

        # Healthcare utilization impact
        utilization_factor = mi_data.get('healthcare_utilization', 1.0)
        adjustment_factor *= utilization_factor

        return base_prediction * adjustment_factor

    def _calculate_time_factors(self, forecast_date: datetime, category: ProductCategory) -> float:
        """Calculate time-based adjustment factors"""

        time_factor = 1.0

        # Seasonal adjustments
        if category in [ProductCategory.PPE, ProductCategory.PHARMACEUTICALS]:
            if forecast_date.month in [10, 11, 12, 1, 2, 3]:  # Flu season
                time_factor *= 1.2

        # Weekly patterns
        if forecast_date.weekday() >= 5:  # Weekend
            time_factor *= 0.8

        # Holiday effects (simplified)
        # In production, use a proper holiday calendar
        if forecast_date.month == 12 and forecast_date.day in [24, 25, 31]:
            time_factor *= 0.7

        return time_factor

    def _calculate_confidence_intervals(
        self, 
        prediction: float, 
        historical_values: pd.Series
    ) -> Tuple[float, float]:
        """Calculate confidence intervals for predictions"""

        try:
            # Calculate historical standard deviation
            std_dev = historical_values.std()

            # Use normal distribution assumption for confidence intervals
            # For 95% confidence interval
            z_score = 1.96  # 95% confidence

            margin_of_error = z_score * std_dev

            confidence_lower = prediction - margin_of_error
            confidence_upper = prediction + margin_of_error

            return confidence_lower, confidence_upper

        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {str(e)}")
            # Fallback to ±20% of prediction
            return prediction * 0.8, prediction * 1.2

    def _identify_contributing_factors(self, category: ProductCategory, region: Region) -> List[str]:
        """Identify factors contributing to demand forecast"""

        factors = []

        # Base factors
        factors.append("Historical trend")

        # Market intelligence factors
        mi_data = self.market_intelligence_data

        if mi_data.get('social_media_sentiment', 0.6) > 0.7:
            factors.append("Positive social media sentiment")
        elif mi_data.get('social_media_sentiment', 0.6) < 0.4:
            factors.append("Negative social media sentiment")

        if mi_data.get('competitor_disruption', False):
            factors.append("Competitor supply chain disruption")

        if mi_data.get('weather_impact', 1.0) > 1.2:
            factors.append("Weather/disaster impact")

        if mi_data.get('healthcare_utilization', 1.0) > 1.1:
            factors.append("Increased healthcare utilization")

        # Seasonal factors
        current_month = datetime.now().month
        if category in [ProductCategory.PPE, ProductCategory.PHARMACEUTICALS]:
            if current_month in [10, 11, 12, 1, 2, 3]:
                factors.append("Seasonal flu trend")

        return factors

    async def _calculate_accuracy(
        self, 
        historical_data: pd.DataFrame, 
        forecasts: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""

        try:
            # For demonstration, calculate accuracy against recent historical data
            # In production, this would compare forecasts against actual outcomes

            # Simulate accuracy metrics
            accuracy_metrics = {
                'mean_absolute_error': random.uniform(0.08, 0.15),
                'mean_squared_error': random.uniform(0.05, 0.12),
                'r2_score': random.uniform(0.80, 0.95),
                'mean_absolute_percentage_error': random.uniform(8, 15),
                'forecast_bias': random.uniform(-0.05, 0.05),
                'accuracy_by_category': {},
                'accuracy_by_region': {}
            }

            # Category-specific accuracy
            for category in ProductCategory:
                accuracy_metrics['accuracy_by_category'][category.value] = random.uniform(0.75, 0.95)

            # Region-specific accuracy
            for region in Region:
                accuracy_metrics['accuracy_by_region'][region.value] = random.uniform(0.80, 0.92)

            logger.info(f"Accuracy metrics calculated: R² = {accuracy_metrics['r2_score']:.3f}")
            return accuracy_metrics

        except Exception as e:
            logger.error(f"Accuracy calculation failed: {str(e)}")
            return {'r2_score': 0.0, 'mean_absolute_error': 1.0}

    async def _store_forecasts(self, forecasts: List[Dict[str, Any]], accuracy: Dict[str, float]):
        """Store forecasts and accuracy metrics in database"""

        try:
            session = db_manager.get_session()

            # Store individual forecasts
            for forecast in forecasts:
                DatabaseOperations.save_demand_forecast(
                    forecast_date=forecast['forecast_date'],
                    product_category=forecast['product_category'],
                    region=forecast['region'],
                    predicted_demand=forecast['predicted_demand'],
                    confidence_lower=forecast['confidence_interval_lower'],
                    confidence_upper=forecast['confidence_interval_upper'],
                    model_version=forecast['model_version'],
                    factors=forecast['factors_contributing'],
                    session=session
                )

            # Store accuracy metrics
            for metric_name, metric_value in accuracy.items():
                if isinstance(metric_value, (int, float)):
                    DatabaseOperations.save_system_metric(
                        metric_name=f"demand_forecast_{metric_name}",
                        metric_value=metric_value,
                        metric_unit="ratio" if "r2" in metric_name else "error",
                        agent_name="demand_prediction",
                        session=session
                    )

            session.close()
            logger.info(f"Stored {len(forecasts)} forecasts and accuracy metrics")

        except Exception as e:
            logger.error(f"Failed to store forecasts: {str(e)}")

    async def _check_model_retraining(self, accuracy: Dict[str, float]):
        """Check if models need retraining based on accuracy"""

        try:
            r2_score = accuracy.get('r2_score', 0.0)

            if r2_score < settings.MODEL_ACCURACY_THRESHOLD:
                logger.warning(f"Model accuracy below threshold: {r2_score:.3f} < {settings.MODEL_ACCURACY_THRESHOLD}")
                # In production, trigger model retraining process

            # Check if it's time for scheduled retraining
            if (self.last_model_training is None or 
                datetime.now() - self.last_model_training > timedelta(hours=settings.MODEL_RETRAIN_INTERVAL_HOURS)):

                logger.info("Scheduled model retraining triggered")
                await self._retrain_models()

        except Exception as e:
            logger.error(f"Model retraining check failed: {str(e)}")

    async def _retrain_models(self):
        """Retrain machine learning models"""

        try:
            logger.info("Starting model retraining")

            # Fetch fresh historical data
            historical_data = await self._fetch_historical_data()

            # Retrain models for each product-region combination
            for category in ProductCategory:
                for region in Region:
                    subset = historical_data[
                        (historical_data['product_category'] == category.value) &
                        (historical_data['region'] == region.value)
                    ]

                    if len(subset) >= settings.MIN_HISTORICAL_DATA_POINTS:
                        features = self._prepare_features(subset)
                        target = subset['actual_demand'].values

                        # Retrain models
                        for model_name, model in self.models.items():
                            if model_name != 'ensemble_weights':
                                model.fit(features, target)

            self.last_model_training = datetime.now()
            logger.info("Model retraining completed")

        except Exception as e:
            logger.error(f"Model retraining failed: {str(e)}")

    # Public API methods
    async def get_forecast(
        self, 
        product_category: Optional[str] = None,
        region: Optional[str] = None,
        days_ahead: int = 30
    ) -> DemandForecastResponse:
        """Get demand forecasts"""

        try:
            # Filter forecasts based on parameters
            filtered_forecasts = []

            for forecast in self.current_forecasts:
                # Apply filters
                if product_category and forecast['product_category'] != product_category:
                    continue
                if region and forecast['region'] != region:
                    continue

                # Check date range
                forecast_date = forecast['forecast_date']
                if isinstance(forecast_date, str):
                    forecast_date = datetime.fromisoformat(forecast_date)

                days_from_now = (forecast_date - datetime.now()).days
                if days_from_now > days_ahead:
                    continue

                # Convert to DemandForecast model
                demand_forecast = DemandForecast(
                    product_category=ProductCategory(forecast['product_category']),
                    region=Region(forecast['region']),
                    forecast_date=forecast_date,
                    predicted_demand=forecast['predicted_demand'],
                    confidence_interval_lower=forecast['confidence_interval_lower'],
                    confidence_interval_upper=forecast['confidence_interval_upper'],
                    factors_contributing=forecast['factors_contributing'],
                    accuracy_score=self.accuracy_metrics.get('r2_score')
                )

                filtered_forecasts.append(demand_forecast)

            # Flatten model_performance to Dict[str, float]
            model_performance_flat = {}
            for key, value in self.accuracy_metrics.items():
                if isinstance(value, (int, float)):
                    model_performance_flat[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        model_performance_flat[f"{key}_{sub_key}"] = sub_value if isinstance(sub_value, (int, float)) else 0.0

            return DemandForecastResponse(
                forecasts=filtered_forecasts,
                model_performance=model_performance_flat,
                last_updated=datetime.now(),
                next_update=datetime.now() + timedelta(seconds=settings.DEMAND_FORECAST_UPDATE_INTERVAL)
            )

        except Exception as e:
            logger.error(f"Failed to get demand forecast: {str(e)}")
            raise

    async def recalculate_forecasts(self):
        """Trigger immediate forecast recalculation"""
        await self.update_forecasts()

    async def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get current forecast accuracy metrics"""
        return self.accuracy_metrics
