#File: backend/utils/ml_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLUtils:
    """
    Machine Learning utilities for Health First AI Supply Chain

    Provides:
    - Model training and evaluation
    - Feature engineering
    - Model selection and hyperparameter tuning
    - Ensemble methods
    - Model persistence
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}

    def prepare_features(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None,
        create_lag_features: bool = True,
        lag_periods: List[int] = [1, 7, 30]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning models

        Args:
            data: Input DataFrame
            target_column: Name of target column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            create_lag_features: Whether to create lag features
            lag_periods: Lag periods for feature creation

        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            logger.info("Preparing features for ML models")

            # Make a copy of the data
            df = data.copy()

            # Sort by date if available
            if 'date' in df.columns:
                df = df.sort_values('date')

            # Separate target
            target = df[target_column]
            features_df = df.drop(columns=[target_column])

            # Handle categorical columns
            if categorical_columns:
                for col in categorical_columns:
                    if col in features_df.columns:
                        if col not in self.encoders:
                            self.encoders[col] = LabelEncoder()
                            features_df[col] = self.encoders[col].fit_transform(features_df[col].astype(str))
                        else:
                            features_df[col] = self.encoders[col].transform(features_df[col].astype(str))

            # Create lag features
            if create_lag_features and len(df) > max(lag_periods):
                for lag in lag_periods:
                    features_df[f'{target_column}_lag_{lag}'] = target.shift(lag)

                    # Create rolling statistics
                    features_df[f'{target_column}_rolling_mean_{lag}'] = target.rolling(window=lag).mean()
                    features_df[f'{target_column}_rolling_std_{lag}'] = target.rolling(window=lag).std()

            # Create time-based features if date column exists
            if 'date' in df.columns:
                date_col = pd.to_datetime(df['date'])
                features_df['year'] = date_col.dt.year
                features_df['month'] = date_col.dt.month
                features_df['day_of_week'] = date_col.dt.dayofweek
                features_df['quarter'] = date_col.dt.quarter
                features_df['is_weekend'] = (date_col.dt.dayofweek >= 5).astype(int)
                features_df['is_month_start'] = date_col.dt.is_month_start.astype(int)
                features_df['is_month_end'] = date_col.dt.is_month_end.astype(int)

            # Create interaction features for important combinations
            numerical_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                # Create a few interaction features (limit to avoid explosion)
                important_cols = numerical_cols[:5]  # Limit to first 5 numerical columns
                for i, col1 in enumerate(important_cols):
                    for col2 in important_cols[i+1:]:
                        features_df[f'{col1}_x_{col2}'] = features_df[col1] * features_df[col2]

            # Remove rows with NaN values (created by lag features)
            initial_rows = len(features_df)
            features_df = features_df.dropna()
            target = target.loc[features_df.index]
            final_rows = len(features_df)

            if final_rows < initial_rows:
                logger.info(f"Dropped {initial_rows - final_rows} rows due to NaN values")

            # Store feature names
            self.feature_names['current'] = features_df.columns.tolist()

            logger.info(f"Features prepared: {features_df.shape[1]} features, {features_df.shape[0]} samples")
            return features_df, target

        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            raise

    def train_ensemble_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "demand_forecast",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train ensemble model with multiple algorithms

        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name for the model
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility

        Returns:
            Dictionary with model performance metrics
        """
        try:
            logger.info(f"Training ensemble model: {model_name}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            # Store scaler
            self.scalers[model_name] = scaler

            # Define base models
            models = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state
                )
            }

            # Train models and collect predictions
            model_performances = {}
            train_predictions = {}
            test_predictions = {}

            for name, model in models.items():
                logger.info(f"Training {name}")

                # Train model
                if name in ['linear_regression', 'ridge']:
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)

                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, train_pred)
                test_metrics = self._calculate_metrics(y_test, test_pred)

                model_performances[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }

                train_predictions[name] = train_pred
                test_predictions[name] = test_pred

            # Create ensemble predictions (weighted average)
            weights = self._calculate_ensemble_weights(model_performances)

            ensemble_train_pred = np.zeros(len(y_train))
            ensemble_test_pred = np.zeros(len(y_test))

            for name, weight in weights.items():
                ensemble_train_pred += weight * train_predictions[name]
                ensemble_test_pred += weight * test_predictions[name]

            # Calculate ensemble metrics
            ensemble_train_metrics = self._calculate_metrics(y_train, ensemble_train_pred)
            ensemble_test_metrics = self._calculate_metrics(y_test, ensemble_test_pred)

            # Store ensemble model
            ensemble_model = {
                'models': {name: perf['model'] for name, perf in model_performances.items()},
                'weights': weights,
                'scaler': scaler,
                'feature_names': X.columns.tolist(),
                'train_metrics': ensemble_train_metrics,
                'test_metrics': ensemble_test_metrics,
                'trained_at': datetime.now()
            }

            self.models[model_name] = ensemble_model

            # Prepare results
            results = {
                'model_name': model_name,
                'ensemble_performance': {
                    'train_metrics': ensemble_train_metrics,
                    'test_metrics': ensemble_test_metrics
                },
                'individual_models': {
                    name: {
                        'train_metrics': perf['train_metrics'],
                        'test_metrics': perf['test_metrics']
                    }
                    for name, perf in model_performances.items()
                },
                'ensemble_weights': weights,
                'feature_importance': self._get_feature_importance(ensemble_model, X.columns),
                'cross_validation': self._perform_cross_validation(X, y, ensemble_model)
            }

            logger.info(f"Ensemble model trained successfully. Test R²: {ensemble_test_metrics['r2_score']:.3f}")
            return results

        except Exception as e:
            logger.error(f"Ensemble model training failed: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2_score': float(r2_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred))
        }

    def _calculate_ensemble_weights(self, model_performances: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance"""

        # Use R² score for weighting (higher is better)
        scores = {}
        for name, perf in model_performances.items():
            r2 = perf['test_metrics']['r2_score']
            # Ensure non-negative weights
            scores[name] = max(0, r2)

        # Normalize weights
        total_score = sum(scores.values())
        if total_score > 0:
            weights = {name: score / total_score for name, score in scores.items()}
        else:
            # Equal weights if all models perform poorly
            weights = {name: 1.0 / len(scores) for name in scores.keys()}

        return weights

    def _get_feature_importance(self, ensemble_model: Dict[str, Any], feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from ensemble model"""

        try:
            importances = np.zeros(len(feature_names))
            weights = ensemble_model['weights']

            # Aggregate importance from tree-based models
            for name, model in ensemble_model['models'].items():
                if hasattr(model, 'feature_importances_'):
                    weight = weights.get(name, 0)
                    importances += weight * model.feature_importances_

            # Normalize
            if importances.sum() > 0:
                importances = importances / importances.sum()

            # Create feature importance dictionary
            feature_importance = {
                feature_names[i]: float(importances[i])
                for i in range(len(feature_names))
            }

            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return feature_importance

        except Exception as e:
            logger.error(f"Feature importance calculation failed: {str(e)}")
            return {}

    def _perform_cross_validation(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        ensemble_model: Dict[str, Any],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Perform cross-validation on the ensemble model"""

        try:
            cv_scores = {}

            # Perform CV for each base model
            for name, model in ensemble_model['models'].items():
                if name in ['linear_regression', 'ridge']:
                    # Use scaled features for linear models
                    scaler = ensemble_model['scaler']
                    X_scaled = pd.DataFrame(
                        scaler.transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
                else:
                    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

                cv_scores[name] = {
                    'mean_score': float(scores.mean()),
                    'std_score': float(scores.std()),
                    'scores': scores.tolist()
                }

            # Calculate weighted ensemble CV score
            weights = ensemble_model['weights']
            ensemble_cv_score = sum(
                weights[name] * cv_scores[name]['mean_score']
                for name in cv_scores.keys()
            )

            return {
                'individual_models': cv_scores,
                'ensemble_cv_score': float(ensemble_cv_score),
                'cv_folds': cv_folds
            }

        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            return {}

    def predict(
        self, 
        model_name: str, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained ensemble model

        Args:
            model_name: Name of the trained model
            X: Feature matrix for prediction

        Returns:
            Tuple of (predictions, prediction_intervals)
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

            ensemble_model = self.models[model_name]
            models = ensemble_model['models']
            weights = ensemble_model['weights']
            scaler = ensemble_model['scaler']

            # Ensure features match training features
            expected_features = ensemble_model['feature_names']
            if list(X.columns) != expected_features:
                logger.warning("Feature mismatch detected. Attempting to align features.")
                X = X.reindex(columns=expected_features, fill_value=0)

            # Make predictions with each model
            predictions = {}

            for name, model in models.items():
                if name in ['linear_regression', 'ridge']:
                    # Use scaled features for linear models
                    X_scaled = pd.DataFrame(
                        scaler.transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)

                predictions[name] = pred

            # Calculate ensemble prediction
            ensemble_pred = np.zeros(len(X))
            for name, weight in weights.items():
                ensemble_pred += weight * predictions[name]

            # Calculate prediction intervals (simplified approach)
            # In practice, you might use more sophisticated methods like quantile regression
            prediction_std = np.std([predictions[name] for name in predictions.keys()], axis=0)
            confidence_interval = 1.96 * prediction_std  # 95% confidence interval

            prediction_intervals = np.column_stack([
                ensemble_pred - confidence_interval,
                ensemble_pred + confidence_interval
            ])

            logger.info(f"Generated {len(ensemble_pred)} predictions using ensemble model")
            return ensemble_pred, prediction_intervals

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
        param_grid: Dict[str, List] = None,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to tune
            param_grid: Parameter grid for tuning
            cv_folds: Number of CV folds

        Returns:
            Dictionary with best parameters and performance
        """
        try:
            logger.info(f"Tuning hyperparameters for {model_type}")

            # Default parameter grids
            default_param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                },
                'ridge': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }

            if param_grid is None:
                param_grid = default_param_grids.get(model_type, {})

            # Initialize model
            if model_type == 'random_forest':
                model = RandomForestRegressor(random_state=42)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(random_state=42)
            elif model_type == 'ridge':
                model = Ridge()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Perform grid search
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X, y)

            results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'best_model': grid_search.best_estimator_,
                'cv_results': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                }
            }

            logger.info(f"Hyperparameter tuning completed. Best score: {results['best_score']:.3f}")
            return results

        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise

    def save_model(self, model_name: str, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            return False

    def load_model(self, model_name: str, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary of trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            model = self.models[model_name]

            summary = {
                'model_name': model_name,
                'trained_at': model['trained_at'].isoformat(),
                'feature_count': len(model['feature_names']),
                'feature_names': model['feature_names'],
                'train_metrics': model['train_metrics'],
                'test_metrics': model['test_metrics'],
                'ensemble_weights': model['weights'],
                'model_types': list(model['models'].keys())
            }

            return summary

        except Exception as e:
            logger.error(f"Model summary generation failed: {str(e)}")
            return {}

    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare performance of multiple models"""
        try:
            comparison_data = []

            for model_name in model_names:
                if model_name in self.models:
                    model = self.models[model_name]
                    test_metrics = model['test_metrics']

                    comparison_data.append({
                        'model_name': model_name,
                        'r2_score': test_metrics['r2_score'],
                        'mae': test_metrics['mae'],
                        'rmse': test_metrics['rmse'],
                        'mape': test_metrics['mape']
                    })

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('r2_score', ascending=False)

            logger.info(f"Model comparison completed for {len(comparison_df)} models")
            return comparison_df

        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return pd.DataFrame()