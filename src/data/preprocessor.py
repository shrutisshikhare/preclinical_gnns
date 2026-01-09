"""
Data Preprocessor
=================

Module for preprocessing DepMap data including:
- Normalization
- Feature selection
- Missing value imputation
- Train/validation/test splitting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess DepMap datasets for GNN models."""
    
    def __init__(
        self,
        normalization: str = "standard",
        imputation: str = "mean",
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None
    ):
        """
        Initialize data preprocessor.
        
        Parameters
        ----------
        normalization : str
            Normalization method: 'standard', 'minmax', 'robust', or 'none'
        imputation : str
            Imputation method: 'mean', 'median', 'knn', or 'none'
        feature_selection : str, optional
            Feature selection method: 'variance', 'kbest', or None
        n_features : int, optional
            Number of features to select (for 'kbest' method)
        """
        self.normalization = normalization
        self.imputation = imputation
        self.feature_selection = feature_selection
        self.n_features = n_features
        
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.selected_features = None
        
    def fit_transform(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target : pd.Series, optional
            Target variable (for supervised feature selection)
            
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        logger.info("Fitting and transforming data")
        
        # Handle missing values
        if self.imputation != "none":
            data = self._impute(data, fit=True)
        
        # Normalize
        if self.normalization != "none":
            data = self._normalize(data, fit=True)
        
        # Select features
        if self.feature_selection is not None:
            data = self._select_features(data, target, fit=True)
        
        logger.info(f"Transformed data shape: {data.shape}")
        
        return data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        # Handle missing values
        if self.imputation != "none":
            data = self._impute(data, fit=False)
        
        # Normalize
        if self.normalization != "none":
            data = self._normalize(data, fit=False)
        
        # Select features
        if self.feature_selection is not None and self.selected_features is not None:
            data = data[self.selected_features]
        
        return data
    
    def _impute(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Impute missing values."""
        if fit:
            if self.imputation == "mean":
                self.imputer = SimpleImputer(strategy="mean")
            elif self.imputation == "median":
                self.imputer = SimpleImputer(strategy="median")
            elif self.imputation == "knn":
                self.imputer = KNNImputer(n_neighbors=5)
            else:
                raise ValueError(f"Unknown imputation method: {self.imputation}")
            
            logger.info(f"Fitting {self.imputation} imputer")
            imputed_values = self.imputer.fit_transform(data.values)
        else:
            if self.imputer is None:
                raise ValueError("Imputer not fitted")
            imputed_values = self.imputer.transform(data.values)
        
        return pd.DataFrame(imputed_values, index=data.index, columns=data.columns)
    
    def _normalize(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Normalize data."""
        if fit:
            if self.normalization == "standard":
                self.scaler = StandardScaler()
            elif self.normalization == "minmax":
                self.scaler = MinMaxScaler()
            elif self.normalization == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization}")
            
            logger.info(f"Fitting {self.normalization} scaler")
            normalized_values = self.scaler.fit_transform(data.values)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            normalized_values = self.scaler.transform(data.values)
        
        return pd.DataFrame(normalized_values, index=data.index, columns=data.columns)
    
    def _select_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        fit: bool = False
    ) -> pd.DataFrame:
        """Select features."""
        if fit:
            if self.feature_selection == "variance":
                self.feature_selector = VarianceThreshold(threshold=0.01)
                logger.info("Selecting features by variance")
                mask = self.feature_selector.fit_transform(data.values)
                self.selected_features = data.columns[self.feature_selector.get_support()].tolist()
            
            elif self.feature_selection == "kbest":
                if target is None:
                    raise ValueError("Target required for 'kbest' feature selection")
                if self.n_features is None:
                    raise ValueError("n_features required for 'kbest' feature selection")
                
                self.feature_selector = SelectKBest(f_regression, k=self.n_features)
                logger.info(f"Selecting top {self.n_features} features")
                self.feature_selector.fit(data.values, target.values)
                self.selected_features = data.columns[self.feature_selector.get_support()].tolist()
            
            else:
                raise ValueError(f"Unknown feature selection method: {self.feature_selection}")
            
            logger.info(f"Selected {len(self.selected_features)} features")
        
        return data[self.selected_features]
    
    @staticmethod
    def train_test_split_graphs(
        data: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Parameters
        ----------
        data : pd.DataFrame
            Feature data
        target : pd.Series
            Target variable
        test_size : float
            Proportion of test set
        val_size : float
            Proportion of validation set (from remaining data)
        random_state : int
            Random seed
            
        Returns
        -------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, target, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def filter_low_variance_features(
        data: pd.DataFrame,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Remove low variance features.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        threshold : float
            Variance threshold
            
        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        variances = data.var()
        selected_cols = variances[variances > threshold].index.tolist()
        
        logger.info(f"Filtered {len(data.columns) - len(selected_cols)} low variance features")
        
        return data[selected_cols]
