"""
Model factory for creating different types of feature selection models.

This module provides a factory pattern for creating and configuring
different types of feature selection models used in the simulations.
"""

import time
from typing import Dict, Any, Callable, Tuple
import numpy as np

# Import from core EFS package
from efs.core.efs import EFS, EFSCV
from efs.mse import create_mse_scorer

# Import from simulation package
from efs_experiments.models.data_smearing import SmearedFS
from efs_experiments.models.bagging import BaggedFS

# Import parameter utilities
from ..config.parameter_loader import get_m_grid


class ModelFactory:
    """
    Factory for creating and fitting different types of feature selection models.
    
    This class encapsulates the model creation logic that was originally
    scattered throughout simulation_main.py.
    """
    
    def __init__(self, params: Dict[str, Any], cv_value: int, make_k_scorer: Callable):
        """
        Initialize the model factory.
        
        Parameters
        ----------
        params : dict
            Simulation parameters containing model configuration
        cv_value : int
            Cross-validation fold count
        make_k_scorer : callable
            Function to create k-specific scorers
        """
        self.params = params
        self.cv_value = cv_value
        self.make_k_scorer = make_k_scorer
        
    def create_bagged_fs(self, seed: int, sim_num: int) -> Tuple[BaggedFS, float]:
        """
        Create and fit a BaggedFS model.
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        model = BaggedFS(
            k_max=self.params['model']['k_max'],
            n_estimators=self.params['model']['bagged_fs']['n_estimators'],
            random_state=seed + sim_num,
            method=self.params['model'].get('method', 'fs'),
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def create_smeared_fs(self, seed: int, sim_num: int) -> Tuple[SmearedFS, float]:
        """
        Create and fit a SmearedFS model.
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        model = SmearedFS(
            k_max=self.params['model']['k_max'],
            n_estimators=self.params['model']['smeared_fs']['n_estimators'],
            noise_scale=self.params['model']['smeared_fs']['param_grid']['noise_scale'],
            random_state=seed + sim_num,
            method=self.params['model'].get('method', 'fs'),
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def create_efscv(self, seed: int, sim_num: int) -> Tuple[EFSCV, float]:
        """
        Create and fit an EFSCV model.
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        # Calculate m_grid
        m_grid = get_m_grid(
            self.params['model']['m_grid'],
            self.params['data']['n_predictors']
        )
        
        model = EFSCV(
            k_max=self.params['model']['k_max'],
            m_grid=m_grid,
            n_estimators=self.params['model']['efscv']['n_estimators'],
            n_resample_iter=self.params['model']['efscv']['n_resample_iter'],
            method=self.params['model'].get('method', 'fs'),
            random_state=seed + sim_num,
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def create_forward_selection(self, seed: int, sim_num: int) -> Tuple[EFSCV, float]:
        """
        Create and fit a Forward Selection model (EFSCV with specific parameters).
        
        Parameters
        ----------
        seed : int
            Random seed
        sim_num : int
            Simulation number
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        start_time = time.time()
        
        model = EFSCV(
            k_max=self.params['model']['k_max'],
            m_grid=[self.params['data']['n_predictors']],  # Use all features
            n_estimators=1,
            n_resample_iter=0,
            method=self.params['model'].get('method', 'fs'),
            random_state=seed + sim_num,
            cv=self.cv_value,
            scoring=self.make_k_scorer
        )
        
        return model, start_time
    
    def fit_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                  start_time: float) -> Tuple[Any, float]:
        """
        Fit a model and return it with timing information.
        
        Parameters
        ----------
        model : Any
            Model to fit
        X_train : ndarray
            Training features
        y_train : ndarray
            Training targets
        start_time : float
            Time when model creation started
            
        Returns
        -------
        tuple
            (fitted_model, fitting_time)
        """
        model.fit(X_train, y_train)
        fitting_time = time.time() - start_time
        return model, fitting_time 