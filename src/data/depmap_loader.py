"""
DepMap Data Loader
==================

Module for loading and handling DepMap datasets including:
- Gene expression data
- Mutation data
- Drug response data (PRISM, CTRPv2, etc.)
- Cell line metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DepMapDataLoader:
    """Load and manage DepMap datasets."""
    
    def __init__(self, data_dir: str):
        """
        Initialize DepMap data loader.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing DepMap data files
        """
        self.data_dir = Path(data_dir)
        self.expression_data = None
        self.mutation_data = None
        self.drug_response_data = None
        self.cell_line_metadata = None
        
    def load_expression_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load gene expression data.
        
        Parameters
        ----------
        file_path : str, optional
            Path to expression data file. If None, uses default path.
            
        Returns
        -------
        pd.DataFrame
            Gene expression matrix (cell lines x genes)
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "expression.csv"
        
        logger.info(f"Loading expression data from {file_path}")
        self.expression_data = pd.read_csv(file_path, index_col=0)
        logger.info(f"Loaded expression data: {self.expression_data.shape}")
        
        return self.expression_data
    
    def load_mutation_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load mutation data.
        
        Parameters
        ----------
        file_path : str, optional
            Path to mutation data file. If None, uses default path.
            
        Returns
        -------
        pd.DataFrame
            Mutation matrix (cell lines x genes)
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "mutations.csv"
        
        logger.info(f"Loading mutation data from {file_path}")
        self.mutation_data = pd.read_csv(file_path, index_col=0)
        logger.info(f"Loaded mutation data: {self.mutation_data.shape}")
        
        return self.mutation_data
    
    def load_drug_response_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load drug response/perturbation data.
        
        Parameters
        ----------
        file_path : str, optional
            Path to drug response data file. If None, uses default path.
            
        Returns
        -------
        pd.DataFrame
            Drug response data (cell lines x drugs)
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "drug_response.csv"
        
        logger.info(f"Loading drug response data from {file_path}")
        self.drug_response_data = pd.read_csv(file_path, index_col=0)
        logger.info(f"Loaded drug response data: {self.drug_response_data.shape}")
        
        return self.drug_response_data
    
    def load_cell_line_metadata(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load cell line metadata.
        
        Parameters
        ----------
        file_path : str, optional
            Path to metadata file. If None, uses default path.
            
        Returns
        -------
        pd.DataFrame
            Cell line metadata
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "cell_line_metadata.csv"
        
        logger.info(f"Loading cell line metadata from {file_path}")
        self.cell_line_metadata = pd.read_csv(file_path, index_col=0)
        logger.info(f"Loaded metadata for {len(self.cell_line_metadata)} cell lines")
        
        return self.cell_line_metadata
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available DepMap datasets.
        
        Returns
        -------
        dict
            Dictionary containing all loaded datasets
        """
        datasets = {}
        
        try:
            datasets['expression'] = self.load_expression_data()
        except FileNotFoundError:
            logger.warning("Expression data not found")
        
        try:
            datasets['mutations'] = self.load_mutation_data()
        except FileNotFoundError:
            logger.warning("Mutation data not found")
        
        try:
            datasets['drug_response'] = self.load_drug_response_data()
        except FileNotFoundError:
            logger.warning("Drug response data not found")
        
        try:
            datasets['metadata'] = self.load_cell_line_metadata()
        except FileNotFoundError:
            logger.warning("Cell line metadata not found")
        
        return datasets
    
    def get_common_cell_lines(self) -> List[str]:
        """
        Get cell lines present in all loaded datasets.
        
        Returns
        -------
        list
            List of common cell line identifiers
        """
        indices = []
        
        if self.expression_data is not None:
            indices.append(set(self.expression_data.index))
        if self.mutation_data is not None:
            indices.append(set(self.mutation_data.index))
        if self.drug_response_data is not None:
            indices.append(set(self.drug_response_data.index))
        
        if not indices:
            return []
        
        common = set.intersection(*indices)
        logger.info(f"Found {len(common)} common cell lines")
        
        return sorted(list(common))
