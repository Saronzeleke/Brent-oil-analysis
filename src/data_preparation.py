import pandas as pd
import numpy as np
from typing import Tuple, Optional

class OilDataProcessor:
    """Process and prepare oil price data for analysis"""
    
    def __init__(self, price_path: str, events_path: str):
        """
        Initialize with data paths
        
        Args:
            price_path: Path to Brent oil prices CSV
            events_path: Path to events CSV
        """
        self.price_path = price_path
        self.events_path = events_path
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare both price and events data
        
        Returns:
            Tuple of (price_df, events_df)
        """
        # Load price data
        price_df = pd.read_csv(self.price_path, parse_dates=['Date'])
        price_df.set_index('Date', inplace=True)
        
        # Ensure chronological order
        price_df = price_df.sort_index()
        
        # Calculate returns
        price_df['log_return'] = np.log(price_df['Price']) - np.log(price_df['Price'].shift(1))
        price_df['simple_return'] = price_df['Price'].pct_change()
        
        # Load events data
        events_df = pd.read_csv(self.events_path, parse_dates=['date'])
        events_df.set_index('date', inplace=True)
        events_df = events_df.sort_index()
        
        return price_df, events_df
    
    def prepare_for_modeling(self, 
                           price_df: pd.DataFrame,
                           use_log_returns: bool = True) -> np.ndarray:
        """
        Prepare data for Bayesian modeling
        
        Args:
            price_df: Processed price DataFrame
            use_log_returns: Whether to use log returns (stationary)
            
        Returns:
            NumPy array of data for modeling
        """
        if use_log_returns:
            # Use log returns for stationarity
            data = price_df['log_return'].dropna().values
        else:
            # Use original prices (non-stationary)
            data = price_df['Price'].dropna().values
            
        return data
    
    def create_event_features(self, 
                            price_df: pd.DataFrame,
                            events_df: pd.DataFrame,
                            window_days: int = 30) -> pd.DataFrame:
        """
        Create features for event impact analysis
        
        Args:
            price_df: Price DataFrame
            events_df: Events DataFrame
            window_days: Days to consider around each event
            
        Returns:
            DataFrame with event features
        """
        # Create copy to avoid modifying original
        result_df = price_df.copy()
        
        # Initialize event columns
        result_df['has_event'] = 0
        result_df['event_type'] = 'none'
        result_df['days_since_event'] = np.nan
        
        # Mark events in the price data
        for event_date in events_df.index:
            if event_date in result_df.index:
                result_df.loc[event_date, 'has_event'] = 1
                result_df.loc[event_date, 'event_type'] = events_df.loc[event_date, 'event_type']
                
                # Calculate days since event for window
                mask = (result_df.index >= event_date) & \
                       (result_df.index <= event_date + pd.Timedelta(days=window_days))
                result_df.loc[mask, 'days_since_event'] = \
                    (result_df.loc[mask].index - event_date).days
        
        return result_df