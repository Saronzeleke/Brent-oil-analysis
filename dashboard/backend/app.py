from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data')

def load_price_data():
    """Load and prepare price data"""
    df = pd.read_csv(os.path.join(DATA_PATH, 'BrentOilPrices.csv'), parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df['log_return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    return df

def load_events_data():
    """Load events data"""
    df = pd.read_csv(os.path.join(DATA_PATH, 'events.csv'), parse_dates=['date'])
    return df

def load_change_point_results():
    """Load change point analysis results"""
    results_path = os.path.join(os.path.dirname(__file__), '../../results/change_point_analysis.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}

# Initialize data
price_df = load_price_data()
events_df = load_events_data()
change_point_results = load_change_point_results()

@app.route('/api/prices', methods=['GET'])
def get_prices():
    """Get price data with optional date filtering"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = price_df.copy()
    
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    # Convert to list of dicts for JSON
    data = []
    for date, row in df.iterrows():
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': float(row['Price']) if not pd.isna(row['Price']) else None,
            'log_return': float(row['log_return']) if not pd.isna(row['log_return']) else None
        })
    
    return jsonify({
        'data': data,
        'count': len(data),
        'date_range': {
            'start': df.index.min().strftime('%Y-%m-%d') if len(df) > 0 else None,
            'end': df.index.max().strftime('%Y-%m-%d') if len(df) > 0 else None
        }
    })

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get events data"""
    events = events_df.to_dict('records')
    
    # Convert datetime to string
    for event in events:
        if isinstance(event['date'], pd.Timestamp):
            event['date'] = event['date'].strftime('%Y-%m-%d')
    
    return jsonify({
        'events': events,
        'count': len(events)
    })

@app.route('/api/change_points', methods=['GET'])
def get_change_points():
    """Get change point analysis results"""
    return jsonify(change_point_results)

@app.route('/api/analysis/metrics', methods=['GET'])
def get_analysis_metrics():
    """Get key analysis metrics"""
    # Calculate basic metrics
    metrics = {
        'current_price': float(price_df['Price'].iloc[-1]) if len(price_df) > 0 else None,
        'avg_price': float(price_df['Price'].mean()),
        'price_volatility': float(price_df['log_return'].std() * np.sqrt(252)),  # Annualized
        'total_events': len(events_df),
        'event_types': events_df['event_type'].value_counts().to_dict()
    }
    
    # Add change point metrics if available
    if change_point_results:
        metrics['change_point_date'] = change_point_results.get('change_date')
        metrics['impact_metrics'] = change_point_results.get('impact_metrics', {})
    
    return jsonify(metrics)

@app.route('/api/analysis/event_impact/<event_id>', methods=['GET'])
def get_event_impact(event_id):
    """Analyze impact of specific event"""
    try:
        event_idx = int(event_id)
        if 0 <= event_idx < len(events_df):
            event = events_df.iloc[event_idx]
            event_date = event['date']
            
            # Calculate impact metrics
            window_days = 30
            start_date = event_date - pd.Timedelta(days=window_days)
            end_date = event_date + pd.Timedelta(days=window_days)
            
            # Get prices around event
            mask = (price_df.index >= start_date) & (price_df.index <= end_date)
            event_prices = price_df.loc[mask].copy()
            
            if len(event_prices) > 0:
                # Calculate pre and post event averages
                pre_event = event_prices[event_prices.index < event_date]
                post_event = event_prices[event_prices.index > event_date]
                
                impact = {
                    'event': event.to_dict(),
                    'pre_event_avg': float(pre_event['Price'].mean()) if len(pre_event) > 0 else None,
                    'post_event_avg': float(post_event['Price'].mean()) if len(post_event) > 0 else None,
                    'price_change_pct': None,
                    'volatility_change': None
                }
                
                if impact['pre_event_avg'] and impact['post_event_avg']:
                    impact['price_change_pct'] = (
                        (impact['post_event_avg'] - impact['pre_event_avg']) / 
                        impact['pre_event_avg'] * 100
                    )
                
                return jsonify(impact)
    
    except (ValueError, KeyError):
        pass
    
    return jsonify({'error': 'Event not found'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)