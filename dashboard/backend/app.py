from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import traceback

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all routes

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '../../results')

def load_price_data():
    """Load and prepare price data"""
    try:
        file_path = os.path.join(DATA_PATH, 'BrentOilPrices.csv')
        if not os.path.exists(file_path):
            print(f"Price file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Clean data
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price'])
        
        # Calculate returns
        df['log_return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
        df['simple_return'] = df['Price'].pct_change()
        
        return df
    except Exception as e:
        print(f"Error loading price data: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

def load_events_data():
    """Load events data"""
    try:
        file_path = os.path.join(DATA_PATH, 'events.csv')
        if not os.path.exists(file_path):
            print(f"Events file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.sort_values('date')
        
        # Clean data
        if 'impact_direction' not in df.columns:
            df['impact_direction'] = 'neutral'
        
        return df
    except Exception as e:
        print(f"Error loading events data: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()

def load_change_point_results():
    """Load change point analysis results"""
    try:
        results_path = os.path.join(RESULTS_PATH, 'change_point_analysis.json')
        if not os.path.exists(results_path):
            print(f"Change point results not found: {results_path}")
            return {}
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Convert date strings back to datetime if needed
        if 'change_date' in data and data['change_date']:
            try:
                data['change_date'] = datetime.fromisoformat(data['change_date'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
            except:
                pass  # Keep as is if conversion fails
        
        return data
    except Exception as e:
        print(f"Error loading change point results: {e}")
        print(traceback.format_exc())
        return {}

# Initialize data
price_df = load_price_data()
events_df = load_events_data()
change_point_results = load_change_point_results()

@app.route('/api/prices', methods=['GET'])
def get_prices():
    """Get price data with optional date filtering"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if price_df.empty:
            return jsonify({
                'data': [],
                'count': 0,
                'date_range': {'start': None, 'end': None},
                'error': 'No price data available'
            }), 200
        
        df = price_df.copy()
        
        if start_date:
            try:
                df = df[df.index >= pd.to_datetime(start_date)]
            except:
                pass
        
        if end_date:
            try:
                df = df[df.index <= pd.to_datetime(end_date)]
            except:
                pass
        
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
    except Exception as e:
        print(f"Error in /api/prices: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get events data"""
    try:
        if events_df.empty:
            return jsonify({
                'events': [],
                'count': 0,
                'message': 'No events data available'
            }), 200
        
        # Convert to list of dicts
        events = []
        for _, row in events_df.iterrows():
            event_dict = row.to_dict()
            if 'date' in event_dict and hasattr(event_dict['date'], 'strftime'):
                event_dict['date'] = event_dict['date'].strftime('%Y-%m-%d')
            events.append(event_dict)
        
        return jsonify({
            'events': events,
            'count': len(events)
        })
    except Exception as e:
        print(f"Error in /api/events: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/change_points', methods=['GET'])
def get_change_points():
    """Get change point analysis results"""
    try:
        return jsonify(change_point_results)
    except Exception as e:
        print(f"Error in /api/change_points: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/analysis/metrics', methods=['GET'])
def get_analysis_metrics():
    """Get key analysis metrics"""
    try:
        metrics = {
            'current_price': None,
            'avg_price': None,
            'price_volatility': None,
            'total_events': 0,
            'event_types': {}
        }
        
        # Calculate basic metrics from price data
        if not price_df.empty:
            if 'Price' in price_df.columns and len(price_df) > 0:
                metrics['current_price'] = float(price_df['Price'].iloc[-1])
                metrics['avg_price'] = float(price_df['Price'].mean())
            
            if 'log_return' in price_df.columns:
                metrics['price_volatility'] = float(price_df['log_return'].std() * np.sqrt(252))  # Annualized
        
        # Calculate event metrics
        if not events_df.empty:
            metrics['total_events'] = len(events_df)
            if 'event_type' in events_df.columns:
                metrics['event_types'] = events_df['event_type'].value_counts().to_dict()
        
        # Add change point metrics if available
        if change_point_results:
            if 'change_date' in change_point_results:
                metrics['change_point_date'] = change_point_results.get('change_date')
            if 'impact_metrics' in change_point_results:
                metrics['impact_metrics'] = change_point_results.get('impact_metrics', {})
        
        return jsonify(metrics)
    except Exception as e:
        print(f"Error in /api/analysis/metrics: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/analysis/event_impact/<int:event_id>', methods=['GET'])
def get_event_impact(event_id):
    """Analyze impact of specific event"""
    try:
        if events_df.empty or price_df.empty:
            return jsonify({
                'error': 'No data available',
                'message': 'Price or events data not loaded'
            }), 404
        
        if event_id < 0 or event_id >= len(events_df):
            return jsonify({
                'error': 'Event not found',
                'message': f'Event ID {event_id} out of range'
            }), 404
        
        event = events_df.iloc[event_id]
        event_date = event['date']
        
        # Calculate impact metrics
        window_days = 30
        start_date = event_date - pd.Timedelta(days=window_days)
        end_date = event_date + pd.Timedelta(days=window_days)
        
        # Get prices around event
        mask = (price_df.index >= start_date) & (price_df.index <= end_date)
        event_prices = price_df.loc[mask].copy()
        
        if len(event_prices) == 0:
            return jsonify({
                'event': event.to_dict(),
                'pre_event_avg': None,
                'post_event_avg': None,
                'price_change_pct': None,
                'volatility_change': None,
                'message': 'No price data available for event period'
            })
        
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
        
        # Convert event date to string
        if hasattr(impact['event']['date'], 'strftime'):
            impact['event']['date'] = impact['event']['date'].strftime('%Y-%m-%d')
        
        # Calculate percentage change
        if impact['pre_event_avg'] is not None and impact['post_event_avg'] is not None:
            if impact['pre_event_avg'] != 0:
                impact['price_change_pct'] = float(
                    (impact['post_event_avg'] - impact['pre_event_avg']) / 
                    impact['pre_event_avg'] * 100
                )
        
        return jsonify(impact)
    except Exception as e:
        print(f"Error in /api/analysis/event_impact/{event_id}: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'data_status': {
                'price_data_loaded': not price_df.empty,
                'events_data_loaded': not events_df.empty,
                'change_points_loaded': bool(change_point_results)
            }
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("Starting Brent Oil Analysis API Server...")
    print(f"Price data loaded: {len(price_df)} records")
    print(f"Events data loaded: {len(events_df)} events")
    print(f"Change points loaded: {'Yes' if change_point_results else 'No'}")
    print("Server running on http://localhost:5000")
    print("API Endpoints:")
    print("  GET /api/prices - Get price data")
    print("  GET /api/events - Get events data")
    print("  GET /api/change_points - Get change point analysis")
    print("  GET /api/analysis/metrics - Get analysis metrics")
    print("  GET /api/analysis/event_impact/<id> - Analyze event impact")
    print("  GET /api/health - Health check")
    
    app.run(debug=True, port=5000, host='localhost')
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# import json
# from datetime import datetime
# import os

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend

# # Load data
# DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data')

# def load_price_data():
#     """Load and prepare price data"""
#     df = pd.read_csv(os.path.join(DATA_PATH, 'BrentOilPrices.csv'), parse_dates=['Date'])
#     df.set_index('Date', inplace=True)
#     df['log_return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
#     return df

# def load_events_data():
#     """Load events data"""
#     df = pd.read_csv(os.path.join(DATA_PATH, 'events.csv'), parse_dates=['date'])
#     return df

# def load_change_point_results():
#     """Load change point analysis results"""
#     results_path = os.path.join(os.path.dirname(__file__), '../../results/change_point_analysis.json')
#     if os.path.exists(results_path):
#         with open(results_path, 'r') as f:
#             return json.load(f)
#     return {}

# # Initialize data
# price_df = load_price_data()
# events_df = load_events_data()
# change_point_results = load_change_point_results()

# @app.route('/api/prices', methods=['GET'])
# def get_prices():
#     """Get price data with optional date filtering"""
#     start_date = request.args.get('start_date')
#     end_date = request.args.get('end_date')
    
#     df = price_df.copy()
    
#     if start_date:
#         df = df[df.index >= pd.to_datetime(start_date)]
#     if end_date:
#         df = df[df.index <= pd.to_datetime(end_date)]
    
#     # Convert to list of dicts for JSON
#     data = []
#     for date, row in df.iterrows():
#         data.append({
#             'date': date.strftime('%Y-%m-%d'),
#             'price': float(row['Price']) if not pd.isna(row['Price']) else None,
#             'log_return': float(row['log_return']) if not pd.isna(row['log_return']) else None
#         })
    
#     return jsonify({
#         'data': data,
#         'count': len(data),
#         'date_range': {
#             'start': df.index.min().strftime('%Y-%m-%d') if len(df) > 0 else None,
#             'end': df.index.max().strftime('%Y-%m-%d') if len(df) > 0 else None
#         }
#     })

# @app.route('/api/events', methods=['GET'])
# def get_events():
#     """Get events data"""
#     events = events_df.to_dict('records')
    
#     # Convert datetime to string
#     for event in events:
#         if isinstance(event['date'], pd.Timestamp):
#             event['date'] = event['date'].strftime('%Y-%m-%d')
    
#     return jsonify({
#         'events': events,
#         'count': len(events)
#     })

# @app.route('/api/change_points', methods=['GET'])
# def get_change_points():
#     """Get change point analysis results"""
#     return jsonify(change_point_results)

# @app.route('/api/analysis/metrics', methods=['GET'])
# def get_analysis_metrics():
#     """Get key analysis metrics"""
#     # Calculate basic metrics
#     metrics = {
#         'current_price': float(price_df['Price'].iloc[-1]) if len(price_df) > 0 else None,
#         'avg_price': float(price_df['Price'].mean()),
#         'price_volatility': float(price_df['log_return'].std() * np.sqrt(252)),  # Annualized
#         'total_events': len(events_df),
#         'event_types': events_df['event_type'].value_counts().to_dict()
#     }
    
#     # Add change point metrics if available
#     if change_point_results:
#         metrics['change_point_date'] = change_point_results.get('change_date')
#         metrics['impact_metrics'] = change_point_results.get('impact_metrics', {})
    
#     return jsonify(metrics)

# @app.route('/api/analysis/event_impact/<event_id>', methods=['GET'])
# def get_event_impact(event_id):
#     """Analyze impact of specific event"""
#     try:
#         event_idx = int(event_id)
#         if 0 <= event_idx < len(events_df):
#             event = events_df.iloc[event_idx]
#             event_date = event['date']
            
#             # Calculate impact metrics
#             window_days = 30
#             start_date = event_date - pd.Timedelta(days=window_days)
#             end_date = event_date + pd.Timedelta(days=window_days)
            
#             # Get prices around event
#             mask = (price_df.index >= start_date) & (price_df.index <= end_date)
#             event_prices = price_df.loc[mask].copy()
            
#             if len(event_prices) > 0:
#                 # Calculate pre and post event averages
#                 pre_event = event_prices[event_prices.index < event_date]
#                 post_event = event_prices[event_prices.index > event_date]
                
#                 impact = {
#                     'event': event.to_dict(),
#                     'pre_event_avg': float(pre_event['Price'].mean()) if len(pre_event) > 0 else None,
#                     'post_event_avg': float(post_event['Price'].mean()) if len(post_event) > 0 else None,
#                     'price_change_pct': None,
#                     'volatility_change': None
#                 }
                
#                 if impact['pre_event_avg'] and impact['post_event_avg']:
#                     impact['price_change_pct'] = (
#                         (impact['post_event_avg'] - impact['pre_event_avg']) / 
#                         impact['pre_event_avg'] * 100
#                     )
                
#                 return jsonify(impact)
    
#     except (ValueError, KeyError):
#         pass
    
#     return jsonify({'error': 'Event not found'}), 404

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)