import React, { useState, useEffect, useCallback } from 'react';
import { 
  LineChart, Line, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, Scatter
} from 'recharts';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";
import './App.css';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [priceData, setPriceData] = useState([]);
  const [eventsData, setEventsData] = useState([]);
  const [changePoints, setChangePoints] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [eventImpact, setEventImpact] = useState(null);
  const [dateRange, setDateRange] = useState({
    start: new Date('2000-01-01'),
    end: new Date()
  });
  const [loading, setLoading] = useState(true);

  // Use useCallback to memoize fetchData
  const fetchData = useCallback(async () => {
    setLoading(true);
    
    try {
      // Format dates for API
      const startDateStr = dateRange.start.toISOString().split('T')[0];
      const endDateStr = dateRange.end.toISOString().split('T')[0];
      
      // Fetch prices
      const priceResponse = await fetch(
        `${API_BASE_URL}/prices?start_date=${startDateStr}&end_date=${endDateStr}`
      );
      if (!priceResponse.ok) {
        throw new Error(`Price API failed: ${priceResponse.status}`);
      }
      const priceResult = await priceResponse.json();
      setPriceData(priceResult.data || []);

      // Fetch events
      const eventsResponse = await fetch(`${API_BASE_URL}/events`);
      if (!eventsResponse.ok) {
        throw new Error(`Events API failed: ${eventsResponse.status}`);
      }
      const eventsResult = await eventsResponse.json();
      setEventsData(eventsResult.events || []);

      // Fetch change points
      const cpResponse = await fetch(`${API_BASE_URL}/change_points`);
      if (cpResponse.ok) {
        const cpResult = await cpResponse.json();
        setChangePoints(cpResult);
      }

      // Fetch metrics
      const metricsResponse = await fetch(`${API_BASE_URL}/analysis/metrics`);
      if (metricsResponse.ok) {
        const metricsResult = await metricsResponse.json();
        setMetrics(metricsResult);
      }

    } catch (error) {
      console.error('Error fetching data:', error);
      // Set default empty states
      setPriceData([]);
      setEventsData([]);
      setMetrics({});
    } finally {
      setLoading(false);
    }
  }, [dateRange.start, dateRange.end]);

  // Fetch initial data
  useEffect(() => {
    fetchData();
  }, [fetchData]); // Now fetchData is in dependencies

  const handleDateChange = (dates) => {
    const [start, end] = dates;
    if (start && end) {
      setDateRange({ start, end });
    }
  };

  const handleEventSelect = async (event, index) => {
    setSelectedEvent(event);
    setEventImpact(null); // Reset previous impact
    
    try {
      const response = await fetch(`${API_BASE_URL}/analysis/event_impact/${index}`);
      if (response.ok) {
        const impactData = await response.json();
        setEventImpact(impactData);
      }
    } catch (error) {
      console.error('Error fetching event impact:', error);
    }
  };

  // Prepare data for charts - safely handle missing data
  const prepareChartData = () => {
    if (!Array.isArray(priceData) || priceData.length === 0) {
      return [];
    }
    
    return priceData
      .filter(item => item && item.date && item.price != null)
      .map(item => ({
        date: item.date,
        price: parseFloat(item.price) || 0,
        logReturn: parseFloat(item.log_return) || 0
      }));
  };

  // Combine price data with events
  const combinedData = prepareChartData().map(item => {
    const event = eventsData.find(e => e && e.date === item.date);
    return {
      ...item,
      eventType: event ? event.event_type : null,
      eventDescription: event ? event.event_description : null,
      hasEvent: !!event
    };
  });

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <div className="loading-text">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="header">
        <h1>Brent Oil Price Analysis Dashboard</h1>
        <div className="date-picker-container">
          <div className="date-range-label">
            <label>Date Range:</label>
            <DatePicker
              selected={dateRange.start}
              onChange={handleDateChange}
              startDate={dateRange.start}
              endDate={dateRange.end}
              selectsRange
              dateFormat="yyyy-MM-dd"
              className="date-picker"
              maxDate={new Date()}
              isClearable
              placeholderText="Select date range"
            />
          </div>
          <button onClick={fetchData} className="refresh-btn">
            Refresh Data
          </button>
        </div>
      </header>

      <div className="metrics-panel">
        <div className="metric-card">
          <h3>Current Price</h3>
          <p className="metric-value">
            {metrics.current_price ? `$${metrics.current_price.toFixed(2)}` : 'N/A'}
          </p>
        </div>
        <div className="metric-card">
          <h3>Average Price</h3>
          <p className="metric-value">
            {metrics.avg_price ? `$${metrics.avg_price.toFixed(2)}` : 'N/A'}
          </p>
        </div>
        <div className="metric-card">
          <h3>Volatility</h3>
          <p className="metric-value">
            {metrics.price_volatility ? `${(metrics.price_volatility * 100).toFixed(2)}%` : 'N/A'}
          </p>
        </div>
        <div className="metric-card">
          <h3>Total Events</h3>
          <p className="metric-value">{metrics.total_events || 0}</p>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart-card">
          <div className="chart-header">
            <h2>Historical Brent Oil Prices</h2>
            <span className="chart-subtitle">
              {combinedData.length > 0 ? 
                `${combinedData.length} data points from ${combinedData[0].date} to ${combinedData[combinedData.length - 1].date}` : 
                'No data available'}
            </span>
          </div>
          {combinedData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart 
                data={combinedData}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="date" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  tickFormatter={(date) => {
                    try {
                      return new Date(date).toLocaleDateString('en-US', { 
                        year: 'numeric', 
                        month: 'short', 
                        day: 'numeric' 
                      });
                    } catch {
                      return date;
                    }
                  }}
                  stroke="#666"
                />
                <YAxis 
                  label={{ 
                    value: 'Price (USD)', 
                    angle: -90, 
                    position: 'insideLeft',
                    offset: -10,
                    style: { textAnchor: 'middle' }
                  }}
                  stroke="#666"
                  tickFormatter={(value) => `$${value.toFixed(2)}`}
                />
                <Tooltip 
                  formatter={(value) => {
                    if (value === null || value === undefined || isNaN(value)) {
                      return ['N/A', 'Price'];
                    }
                    return [`$${parseFloat(value).toFixed(2)}`, 'Price'];
                  }}
                  labelFormatter={(label) => {
                    try {
                      return new Date(label).toLocaleDateString('en-US', { 
                        weekday: 'short',
                        year: 'numeric', 
                        month: 'long', 
                        day: 'numeric' 
                      });
                    } catch {
                      return label;
                    }
                  }}
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #ccc',
                    borderRadius: '4px'
                  }}
                />
                <Legend 
                  verticalAlign="top" 
                  height={36}
                  iconType="circle"
                />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6, strokeWidth: 0 }}
                  name="Brent Oil Price"
                  connectNulls
                />
                {combinedData.some(d => d.hasEvent) && (
                  <Scatter
                    data={combinedData.filter(d => d.hasEvent)}
                    fill="#ff7300"
                    shape="triangle"
                    name="Historical Events"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="no-data-message">
              <p>No price data available for the selected date range.</p>
              <button onClick={fetchData} className="retry-btn">
                Try Loading Data
              </button>
            </div>
          )}
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <h2>Price Log Returns (Volatility)</h2>
            <span className="chart-subtitle">
              Daily log returns showing price volatility over time
            </span>
          </div>
          {combinedData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart 
                data={combinedData}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="date" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  tickFormatter={(date) => {
                    try {
                      return new Date(date).toLocaleDateString('en-US', { 
                        year: 'numeric', 
                        month: 'short', 
                        day: 'numeric' 
                      });
                    } catch {
                      return date;
                    }
                  }}
                  stroke="#666"
                />
                <YAxis 
                  label={{ 
                    value: 'Log Return', 
                    angle: -90, 
                    position: 'insideLeft',
                    offset: -10,
                    style: { textAnchor: 'middle' }
                  }}
                  stroke="#666"
                />
                <Tooltip 
                  formatter={(value) => {
                    if (value === null || value === undefined || isNaN(value)) {
                      return ['N/A', 'Log Return'];
                    }
                    return [parseFloat(value).toFixed(6), 'Log Return'];
                  }}
                  labelFormatter={(label) => {
                    try {
                      return new Date(label).toLocaleDateString('en-US', { 
                        weekday: 'short',
                        year: 'numeric', 
                        month: 'long', 
                        day: 'numeric' 
                      });
                    } catch {
                      return label;
                    }
                  }}
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #ccc',
                    borderRadius: '4px'
                  }}
                />
                <Legend 
                  verticalAlign="top" 
                  height={36}
                  iconType="circle"
                />
                <Bar 
                  dataKey="logReturn" 
                  fill="#82ca9d" 
                  name="Log Return"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="no-data-message">
              <p>No return data available for the selected date range.</p>
            </div>
          )}
        </div>
      </div>

      <div className="events-panel">
        <div className="events-list">
          <div className="section-header">
            <h2>Key Historical Events</h2>
            <span className="section-subtitle">
              {eventsData.length} events loaded
            </span>
          </div>
          {eventsData.length > 0 ? (
            <div className="events-table-container">
              <table className="events-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Impact</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {eventsData.map((event, index) => {
                    if (!event) return null;
                    return (
                      <tr 
                        key={`${event.date}-${index}`}
                        className={selectedEvent?.date === event.date ? 'selected' : ''}
                      >
                        <td>{event.date || 'Unknown'}</td>
                        <td>
                          <span className="event-type-badge">
                            {event.event_type || 'Unknown'}
                          </span>
                        </td>
                        <td className="event-description">
                          {event.event_description || 'No description'}
                        </td>
                        <td>
                          <span className={`impact-badge ${event.impact_direction || 'neutral'}`}>
                            {event.impact_direction || 'neutral'}
                          </span>
                        </td>
                        <td>
                          <button 
                            onClick={() => handleEventSelect(event, index)}
                            className="analyze-btn"
                            disabled={!event.date}
                          >
                            Analyze Impact
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="no-data-message">
              <p>No events data available.</p>
            </div>
          )}
        </div>

        {eventImpact && (
          <div className="event-impact">
            <div className="section-header">
              <h2>Event Impact Analysis</h2>
              {eventImpact.error && (
                <span className="error-message">{eventImpact.error}</span>
              )}
            </div>
            <div className="impact-metrics">
              <div className="impact-metric">
                <h3>Event</h3>
                <p className="impact-value">{eventImpact.event?.event_description || 'Unknown event'}</p>
              </div>
              <div className="impact-metric">
                <h3>Pre-Event Average</h3>
                <p className="impact-value">
                  {eventImpact.pre_event_avg ? `$${eventImpact.pre_event_avg.toFixed(2)}` : 'N/A'}
                </p>
              </div>
              <div className="impact-metric">
                <h3>Post-Event Average</h3>
                <p className="impact-value">
                  {eventImpact.post_event_avg ? `$${eventImpact.post_event_avg.toFixed(2)}` : 'N/A'}
                </p>
              </div>
              <div className="impact-metric">
                <h3>Price Change</h3>
                <p className={`impact-value ${
                  eventImpact.price_change_pct > 0 ? 'positive' : 
                  eventImpact.price_change_pct < 0 ? 'negative' : 'neutral'
                }`}>
                  {eventImpact.price_change_pct != null ? 
                    `${eventImpact.price_change_pct > 0 ? '+' : ''}${eventImpact.price_change_pct.toFixed(2)}%` : 
                    'N/A'}
                </p>
              </div>
            </div>
            {eventImpact.pre_event_avg && eventImpact.post_event_avg && (
              <div className="impact-summary">
                <p>
                  The {eventImpact.event?.event_type?.toLowerCase() || 'event'} on {eventImpact.event?.date || 'this date'} 
                  {' '}resulted in a {eventImpact.price_change_pct > 0 ? 'price increase' : 'price decrease'} of{' '}
                  {Math.abs(eventImpact.price_change_pct || 0).toFixed(2)}%.
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {changePoints && changePoints.change_date && (
        <div className="change-points-panel">
          <div className="section-header">
            <h2>Change Point Analysis Results</h2>
            <span className="section-subtitle">
              Bayesian change point detection results
            </span>
          </div>
          <div className="cp-content">
            <div className="cp-metric">
              <h3>Most Probable Change Date</h3>
              <p className="cp-value">{changePoints.change_date || 'N/A'}</p>
            </div>
            <div className="cp-metric">
              <h3>Mean Before Change</h3>
              <p className="cp-value">
                {changePoints.impact_metrics?.mean_before ? 
                  `$${changePoints.impact_metrics.mean_before.toFixed(2)}` : 'N/A'}
              </p>
            </div>
            <div className="cp-metric">
              <h3>Mean After Change</h3>
              <p className="cp-value">
                {changePoints.impact_metrics?.mean_after ? 
                  `$${changePoints.impact_metrics.mean_after.toFixed(2)}` : 'N/A'}
              </p>
            </div>
            <div className="cp-metric">
              <h3>Percentage Change</h3>
              <p className={`cp-value ${
                changePoints.impact_metrics?.percent_change > 0 ? 'positive' : 
                changePoints.impact_metrics?.percent_change < 0 ? 'negative' : 'neutral'
              }`}>
                {changePoints.impact_metrics?.percent_change != null ? 
                  `${changePoints.impact_metrics.percent_change > 0 ? '+' : ''}${changePoints.impact_metrics.percent_change.toFixed(2)}%` : 
                  'N/A'}
              </p>
            </div>
          </div>
          {changePoints.nearby_events && changePoints.nearby_events.length > 0 && (
            <div className="nearby-events">
              <h3>Nearby Historical Events (±30 days)</h3>
              <ul>
                {changePoints.nearby_events.map((event, index) => (
                  <li key={index}>
                    <strong>{event.date}</strong>: {event.event_type} - {event.event_description}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;