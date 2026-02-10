import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, ScatterChart, Scatter
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

  // Fetch initial data
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    
    try {
      // Fetch prices
      const priceResponse = await fetch(
        `${API_BASE_URL}/prices?start_date=${dateRange.start.toISOString().split('T')[0]}&end_date=${dateRange.end.toISOString().split('T')[0]}`
      );
      const priceResult = await priceResponse.json();
      setPriceData(priceResult.data);

      // Fetch events
      const eventsResponse = await fetch(`${API_BASE_URL}/events`);
      const eventsResult = await eventsResponse.json();
      setEventsData(eventsResult.events);

      // Fetch change points
      const cpResponse = await fetch(`${API_BASE_URL}/change_points`);
      const cpResult = await cpResponse.json();
      setChangePoints(cpResult);

      // Fetch metrics
      const metricsResponse = await fetch(`${API_BASE_URL}/analysis/metrics`);
      const metricsResult = await metricsResponse.json();
      setMetrics(metricsResult);

    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDateChange = (dates) => {
    const [start, end] = dates;
    setDateRange({ start, end });
  };

  const handleEventSelect = async (event, index) => {
    setSelectedEvent(event);
    
    try {
      const response = await fetch(`${API_BASE_URL}/analysis/event_impact/${index}`);
      const impactData = await response.json();
      setEventImpact(impactData);
    } catch (error) {
      console.error('Error fetching event impact:', error);
    }
  };

  // Prepare data for charts
  const prepareChartData = () => {
    return priceData.map(item => ({
      date: item.date,
      price: item.price,
      logReturn: item.log_return
    }));
  };

  // Combine price data with events
  const combinedData = prepareChartData().map(item => {
    const event = eventsData.find(e => e.date === item.date);
    return {
      ...item,
      eventType: event ? event.event_type : null,
      eventDescription: event ? event.event_description : null,
      hasEvent: !!event
    };
  });

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  return (
    <div className="App">
      <header className="header">
        <h1>Brent Oil Price Analysis Dashboard</h1>
        <div className="date-picker-container">
          <DatePicker
            selected={dateRange.start}
            onChange={handleDateChange}
            startDate={dateRange.start}
            endDate={dateRange.end}
            selectsRange
            dateFormat="yyyy-MM-dd"
            className="date-picker"
          />
          <button onClick={fetchData} className="refresh-btn">
            Refresh Data
          </button>
        </div>
      </header>

      <div className="metrics-panel">
        <div className="metric-card">
          <h3>Current Price</h3>
          <p className="metric-value">
            ${metrics.current_price ? metrics.current_price.toFixed(2) : 'N/A'}
          </p>
        </div>
        <div className="metric-card">
          <h3>Average Price</h3>
          <p className="metric-value">
            ${metrics.avg_price ? metrics.avg_price.toFixed(2) : 'N/A'}
          </p>
        </div>
        <div className="metric-card">
          <h3>Volatility</h3>
          <p className="metric-value">
            {metrics.price_volatility ? (metrics.price_volatility * 100).toFixed(2) + '%' : 'N/A'}
          </p>
        </div>
        <div className="metric-card">
          <h3>Total Events</h3>
          <p className="metric-value">{metrics.total_events || 'N/A'}</p>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart-card">
          <h2>Historical Brent Oil Prices</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={combinedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                angle={-45}
                textAnchor="end"
                height={80}
                tickFormatter={(date) => new Date(date).toLocaleDateString()}
              />
              <YAxis 
                label={{ value: 'Price (USD)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => [`$${value.toFixed(2)}`, 'Price']}
                labelFormatter={(label) => new Date(label).toLocaleDateString()}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#8884d8" 
                strokeWidth={2}
                dot={false}
                name="Price"
              />
              <Scatter
                data={combinedData.filter(d => d.hasEvent)}
                fill="#ff7300"
                shape="triangle"
                name="Events"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h2>Price Log Returns (Volatility)</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={combinedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                angle={-45}
                textAnchor="end"
                height={80}
                tickFormatter={(date) => new Date(date).toLocaleDateString()}
              />
              <YAxis 
                label={{ value: 'Log Return', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => [value.toFixed(4), 'Log Return']}
                labelFormatter={(label) => new Date(label).toLocaleDateString()}
              />
              <Legend />
              <Bar 
                dataKey="logReturn" 
                fill="#82ca9d" 
                name="Log Return"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="events-panel">
        <div className="events-list">
          <h2>Key Historical Events</h2>
          <div className="events-table">
            <table>
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
                {eventsData.map((event, index) => (
                  <tr 
                    key={index}
                    className={selectedEvent?.date === event.date ? 'selected' : ''}
                  >
                    <td>{event.date}</td>
                    <td>{event.event_type}</td>
                    <td>{event.event_description}</td>
                    <td>
                      <span className={`impact-badge ${event.impact_direction}`}>
                        {event.impact_direction}
                      </span>
                    </td>
                    <td>
                      <button 
                        onClick={() => handleEventSelect(event, index)}
                        className="analyze-btn"
                      >
                        Analyze Impact
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {eventImpact && (
          <div className="event-impact">
            <h2>Event Impact Analysis</h2>
            <div className="impact-metrics">
              <div className="impact-metric">
                <h3>Event</h3>
                <p>{eventImpact.event?.event_description}</p>
              </div>
              <div className="impact-metric">
                <h3>Pre-Event Average</h3>
                <p>${eventImpact.pre_event_avg?.toFixed(2) || 'N/A'}</p>
              </div>
              <div className="impact-metric">
                <h3>Post-Event Average</h3>
                <p>${eventImpact.post_event_avg?.toFixed(2) || 'N/A'}</p>
              </div>
              <div className="impact-metric">
                <h3>Price Change</h3>
                <p className={eventImpact.price_change_pct > 0 ? 'positive' : 'negative'}>
                  {eventImpact.price_change_pct ? 
                    `${eventImpact.price_change_pct > 0 ? '+' : ''}${eventImpact.price_change_pct.toFixed(2)}%` : 
                    'N/A'}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {changePoints && (
        <div className="change-points-panel">
          <h2>Change Point Analysis Results</h2>
          <div className="cp-content">
            <div className="cp-metric">
              <h3>Most Probable Change Date</h3>
              <p>{changePoints.change_date || 'N/A'}</p>
            </div>
            <div className="cp-metric">
              <h3>Mean Before Change</h3>
              <p>${changePoints.impact_metrics?.mean_before?.toFixed(2) || 'N/A'}</p>
            </div>
            <div className="cp-metric">
              <h3>Mean After Change</h3>
              <p>${changePoints.impact_metrics?.mean_after?.toFixed(2) || 'N/A'}</p>
            </div>
            <div className="cp-metric">
              <h3>Percentage Change</h3>
              <p className={changePoints.impact_metrics?.percent_change > 0 ? 'positive' : 'negative'}>
                {changePoints.impact_metrics?.percent_change ? 
                  `${changePoints.impact_metrics.percent_change > 0 ? '+' : ''}${changePoints.impact_metrics.percent_change.toFixed(2)}%` : 
                  'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;