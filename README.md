# Brent Oil Price Analysis: Structural Break Detection

## Project Overview

This project analyzes historical Brent crude oil prices (1987-2022) to identify structural breaks and assess the impact 

of geopolitical events, OPEC decisions, and economic shocks using Bayesian change point detection.

## Key Objectives

1. Perform comprehensive time series analysis of Brent oil prices

2. Identify structural breaks using Bayesian change point detection with PyMC

3. Correlate detected change points with historical events

4. Generate actionable insights for energy market stakeholders

## Data Description

- **Primary Dataset**: Daily Brent oil prices from May 20, 1987 to September 30, 2022

  - Date: Day-month-year format (e.g., 20-May-87)

  - Price: USD per barrel

- **Event Dataset**: 15 key geopolitical, economic, and OPEC events with dates and impact classifications

## Project Structure

project-root/

├── README.md # This file

├── docs/ # Documentation

│ ├── analysis_workflow.docx # Analysis methodology

│ └── assumptions_limitations.txt # Constraints and caveats

├── data/ # Data files

│ ├── brent_oil_prices.csv # Main price dataset

│ └── events.csv # Historical events

├── notebooks/ # Jupyter notebooks

│ └── exploratory_EDA.ipynb # Exploratory data analysis

└── src/ # Source code

└── change_point_model.py # Change point detection implementation


## Installation & Setup

### Prerequisites

- Python 3.8+

- Git

### Environment Setup

# Clone repository

git clone https://github.com/Saronzeleke/Brent-oil-analysis.git

cd Brent-oil-analysis

# Create virtual environment

python -m venv my_env

# Activate virtual environment

# On Windows:

my_env\Scripts\activate

# On Mac/Linux:

source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

Required Packages

Create requirements.txt:


# Usage

1. Exploratory Data Analysis

jupyter notebook notebooks/exploratory_EDA.ipynb

2. Run Change Point Detection

python src/change_point_model.py

3. Generate Reports

Review docs/analysis_workflow.docx for methodology

Check docs/assumptions_limitations.txt for analysis constraints

# Analysis Workflow

Data Loading & Validation: Load and clean historical price data

Time Series Analysis: Trend, stationarity, and volatility assessment

Event Research: Compile and categorize historical events

Change Point Detection: Bayesian structural break identification

Impact Analysis: Correlate breaks with historical events

Insight Generation: Produce stakeholder reports

# Key Deliverables

Complete time series analysis with visualizations

Bayesian change point detection model

Event-impact correlation analysis

Comprehensive documentation

# Communication Channels

Technical Reports: Detailed Jupyter notebooks

Executive Summaries: 1-2 page PDF documents

Interactive Visualizations: Plotly dashboards

Repository Documentation: GitHub README and wiki

# Contributing

Fork the repository

Create a feature branch

Commit changes with clear messages

Push to the branch

Open a Pull Request

# License

MIT License - see LICENSE file for details

# Contact

Email: Sharonkuye369@gmail.com