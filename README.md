# MagicBricks Real Estate Analysis & Visualization 🏠📊

This repository contains a complete end-to-end data science project focused on the Indian rental real estate market. The project involves scraping, cleaning, and analyzing property data from **MagicBricks** for five major Indian metros: **Bangalore, Chennai, Hyderabad, Mumbai, and Pune**.

## � Project Structure

Collaborative analysis is split into two primary notebooks representing the data pipeline:

### 1. [EDAProject.ipynb](./EDAProject.ipynb) (Data Acquisition & Cleaning)
This notebook focuses on the technical foundation of the project:
- **Web Scraping**: Custom scraper built with `BeautifulSoup` and `requests` to extract ~1,500 property listings.
- **Data Engineering**: Standardizing raw text data (converting "Lac/Cr" to numeric, extracting area from strings).
- **Quality Control**: Handling missing values, removing duplicates, and standardizing city-wise neighborhood names.
- **Output**: Generates the `MagicBricksProject_Scraped_Data.csv` and initiates the transition to the cleaned dataset.

### 2. [Data Analysis and Visualization .ipynb](./Data Analysis and Visualization .ipynb) (Deep-Dive Analysis)
This notebook extracts market intelligence through exploratory data analysis and high-impact visualizations:
- **Statistical EDA**: Summary statistics and outlier detection across cities.
- **Visual Insights**:
    - **Price Distribution**: Identifying the "Mumbai Premium" vs. more affordable markets.
    - **Physical Attributes**: Analyzing the correlation between Area, BHK, and Bathrooms.
    - **Preference Patterns**: Visualizing Tenant Preferences and Furnishing trends.
    - **Geospatial Hotspots**: Mapping Top 10 Most and Least expensive localities.
- **Correlation Mapping**: Using heatmaps and pairplots to understand the variables most impacting rental valuation.

## 🚀 Key Insights
- **The Mumbai Premium**: Mumbai's average rent per sqft is nearly **3x** that of Hyderabad, despite similar property configurations.
- **Standardization**: High correlation (0.86) between BHK and Bathrooms confirms standard urban apartment designs.
- **Vastu Trends**: A massive inventory skew towards **East-facing** properties reflects strong cultural buyer/renter demand.
- **Market Liquidity**: Over **90%** of listings are available for immediate possession.

## �️ Tech Stack
- **Web Scraping**: `BeautifulSoup`, `requests`, `re`
- **Data Science**: `pandas`, `numpy`
- **Visualization**: `seaborn`, `matplotlib`
- **Utility**: `time`, `warnings`

## 🧹 Dataset Info
The final dataset consists of **1,233+ clean listings** with features including:
- BHK, Price, Area (sqft)
- Property Type (Flat/House/Villa)
- Furnishing Status
- Facing Direction & Overlooking Views
- Bathroom/Balcony Counts
- Tenant Preference (Family/Bachelors)

## 🚀 How to Use
1. **Data Gathering**: Run `EDAProject.ipynb` if you wish to see how the scraping and cleaning logic was built.
2. **Analysis**: Run `Data Analysis and Visualization .ipynb` to generate the market analysis and interactive plots.

---
**Author**: [Kunal]
*Transforming raw real estate listings into actionable market insights.*

