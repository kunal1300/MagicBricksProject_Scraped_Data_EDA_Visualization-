# MagicBricks Real Estate Rental Analysis 🏠

A comprehensive end-to-end data science project that analyzes rental property data across major Indian cities using web scraping, data cleaning, and advanced visualization techniques.

## 📊 Project Overview

This project performs an in-depth analysis of residential rental properties across five major Indian cities: **Bangalore, Chennai, Hyderabad, Mumbai, and Pune**. The analysis provides insights into rental prices, property characteristics, tenant preferences, and market trends.

## 🎯 Key Features

- **Web Scraping**: Automated data collection from MagicBricks using BeautifulSoup
- **Data Cleaning**: Comprehensive preprocessing including duplicate removal, price standardization, and location normalization
- **Exploratory Data Analysis**: Statistical analysis of 1,271 rental properties across 13 features
- **Advanced Visualizations**: 15+ detailed visualizations including heatmaps, scatter plots, box plots, and distribution analyses

## 📁 Project Structure

```
├── EDAProject.ipynb                              # Data scraping & cleaning pipeline
├── Data_Analysis_and_Visualization_.ipynb        # Analysis & visualization
├── MagicBricksProject_Scraped_Data.csv          # Raw scraped data
└── MagicBricksProject_Cleaned_Data.csv          # Processed dataset
```

## 🔧 Technologies Used

- **Python 3.x**
- **Data Collection**: BeautifulSoup, Requests
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Analysis**: Statistical methods, correlation analysis

## 📈 Dataset Features

The cleaned dataset contains **1,271 properties** with the following attributes:

| Feature | Description |
|---------|-------------|
| City | Location city (Bangalore, Chennai, Hyderabad, Mumbai, Pune) |
| BHK | Number of bedrooms (1 BHK to 4+ BHK) |
| Location | Specific locality within the city |
| Price (₹) | Monthly rental price in INR |
| Area (sqft) | Property area in square feet |
| Property Type | Flat, House, Villa, etc. |
| Furnishing | Furnished, Semi-Furnished, Unfurnished |
| Property Facing | Direction the property faces |
| Overlooking | View from the property |
| Bathroom | Number of bathrooms |
| Balcony | Number of balconies |
| Tenant Preferred | Type of tenants preferred |
| Availability | Property availability status |

## 🔍 Analysis Highlights

### 1. **Price Analysis**
- City-wise average rental price comparison
- Top 5 most and least expensive locations per city
- Price distribution across different property types
- Price vs. area correlation analysis

### 2. **Property Characteristics**
- BHK distribution across cities
- Furnishing status impact on pricing
- Property type distribution (Flat, House, Villa)
- Bathroom and balcony analysis

### 3. **Market Insights**
- Tenant preference patterns
- Property facing direction trends
- Correlation matrix for numeric features
- Multi-variable relationship analysis using pairplots

### 4. **Visualizations**
- **Heatmaps**: BHK vs City rental prices, correlation matrices
- **Box & Violin Plots**: Price distributions with outlier detection
- **Scatter Plots**: Price vs area relationships with city-wise coloring
- **Bar Charts**: City-wise comparisons, property type distributions
- **Count Plots**: Categorical feature distributions

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy seaborn matplotlib beautifulsoup4 requests
```

### Running the Project

1. **Data Collection & Cleaning**:
   ```bash
   jupyter notebook EDAProject.ipynb
   ```
   - Scrapes data from MagicBricks
   - Cleans and preprocesses the dataset
   - Saves cleaned data to CSV

2. **Analysis & Visualization**:
   ```bash
   jupyter notebook Data_Analysis_and_Visualization_.ipynb
   ```
   - Loads cleaned dataset
   - Performs exploratory data analysis
   - Generates comprehensive visualizations

## 📊 Key Findings

- **Mumbai** has the highest average rental prices among all cities
- **3 BHK properties** are the most common configuration across cities
- **Furnished properties** command premium pricing compared to unfurnished ones
- **Flats** are the most common property type, ideal for city living
- Strong positive correlation between **property area and rental price**
- **Family tenants** are the most preferred tenant type across cities

## 🛠️ Data Cleaning Process

The data cleaning pipeline includes:

1. **Duplicate Removal**: Eliminated duplicate entries based on key features
2. **Price Standardization**: Converted price strings (Lac, Cr) to numeric values
3. **Location Normalization**: Standardized locality names across cities
4. **Missing Value Handling**: Addressed null values in critical columns
5. **Data Type Conversion**: Ensured proper data types for analysis
6. **Outlier Detection**: Identified and documented extreme values

## 📝 Data Sources

- **Primary Source**: [MagicBricks](https://www.magicbricks.com) - India's leading real estate marketplace
- **Collection Method**: Web scraping with ethical rate limiting
- **Time Period**: Current listings (snapshot-based)

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end data science workflow
- Web scraping best practices
- Data cleaning and preprocessing techniques
- Statistical analysis and correlation studies
- Advanced data visualization with Seaborn and Matplotlib
- Real-world data handling challenges

## 📌 Future Enhancements

- [ ] Add time-series analysis for price trends
- [ ] Implement machine learning models for price prediction
- [ ] Create interactive dashboards using Plotly or Dash
- [ ] Expand dataset to include more cities
- [ ] Add amenities analysis (parking, gym, swimming pool)
- [ ] Perform geospatial analysis with property location mapping

## 📄 License

This project is available for educational and research purposes.

## 👤 Author

**KUNAL SOLANKI**

## 🙏 Acknowledgments

- MagicBricks for providing the rental property data
- Python data science community for excellent libraries
- Open-source contributors

---

