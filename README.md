# MagicBricks Real Estate Data Analysis & Visualization

## 📌 Project Overview
This project involves the comprehensive analysis and visualization of rental property data scraped from **MagicBricks**. The dataset covers real estate listings across **5 major Indian cities**: Hyderabad, Bangalore, Mumbai, Pune, and Chennai. 

The primary goal is to gain insights into the rental market, understand price trends, analyze property characteristics, and identify tenant preferences.

## 📂 Dataset
- **Source**: MagicBricks (Property for Rent)
- **Raw Data**: `MagicBricksProject_Scraped_Data.csv`
- **Cleaned Data**: `MagicBricksProject_Cleaned_Data.csv`

### Data Dictionary
| Column | Description |
|:--- |:--- |
| **City** | City where the property is located (e.g., Hyderabad, Mumbai) |
| **BHK** | Number of Bedrooms, Hall, and Kitchen |
| **Location** | Specific locality or neighborhood |
| **Price** | Monthly rental price (INR) |
| **Area (sqft)** | Built-up area in square feet |
| **Property Type** | Type of property (e.g., Flat, House, Villa) |
| **Furnishing** | Furnishing status (Furnished, Semi-Furnished, Unfurnished) |
| **Property Facing** | Cardinal direction the property faces |
| **Overlooking** | Views/Landmarks visible (e.g., Park, Main Road) |
| **Bathroom** | Number of bathrooms |
| **Balcony** | Number of balconies |
| **Tenant Preferred** | Preferred tenant type (Bachelors, Family) |
| **Availability** | Possession status (Immediately, From Date) |

## 🛠 Tech Stack
- **Language**: Python
- **Libraries**:
  - `pandas` & `numpy` (Data Manupulation)
  - `matplotlib` & `seaborn` (Visualization)
  - `BeautifulSoup` (Web Scraping - *used for data collection*)

## 📊 Key Visualizations & Insights
The project includes a series of visualizations to explore the data:

1.  **Rental Price by City**: Comparison of average rental prices across the simple cities.
2.  **Price Distribution**: Understanding the spread of rental prices.
3.  **BHK Distribution**: Most common property configurations (1BHK, 2BHK, etc.).
4.  **Price vs. Area**: Scatter plot analyzing the relationship between property size and rent.
5.  **Furnishing Analysis**: Impact of furnishing on rental value.
6.  **Property Type Distribution**: Breakdown of Flats vs. Independent Houses/Villas.
7.  **Tenant Preference**: Analysis of owner preferences for Bachelors vs. Families.
8.  **Property Facing**: Distribution of properties based on facing direction.
9.  **Price Heatmaps**: visual representation of price density.
10. **Bathroom & Balcony Analysis**: availability of amenities.
11. **Location Analysis**: 
    - Top Expensive Locations
    - Least Expensive Locations
12. **Correlation Matrix**: Heatmap showing correlations between numerical variables (Price, Area, BHK, etc.).
13. **Availability Status**: Proportion of properties available immediately vs. later.
14. **Price Per Sqft**: Analysis of the cost efficiency of rental spaces.

## 🚀 How to Run
1.  Ensure you have Python installed along with the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Data Analysis and Visualization .ipynb"
    ```
3.  Run the cells sequentially to reproduce the analysis and generate visualizations.

## 📷 Saved Visualizations
All generated plots are saved as PNG files in the project directory (e.g., `1_rental_price_by_city.png`, `12_correlation_heatmap.png`).
