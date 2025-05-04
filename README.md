

## 📚 Table of Contents

- [ Project Data Analytics: Brazilian E-Commerce Public Dataset by Olist ](#-project-data-analytics-brazilian-e-commerce-public-dataset-by-olist-)
  - [📚 Table of Contents](#-table-of-contents)
  - [🎯 Introduction](#-introduction)
  - [💻 Installation](#-installation)
  - [🔄 Project Workflow](#-project-workflow)
  - [📈 Dashboard Explanation](#-dashboard-explanation)


## 🎯 Introduction

This project focuses on the analysis of a public dataset provided by Olist, a Brazilian e-commerce company. The dataset includes information about customer orders, products, payments, reviews, and more. The goal of this project is to extract meaningful insights from the data that can help improve business strategies and customer experience.

## 💻 Installation

Follow these steps to get the project up and running on your local machine:

1. **Clone the repository**

   Clone the repository using git:

   ```bash
   git clone [https://github.com/adityasinha27/Brazilian_ecommerce.git]
   ```

2. **Set up a virtual environment** (Optional)

   It's recommended to create a virtual environment to keep the dependencies required by this project separate from your system's Python environment. Here's how you can create a virtual environment:

   ```bash
   python3 -m venv env
   ```

   Activate the virtual environment:

   On Windows:

   ```bash
   .\env\Scripts\activate
   ```

   On Unix or MacOS:

   ```ls
   source env/bin/activate
   ```

3. **Install the dependencies**

   Navigate to the project directory and install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project**

   Now you can run the project easily by click on the run all button in the jupyter notebook

## 🔄 Project Workflow

1. **Data Wrangling**
    
   The first step of the project is to clean and prepare the data for analysis. The data wrangling process includes:

   - Handling missing values
   - Handling duplicate values
   - Handling incorrect data types
   - Handling outliers
   - Handling inconsistent data
   - Handling data redundancy
   - Handling data normalization

2. **Exploratory Data Anlaysis**

    The second step of the project is to explore the data and extract meaningful insights. The exploratory data analysis process includes:
    
    - Order Analysis
    - Product Analysis
    - Payment Analysis
    - Customer Analysis
    - Reviews Analysis
    - Sellers Analysis

3. **Data Visualization**
   
    The third step of the project is to visualize the data and extract meaningful insights. The data visualization process includes:
    
    - Product Category with most outstanding reviews given by customers
    - Visualize OTD(On Time Delivery) rate of orders
    - Visualize the demographics of customers
    - Visualize the frequency of purchases made by customers
    - Visualize the monetary value of purchases made by customers
    - Visualize the recent purchases made by customers
    - Visualize the customers segmentation based on RFM analysis
  
4. **Dashboard**

    The last step of the project is to create a dashboard that can be used to visualize the data and extract meaningful insights. The dashboard includes:
    
    - Product Category sales performance
    - Daily sales performance
  
   Key Performance Indicators (KPIs): Display total sales, average rating, and average sales per order.

      Visualizations:
      
      Sales by Product Line: Horizontal bar chart showcasing sales distribution across product categories.
      
      Daily Sales: Bar chart illustrating sales trends over days.
      
      Predictive Analytics:
      
      Predict total order price based on input features like freight value and price.
      
      Estimate delivery time using a Random Forest Regressor.
      
      Classify review scores using a Random Forest Classifier.
      
      Monthly Sales Forecasting: Utilize an LSTM model to forecast sales for the upcoming month.
      
      Data Export: Download prediction results in CSV or Excel formats.
      
      Dark Mode UI: Enhanced user interface with a dark theme for better readability

## 📈 Dashboard Explanation

Follow these steps to run the dashboard on your local machine:

1. **Run the Streamlit app**

   Navigate to the project directory and run the Streamlit app using the following command:

   ```sh
   streamlit run dashboard/dashboard.py
   ```

   This will start the Streamlit server and open a new page in your default web browser with the URL of the Streamlit app.

   

2. **Use the dashboard**

The dashboard provides a visual representation of sales performance data. Here's how to use it:

- **Filter the data**

   Use the filters in the sidebar to select the city, customer segmentation, and date range you're interested in. The dashboard will update automatically to reflect your selections.

- **View key performance indicators**

   At the top of the dashboard, you'll see key performance indicators (KPIs) such as total sales, average rating, and average sales per order.

- **Explore the charts**

  
![Screenshot 2025-04-25 074332](https://github.com/user-attachments/assets/da77a828-b9c4-47dc-8005-21c3a21910e1)

![Screenshot 2025-05-01 135907](https://github.com/user-attachments/assets/81c0400e-6b58-4c6f-a599-7b4c5bb90581)

![f6f921c4e650d74d37c1b9329ba0c1ba8d8045faae554661054e37d6](https://github.com/user-attachments/assets/9e5c4c8b-1e32-42c3-8a8d-949e601ffc0d)


   The dashboard includes two charts: Sales by Product Line and Sales by Day. These charts provide a visual representation of the sales data based on your filter selections.

```python
print("Thank you for reading! 🙏")
```

**Thank you for reading! 🙏**


