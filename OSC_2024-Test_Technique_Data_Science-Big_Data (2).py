#!/usr/bin/env python
# coding: utf-8

# # SHOP SMART

# # Data Preparation and Exploration

# ## Importing the data

# In[4]:


#import libraries
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_squared_error
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[5]:


# Load CSV files into DataFrames
store = pd.read_excel(r"C:\Users\Yong Justice\OSC 2024 Summer Challenge Data Science Big Data\Online Retail.xlsx")


# ### Data Inspection

# In[7]:


#Data_Inspection
store.shape
store.info()
store.head()


# In[8]:


#handle missing values
store["Description"].fillna(value='No Description', inplace=True)


# In[9]:


store.dropna(inplace=True)


# In[10]:


#Convert data types
store['InvoiceNo'] = pd.to_numeric(store['InvoiceNo'], errors='coerce')


# In[11]:


#Create New Features
store["TotalSpent"]=store["Quantity"] * store["UnitPrice"]
store['PurchaseFrequency'] = store.groupby('CustomerID')['InvoiceNo'].transform('count')
store['Month'] = store['InvoiceDate'].dt.month
# Convert DayOfWeek (numeric) to DayName (string)
store['Day_of_the_week'] = pd.to_datetime(store['InvoiceDate']).dt.strftime('%A') 
store['Month_Name'] = pd.to_datetime(store['InvoiceDate']).dt.strftime('%B')
store["DayOfWeek"] = pd.to_datetime(store["InvoiceDate"]).dt.dayofweek


# ## Exploratory Data Analysis(EDA)

# In[13]:


# Print object type, shape, and head
print("store type:", type(store))
print("store shape:", store.shape)
store.head()


# In[14]:


#Descriptive Statistics
store.describe()


# ##### Univariate Analysis(Visuals)

# In[16]:


# Top 10 Products 
top_products = store['Description'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar')
plt.title('Top 10 Products Sold')
plt.xlabel('Product')
plt.ylabel('Number of Times Purchased')
plt.xticks(rotation=45, ha='right')
plt.show()
plt.savefig('top_products.png')


# In[49]:


# Time-Based Product Trends
# Ensure datetime
store['InvoiceDate'] = pd.to_datetime(store['InvoiceDate']) 

# Monthly Trend of a Specific Product 
product_name = 'WHITE METAL LANTERN' 
product_data = store[store['Description'] == product_name]
monthly_trend = product_data.groupby(product_data['InvoiceDate'].dt.month).size()

plt.figure(figsize=(10, 5))
monthly_trend.plot(kind='line', marker='o')
plt.title(f'Monthly Sales Trend for "{product_name}"')
plt.xlabel('Month')
plt.ylabel('Number of Sales')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                              'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()
   
plt.savefig('trend.png')


# In[51]:


plt.figure(figsize=(8, 5))
plt.hist(store['UnitPrice'], bins=10)  

plt.title('Price Distribution of Products')
plt.xlabel('Unit Price')
plt.ylabel('Frequency') 



plt.show() 
plt.savefig('productprice.png')


# In[53]:


country_counts = store['Country'].value_counts()

plt.figure(figsize=(10, 6))
plt.barh(country_counts.index, country_counts.values)

plt.title('Highest Customer Country')
plt.xlabel('Number of Purchases')
plt.ylabel('Country')


plt.show()
plt.savefig('highst customer.png')


# ## Build Model

# In[55]:


# Prepare data for clustering (select relevant features)
features_for_clustering = ['TotalSpent', 'PurchaseFrequency']
X = store[features_for_clustering]

# Determine optimal number of clusters (elbow method)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Apply KMeans clustering 
kmeans = KMeans(n_clusters=3, random_state=42) # Choose optimal k based on elbow method
store['Cluster'] = kmeans.fit_predict(X)

# Analyze and visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PurchaseFrequency', y='TotalSpent', hue='Cluster', data=store, palette='viridis')
plt.title('Customer Segmentation')
plt.show()
plt.savefig('elbow_k.png')


# ###  Predictive Modeling

# In[23]:


#  I'll demonstrate regression

# Select features and target variable 
features = ['PurchaseFrequency', 'DayOfWeek', 'Month', 'Cluster'] 
target = 'TotalSpent'

X = store[features]
y = store[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model 
rmse = mean_squared_error(y_test, y_pred, squared=False) 
print('RMSE:', rmse) 


# In[56]:


# 1. Predictions vs. Actual Values (Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)  # Alpha for transparency
plt.xlabel('Actual Total Spent')
plt.ylabel('Predicted Total Spent')
plt.title('Actual vs. Predicted Total Spent')

#diagonal line (perfect predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         linestyle='--', color='red', linewidth=2)
plt.show() 
plt.savefig('diagonal.png')
# 2. Residual Plot (Analyzing Errors)
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Total Spent')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at zero
plt.show()
plt.savefig('residual.png')
# 3. Distribution of Residuals (Histogram)
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=10)
plt.title('Distribution of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
plt.savefig('hist.png')


# # Interpretation and Recommendations

# ##### This document presents the findings and recommendations derived from analyzing ShopSmart's customer data. We used Python along with libraries like Pandas, Matplotlib, and Scikit-learn to conduct our analysis.
# 
# 1. Exploratory Data Analysis (EDA): Key Findings
#  Top Products: We identified the top-selling products . This informs inventory management and potential promotional strategies.
# Customer Spending: Describe the distribution of customer spending. Are there distinct groups of high and low spenders?
# Popular Shopping Days/Times: Highlight any patterns in shopping days and times.  This is crucial for staffing and promotional scheduling.
# Customer Segmentation
# 
#    We identified three distinct customer segments: 'Loyal Customers' who spend the most and shop frequently, 'Potential Customers' with moderate spending and frequency, and 'Infrequent Customers' with low engagement.  
# 
# 2. Predictive Modeling (Regression Example):
# Model Goal: aimed to predict the total amount a customer would spend based on features like purchase frequency and day of the week.
# Model Performance: Report the RMSE value you obtained. A lower RMSE suggests better predictive accuracy.
# 
# 
# 3. Recommendations for ShopSmart
# 
# Based on our analysis, we recommend the following strategies to enhance ShopSmart's marketing efforts:
# Targeted Promotions:
#     Loyal Customers: Implement a loyalty program with exclusive discounts and early access to new products to retain this valuable segment. 
#     Potential Customers: Design personalized recommendations and targeted promotions to encourage them to transition into more frequent shoppers. Consider free shipping offers or discounts on related products.
#     Infrequent Customers: Conduct surveys to better understand their needs and preferences. Craft enticing entry-level promotions or bundle offers to attract their attention.
# Optimize Operations:
#     Staffing and Inventory: Adjust staffing levels and inventory based on identified peak shopping days and times to improve customer experience.
#     Product Bundling:If you did co-purchase analysis, recommend relevant product bundles to increase sales.
# Personalized Marketing:
#     Leverage customer segmentation insights to personalize marketing messages and product recommendations. Consider email campaigns tailored to specific customer segments' interests. 
# Further Exploration:
#     Price Sensitivity:Analyze if and how price influences purchasing decisions.  
#     Customer Lifetime Value: Predict customer lifetime value to allocate marketing resources more efficiently. 
#     Channel Optimization:If you have data on marketing channels, analyze their effectiveness. 
# 
# 4. Conclusion
# 
# This data-driven analysis provides valuable insights into ShopSmart's customer behavior and offers actionable recommendations to optimize marketing strategies. By implementing these recommendations, ShopSmart can enhance customer engagement, increase sales, and drive business growth.
# 

# #### Analyses for ShopSmart 
# #### OSC Challenge
# ###### Yong Justice Animbom Numfor

# In[ ]:




