"""
E-commerce Sales Analysis Project
---------------------------------
Author   : Tahmid Rahman Siddiki
Role     : Aspiring Data Analyst
Email    : tahmid4030@gmail.com
LinkedIn : www.linkedin.com/in/tahmid-rahman-a80a82163
GitHub   : https://github.com/tahmidrs
Location : [Dhaka, Bangladesh]
Date     : September 2025
Version  : 1.0

Description:
This project analyzes e-commerce sales data (2010–2011),
covering data cleaning, exploratory analysis, customer
segmentation (RFM), and business recommendations.
"""



import pandas as pd

# 1) load (use latin1 because this file often needs it)
df = pd.read_csv(r"C:\Users\User.DESKTOP-5903S8A\Downloads\data.csv", encoding="latin1")


# 2) quick overview
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("\ninfo():")
print(df.info())

# 3) missing values
print("\nmissing values per column:")
print(df.isna().sum())

# 4) duplicates
print("\n# exact duplicate rows:", df.duplicated().sum())

# 5) quantity / price checks
print("\nQuantity stats:")
print(df["Quantity"].describe())
print("\nUnitPrice stats:")
print(df["UnitPrice"].describe())

# 6) rows with Quantity <= 0 (likely returns)
neg_q = df[df["Quantity"] <= 0]
print("\nrows with Quantity <= 0 (sample 5):")
print(neg_q.head(5))
print("count:", neg_q.shape[0])

# 7) parse InvoiceDate (use dayfirst=True for this dataset)
df["InvoiceDate_parsed"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
print("\nInvoiceDate dtype after parse:", df["InvoiceDate_parsed"].dtype)
print("missing parsed dates:", df["InvoiceDate_parsed"].isna().sum())
print("sample parsed dates:")
print(df["InvoiceDate_parsed"].head(5))

# 8) countries overview
print("\nunique countries:", df["Country"].nunique())
print(df["Country"].value_counts().head(10))

# 1) Drop duplicates
df = df.drop_duplicates()

# 2) Drop missing Description
df = df.dropna(subset=["Description"])

# 3) Drop missing CustomerID
df = df.dropna(subset=["CustomerID"])

# 4) Remove rows where Quantity <= 0 or UnitPrice <= 0
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 5) Fix InvoiceDate parsing (use correct format: month/day/year hour:minute)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M", errors="coerce")

# Check if parsing worked
print("missing parsed dates:", df["InvoiceDate"].isna().sum())
print("sample dates:", df["InvoiceDate"].head(5))

# 6) Reset index
df = df.reset_index(drop=True)

print("shape after cleaning:", df.shape)

import matplotlib.pyplot as plt

# 1) Total sales column (Quantity × UnitPrice)
df["Sales"] = df["Quantity"] * df["UnitPrice"]

# 2) Top 10 selling products (by total sales)
top_products = df.groupby("Description")["Sales"].sum().sort_values(ascending=False).head(10)
print("Top 10 products by total sales:")
print(top_products)

# 3) Top 10 countries (excluding UK)
top_countries = df[df["Country"] != "United Kingdom"].groupby("Country")["Sales"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 countries (excluding UK):")
print(top_countries)

# 4) Monthly sales trend
df["Month"] = df["InvoiceDate"].dt.to_period("M")
monthly_sales = df.groupby("Month")["Sales"].sum()

# Plot monthly sales
monthly_sales.plot(kind="line", figsize=(10,5), marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales (£)")
plt.show()

# Top 10 customers by total spend
top_customers = df.groupby("CustomerID")["Sales"].sum().sort_values(ascending=False).head(10)
print("Top 10 customers by sales:")
print(top_customers)

# Average order value per customer
avg_order_value = df.groupby("CustomerID")["Sales"].mean().mean()
print("\nAverage order value (overall):", round(avg_order_value, 2))

# Number of unique customers
print("\nNumber of unique customers:", df["CustomerID"].nunique())

import datetime as dt

# Reference date = max invoice date in dataset
ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# RFM calculation
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (ref_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",  # Frequency (unique orders)
    "Sales": "sum"           # Monetary
})

rfm.rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "Sales": "Monetary"
}, inplace=True)

# Quick look
print(rfm.head(10))

# Create R, F, M scores (1 to 5)
rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)

# Combine into one RFM score
rfm["RFM_Segment"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
rfm["RFM_Score"] = rfm[["R_Score","F_Score","M_Score"]].sum(axis=1)

print(rfm.head(10))

# Define segments based on RFM_Score
def segment_customer(score):
    if score >= 13:
        return "VIP"
    elif score >= 10:
        return "Loyal"
    elif score >= 7:
        return "Potential"
    elif score >= 4:
        return "At Risk"
    else:
        return "Lost"

rfm["Segment"] = rfm["RFM_Score"].apply(segment_customer)

# Count customers in each segment
print(rfm["Segment"].value_counts())

# Average sales by segment
print("\nAverage Monetary value per segment:")
print(rfm.groupby("Segment")["Monetary"].mean())