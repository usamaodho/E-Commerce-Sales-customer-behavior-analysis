# E-Commerce Sales & Customer Analytics

## Project Overview

This project is an end-to-end exploratory analytics notebook for an e-commerce transaction dataset. The goal is to turn raw sales transactions into actionable business insights around **revenue, products, customers, retention, operations, demand patterns, and product associations**.

The analysis works with **541,909 transaction records and 8 columns**, including invoice number, product information, quantity, invoice date, unit price, customer ID, and country.

The notebook combines data cleaning, exploratory data analysis, customer segmentation, cohort/retention analysis, operational analysis, and market basket analysis.

---

## Key Features

- Revenue and sales performance analysis
- Monthly revenue and seasonality analysis
- Top-product and underperforming-product analysis
- Country-level sales analysis
- RFM (Recency, Frequency, Monetary) customer segmentation
- Identification of high-value and at-risk customers
- Repeat-customer vs. one-time-customer revenue analysis
- Customer cohort and retention analysis
- Average Order Value (AOV) analysis
- Return-rate analysis
- Day-of-week and hourly demand analysis
- Market Basket Analysis using the Apriori algorithm
- Association-rule analysis using support, confidence, and lift
- Product association network visualization
- Business-focused recommendations based on analytical findings

---

## Business Problems Addressed

The project is structured around 10 practical business questions:

1. **Revenue & Sales:** Which months generated the highest revenue, and is there a seasonal pattern?
2. **Product Performance:** Which are the top 10 products by total revenue, and which products are consistently underperforming?
3. **Geographic Performance:** Which countries contribute the most to total sales, and how does the UK compare with international markets?
4. **Customer Value:** Who are the most valuable customers, and who is at risk of churning?
5. **Customer Loyalty:** What percentage of revenue comes from repeat customers versus one-time buyers?
6. **Retention:** How does customer retention change month over month, and which cohort has the best long-term retention?
7. **Order Value:** What is the Average Order Value (AOV), and how has it changed over time?
8. **Returns:** What proportion of transactions are returns, and which products have the highest return rate?
9. **Demand Patterns:** Which day of the week and hour of the day receive the most orders?
10. **Product Associations:** Which products are frequently bought together in the same invoice?

These questions connect the technical analysis directly to inventory planning, marketing, customer retention, product strategy, operations, and cross-selling.

---

## Dataset

The notebook loads the dataset from:

```python
data = pd.read_csv('data.csv', encoding='latin-1')
```

The original dataset contains:

| Column | Description |
|---|---|
| `InvoiceNo` | Transaction/invoice identifier |
| `StockCode` | Product identifier |
| `Description` | Product description |
| `Quantity` | Number of units purchased |
| `InvoiceDate` | Transaction date and time |
| `UnitPrice` | Price per unit |
| `CustomerID` | Customer identifier |
| `Country` | Customer country |

Dataset size:

- **Rows:** 541,909
- **Columns:** 8
- **Initial memory usage:** approximately 33.1 MB

---

## Technologies & Libraries

The notebook is implemented in Python and uses:

- **Pandas** — data loading, cleaning, transformation, aggregation, and analysis
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — statistical/data visualizations
- **NetworkX** — product association network graphs
- **mlxtend** — Apriori frequent-itemset mining and association rules

Main imports include:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
```

---

## What I Worked On

### 1. Data Loading & Initial Inspection

I loaded the transaction data and inspected its structure using `head()`, `info()`, and `shape`.

This helped establish:

- Dataset dimensions
- Column names
- Data types
- Missing-value counts
- Memory usage
- Initial data quality issues

---

### 2. Data Type Correction

Some columns did not initially have the most appropriate data types.

I converted:

```python
data['InvoiceDate'] = data['InvoiceDate'].astype('datetime64[ns]')
data['Country'] = data['Country'].astype('string')
data['Description'] = data['Description'].astype('string')
```

This made the data more suitable for date-based analysis, grouping, and text handling.

---

### 3. Missing-Value Handling

The initial inspection identified missing values in:

- `Description`: 1,454 missing values
- `CustomerID`: 135,080 missing values

I handled these missing values in the notebook by:

```python
data.fillna({'CustomerID': data['CustomerID'].mean()}, inplace=True)
data['Description'] = data['Description'].bfill()
```

After cleaning, the notebook verified that the columns contained no remaining null values.

> Note: These are the exact missing-value strategies implemented in the notebook. For a production analytics pipeline, customer identifiers and product descriptions may require a more domain-specific treatment.

---

## Problems Encountered and How I Solved Them

### Problem 1: Missing Customer IDs

A large number of transactions had missing `CustomerID` values.

**Solution implemented:**  
I replaced missing customer IDs with the mean customer ID and then verified the result.

**Why it mattered:**  
Customer-level analysis such as RFM segmentation requires a populated customer identifier.

---

### Problem 2: Missing Product Descriptions

Some transactions had missing product descriptions.

**Solution implemented:**  
I used backward filling (`bfill`) for the `Description` column and checked the dataset again for null values.

**Why it mattered:**  
Product descriptions are needed for product-level reporting and market basket analysis.

---

### Problem 3: Incorrect Data Types

`InvoiceDate` was initially stored as an object/string rather than a datetime value.

**Solution implemented:**  
I explicitly converted it to `datetime64[ns]`.

**Why it mattered:**  
Datetime conversion is necessary for monthly, daily, hourly, trend, and cohort analysis.

---

### Problem 4: Negative Quantities and Prices

The descriptive statistics showed negative values in `Quantity` and `UnitPrice`.

For example, the dataset contained:

- Minimum quantity: `-80995`
- Minimum unit price: `-11062.06`

Negative quantities are relevant to return/cancellation analysis and therefore should not simply be treated as ordinary sales.

**Solution implemented:**  
The notebook uses negative quantities as part of the return analysis, where transactions with negative quantities are interpreted as returns.

---

### Problem 5: Customer Segmentation

Raw customer transactions do not directly indicate which customers are valuable, loyal, at risk, or lost.

**Solution implemented:**  
I built an RFM model using:

- **Recency** — how recently the customer purchased
- **Frequency** — how often the customer purchased
- **Monetary** — how much the customer spent

Each metric was converted into a 1–5 score using quantile-based scoring.

The scores were combined into:

```text
RFM_Score = R_Score + F_Score + M_Score
```

Customers were then segmented into:

- **Champions**
- **Loyal Customers**
- **At Risk**
- **Lost Customers**

---

### Problem 6: Identifying Cross-Sell Opportunities

The business may know individual product sales but still not know which products customers commonly purchase together.

**Solution implemented:**  
I transformed invoice-level transactions into a basket format and applied the **Apriori algorithm**.

The notebook uses:

- `min_support = 0.02`
- `max_len = 3`

Association rules are then evaluated using **lift**, with:

- `min_threshold = 1.5`

The results are sorted by lift to highlight strong product associations.

---

## RFM Customer Segmentation

The notebook creates RFM scores using quantile-based scoring.

### Recency

Recent customers receive higher recency scores:

```python
pd.qcut(
    rfm['Recency'],
    q=5,
    labels=[5,4,3,2,1]
)
```

### Frequency

Customers with higher purchase frequency receive higher frequency scores.

### Monetary

Customers with higher monetary value receive higher monetary scores.

### Overall Score

The three scores are added together:

```python
rfm['RFM_Score'] = (
    rfm['R_Score'] +
    rfm['F_Score'] +
    rfm['M_Score']
)
```

### Segment Rules

| RFM Score | Segment |
|---:|---|
| 13+ | Champions |
| 10–12 | Loyal Customers |
| 7–9 | At Risk |
| Below 7 | Lost Customers |

The resulting segment summary in the notebook contains:

| Segment | Customers | Avg. Recency | Avg. Frequency | Avg. Monetary |
|---|---:|---:|---:|---:|
| At Risk | 1,115 | 81.80 | 2.32 | 628.63 |
| Champions | 949 | 13.47 | 18.00 | 7,730.30 |
| Lost Customers | 1,319 | 193.93 | 1.26 | 253.98 |
| Loyal Customers | 990 | 43.08 | 4.62 | 1,389.66 |

---

## Market Basket Analysis

For product association analysis, transactions are converted into a basket representation.

The notebook then encodes the basket into boolean values and applies Apriori:

```python
basket_encoded = basket.apply(lambda col: col > 0)

frequent_itemsets = apriori(
    basket_encoded,
    min_support=0.02,
    use_colnames=True,
    max_len=3
)
```

Association rules are generated using lift:

```python
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.5,
    num_itemsets=len(frequent_itemsets)
)
```

The notebook produces:

- Top association rules
- Support values
- Confidence values
- Lift values
- A top-10 association-rule table
- A product association network graph

The network graph represents product relationships as directed edges, with lift used as the edge weight.

---

## Analysis Areas

### Revenue & Sales Performance

Used to understand monthly revenue trends and seasonality for better inventory, marketing, and staffing decisions.

### Product Performance

Used to identify high-revenue products and underperforming SKUs.

### Geographic Analysis

Used to compare UK performance with international markets.

### Customer Analytics

Used to understand customer value, loyalty, churn risk, and repeat purchasing.

### Retention & Cohorts

Used to evaluate how customer retention changes over time and compare customer cohorts.

### Orders & Operations

Used to monitor Average Order Value and return behavior.

### Time & Demand Patterns

Used to identify peak days and hours for orders.

### Market Basket Analysis

Used to discover products that are frequently purchased together and support cross-selling recommendations.

---

## Key Insights

The notebook's final insights identify several important business opportunities:

- Monthly revenue analysis can reveal peak sales periods and seasonality.
- Product revenue ranking highlights major revenue drivers and weaker SKU groups.
- RFM segmentation identifies high-value customers and customers who may be at risk.
- Repeat-buyer analysis helps quantify the contribution of loyal customers.
- Return and demand analysis can expose products with higher return risk and useful operational time windows.
- Market basket analysis identifies product relationships that can support “frequently bought together” recommendations.

---

## Business Recommendations

Based on the analysis, the notebook recommends:

1. Focus on high-revenue product groups while reviewing low-performing SKUs for markdowns or bundle offers.
2. Prioritize retention campaigns for **At Risk** customers.
3. Reward **Champions** to increase customer lifetime value.
4. Use repeat-customer revenue analysis to strengthen loyalty strategies.
5. Investigate products with high return rates by improving descriptions, quality checks, and return-policy communication.
6. Schedule marketing promotions and customer support around peak order windows.
7. Use product association rules to create cross-sell and “frequently bought together” recommendations.

---

## Output Visualizations

The notebook includes analytical visualizations and saves market basket outputs such as:

- `q10_association_rules_table.png`
- `q10_network_graph.png`

The association network graph provides a visual representation of relationships between frequently purchased products.

---

## Project Workflow

```text
Raw E-Commerce Data
        |
        v
Data Loading
        |
        v
Data Inspection
        |
        v
Data Type Correction
        |
        v
Missing-Value Handling
        |
        v
Exploratory Analysis
        |
        +--------------------+
        |                    |
        v                    v
Sales/Product Analysis   Customer Analysis
        |                    |
        |                    v
        |               RFM Segmentation
        |                    |
        |                    v
        |               Cohort/Retention
        |
        +--------------------+
        |
        v
Orders & Returns Analysis
        |
        v
Time & Demand Analysis
        |
        v
Market Basket Analysis
        |
        v
Business Insights & Recommendations
```

---

## Project Highlights

### Data Analytics
- Worked with a large transaction dataset containing over half a million records.
- Performed structured data inspection and cleaning.
- Converted data into analysis-ready formats.

### Customer Analytics
- Implemented RFM analysis from transaction-level data.
- Created actionable customer segments.
- Identified Champions, Loyal Customers, At-Risk customers, and Lost Customers.

### Business Intelligence
- Connected technical analysis to real business questions.
- Produced recommendations for revenue growth, retention, inventory, operations, and marketing.

### Association Mining
- Implemented Apriori frequent-itemset mining.
- Generated association rules using support, confidence, and lift.
- Created a network graph to visualize product relationships.

---

## Limitations & Possible Improvements

The current notebook is primarily an analytical project. Some areas could be improved for a production-grade system:

- Use a more appropriate strategy for missing `CustomerID` values instead of mean imputation, because customer IDs are identifiers rather than continuous measurements.
- Validate and handle negative prices separately from returns.
- Add automated data-quality checks.
- Add statistical validation for important business conclusions.
- Build an interactive dashboard using tools such as Power BI, Tableau, or Streamlit.
- Automate the analysis pipeline for new transaction data.
- Add unit tests and reusable functions for the data-cleaning and analytics steps.
- Store processed datasets and analytical outputs in a structured data pipeline.

---

## Conclusion

This project demonstrates how raw e-commerce transaction data can be transformed into useful business intelligence.

The main strength of the project is that it does not stop at basic sales charts. It combines **sales analysis, customer segmentation, retention analysis, operational metrics, return analysis, demand patterns, and market basket analysis** to answer practical business questions.

The final output provides actionable directions for:

- Revenue growth
- Customer retention
- Customer lifetime value
- Product strategy
- Inventory and operational planning
- Marketing timing
- Cross-selling

Overall, the notebook delivers a complete exploratory analytics workflow from **raw transaction data → data cleaning → analysis → customer intelligence → product associations → business recommendations**.

---

## How to Run

1. Clone the repository.
2. Make sure Python 3 is installed.
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn networkx mlxtend
```

4. Place `data.csv` in the project directory.
5. Open the Jupyter Notebook.
6. Run the cells from top to bottom.

---

## Repository Structure

A recommended GitHub structure is:

```text
project/
│
├── data.csv
├── ecommerce_analysis.ipynb
├── q10_association_rules_table.png
├── q10_network_graph.png
└── README.md
```

> If the dataset is not suitable for public redistribution, do not upload the raw `data.csv` to GitHub. Instead, provide instructions for obtaining the dataset or include a small sample dataset.

---

## Skills Demonstrated

- Python
- Pandas
- NumPy
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Business Analytics
- Customer Segmentation
- RFM Analysis
- Cohort Analysis
- Retention Analysis
- Revenue Analysis
- Product Performance Analysis
- Return Analysis
- Time-Series/Time-Based Analysis
- Market Basket Analysis
- Apriori Algorithm
- Association Rules
- Network Graph Visualization
- Business Problem Solving
