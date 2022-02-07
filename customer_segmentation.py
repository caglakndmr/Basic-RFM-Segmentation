# Made by caglakndmr @GitHub
# A mini project for basic RFM customer segmentation, to better understand the concept.


# Dataset used
#
# Online Retail Dataset
# https://archive.ics.uci.edu/ml/datasets/online+retail

# Atribute Information
#
# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction.
# If this code starts with letter 'C', it indicates a cancellation.
# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
# Description: Product (item) name. Nominal.
# Quantity: The quantities of each product (item) per transaction. Numeric.
# InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
# UnitPrice: Unit price. Numeric, Product price per unit in sterling.
# CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
# Country: Country name. Nominal, the name of the country where each customer resides.

import pandas as pd
import datetime as dt
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

df_ = pd.read_excel("online_retail.xlsx")
df = df_.copy()

# Checking to see if there are any null values we can get rid of.
df.isnull().sum()

# Checking for abnormalties and outlier values.
# There are below zero values for min "Quantity" and "UnitPrice" values, and there seems to be outlier values (max).
df.describe().T


def find_thresholds(dataframe, variable):
    """
    Calculates and returns the thresholds to check for outliers.

    Parameters
    ----------
    dataframe : pandas.DataFrame
                The dataframe that contains the columns that will be checked for outliers.
    variable : str
               Name of the column that contains the values.

    Returns
    -------
    low_limit : float
                The lower threshold.
    up_limit : float
               The upper threshold.
    """
    q1 = dataframe[variable].quantile(0.25)
    q3 = dataframe[variable].quantile(0.75)
    q_range = q3 - q1
    low_limit = q1 - 1.5 * q_range
    up_limit = q3 + 1.5 * q_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """
    Checks for and handles outlier values inside the specified column of dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
                The dataframe that contains the columns will be checked for outliers.
    variable : str
               Name of the column that contains the values.

    Returns
    -------
    None
    """
    low_limit, up_limit = find_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def data_preprocess(dataframe):
    """
    Data preprocessing. Drops all null values, removes cancelled orders, handles outliers and creates a new column for
    the total prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
                The dataframe that contains the columns that will be checked for outliers.

    Returns
    -------
    dataframe : pandas.DataFrame
        The processed dataframe.

    """
    # Fixed the min "UnitPrice" value.
    dataframe.dropna(inplace=True)

    # Invoice numbers starting with "C" are cancelled orders. We will need to get rid of these entries.
    # Fixed the min "Quantity" value.
    dataframe = dataframe[~dataframe["InvoiceNo"].str.contains("C", na=False)]

    # Handling outliers.
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "UnitPrice")

    # Calculating the total price for each entry and creating a column.
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["UnitPrice"]
    return dataframe


df = data_preprocess(df)

# Checking the last invoice date, so we can adjust today's date accordingly for our Recency metric.
df["InvoiceDate"].max()
todays_date = dt.datetime(2011, 12, 11)


def create_segment_dataframe(dataframe):
    """
    Calculates the Recency, Frequency and Monetary metrics as well as the RF(M) Score from the given dataframe,
    and segments the customers into the new dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
                The dataframe customers will be segmented from.

    Returns
    -------
    dataframe : pandas.DataFrame
        The new dataframe of segmented customers.

    """
    # Creating new DataFrame for the RFM metrics.
    rfm = dataframe.groupby("CustomerID").agg({"InvoiceDate": lambda invoice_date: (todays_date - invoice_date.max()).days,
                                               "InvoiceNo": lambda invoice_no: invoice_no.nunique(),
                                               "TotalPrice": lambda total_price: total_price.sum()})
    # Changing the column names
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # If there are customers whose monetary values are 0, we can get rid of these entries.
    rfm = rfm[rfm["Monetary"] > 0]

    # Calculating the R, F, and M scores.
    rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["FrequencyScore"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["MonetaryScore"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    # Calculating RF(M) score. In the end the "Monetary" metric will not have any effect on the segmentation.
    rfm["RF_Score"] = rfm["RecencyScore"].astype(str) + rfm["FrequencyScore"].astype(str)

    # Creating segments.
    segment_map = {
        r"[1-2][1-2]": "Hibernating",
        r"[1-2][3-4]": "At Risk",
        r"[1-2]5": "Can't Lose",
        r"3[1-2]": "About to Sleep",
        r"33": "Need Attention",
        r"[3-4][4-5]": "Loyal Customer",
        r"41": "Promising",
        r"51": "New Customer",
        r"[4-5][2-3]": "Potential Loyalist",
        r"5[4-5]": "Champion"
    }

    # Creating a new column for customer segments and assigning their values.
    rfm["CustomerSegment"] = rfm["RF_Score"].replace(segment_map, regex=True)
    # Fixing CustomerID index
    rfm.index = rfm.index.astype(int)
    return rfm


rfm = create_segment_dataframe(df)
# only_segments = rfm[["RF_Score","CustomerSegment"]]
