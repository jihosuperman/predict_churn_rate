from pyspark.sql.functions import to_date, substring, concat_ws
from pyspark.sql import DataFrame
import pandas as pd


def active_day_per_total(transaction_df):
    transaction_df = (transaction_df
                       .withColumn('transaction_date', 
                                   concat_ws('-', substring(transaction_df['transaction_date'], 1, 4), substring(transaction_df['transaction_date'], 5, 2), substring(transaction_df['transaction_date'], 7, 2)))
                    )
    transaction_df = (transaction_df
                       .withColumn('transaction_date', to_date(transaction_df['transaction_date'], 'yyyy-MM-dd'))
                    )
    transaction_df = (transaction_df
                       .withColumn('membership_expire_date', 
                                   concat_ws('-', substring(transaction_df['membership_expire_date'], 1, 4), substring(transaction_df['membership_expire_date'], 5, 2), substring(transaction_df['membership_expire_date'], 7, 2)))
                    )
    transaction_df = (transaction_df
                       .withColumn('membership_expire_date', to_date(transaction_df['membership_expire_date'], 'yyyy-MM-dd'))
                    )
    
    return transaction_df

def multi_join_left(df1, df2, df3, df4):
    total_df = df1.join(df2, on='msno', how='left').join(df3, on='msno', how='left').join(df4, on='msno', how='left')
    
    return total_df

def multi_join_inner(*args):
    for df in args:
      total_df = df1.join(df2, on='msno', how='inner').join(df3, on='msno', how='inner').join(df4, on='msno', how='inner')
    
    return total_df

def sparkDF_to_csv(df, address):
    df.to_Pandas().to_csv(address)
    print('Finish!!')