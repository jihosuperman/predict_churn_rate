from pyspark.sql.functions import to_date, substring, concat_ws
from pyspark.sql import DataFrame


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
