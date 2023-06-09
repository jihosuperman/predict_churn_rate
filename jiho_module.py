from pyspark.sql.functions import to_date, substring, concat_ws, lag, when, datediff
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
import pandas as pd
import seaborn as sns
from functools import reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def int_to_date(transaction_df, col_name):
    transaction_df = (transaction_df
                       .withColumn(
                            col_name, 
                            concat_ws(
                                '-', 
                                substring(transaction_df[col_name], 1, 4), 
                                substring(transaction_df[col_name], 5, 2), 
                                substring(transaction_df[col_name], 7, 2)))
                    )
    transaction_df = (transaction_df
                       .withColumn(
                            col_name, 
                            to_date(
                                transaction_df[col_name], 
                                'yyyy-MM-dd'))
                    )
    
    return transaction_df

def add_subscribe_period(transaction_df):
    transaction_df = (transaction_df
                      .withColumn('prev_expire_date', lag('membership_expire_date')
                                                      .over(Window.partitionBy('msno_num').orderBy('transaction_date','membership_expire_date')))
                      )
    
    transaction_df = transaction_df.withColumn('periods', when(transaction_df.is_cancel == 0, datediff(transaction_df['membership_expire_date'], transaction_df['transaction_date']))
                                             .otherwise(datediff(transaction_df['membership_expire_date'], transaction_df['prev_expire_date']))
                                             )
    
    return transaction_df

def join_dataframes(dataframes: list, join_condition: str) -> DataFrame:
    """
    Takes a list of Spark DataFrames and a join condition, and performs an inner join on all of them.
    """
    return reduce(lambda df1, df2: df1.join(df2, on='msno',how=join_condition), dataframes)

def sparkDF_to_csv(df: DataFrame, address: str):
    """
    Take a Spark DataFrame and export it as csv format.
    """
    df.toPandas().to_csv(address)
    print('Finish!!')

def print_model_result(model, X_test, y_test):
   pred_test2 = model.predict(X_test)

   print("accuracy:", accuracy_score(y_test, pred_test2))
   print("precision:", precision_score(y_test, pred_test2))
   print("recall:", recall_score(y_test, pred_test2))
   print("f1 score:", f1_score(y_test, pred_test2))
   print(confusion_matrix(y_test, pred_test2, labels=[1,0]))

def print_grid_result(model, X_test, y_test):
   print('best parameters : ', model.best_params_)
   print('best score : ', model.best_score_)
   best_test2 = model.best_estimator_
   pred_test2 = model.predict(X_test)

   print("accuracy:", accuracy_score(y_test, pred_test2))
   print("precision:", precision_score(y_test, pred_test2))
   print("recall:", recall_score(y_test, pred_test2))
   print("f1 score:", f1_score(y_test, pred_test2))
   print(confusion_matrix(y_test, pred_test2))

def compare_density(df, col_name, ax=None):
    sns.kdeplot(df, x=col_name, hue='is_churn', common_norm=False, ax=ax)