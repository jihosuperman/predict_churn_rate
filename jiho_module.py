import dask as dd

def concat_and_check(df1, df2):
    df_new = dd.concat([df1, df2]).reset_index(drop=True)
    check = True
    if len(df1) + len(df2) == len(df_new):
        check = True
    else:
        check = False

    print(check)
    return df_new



def active_day_per_total(log_data):
    user_active_count = log_data.groupBy("msno").count()