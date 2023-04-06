def active_day_per_total(log_data):
    user_active_count = log_data.groupBy("msno").count()