import pandas as pd
import numpy as np
import datetime

# 각 데이터 불러올 때 컬러명 변경 절대 금지!

class TrainingModule():

    def __init__(self):
        self.train = None
        self.members = None
        self.transaction = None
        self.userlog = None
        self.originalage1 = None
        self.originalage2 = None       
        
    def importData(self, train_path=None, member_path=None, transaction_path=None, userlog_path=None):  
        ##### Train data 불러오기 #####
        if train_path == None:
            self.train == None
        else:
            self.train = pd.read_parquet(train_path)

        ##### Members data 불러오기 #####
        if member_path == None:
            self.members == None
        else:
            self.members = pd.read_parquet(member_path)
            self.members.gender.fillna('donknow', inplace=True)  
            self.members.registration_init_time = pd.to_datetime(self.members.registration_init_time, format='%Y-%m-%d')
                
        ##### Transaction data 불러오기 & 필터링 #####
        if transaction_path == None:
            self.transaction == None
        else:
            self.transaction = pd.read_parquet(transaction_path)
            self.transaction.transaction_date = pd.to_datetime(self.transaction.transaction_date, format='%Y-%m-%d')
            self.transaction.membership_expire_date = pd.to_datetime(self.transaction.membership_expire_date, format='%Y-%m-%d')
            # 아이디와 거래일의 중복값 중 첫번째 & 구독 취소 유저가 아닌 경우를 제거
            transaction_filter1 = (~((self.transaction.duplicated(['msno_num', 'transaction_date'], keep='first')) & (self.transaction['is_cancel'] == 0)))

            # 위에서 필터링되지 않은 같은 구독 취소 거래가 중복된 값 제거
            cols = list(self.transaction.columns)[1:9]
            cols.remove('membership_expire_date')
            transaction_filter2 = (~((self.transaction.duplicated(cols, keep='first')) & (self.transaction['is_cancel'] == 1)))

            # 멤버쉽 만료일 > 거래일 인 경우만 남김
            transaction_filter3 = (self.transaction['transaction_date'] < self.transaction['membership_expire_date'])

            min_timestamp = pd.Timestamp(datetime.date(2005,1,1))
            max_timestamp = pd.Timestamp(datetime.date(2018,4,30))

            transaction_filter4 = ((self.transaction['membership_expire_date'] >= min_timestamp) & (self.transaction['membership_expire_date'] <= max_timestamp))

            self.transaction = self.transaction[transaction_filter1&transaction_filter2&transaction_filter3&transaction_filter4]

        ##### Userlog data 불러오기 #####
        if userlog_path == None:
            self.userlog == None
        else:
            self.userlog = pd.read_parquet(userlog_path)
            self.userlog.date = pd.to_datetime(self.userlog.date, format='%Y-%m-%d')

            filter1 = (
                        (self.userlog['total_secs'] >= 0)&
                        (self.userlog['num_25'] >= 0)&
                        (self.userlog['num_50'] >= 0)&
                        (self.userlog['num_75'] >= 0)&
                        (self.userlog['num_985'] >= 0)&
                        (self.userlog['num_100'] >= 0)&
                        (self.userlog['num_unq'] >= 0)
                    )
                
            filter2 = (self.userlog['total_secs'] <= 86400)

            def remove_quantile(df):
                for col in ['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq']:
                    df2 = df[df[col] > 0]
                    level_1q = df2[col].quantile(0.25)
                    level_3q = df2[col].quantile(0.75)
                    IQR = level_3q - level_1q
                    rev_range = 1.5
                    df = df[(df[col] <= level_3q+(rev_range*IQR)) & (df[col] >= level_1q-(rev_range*IQR))]
                return df
                
            self.userlog = remove_quantile(self.userlog[filter1 & filter2])
    
    def pushTrain(self): ### 트레인데이터 푸쉬
        
        return self.train

    def gender(self): # 그대로 사용 # 젠더 드랍  ############### 완료
        gender_original =self.members[['msno_num','gender']].copy()
        gender_label = {'male':1, 'female':2, 'donknow':0}

        gender_original['gender'] = gender_original['gender'].replace(gender_label)

        gender_drop = self.members[['msno_num']].copy()
        
        gender_result = [gender_original, gender_drop]

        return gender_result
        
    def age(self): # 0 : 그대로 사용 # 1 : 15 ~ 64세 와 나머지 ############# 완료
    
        original_age_1 = self.members[['msno_num','bd']]
        original_age_1.loc[original_age_1['bd'] < 0, 'bd'] = 0
        original_age_1.loc[original_age_1['bd'] >= 100, 'bd'] = 100
        
        original_age_2 = self.members[['msno_num','bd']]
        original_age_2['bd'] = original_age_2['bd'].apply(lambda x: x if (x >= 15) & (x <=64) else -1)
        # Null : -1
        self.originalage1 = original_age_1
        self.originalage2 = original_age_2 

        original_age_result = [original_age_1, original_age_2]

        return original_age_result


    def ageCategory(self): ############### 완료
        age_list = [self.originalage1, self.originalage2]
        #print(f'type(self.originalage1) : {type(self.originalage1)}')
        #print(f'type(self.originalage2) : {type(self.originalage2)}')

        age_category_result = []
        for idx, item in enumerate(age_list):
            #print(f'type(item) : {type(item)}')
            bins = [15, 26, 36, 46, 56, 64]
            labels = [0,1,2,3,4]
            age_df = item.copy()
            age_df['bd'] = pd.cut(age_df['bd'], bins=bins, labels=labels)
            age_df['bd'] = age_df['bd'].cat.add_categories([5]).fillna(5)
            age_df['bd'] = age_df['bd'].astype(int)
            age_category_result.append(age_df)

            age_df1 = item.copy()
            bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
            labels = [0,1,2,3,4,5,6,7,8,9]
            age_df1['bd'] = pd.cut(age_df1['bd'], bins=bins, labels=labels)
            age_df1['bd'] = age_df1['bd'].cat.add_categories([10]).fillna(10)
            age_df['bd'] = age_df['bd'].astype(int)
            age_category_result.append(age_df1)

            age_df2 = item.copy()
            age_df2['bd'] = age_df2['bd'].apply(lambda x: -1 if( (x < 16) or (x > 64)) else x)
            age_df2['bd'] = age_df2['bd'].apply(lambda x: 0 if (x != -1) & (x >=25) else x)
            age_df2['bd'] = age_df2['bd'].apply(lambda x: 1 if (x != -1) & (x <=25) else x)
            age_category_result.append(age_df2)

        return age_category_result
            

    def cityBinary(self): # 그대로  # 이분법 ################# 완료
        city_original = self.members[['msno_num','city']]
        
        self.members['city'] = np.where(self.members['city'] == 1, 1, 0)
        city_binary = self.members[['msno_num','city']]

        city_result = [city_original , city_binary]

        return city_result

            
    def autoRenew(self): # 연속 , # 그냥여부 ##################### 완료
        autoRenewSum = self.transaction[['msno_num','is_auto_renew']].groupby('msno_num').sum().reset_index()
        autoRenewMax = self.transaction[['msno_num','is_auto_renew']].groupby('msno_num').max().reset_index()

        auto_renew_result = [autoRenewSum, autoRenewMax]

        return auto_renew_result


    def paymentMethodId(self): # 지불방식 ################### 완료
        paymentMethod = self.transaction[['msno_num','payment_method_id']].groupby('msno_num')['payment_method_id'].apply(lambda x: x.value_counts().index[0]).to_frame().reset_index()
        
        return paymentMethod


    def dicountRecord(self): # 할인받은 기록 ####################### 완료
        discount = (self.transaction.assign(discount_record=lambda x: x['actual_amount_paid'] - x['plan_list_price'])
                .groupby('msno_num', as_index=False)['discount_record']
                .sum()
                .assign(discount_record=lambda x: (x['discount_record'] > 0).astype(int)))

        return discount
    

    def initRegistration(self): # year_month   # year ######################### 완료
        registration_year_month = self.members[['msno_num','registration_init_time']]
        registration_year_month['registration_init_time'] = pd.to_datetime(registration_year_month['registration_init_time']).dt.to_period('M').astype(str).str.replace('-', '').astype(int)
        
        registration_year = self.members[['msno_num','registration_init_time']] 
        registration_year['registration_init_time'] = registration_year['registration_init_time'].dt.year

        init_registration_result = [registration_year_month, registration_year]

        return init_registration_result
    

    def delayTran(self): ##### 연속형 /// 카테고리 ################## 완료 
        df1 = self.transaction.groupby('msno_num')['transaction_date'].min().reset_index()

        df2 = pd.merge(df1, self.members[['msno_num', 'registration_init_time']], on='msno_num')
        df2.rename(columns={'transaction_date': 'first_transaction_date'}, inplace=True)

        df2['reg_first_subs'] = round(((df2['first_transaction_date'] - df2['registration_init_time']).dt.days/360), 0)
        continue_delay = df2[['msno_num', 'reg_first_subs']]
       
        category_delay = continue_delay.copy()
        bins = [-2, 0, 11, 99]
        labels = ['0', '1-11', '12+']
        category_delay['reg_first_subs'] = pd.cut(category_delay['reg_first_subs'], bins=bins, labels=labels)

        delay_transaction = [continue_delay, category_delay]
        
        return delay_transaction

    def membershipDuration(self): # 멤버쉽 유지기간 ########################### 완료
       
        membership_duration = self.transaction[['msno_num','transaction_date','membership_expire_date']]
        membership_duration['year_month'] =  membership_duration['transaction_date'].dt.to_period('M')
        membership_duration = membership_duration.drop_duplicates(subset=['msno_num', 'year_month'], keep='last')
        membership_duration = membership_duration['msno_num'].value_counts().reset_index()
        membership_duration.columns = ['msno_num', 'duration']
    
        return  membership_duration
        
    def isCancel(self): # 구독취소여부 ################################## 완료
        is_cancel = self.transaction.groupby('msno_num').agg({'is_cancel':'sum'}).reset_index()
        is_cancel['is_cancel'] = np.where(is_cancel['is_cancel'] > 0, 1, 0)
            
        return is_cancel[['msno_num','is_cancel']]

    def longTermUnconnect(self): ### 연속형  #### 30일 미접속
        df = self.userlog[['msno_num','date']].sort_values(['msno_num', 'date'])
        df['unconnected'] = df.groupby('msno_num')['date'].diff().fillna(pd.Timedelta(seconds=0))
        df['unconnected'] = df['unconnected'].dt.days.fillna(0).astype(int)

        df = df.groupby('msno_num')[['unconnected']].max().reset_index()
        
        df1 = df.copy()
        
        countinue_long_unconnect = df[['msno_num','unconnected']]
        

        labels=[0,1,2,3,4,5]
        bins=[0,8,16,31,121,366,np.inf]
        
        df1['unconnected'] = pd.cut(df['unconnected'], labels=labels, bins=bins, right=False)
        df1['unconnected'] = df1['unconnected'].astype(int)
        df1 = df1.reset_index()

        days_30_long_unconnect = df1[['msno_num','unconnected']].copy()
        
        log_unconnect = [countinue_long_unconnect, days_30_long_unconnect]

        return log_unconnect
            
      
    def userLogGroup(self):
        self.userlog['date'] = self.userlog['date'].dt.strftime('%Y-%m')
        df = self.userlog.groupby(['msno_num', 'date']).agg(log_count=('date','count'),
                                                            num_25=('num_25','mean'),
                                                            num_50=('num_50','mean'),
                                                            num_75=('num_75','mean'),
                                                            num_985=('num_985','mean'),
                                                            num_100=('num_100','mean'),
                                                            num_unq=('num_unq','mean'),
                                                            total_secs=('total_secs','sum')
                                                            )
            
        df = df.groupby('msno_num').agg(log_count=('log_count','mean'),
                                    num_25=('num_25','mean'),
                                    num_50=('num_50','mean'),
                                    num_75=('num_75','mean'),
                                    num_985=('num_985','mean'),
                                    num_100=('num_100','mean'),
                                    num_unq=('num_unq','mean'),
                                    total_secs=('total_secs','mean')
                                    )
    
        
        cols = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100']
        num_25_75 = df.loc[:, cols[1:3]].sum(axis=1)
        num_75_100 = df.loc[:, cols[3:]].sum(axis=1)
        total = df.loc[:, cols].sum(axis=1)
        df['per_25'] = df['num_25'].div(total, axis=0)
        df['per_25_75'] = num_25_75.div(total, axis=0)
        df['per_100'] = num_75_100.div(total, axis=0)

        new_col = 'total_plays_per_unique'
        df[new_col] = total.div(df['num_unq'], axis=0)
        df = df.reset_index()

        user_log_group = df[['msno_num','total_secs', 'per_25', 'per_25_75', 'per_100', 'total_plays_per_unique']].copy()
        
        return user_log_group
        



