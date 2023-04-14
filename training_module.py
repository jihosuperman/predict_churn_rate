import pandas as pd
import numpy as np
import datetime

# 각 데이터 불러올 때 컬러명 변경 절대 금지!

class TrainingModule():

    def __init__(self):
        self.train = None
        self.members = None
        self.transcation = None
        self.userlog = None

    def import_data(self, train_path=None, memeber_path=None, transaction_path=None, userlog_path=None):
        try:   
            ##### Train data 불러오기 #####
            if train_path == None:
                self.train == None
            else:
                self.train = pd.read_parquet(train_path)
                self.train.date = pd.to_datetime(self.train.date, format='%Y-%m-%d')

            ##### Members data 불러오기 #####
            if memeber_path == None:
                self.members == None
            else:
                self.members = pd.read_parquet(memeber_path)
                self.members.gender.fillna('donknow')     ######## 젠더 Null 값 donknow 추가 ############ 수정
                self.members.registration_init_time = pd.to_datetime(self.members.registration_init_time, format='%Y-%m-%d')

                
            ##### Transaction data 불러오기 & 필터링 #####
            if transaction_path == None:
                self.transaction == None
            else:
                self.transaction = pd.read_parquet(transaction_path)
                self.transaction.date = pd.to_datetime(self.transaction.date, format='%Y-%m-%d')
                # 아이디와 거래일의 중복값 중 첫번째 & 구독 취소 유저가 아닌 경우를 제거
                transaction_filter1 = (~((self.transcation.duplicated(['msno_num', 'transaction_date'], keep='first')) & (self.transcation['is_cancel'] == 0)))

                # 위에서 필터링되지 않은 같은 구독 취소 거래가 중복된 값 제거
                cols = list(self.transcation.columns)[1:9]
                cols.remove('membership_expire_date')
                transaction_filter2 = (~((self.transcation.duplicated(cols, keep='first')) & (self.transcation['is_cancel'] == 1)))

                # 멤버쉽 만료일 > 거래일 인 경우만 남김
                transaction_filter3 = (self.transcation['transaction_date'] < self.transcation['membership_expire_date'])

                min_timestamp = pd.Timestamp(datetime.date(2005,1,1))
                max_timestamp = pd.Timestamp(datetime.date(2018,4,30))

                transaction_filter4 = ((self.transcation['membership_expire_date'] >= min_timestamp) & (self.transcation['membership_expire_date'] <= max_timestamp))

                self.transaction = self.transaction[transaction_filter1&transaction_filter2&transaction_filter3&transaction_filter4]

            ##### Userlog data 불러오기 #####
            if userlog_path == None:
                self.userlog == None
            else:
                self.userlog = pd.read_parquet(userlog_path)
                self.userlog.date = pd.to_datetime(self.userlog.date, format='%Y-%m-%d')
        except:
            print('Something Wrong with import_data() method')

    def userlog_filtering(self, use=0):
        try:
            if use==0:
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

                return self.userlog
        except:
            print("Something is wrong with userlog_filtering()")
    

    def age(self, use): # 0 : 그대로 사용 # 1 : 15 ~ 64세만 사용
        try:
            if int(use) == 0:
                self.members.loc[self.members['bd'] < 0, 'bd'] = 0
                self.members.loc[self.members['bd'] >= 100, 'bd'] = 100

                return self.members
            else:
                self.members['bd'] = self.members['bd'].apply(lambda x: x if (x >= 15) & (x <=64) else -1)
                # Null : -1
                return self.members
        except:
            print('Wrong input')

    def ageCategory(self, use): # 0 : 숫자 그대로 사용 / 1 : 10세 단위로 구분  / 2: 5세 단위 구분 / 3: 25세 이하 이상
        try:
            if int(use) == 0:
                return self.members
            
            elif int(use) == 1:
                bins = [15, 26, 36, 46, 56, 64]
                labels = ['16-26','26-36','36-46','46-56','56-64']
                self.members['bd'] = pd.members.cut(self.members['bd'], bins=bins, labels=labels)
                self.members['bd'] = self.members['bd'].apply(lambda x: '이상' if type(x) is int else x )
                return self.members
            elif int(use) == 2:
                bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
                labels = ['16-26','26-36','36-46','46-56','56-64']
                self.members['bd'] = self.members.cut(self.members['bd'], bins=bins, labels=labels)
                self.members['bd'] = self.members['bd'].apply(lambda x: '이상' if type(x) is int else x )
                return self.members
            elif int(use) == 3:
                self.members['bd'] = self.members['bd'].apply(lambda x: -1 if( (x < 16) or (x > 64)) else x)
                ## 16세 미만 또는 64세 초과  = -1
                self.members['bd'] = self.members['bd'].apply(lambda x: 1 if (x != -1) & (x <=25) else 0)
                ## -1 (이상한 값) 아니고 25세 이하 = 1 나머지 0
                return self.members
        except:
            print('Wrong input')

    def cityBinary(self, use): # 0 : 숫자 그대로 사용 , 1 : binary
        try:    
            if int(use) == 0:
                return self.members
            else:            
                self.members['city'] = np.where(self.members['city'] == 1, 1, 0)
                return self.members
        except:
            print('Something is wrong with cityBinary()')
            
    
    def autoRenew(self, use):
        try:
            if int(use) == 0:
                df = self.transaction.groupby('msno_num')[['is_auto_renew']].sum()
                self.members = self.members.merge(df, on='msno_num', how='inner')
                return self.members
            elif int(use) == 1:
                df = self.transaction.groupby('msno_num')[['is_auto_renew']].max()
                self.members = self.members.merge(df, on='msno_num', how='inner')
                return self.members
            else:
                print('Wrong_input in auto_renew')
        except:
            print('Something is wrong with autoRenew()')

    def paymentMethodId(self, use):
        try:
            if int(use) == 0:
                self.transaction['payment_method_id'] = pd.cut(self.transaction['payment_method_id'],
                                                            bins=[1,12,42],
                                                            right=False,
                                                            labels=[0,1,2])
                self.members = self.members.merge(self.transaction[['msno_num', 'payment_method_id']], on='msno_num', how='inner')
                return self.members
        except:
            print('Something is wrong with paymentMethodId()')
    
    def dicountRecord(self,use): # 0 : 할인 X # 1 : 할인 O
        try:
            if int(use) == 0:
                self.transaction['discount_record'] = self.transaction['plan_list_price'] - self.transaction['actual_amount_paid']
                df = self.transaction.groupby('msno_num')[['discount_record']].sum()
                df['discount_record'] = np.where(df['discount_record'] > 0, 1, 0)
                self.members = self.members.merge(self.transaction[['msno_num', 'discount_record']], on='msno_num', how='inner')

                return self.members
        except:
            print('Something is wrong with dicountRecord()')
    
    def delayTran(self, use): ## 가입 후 첫 거래일 까지 걸린 시간
        try:
            if int(use) == 0:
                transaction = self.transaction.groupby('msno_num')[['transaction_date']].min()
                df = pd.merge(transaction, pd.merge(transaction, self.members, how='inner', on='msno_num'), how='inner', on='msno_num')
                df['reg_first_subs'] = round(((df['transaction_date'] - df['registration_init_time']).dt.days/360),0)
                self.members = self.members.merge(df[['msno_num', 'reg_first_subs']], on='msno_num', how='inner')
                return self.members
        except:
            print("Something is wrong with delayTran()")

    def membershipDuration(self, use = 0): # 멤버쉽 유지기간
        try:
            if int(use) ==0:   
                transaction = self.transaction[['msno_num','transaction_date','membership_expire_date']]
                transaction['year_month'] =  transaction['transaction_date'].dt.to_period('M')
                transaction = transaction.drop_duplicates(subset=['msno_num', 'year_month'], keep='last')
                transaction = transaction['msno_num'].value_counts().reset_index()
                transaction.columns = ['msno_num', 'duration']
                self.members = pd.merge(self.members,transaction , on='msno_num')
                return  self.members

        except:
            print('Something is wrong with membershipDuration')
        
    def isCancel(self, use): # 구독취소여부
        try:
            if int(use) == 0:
                is_cancel = self.transaction.groupby('msno_num').agg({'is_cancel':'sum'}).reset_index()
                is_cancel['is_cancel'] = np.where(is_cancel['is_cancel'] > 1, 1, 0)
                self.members = self.members.merge(is_cancel[['msno_num', 'is_cancel']], on='msno_num', how='inner')
                return self.members
        except:
            print('Something is wrong with isCancel()')

    def longTermUnconnect(self, use):
        try:
            if use == 0:
                df = self.userlog.sort_values(['msno_num', 'date'])
                df['unconectted'] = self.userlog.groupby('msno_num')['date'].diff().fillna(pd.Timedelta(seconds=0))
                df = df.groupby('msno_num')['unconectted'].max()

                self.members.merge(df, on='msno_num', how='inner')
                return self.members
            
            elif use == 1:
                df = self.userlog.sort_values(['msno_num', 'date'])
                df['unconectted'] = self.userlog.groupby('msno_num')['date'].diff().fillna(pd.Timedelta(seconds=0))
                df = df.groupby('msno_num')['unconectted'].max()
                df['unconectted'] = df['unconectted'].apply(lambda x: 1 if x >= 30 else 0)

                self.members.merge(df, on='msno_num', how='inner')
                return self.members
            else:
                print("Wrong input in longTermUnconnect()")
        except:
            print("Something is wrong with longTermUnconnect()")
        
    def userLogGroup(self, use):
        try:
            if int(use) == 0:
                self.userlog['date'] = self.userlog['date'].dt.strftime('%Y-%m')
                df = self.userlog.groupby(['msno_num', 'date']).agg(log_count=('date','count'),
                                                                    num_25=('num_25','mean'),
                                                                    num_50=('num_50','mean'),
                                                                    num_75=('num_75','mean'),
                                                                    num_985=('num_985','mean'),
                                                                    num_100=('num_100','mean'),
                                                                    total_secs=('total_secs','sum')
                                                                    )
                
                df = df.groupby('msno').agg(log_count=('log_count','mean'),
                                            num_25=('num_25','mean'),
                                            num_50=('num_50','mean'),
                                            num_75=('num_75','mean'),
                                            num_985=('num_985','mean'),
                                            num_100=('num_100','mean'),
                                            total_secs=('total_secs','mean')
                                            )

                cols = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100']
                num_25_75 = self.userlog.loc[:, cols[1:3]].sum(axis=1)
                num_75_100 = self.userlog.loc[:, cols[3:]].sum(axis=1)
                total = self.userlog.loc[:, cols].sum(axis=1)
                df['per_25'] = df['num_25'].div(total, axis=0)
                df['per_25_75'] = num_25_75.div(total, axis=0)
                df['per_100'] = num_75_100.div(total, axis=0)

                new_col = 'total_plays_per_unique'
                df[new_col] = total.div(df['num_unq'], axis=0)

                self.members.merge(df, on='msno_num', how='inner')

                return self.members
            else:
                print("Wrong Input in userLogGroup()")

        except:
            print("Someting is Wrong with userLogGroup()")



    def preProcessing(self):
        self.train  
        self.members = None
        self.transcation = None
        self.transcation_grouped = None
        self.userlog = None
        self.userlog_grouped = None