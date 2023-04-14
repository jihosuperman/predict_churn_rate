import pandas as pd
import numpy as np
import datetime

# 각 데이터 불러올 때 컬러명 변경 절대 금지!

class TrainingModule():

    def __init__(self):
        self.train = None
        self.members = None
        self.transcation = None
        self.transcation_grouped = None
        self.userlog = None
        self.userlog_grouped = None

    def import_data(self, train_path=None, memeber_path=None, transaction_path=None, userlog_path=None):
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

    def userlog_filtering(self, condition=1):
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
        
        if condition == 1:
            self.userlog = self.userlog[filter1]
        
        elif condition == 2:
            self.userlog = self.userlog[filter1&filter2]

        elif condition == 3:
            self.userlog = remove_quantile(self.userlog[filter1 & filter2])

        return self.userlog
    
    def gender(self, use): # 성별 컬럼을 사용할거면 1 아니면 0
        try:
            if int(use) == 1 :
                return self.members
            elif int(use) == 0:
                return self.members.loc[:, ~self.members.columns.isin(['gender'])]
        except:
            print('Wrong input')

    def age(self, use): # 0 : 그대로 사용 # 1 : 15 ~ 64세만 사용
        try:
            if int(use) == 0:
                self.members.loc[self.members['bd'] <0, 'bd'] = 0
                self.members.loc[self.members['bd'] >0, 'bd'] = 100
                return self.members
            else:
                self.members['True'] = self.members['bd'].apply(lambda x: 1 if (x >= 15) & (x <=64) else 0)
                self.members = self.members[self.members['True'] == 1]
                return self.members.loc[:, ~self.members.columns.isin(['True'])]
        except:
            print('Wrong input')

    def cityBinary(self): # city 컬럼을 binary 로 사용할때!
        self.members['city'] = np.where(self.members['city'] == 1, 1, 0)
        return self.members.loc[:, ~self.members.columns.isin(['city'])]
    
    def delayTran(self): ## 가입 후 첫 거래일 까지 걸린 시간
        transaction = self.transaction.groupby('msno_num')[['transaction_date']].min()
        df = pd.merge(transaction, pd.merge(transaction, self.member, how='inner', on='msno_num'), how='inner', on='msno_num')
        df['reg_first_subs'] = round(((df['transaction_date'] - df['registration_init_time']).dt.days/360),0)
        self.member = self.member.merge(df[['msno_num', 'reg_first_subs']], on='msno_num', how='inner')
        return self.member
    
    def auto_renew(self, use):
        try:
            if use == 0:
                df = self.transaction.groupby('msno_num')[['is_auto_renew']].sum()
                self.member = self.member.merge(df, on='msno_num', how='inner')
                return self.member
            else:
                df 
        except:
            print('Wrong input')
