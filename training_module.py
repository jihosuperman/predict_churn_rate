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
        self.train = pd.read_parquet(train_path)
        self.members = pd.read_parquet(memeber_path)
        self.transaction = pd.read_parquet(transaction_path)
        self.userlog = pd.read_parquet(userlog_path)

        self.train.date = pd.to_datetime(self.train.date, format='%Y-%m-%d')
        self.members.date = pd.to_datetime(self.members.date, format='%Y-%m-%d')
        self.transaction.date = pd.to_datetime(self.transaction.date, format='%Y-%m-%d')
        self.userlog.date = pd.to_datetime(self.userlog.date, format='%Y-%m-%d')

    def transaction_filtering(self):
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

        return self.transaction
