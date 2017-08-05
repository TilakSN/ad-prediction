from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import datetime as dt
import sys
import time

start = time.time()
try:
    LEVEL = int(sys.argv[-1])
except:
    LEVEL = 1

DEBUG = ['hack', 'small', 'medium', 'large', 'huge'][LEVEL]
DEBUG += '/' + DEBUG

data = []
FILENAME = DEBUG + '%s.csv'
use_cols = ['weekday', 'hour', 'so_count', 'sm_count', 'sc_count', 'a', 'b', 'c', 'd', 'e', 'f', 'Chrome', 'Edge', 'Firefox', 'IE', 'NoBrowser', 'Opera', 'Safari', 'Desktop', 'Mobile', 'NoDevice', 'Tablet']
ALL_COLS = ['ID', 'datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid'] + use_cols + ['duration', 'click']

for filename in ('train', 'test'):
    print('[%7.3f]:' % (time.time() - start), 'Reading', filename)
    df = pd.read_csv(FILENAME % filename)

    # _ = df.pop('ID')

    print('[%7.3f]:' % (time.time() - start), 'Modifying date-time')
    df.datetime = pd.to_datetime(df.datetime)
    df['weekday'] = df.datetime.dt.weekday
    df['hour'] = df.datetime.dt.hour
    df['duration'] = (df.datetime - dt.datetime(2017, 1, 1)).dt.total_seconds() / 3600
    # _ = df.pop('datetime')

    print('[%7.3f]:' % (time.time() - start), 'Modifying site id')
    df.siteid.fillna(value=1, inplace=True)
    df.siteid = df.siteid.astype('int32')#.astype('str')

    print('[%7.3f]:' % (time.time() - start), 'Modifying 3 more')
    # df.offerid = df.offerid.astype('str')
    # df.category = df.category.astype('str')
    # df.merchant = df.merchant.astype('str')

    print('[%7.3f]:' % (time.time() - start), 'Modifying browser id')
    df.browserid.fillna(value='NoBrowser', inplace=True)
    df.replace(to_replace={'browserid': {'Google Chrome': 'Chrome', 'Internet Explorer': 'IE', 'InternetExplorer': 'IE', 'Mozilla': 'Firefox', 'Mozilla Firefox': 'Firefox'}}, inplace=True)
    print('[%7.3f]:' % (time.time() - start), 'Modifying device id')
    df.devid.fillna(value='NoDevice', inplace=True)

    print('[%7.3f]:' % (time.time() - start), 'Merge 1')
    so_count = df.groupby(['siteid', 'offerid']).size().reset_index()
    so_count.columns = ['siteid', 'offerid', 'so_count']
    df = df.merge(so_count)
    # _ = df.pop('offerid')

    print('[%7.3f]:' % (time.time() - start), 'Merge 2')
    sm_count = df.groupby(['siteid', 'merchant']).size().reset_index()
    sm_count.columns = ['siteid', 'merchant', 'sm_count']
    df = df.merge(sm_count)
    # _ = df.pop('merchant')

    print('[%7.3f]:' % (time.time() - start), 'Merge 3')
    sc_count = df.groupby(['siteid', 'category']).size().reset_index()
    sc_count.columns = ['siteid', 'category', 'sc_count']
    df = df.merge(sc_count)
    # _ = df.pop('category')
    # _ = df.pop('siteid')

    print('[%7.3f]:' % (time.time() - start), 'One hot 1')
    df = df.join(pd.get_dummies(df.countrycode))
    # _ = df.pop('countrycode')

    print('[%7.3f]:' % (time.time() - start), 'One hot 2')
    df = df.join(pd.get_dummies(df.browserid))
    # _ = df.pop('browserid')

    df = df.join(pd.get_dummies(df.devid))
    # _ = df.pop('devid')

    print('[%7.3f]:' % (time.time() - start), 'Missing columns')
    for i in use_cols:
        if i not in df.columns:
            df[i] = 0

    df = df[ALL_COLS]
    _ = ALL_COLS.pop()

    print('[%7.3f]:' % (time.time() - start), 'Done :)')
    data.append(df)

train, test = data
print('[%7.3f]:' % (time.time() - start), 'Scaling')
scaler = StandardScaler().fit(train[use_cols])
s_train = scaler.transform(train[use_cols])
train_out = train.click
train_weights = train.duration
s_test = scaler.transform(test[use_cols])
mean, std = train_out.mean(), train_out.std()

# ''' Comment this line to activate Validation split
from sklearn.model_selection import train_test_split
print('[%7.3f]:' % (time.time() - start), 'Sp    lit')
s_train, s_validation, train_out, validation_out, train_weights, validation_weights = train_test_split(s_train, train.click, train.duration, test_size=0.5, random_state=100)
# '''

print('[%7.3f]:' % (time.time() - start), 'Fit')
model = lm.LinearRegression()
model.fit(s_train, train_out, sample_weight=train_weights)

print('[%7.3f]:' % (time.time() - start), 'Validating')
validation_predict = model.predict(s_validation)
validation_score = np.sum((validation_out - (validation_predict * std + mean)) ** 2) / len(validation_out)
roc_score = roc_auc_score(validation_out, validation_predict)
print('[%7.3f]:' % (time.time() - start), 'PREDICT')
predict = model.predict(s_test) * std + mean

print('[%7.3f]:' % (time.time() - start), (1 - validation_score) * 100, end='%\n')
print('[%7.3f]:' % (time.time() - start), roc_score)

print('[%7.3f]:' % (time.time() - start), 'Storing')
df = pd.DataFrame({'ID': test.ID, 'click': predict})
df.to_csv(DEBUG + 'out.csv', index=False)

print('[%7.3f]:' % (time.time() - start), 'Finally Yayyyyyy :P')
