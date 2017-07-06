
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb

get_ipython().magic(u'matplotlib inline')


# In[2]:


train_df = pd.read_csv('train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('test.csv', parse_dates=['timestamp'])
print train_df.shape
print test_df.shape


# In[24]:


plt.figure(figsize=(8,8))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=16)
plt.ylabel('price', fontsize=16)
plt.show()


# In[25]:


plt.figure(figsize=(12,8))
sns.distplot(train_df.price_doc.values, bins=100, kde=True)
plt.xlabel('price', fontsize=16)
plt.show()


# In[26]:


missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values)
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# In[3]:


test_id = test_df.id

bad_index = train_df[train_df.life_sq > train_df.full_sq].index
train_df.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test_df.loc[equal_index, "life_sq"] = test_df.loc[equal_index, "full_sq"]
bad_index = test_df[test_df.life_sq > test_df.full_sq].index
test_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = train_df[train_df.life_sq < 5].index
train_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = test_df[test_df.life_sq < 5].index
test_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = train_df[train_df.full_sq < 5].index
train_df.loc[bad_index, "full_sq"] = np.NaN
bad_index = test_df[test_df.full_sq < 5].index
test_df.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train_df.loc[kitch_is_build_year, "build_year"] = train_df.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train_df[train_df.kitch_sq >= train_df.life_sq].index
train_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test_df[test_df.kitch_sq >= test_df.life_sq].index
test_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train_df[(train_df.kitch_sq == 0).values + (train_df.kitch_sq == 1).values].index
train_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test_df[(test_df.kitch_sq == 0).values + (test_df.kitch_sq == 1).values].index
test_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train_df[(train_df.full_sq > 210) & (train_df.life_sq / train_df.full_sq < 0.3)].index
train_df.loc[bad_index, "full_sq"] = np.NaN
bad_index = test_df[(test_df.full_sq > 150) & (test_df.life_sq / test_df.full_sq < 0.3)].index
test_df.loc[bad_index, "full_sq"] = np.NaN
bad_index = train_df[train_df.life_sq > 300].index
train_df.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test_df[test_df.life_sq > 200].index
test_df.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train_df.product_type.value_counts(normalize= True)
test_df.product_type.value_counts(normalize= True)
bad_index = train_df[train_df.build_year < 1500].index
train_df.loc[bad_index, "build_year"] = np.NaN
bad_index = test_df[test_df.build_year < 1500].index
test_df.loc[bad_index, "build_year"] = np.NaN
bad_index = train_df[train_df.num_room == 0].index
train_df.loc[bad_index, "num_room"] = np.NaN
bad_index = test_df[test_df.num_room == 0].index
test_df.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train_df.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test_df.loc[bad_index, "num_room"] = np.NaN
bad_index = train_df[(train_df.floor == 0).values * (train_df.max_floor == 0).values].index
train_df.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train_df[train_df.floor == 0].index
train_df.loc[bad_index, "floor"] = np.NaN
bad_index = train_df[train_df.max_floor == 0].index
train_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = test_df[test_df.max_floor == 0].index
test_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = train_df[train_df.floor > train_df.max_floor].index
train_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = test_df[test_df.floor > test_df.max_floor].index
test_df.loc[bad_index, "max_floor"] = np.NaN
train_df.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train_df.loc[bad_index, "floor"] = np.NaN
train_df.material.value_counts()
test_df.material.value_counts()
train_df.state.value_counts()
bad_index = train_df[train_df.state == 33].index
train_df.loc[bad_index, "state"] = np.NaN
test_df.state.value_counts()


# In[4]:


train_df.loc[train_df.full_sq == 0, 'full_sq'] = 50
train_df = train_df[train_df.price_doc/train_df.full_sq <= 600000]
train_df = train_df[train_df.price_doc/train_df.full_sq >= 10000]

month_year = (train_df.timestamp.dt.month + train_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test_df.timestamp.dt.month + test_df.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

week_year = (train_df.timestamp.dt.weekofyear + train_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test_df.timestamp.dt.weekofyear + test_df.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

train_df['month'] = train_df.timestamp.dt.month
train_df['dow'] = train_df.timestamp.dt.dayofweek

test_df['month'] = test_df.timestamp.dt.month
test_df['dow'] = test_df.timestamp.dt.dayofweek

train_df['rel_floor'] = train_df['floor'] / train_df['max_floor'].astype(float)
train_df['rel_kitch_sq'] = train_df['kitch_sq'] / train_df['full_sq'].astype(float)

test_df['rel_floor'] = test_df['floor'] / test_df['max_floor'].astype(float)
test_df['rel_kitch_sq'] = test_df['kitch_sq'] / test_df['full_sq'].astype(float)

train_df.apartment_name=train_df.sub_area + train_df['metro_km_avto'].astype(str)
test_df.apartment_name=test_df.sub_area + train_df['metro_km_avto'].astype(str)

train_df['room_size'] = train_df['life_sq'] / train_df['num_room'].astype(float)
test_df['room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)


# In[5]:


rate_2015_q2 = 1
rate_2015_q1 = rate_2015_q2 / .9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
rate_2013_q1 = rate_2013_q2 / 1.0104
rate_2012_q4 = rate_2013_q1 / 0.9832 
rate_2012_q3 = rate_2012_q4 / 1.0277
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011

train_df['average_q_price'] = 1

train_df_2015_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2015].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_df_2015_q2_index, 'average_q_price'] = rate_2015_q2

train_df_2015_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2015].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_df_2015_q1_index, 'average_q_price'] = rate_2015_q1

train_df_2014_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_df_2014_q4_index, 'average_q_price'] = rate_2014_q4

train_df_2014_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_df_2014_q3_index, 'average_q_price'] = rate_2014_q3

train_df_2014_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_df_2014_q2_index, 'average_q_price'] = rate_2014_q2

train_df_2014_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2014].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_df_2014_q1_index, 'average_q_price'] = rate_2014_q1

train_df_2013_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_df_2013_q4_index, 'average_q_price'] = rate_2013_q4

train_df_2013_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_df_2013_q3_index, 'average_q_price'] = rate_2013_q3

train_df_2013_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_df_2013_q2_index, 'average_q_price'] = rate_2013_q2

train_df_2013_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2013].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_df_2013_q1_index, 'average_q_price'] = rate_2013_q1

train_df_2012_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_df_2012_q4_index, 'average_q_price'] = rate_2012_q4

train_df_2012_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_df_2012_q3_index, 'average_q_price'] = rate_2012_q3

train_df_2012_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_df_2012_q2_index, 'average_q_price'] = rate_2012_q2

train_df_2012_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2012].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_df_2012_q1_index, 'average_q_price'] = rate_2012_q1

train_df_2011_q4_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 10].loc[train_df['timestamp'].dt.month <= 12].index
train_df.loc[train_df_2011_q4_index, 'average_q_price'] = rate_2011_q4

train_df_2011_q3_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 7].loc[train_df['timestamp'].dt.month < 10].index
train_df.loc[train_df_2011_q3_index, 'average_q_price'] = rate_2011_q3

train_df_2011_q2_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 4].loc[train_df['timestamp'].dt.month < 7].index
train_df.loc[train_df_2011_q2_index, 'average_q_price'] = rate_2011_q2

train_df_2011_q1_index = train_df.loc[train_df['timestamp'].dt.year == 2011].loc[train_df['timestamp'].dt.month >= 1].loc[train_df['timestamp'].dt.month < 4].index
train_df.loc[train_df_2011_q1_index, 'average_q_price'] = rate_2011_q1

train_df['price_doc'] = train_df['price_doc'] * train_df['average_q_price']


# In[6]:


mult = 1.03
train_df['price_doc'] = train_df['price_doc'] * mult
y_train = train_df["price_doc"]


# In[7]:


x_train = train_df.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
x_test = test_df.drop(["id", "timestamp"], axis=1)

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))

x_train = x_all[:num_train]
x_test = x_all[num_train:]

xgb_params = {
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 500
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_predict = model.predict(dtest)
gunja_output = pd.DataFrame({'id': test_id, 'price_doc': y_predict})


# In[8]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_id= test_df.id
mult = 0.8

y_train = train_df["price_doc"] * mult + 10
x_train = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test_df.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 400
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': test_id, 'price_doc': y_predict})


# In[9]:


df_train = pd.read_csv("train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'])

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

mult = 0.969
y_train = df_train['price_doc'].values * mult + 10
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

train_df['building_name'] = pd.factorize(train_df.sub_area + train_df['metro_km_avto'].astype(str))[0]
test_df['building_name'] = pd.factorize(test_df.sub_area + test_df['metro_km_avto'].astype(str))[0]

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(train_df[col].astype(str) + month_year.astype(str))[0])
   train_df[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(train_df[col].astype(str) + week_year.astype(str))[0])
   train_df[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(test_df[col].astype(str) + month_year.astype(str))[0])
   test_df[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(test_df[col].astype(str) + week_year.astype(str))[0])
   test_df[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)

factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

X_all = df_values.values

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

num_boost_rounds = 450
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})


# In[10]:


first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])
first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
                                    .286*np.log(first_result.price_doc_bruno) ) 
result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])

result["price_doc"] = np.exp( .78*np.log(result.price_doc_follow) +
                              .22*np.log(result.price_doc_gunja) )
                              
result["price_doc"] =result["price_doc"] *0.9915        
result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)
result.head()
result.to_csv('same_result.csv', index=False)

