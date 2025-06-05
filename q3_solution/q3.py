import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

df = pd.read_csv('ethusdt_ticks.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)



df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2


df['obi'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-9)
df['buy_volume'] = np.where(df['side'] == 'buy', df['trade_size'], 0)
df['sell_volume'] = np.where(df['side'] == 'sell', df['trade_size'], 0)
df['tfi'] = (df['buy_volume'] - df['sell_volume']).rolling('10s').sum()


df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
df['volatility'] = df['log_return'].rolling('1min').std()


df['spread'] = df['ask_price'] - df['bid_price']
df['signed_volume'] = np.where(df['side'] == 'buy', df['trade_size'], -df['trade_size'])
df['price_impact'] = df['signed_volume'] * df['spread']
df['trade_count'] = 1
df['intensity'] = df['trade_count'].rolling('10s').sum()

agg = {
    'obi': 'last',
    'tfi': 'last',
    'volatility': 'last',
    'spread': 'mean',
    'price_impact': 'mean',
    'intensity': 'mean',
    'mid_price': 'last'
}
features = df.resample('1min').agg(agg)


features['future_price'] = features['mid_price'].shift(-1)
features['return'] = (features['future_price'] - features['mid_price']) / features['mid_price']
features['label'] = np.where(features['return'] > 0, 1, 0)

features = features.dropna()


X = features.drop(columns=['future_price', 'return', 'label', 'mid_price'])
y = features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.9)

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
