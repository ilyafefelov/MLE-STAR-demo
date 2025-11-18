
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Handle missing values by imputing with the median
for col in train_df.columns:
    if train_df[col].isnull().any():
        train_df[col] = train_df[col].fillna(train_df[col].median())

# Separate features and target
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM model
lgb_params = {
    'objective': 'rmse',
    'metric': 'rmse',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': 42,
}

model = lgb.LGBMRegressor(**lgb_params)
model.fit(X_train, y_train)

# Predictions on the validation set
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)

print(f"Final Validation Performance: {rmse}")
