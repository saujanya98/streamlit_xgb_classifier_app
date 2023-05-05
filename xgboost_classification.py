import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

df = pd.read_csv('drug200.csv')

encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['BP'] = encoder.fit_transform(df['BP'])
df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])

# Define mapping function
def map_drug_names(drug_name):
    drug_map = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4}
    return drug_map[drug_name]

# Apply mapping function to 'Drug' column in a pandas DataFrame
df['Drug'] = df['Drug'].apply(map_drug_names)

X = df.drop('Drug',axis=1).values
y = df['Drug'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Set up parameter grid for grid search
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0]
}

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=5)

# Perform grid search
grid_search = GridSearchCV(xgb_model, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters and train XGBoost classifier
best_params = grid_search.best_params_
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=5, **best_params)
xgb_model.fit(X_train, y_train)

# Get predicted classes on test set
y_pred = xgb_model.predict(X_test)

# Generate classification report
class_report = classification_report(y_test, y_pred)
print(class_report)

cm = confusion_matrix(y_test,y_pred)
ax = sns.heatmap(cm,annot=True,cmap='Blues')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')