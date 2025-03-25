from preprocess import preprocessing_data, standardize_column_values
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import f1_score

print("Load the data to start the Training")

df = pd.read_excel('ml_data_2019.xlsx')

df = preprocessing_data(df)

# Separate inputs and output
X = df.drop(columns=['ÉLIGIBILITÉ_AU_DON.'])
y = df['ÉLIGIBILITÉ_AU_DON.']


categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ],
    remainder='passthrough'
)

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


best_params = {'n_estimators': 141,
                'max_depth': 7,
                'learning_rate': 0.27961974745824497,
                'subsample': 0.8238885878088886,
                'colsample_bytree': 0.8100229723755945,
                'gamma': 0.2896805464731108,
                'reg_alpha': 0.9637458082318096,
                'reg_lambda': 0.6717791469966395,
                'silent': 1,
                'scale_pos_weight': 4,
                'random_state': 2018,
                'verbosity': 0,
                'objective': 'multi:softmax',
                'num_class': 3}


X_p = preprocessor.fit_transform(X)

print("Training Started")

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
models = []
f1_s = []

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_p, y)):
    X_train, X_valid = X_p[train_idx], X_p[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')

    model.fit(
        X_train, y_train,
        #eval_set=[(X_valid, y_valid)],
        #eval_metric='auc',
        #callbacks=[lgb.early_stopping(50, verbose=False)],
        #categorical_feature=categorical_features,
    )
    print("x_valid shape :", X_valid.shape)
    y_pred = model.predict_proba(X_valid)[:, 1]
    y_pred_f1 = model.predict(X_valid)[:]
    score = f1_score(y_valid, y_pred_f1, average='weighted')
    f1_s.append(score)
    models.append(model)

print("Training Finished")

model = models[0]



preprocessor.fit_transform(X)
print("Save Model")
# Save the model and label encoder and preprocessor
joblib.dump({'model': model, 'label_encoder': label_encoder, 'preprocessor': preprocessor}, 'model_pipeline.pkl')
print("Model Saved")
