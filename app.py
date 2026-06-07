import gc

import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt


from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgbm

st.title('Telco Customer Churn Prediction')

st.markdown("""
Customer churn is one of the biggest challenges in the telecom industry.
Losing a customer is always more costly than retaining one — so being able to predict *who might leave* before they actually do is incredibly valuable.

This project is my attempt at building a churn prediction system using three powerful gradient boosting models: **XGBoost**, **CatBoost**, and **LightGBM**.
I trained and tuned each model on real telecom customer data, and this app lets you interact with them directly.

## How to use this app

1. Pick a model from the **sidebar** — XGBoost, CatBoost, or LightGBM
2. Hit **`Performance on Test Dataset`** to see how well the model performs (ROC AUC + curve)
3. Hit **`Prediction on Random Instance`** to test the model on a random customer from the test set
4. Or fill in customer details manually in the sidebar and click **`Predict`** to get a live prediction
5. All results show up in the **[Prediction Result](#prediction-result)** section below

---

[![](https://img.shields.io/badge/GitHub-View%20Source%20Code-100000?logo=github&logoColor=white)](https://github.com/tanmaytalekar13/Customer-Churn-Prediction)

""")

df_churn = pd.read_csv("dataset//Telco-Customer-Churn-dataset-cleaned.csv")
df_train = pd.read_csv('dataset//Telco-Customer-Churn-dataset-Train.csv', index_col=0)
df_test = pd.read_csv('dataset//Telco-Customer-Churn-dataset-Test.csv', index_col=0)

st.header('Churn Data Overview')
st.write('Data Dimension: ' + str(df_churn.shape[0]) + ' rows and ' + str(df_churn.shape[1]) + ' columns.')
st.dataframe(df_churn)


@st.cache_data
def download_dataset(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
    return href


st.markdown(download_dataset(df_churn), unsafe_allow_html=True)

st.markdown("## Prediction Result")


st.sidebar.markdown("## Predict Customer Churn Rate")
classifier_name = st.sidebar.selectbox(
    'Select a Classifier',
    ('XGBoost', 'CatBoost', 'LightGBM')
)


def get_classifier(clf_name):
    if clf_name == 'XGBoost':
        clf = xgb.XGBClassifier()  # init model
        clf.load_model("models/model_xgb.json")
    elif clf_name == 'CatBoost':
        clf = CatBoostClassifier()  # parameters not required.
        clf.load_model('models/model_catboost')
    else:
        clf = lgbm.Booster(model_file='models/model_lgbm.txt')
    return clf


clf = get_classifier(classifier_name)


def get_transformed_data(test_data=None):
    X = df_train.drop("Churn", axis=1)

    if test_data is None:
        test_data = df_test.copy()
    # test dataset
    y_test = test_data['Churn'].values
    X_test = test_data.drop("Churn", axis=1)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = list(set(X.columns) - set(X._get_numeric_data().columns))

    ordinal_encoder = OrdinalEncoder()
    X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])
    X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])

    transformer = RobustScaler()
    X[num_cols] = transformer.fit_transform(X[num_cols])
    X_test[num_cols] = transformer.transform(X_test[num_cols])

    del X
    gc.collect()
    return X_test, y_test


def make_prediction(X_test):
    try:
        # xgboost, catboost
        test_pred = clf.predict_proba(X_test)[:, 1]  # probability of getting 1
    except AttributeError:
        # lgbm load model
        # https://github.com/Microsoft/LightGBM/issues/1217
        test_pred = clf.predict(X_test)
    return test_pred


if st.sidebar.button('Performance on Test Dataset'):
    X_test, y_test = get_transformed_data()
    test_pred = make_prediction(X_test)
    st.write("Performance on The Test Dataset ( ROC AUC Score) : ", roc_auc_score(y_test, test_pred))

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(plt)

if st.sidebar.button('Prediction on Random Instance from Test Data'):
    random_test_instance = df_test.sample(n=1)
    X_test, y_test = get_transformed_data(random_test_instance)
    test_pred = make_prediction(X_test)
    st.markdown(f"Prediction on Random Instance From Test Data : {'**Churned**' if test_pred[0]>0.5 else '**Not Churned**'} (Probability: {test_pred[0] : 0.2f}) ")
    st.write("Random Instance Features")
    st.dataframe(random_test_instance)


st.sidebar.markdown('## User Input')


def binning_feature(feature, value):
    bins = np.linspace(min(df_churn[feature]), max(df_churn[feature]), 4)
    if bins[0] <= value <= bins[1]:
        return 'Low'
    elif bins[1] < value <= bins[2]:
        return 'Medium'
    else:
        return 'High'


def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ('Yes', 'No'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No', 'No phone service'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No'))
    internet_service_type = st.sidebar.selectbox('Internet Service Type', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))

    payment_method = st.sidebar.selectbox('PaymentMethod', (
        'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))

    # tenure filter
    unique_tenure_values = df_churn.tenure.unique()
    min_value, max_value = min(unique_tenure_values), max(unique_tenure_values)

    # tenure slider
    tenure = st.sidebar.slider("Tenure", int(min_value), int(max_value), int(min_value), 1)

    # MonthlyCharges filter
    unique_monthly_charges_values = df_churn.MonthlyCharges.unique()
    min_value, max_value = min(unique_monthly_charges_values), max(unique_monthly_charges_values)

    # MonthlyCharges slider
    monthly_charges = st.sidebar.slider("Monthly Charges", min_value, max_value, float(min_value))

    min_value_total = monthly_charges * tenure
    max_value_total = (monthly_charges * tenure) + 100

    st.sidebar.markdown("**`TotalCharges`** = `MonthlyCharges` * `Tenure` + `Extra Cost ( ~100 )`")

    # TotalCharges slider
    total_charges = st.sidebar.slider("Total Charges", min_value_total, max_value_total)

    # Churn filter
    data = {'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen.lower() == 'yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service_type],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'tenure-binned': binning_feature('tenure', 7),
            'MonthlyCharges-binned': binning_feature('MonthlyCharges', monthly_charges),
            'TotalCharges-binned': binning_feature('TotalCharges', total_charges)
            }

    features = pd.DataFrame(data)

    return features


input_df = user_input_features()

# Use training data column order and types as the reference
X = df_train.drop("Churn", axis=1)
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = list(set(X.columns) - set(X._get_numeric_data().columns))

# Align user_df to have the exact same columns and order as training data
user_df = input_df.copy()
user_df = user_df[X.columns]  # reorder columns to match training set

ordinal_encoder = OrdinalEncoder()
X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])
user_df[cat_cols] = ordinal_encoder.transform(user_df[cat_cols])

transformer = RobustScaler()
X[num_cols] = transformer.fit_transform(X[num_cols])
user_df[num_cols] = transformer.transform(user_df[num_cols])

if st.sidebar.button('Predict'):
    test_pred = make_prediction(user_df)
    st.markdown(f"Prediction result : {'**Churned**' if test_pred[0]>0.5 else '**Not Churned**'} (Probability: {test_pred[0] : 0.2f}) ")
    st.write("User Input Features")
    st.dataframe(input_df)