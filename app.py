import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

st.title('Income Prediction App')
st.header('Upload Training Data Files')

uploaded_file1=st.file_uploader("Choose the adult.csv file (Training data)", type="csv", key='train1')
uploaded_file2=st.file_uploader("Choose the adult_test.csv file (Test data)", type="csv", key='train2')

if uploaded_file1 is not None and uploaded_file2 is not None:
    df_1=pd.read_csv(uploaded_file1, header=None)
    df_2=pd.read_csv(uploaded_file2, header=None)
    
    df=pd.concat([df_1, df_2], ignore_index=True)
    
    df.columns=["age","workclass","fnlwgt","education","education-num","marital-status",
               "occupation","relationship","race","sex","capital-gain","capital-loss",
               "hours-per-week","native-country","income"]
    
    df=df.drop(columns=["capital-gain","capital-loss"])
    
    df=df.apply(lambda x: x.str.strip() if x.dtype=="object" else x)
    
    df=df[df["native-country"] != "?"]
    mode_workclass=df["workclass"].mode()[0]
    df["workclass"]=df["workclass"].apply(lambda x: mode_workclass if x == "?" else x)
    mode_occupation=df["occupation"].mode()[0]
    df["occupation"]=df["occupation"].apply(lambda x: mode_occupation if x == "?" else x)
    
    columns_numerical=["age","fnlwgt","education-num","hours-per-week"]
    for i in columns_numerical:
        df[i]=pd.to_numeric(df[i], errors="coerce")
    
    df.drop_duplicates(inplace=True)
    
    df=df.apply(lambda x: x.str.replace("-"," ") if x.dtype=="object" else x)
    
    df=df.apply(lambda x: x.str.lower() if x.dtype=="object" else x)
    
    df["income"]=df["income"].replace({"<=50k.":"<=50k",">50k.":">50k"})
    
    df["age_group"]=pd.cut(df["age"], bins=4, labels=["young","middle aged","senior","elder"])
    df["hours-per-week-cat"]=pd.cut(df["hours-per-week"], bins=4, labels=["less","ideal","more","extreme"])
    df["education-num-cat"]=pd.cut(df["education-num"], bins=5, labels=["basic","lower","intermediate","higher","very high"])
    
    for i in columns_numerical:
        mean=df[i].mean()
        std=df[i].std()
        threshold=3 * std
        df=df[(df[i] >= mean - threshold) & (df[i] <= mean + threshold)]
    
    df["fnlwgt_log"]=np.log1p(df["fnlwgt"])
    
    df=df.drop(columns=["fnlwgt","fnlwgt_log","education-num","education","hours-per-week","age"])
    
    labelencoder=LabelEncoder()
    columns_ordinal=["workclass","income","age_group","hours-per-week-cat","education-num-cat"]
    encoders={}
    for i in columns_ordinal:
        df[i]=labelencoder.fit_transform(df[i])
        encoders[i]=labelencoder.classes_
    
    df["native-country"]=df["native-country"].apply(lambda x: "usa" if x == "united states" else "others")
    df["race"]=df["race"].apply(lambda x: "white" if x == "white" else "others")
    df["occupation"]=df["occupation"].replace([
        "other service","machine op inspct","transport moving","handlers cleaners",
        "tech support","farming fishing","protective serv","priv house serv",
        "armed forces","sales"], "others")
    df["marital-status"]=df["marital-status"].replace([
        "divorced","separated","widowed","married spouse absent","married af spouse"], "others")
    
    df=pd.get_dummies(df, columns=["occupation","relationship","race","sex","native-country","marital-status"], drop_first=True)
    
    st.write('Processed Data Sample:')
    st.write(df.head())
    
    st.session_state['encoders']=encoders
    st.session_state['mode_workclass']=mode_workclass
    st.session_state['mode_occupation']=mode_occupation
    
    X=df.drop(columns=["income"])
    y=df["income"]
    
    sc=MinMaxScaler()
    X_scaled=pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    st.session_state['scaler']=sc
    
    X_train, X_test, y_train, y_test=train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    
    rfc=RandomForestClassifier(n_estimators=80, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred_rfc=rfc.predict(X_test)
    accuracy_rfc=accuracy_score(y_test, y_pred_rfc)
    st.write(f'Random Forest Classifier Accuracy: {accuracy_rfc:.2f}')
    
    gbc=GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(X_train, y_train)
    y_pred_gbc=gbc.predict(X_test)
    accuracy_gbc=accuracy_score(y_test, y_pred_gbc)
    st.write(f'Gradient Boosting Classifier Accuracy: {accuracy_gbc:.2f}')
    
    logregr=LogisticRegression(max_iter=1000)
    logregr.fit(X_train, y_train)
    y_pred_logreg=logregr.predict(X_test)
    accuracy_logreg=accuracy_score(y_test, y_pred_logreg)
    st.write(f'Logistic Regression Accuracy: {accuracy_logreg:.2f}')
    
    svm=SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm=svm.predict(X_test)
    accuracy_svm=accuracy_score(y_test, y_pred_svm)
    st.write(f'SVM Accuracy: {accuracy_svm:.2f}')
    
    st.session_state['rfc_model']=rfc
    st.session_state['gbc_model']=gbc
    st.session_state['logreg_model']=logregr
    st.session_state['svm_model']=svm
    
    st.header('Predict Income for New Data')
    
    with st.form('prediction_form'):
        st.write('Please input the following features:')
        
        workclass_options=st.session_state['encoders']['workclass']
        age_group_options=st.session_state['encoders']['age_group']
        hours_per_week_cat_options=st.session_state['encoders']['hours-per-week-cat']
        education_num_cat_options=st.session_state['encoders']['education-num-cat']
        
        workclass_input=st.selectbox('Workclass', options=workclass_options)
        age_group_input=st.selectbox('Age Group', options=age_group_options)
        hours_per_week_cat_input=st.selectbox('Hours per Week Category', options=hours_per_week_cat_options)
        education_num_cat_input=st.selectbox('Education Num Category', options=education_num_cat_options)
        
        occupation_options=[col.replace('occupation_', '') for col in df.columns if col.startswith('occupation_')]
        relationship_options=[col.replace('relationship_', '') for col in df.columns if col.startswith('relationship_')]
        race_options=[col.replace('race_', '') for col in df.columns if col.startswith('race_')]
        sex_options=[col.replace('sex_', '') for col in df.columns if col.startswith('sex_')]
        native_country_options=[col.replace('native-country_', '') for col in df.columns if col.startswith('native-country_')]
        marital_status_options=[col.replace('marital-status_', '') for col in df.columns if col.startswith('marital-status_')]
        
        occupation_input=st.selectbox('Occupation', options=occupation_options)
        relationship_input=st.selectbox('Relationship', options=relationship_options)
        race_input=st.selectbox('Race', options=race_options)
        sex_input=st.selectbox('Sex', options=sex_options)
        native_country_input=st.selectbox('Native Country', options=native_country_options)
        marital_status_input=st.selectbox('Marital Status', options=marital_status_options)
        
        model_choice=st.selectbox('Select the model to use for prediction:', ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'SVM'])
        
        submitted=st.form_submit_button('Predict')
        if submitted:

            input_data={}
            
            inputs={
                'workclass': workclass_input,
                'age_group': age_group_input,
                'hours-per-week-cat': hours_per_week_cat_input,
                'education-num-cat': education_num_cat_input
            }
            for col in ['workclass','age_group','hours-per-week-cat','education-num-cat']:
                le=LabelEncoder()
                le.classes_=st.session_state['encoders'][col]
                input_data[col]=le.transform([inputs[col]])[0]
            
            for col in df.columns:
                if col.startswith('occupation_'):
                    input_data[col]=1 if col==f'occupation_{occupation_input}' else 0
                elif col.startswith('relationship_'):
                    input_data[col]=1 if col==f'relationship_{relationship_input}' else 0
                elif col.startswith('race_'):
                    input_data[col]=1 if col==f'race_{race_input}' else 0
                elif col.startswith('sex_'):
                    input_data[col]=1 if col==f'sex_{sex_input}' else 0
                elif col.startswith('native-country_'):
                    input_data[col]=1 if col==f'native-country_{native_country_input}' else 0
                elif col.startswith('marital-status_'):
                    input_data[col]=1 if col==f'marital-status_{marital_status_input}' else 0
                else:
                    continue
            
            input_df=pd.DataFrame([input_data])
            
            X_columns=[col for col in df.columns if col != 'income']
            input_df=input_df.reindex(columns=X_columns, fill_value=0)
            
            sc=st.session_state['scaler']
            input_scaled=sc.transform(input_df)
            
            if model_choice=='Random Forest':
                model=st.session_state['rfc_model']
            elif model_choice=='Gradient Boosting':
                model=st.session_state['gbc_model']
            elif model_choice=='Logistic Regression':
                model=st.session_state['logreg_model']
            elif model_choice=='SVM':
                model=st.session_state['svm_model']
            else:
                st.write('Invalid model choice.')
                model=None
            
            if model is not None:

                prediction=model.predict(input_scaled)
                le_income=LabelEncoder()
                le_income.classes_=st.session_state['encoders']['income']
                predicted_income=le_income.inverse_transform([int(round(prediction[0]))])[0]
                st.write(f'Predicted Income: {predicted_income}')
