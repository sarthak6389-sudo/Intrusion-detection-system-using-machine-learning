import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. SETUP THE PAGE
st.set_page_config(page_title="IDS Dashboard", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS)")
st.markdown("### A Machine Learning approach to detecting network attacks")

# 2. LOAD DATA (Cached so it doesn't reload every time)
@st.cache_data
def load_data():
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate', 'attack', 'level'])

    def fetch(url):
        return pd.read_csv(url, header=None, names=columns)
    
    df_train = fetch(train_url)
    df_test = fetch(test_url)
    return df_train, df_test

with st.spinner('Downloading and processing data...'):
    df_train, df_test = load_data()

# 3. SIDEBAR CONTROLS
st.sidebar.header("Model Settings")
n_trees = st.sidebar.slider("Number of Trees in Random Forest", 10, 200, 50)

# 4. PREPROCESSING
def preprocess(df):
    df['attack_flag'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop(['attack', 'level'], axis=1)
    le = LabelEncoder()
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = le.fit_transform(df[col])
    return df

df_train = preprocess(df_train)
df_test = preprocess(df_test)

X_train = df_train.drop('attack_flag', axis=1)
y_train = df_train['attack_flag']
X_test = df_test.drop('attack_flag', axis=1)
y_test = df_test['attack_flag']

# 5. TRAINING
if st.button("🚀 Train Model"):
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 6. RESULTS
    st.success(f"Model Trained Successfully! Accuracy: {acc*100:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Attack Distribution")
        fig2, ax2 = plt.subplots()
        pd.Series(y_pred).value_counts().plot(kind='bar', ax=ax2, color=['green', 'red'])
        ax2.set_xticklabels(['Normal', 'Attack'])
        st.pyplot(fig2)