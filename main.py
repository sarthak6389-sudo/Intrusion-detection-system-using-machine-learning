import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. AUTOMATIC DATA DOWNLOADER
# ==========================================
# This section downloads the data directly from the internet if you don't have it.
# No more "File Not Found" errors!

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

print("Step 1: Checking for data files...")

def load_data(filename, url):
    if not os.path.exists(filename):
        print(f"   -> '{filename}' not found locally. Downloading from GitHub...")
        try:
            df = pd.read_csv(url, header=None)
            df.to_csv(filename, index=False, header=False) # Save it for next time
            print(f"   -> Download complete: {filename}")
            return df
        except Exception as e:
            print(f"   -> ERROR: Could not download data. Check your internet.\nError: {e}")
            return None
    else:
        print(f"   -> Found '{filename}' locally. Loading...")
        return pd.read_csv(filename, header=None)

# Define column names
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

# Load the data
df_train = load_data("KDDTrain.csv", train_url)
df_test = load_data("KDDTest.csv", test_url)

if df_train is None or df_test is None:
    print("\nCRITICAL FAILURE: Data could not be loaded. Exiting.")
    exit()

# Assign columns
df_train.columns = columns
df_test.columns = columns

# ==========================================
# 2. PREPROCESSING
# ==========================================
print("\nStep 2: Preparing data...")

# Label Encodes 'normal' to 0 and everything else to 1 (Attack)
df_train['attack_flag'] = df_train['attack'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['attack_flag'] = df_test['attack'].apply(lambda x: 0 if x == 'normal' else 1)

# Drop columns we don't need
df_train.drop(['attack', 'level'], axis=1, inplace=True)
df_test.drop(['attack', 'level'], axis=1, inplace=True)

# Convert text columns (like 'tcp', 'http') to numbers
le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    # Fit on combined data to ensure all categories are known
    combined = pd.concat([df_train[col], df_test[col]])
    le.fit(combined)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# Separate inputs (X) and answers (y)
X_train = df_train.drop('attack_flag', axis=1)
y_train = df_train['attack_flag']
X_test = df_test.drop('attack_flag', axis=1)
y_test = df_test['attack_flag']

# ==========================================
# 3. TRAINING
# ==========================================
print("\nStep 3: Training the AI (Random Forest)...")
rf = RandomForestClassifier(n_estimators=50, random_state=42) # n_estimators=50 is faster
rf.fit(X_train, y_train)
print("   -> Training complete.")

# ==========================================
# 4. RESULTS
# ==========================================
print("\nStep 4: Testing the AI...")
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("="*40)
print(f"FINAL ACCURACY: {acc*100:.2f}%")
print("="*40)
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Show Confusion Matrix
print("\nDisplaying Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title('Intrusion Detection Results')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()