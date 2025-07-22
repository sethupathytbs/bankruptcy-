import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("Bankruptcy (2).xlsx")

# Convert 'class' column to numeric (0 = non-bankruptcy, 1 = bankruptcy)
df['class'] = df['class'].map({'non-bankruptcy': 0, 'bankruptcy': 1})

# Streamlit App
st.title(" Bankruptcy Prevention - EDA Dashboard")

# Show dataset
if st.checkbox("Show raw data"):
    st.write(df.head())

# Class distribution
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x=df['class'], ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Feature Correlation")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Boxplots for features
st.subheader("Boxplots for Each Feature")
selected_feature = st.selectbox("Select Feature:", df.columns[:-1])
fig, ax = plt.subplots()
sns.boxplot(x='class', y=selected_feature, data=df, ax=ax)
st.pyplot(fig)
