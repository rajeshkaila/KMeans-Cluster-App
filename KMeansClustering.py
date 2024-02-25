# Import necessary libraries
import subprocess
subprocess.run(["pip", "install", "matplotlib==3.4.3"])
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Streamlit app
def main():
    st.title("K-Means Clustering App")

    # Upload DataFrame CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded file
        malldf = pd.read_csv(uploaded_file)
        
        # Display uploaded DataFrame
        st.subheader("Uploaded DataFrame:")
        st.write(malldf)

        # Exclude non-numeric columns for clustering
        numeric_columns = malldf.select_dtypes(include=[np.number]).columns
        malldf_numeric = malldf[numeric_columns]

        # Select variables for clustering
        st.sidebar.header("Select Variables")
        x_variable = st.sidebar.selectbox("Select X Variable", numeric_columns)
        y_variable = st.sidebar.selectbox("Select Y Variable", numeric_columns)

        # Sidebar for clustering parameters
        st.sidebar.header("Clustering Parameters")
        num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=4)

        # Perform K-Means clustering
        kmeans_model = KMeans(n_clusters=num_clusters, n_init="auto").fit(malldf_numeric[[x_variable, y_variable]])
        malldf['Cluster'] = kmeans_model.labels_

        # Display silhouette score
        silhouette_avg = silhouette_score(malldf_numeric, kmeans_model.labels_)
        st.subheader(f"Silhouette Score: {silhouette_avg:.4f}")

        # Display elbow plot
        st.subheader("Elbow Plot:")
        plot_elbow(malldf_numeric[[x_variable, y_variable]], range(1, 11))

        # Display scatter plot
        st.subheader("Clustering Results:")
        st.scatter_chart(malldf, x=x_variable, y=y_variable, color='Cluster', use_container_width=True)

def plot_elbow(data, n_cluster_values):
    distortions = []
    for n_clusters in n_cluster_values:
        kmeans_model = KMeans(n_clusters=n_clusters, n_init="auto").fit(data)
        distortions.append(kmeans_model.inertia_)

    # Plotting the elbow plot
    fig, ax = plt.subplots()
    ax.plot(n_cluster_values, distortions, marker='o')
    ax.set(xlabel='Number of Clusters', ylabel='Distortion',
           title='Elbow Plot')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
