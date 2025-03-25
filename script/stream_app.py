# initialize an app just by running the python file with streamlit
# libraries
import streamlit as st
import json
import requests
import folium
from streamlit_folium import st_folium
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Function to update active tab
def set_active_tab(tab_name):
    st.session_state.active_tab = tab_name

# here we generated color for the map
def get_gradient_color(value, min_value, max_value):
    """Returns an RGB color between white (255,255,255) and burgundy (128,4,0)"""
    ratio = (value - min_value) / (max_value - min_value) if max_value > min_value else 0
    r = int(255 - (127 * ratio))  # 255 → 128
    g = int(255 - (251 * ratio))  # 255 → 4
    b = int(255 - (255 * ratio))  # 255 → 0
    return f"rgb({r},{g},{b})"





# set title and icon
st.set_page_config(
    page_title="GO-TEC Dashboard",
    page_icon="images/logo.png",
    layout="wide",
    initial_sidebar_state="auto"
)
# Initialize session state for tracking the active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Donnors distribution"
# Load the logos we use
SIMPLE_LOGO = "images/my_logo.png"
LOGO_WITH_TEXT = "images/my_logo_with_text.png"
options = [SIMPLE_LOGO, LOGO_WITH_TEXT]

# set our logo
st.logo(options[1], icon_image=options[0])
# set the hearder in the main age as our brand image
st.image(options[1])  
# Sidebar file uploader
st.sidebar.markdown("<h2 style='color: rgb(128,4,0); font-size: 40px;'>Upload dataset</h2>", unsafe_allow_html=True) # we set the title of the upload section
try:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"]) # we set the uploader form
    st.session_state.df = pd.read_csv(uploaded_file)
except:
    print("")
# if there is an uploaded file we show a message of confirmation
if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")

# Define standard column names for this specific dataset
st.session_state.date_column = "Date de remplissage de la fiche"
st.session_state.last_donation_column = "Si oui preciser la date du dernier don."
st.session_state.eligibility_column = "ELIGIBILITE AU DON"
st.session_state.gender_column = "Genre"
st.session_state.birth_date_column = "Date de naissance"
st.session_state.profession_column = "Niveau d'etude"
st.session_state.marital_status_column = "Situation Matrimoniale (SM)"
st.session_state.profession_column = "Profession"
st.session_state.district_column = "Arrondissement de residence"
st.session_state.city_column = "Ville"
st.session_state.religion_column = "Religion"
st.session_state.has_donated_before_column = "A-t-il (elle) deja donna le sang"
st.session_state.healt_condition = ""


st.sidebar.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True) # straight line for the end of the section

# Main content area with tabs
tab_names = ["Donnors distribution", "Eligibility", "Donors profiling", "Campaign effectiveness", 
             "Donors retention", "Survey/Feedback", "Eligibity prediction"]

# Create tabs
tabs = st.tabs(tab_names)

# Function to display content for each tab
def show_tab_content(tab_index):




############################################################################################################################
############################################################################################################################
#######################################             Donors distribution                  ###################################
############################################################################################################################
############################################################################################################################
    if tab_index == 0:  # 
        col1, col2 = st.columns([0.45,0.45])
        try:
            with col1:

                # we map the distribution according to the "Arrondissement de residence" of each candidate
                # Use session state to prevent map from refreshing on every interaction
                if 'map_data' not in st.session_state:
                    st.session_state.map_data = None

                # load the AMD file for geolocate our targeted arrondissement
                geojson_path = "geoBoundaries-CMR-ADM3.geojson"
                with open(geojson_path, "r", encoding="utf-8") as f:
                    cameroon_geojson = json.load(f)

                # Convert counts to a dictionary for tooltip
                arrondissement_counts_dict = st.session_state.df["Arrondissement de residence"].value_counts().to_dict()

                # **Modify GeoJSON to include "num_candidates" in properties**
                for feature in cameroon_geojson["features"]:
                    name = feature["properties"]["shapeName"]
                    feature["properties"]["num_candidates"] = arrondissement_counts_dict.get(name, 0)

                # we instanciate a map centered on Douala cameroon
                m = folium.Map(location=[4.0511, 9.7679], zoom_start=11)

                # we generate colors for arrondissement
                # Use session state to keep colors consistent
                if 'color_mapping' not in st.session_state:
                    #departments = [feature["properties"]["shapeName"] for feature in cameroon_geojson["features"]]
                    #st.session_state.color_mapping = {dept: f"#{random.randint(0, 0xFFFFFF):06x}" for dept in departments}
                    # Generate color mapping for arrondissements
                    st.session_state.color_mapping = {
                        arrondissement: get_gradient_color(count, st.session_state.df["Arrondissement de residence"].value_counts().min(), st.session_state.df["Arrondissement de residence"].value_counts().max())
                        for arrondissement, count in st.session_state.df["Arrondissement de residence"].value_counts().items()
                    }
                # Ajouter les départements à la carte avec un style gris par défaut et des couleurs au survol
                folium.GeoJson(
                    cameroon_geojson,
                    name="Cameroo   n arrodissement",
                    style_function=lambda feature: {
                        "fillColor": "#CCCCCC",  # Gray color for all areas by default
                        "color": "#999999",      # Darker gray border
                        "weight": 1,
                        "fillOpacity": 0.5,
                    },
                    highlight_function=lambda feature: {
                        "fillColor": st.session_state.color_mapping.get(feature["properties"]["shapeName"], "rgb(255,255,255)"),
                        "color": "black",
                        "weight": 2,
                        "fillOpacity": 0.9,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["shapeName", "num_candidates"],
                        aliases=["Arrondissement:", "Number of candidates:"],
                        labels=True,
                        sticky=True,
                        localize=True,
                        style="background-color: white; color: black; font-size: 12px; padding: 5px;",
                                ),
                ).add_to(m)

                # Afficher la carte dans Streamlit
                st.markdown("<h1 style='font-size: 45px;color:rgb(128,4,0)'>Donnor distribution</h1>", unsafe_allow_html=True)

                # Use key parameter to further control rerendering
                map_data = st_folium(m, width=800, height=400, key="folium_map")

                # Update session state without triggering a rerun
                if map_data and map_data != st.session_state.map_data:
                    st.session_state.map_data = map_data

                # Display information about selected department if available
                if st.session_state.map_data and 'last_active_drawing' in st.session_state.map_data and st.session_state.map_data['last_active_drawing']:
                    st.session_state.selected_dept = st.session_state.map_data['last_active_drawing']['properties']['shapeName']
                    st.write(f"In the arrondissement of {st.session_state.selected_dept} we have :")
                ###
            with col2:
                # Afficher la carte dans Streamlit
                data = pd.DataFrame({
                "Arrondissement": sorted(st.session_state.df["Arrondissement de residence"].unique()),
                #"Latitude": [4.05, 4.06, 4.07, 5.48, 3.87],
                #"Longitude": [9.7, 9.71, 9.72, 10.42, 11.52],
                "Donor Count": list(st.session_state.df["Arrondissement de residence"].value_counts().sort_index())
                })
                fig = px.bar(
                    data, x="Arrondissement", y="Donor Count", 
                    color_discrete_sequence=["rgb(128,4,0)"]
                )
                fig.update_layout(
                    title="Number of People per Month",
                    height=300,  # Set figure height
                )
                st.plotly_chart(fig, use_container_width=True)
                ####

                # Ensure the date column is in datetime format
                st.session_state.df["Date de remplissage de la fiche"] = pd.to_datetime(st.session_state.df["Date de remplissage de la fiche"])
                # Extract month and count occurrences
                monthly_counts = st.session_state.df["Date de remplissage de la fiche"].dt.strftime("%Y-%m").value_counts().sort_index()
                # Convert to DataFrame
                monthly_data = pd.DataFrame({
                    "Month": monthly_counts.index,
                    "Donor Count": monthly_counts.values
                })
                # Create bar chart
                fig = px.area(
                    monthly_data, x="Month", y="Donor Count",
                    color_discrete_sequence=["rgb(128,4,0)"]
                )
                fig.update_traces(fill="tozeroy", 
                                opacity=0.9,
                                mode="lines+markers",
                                marker=dict(size=8,
                                color="rgb(128,4,0) "))
                # Customize layout
                fig.update_layout(
                    title="Number of People per Month",
                    height=300,  # Adjust height
                    xaxis_title="Month",
                    yaxis_title="Donor Count",
                    hovermode="x unified"
                )
                # Show chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

            #####
            try:
                if st.session_state.selected_dept:
                    arr_selected = st.session_state.df["Arrondissement de residence"] == st.session_state.selected_dept
                    df_filtered = st.session_state.df[arr_selected]
                    col1_, col2_,col3_ = st.columns([0.25,0.25,0.5])            
                    with col2_:
                        # Count the number of occurrences for each gender
                    # Compute gender counts
                        gender_counts = df_filtered["Genre"].value_counts()
                        # Create a Plotly donut chart
                        fig = px.pie(
                            names=gender_counts.index, 
                            values=gender_counts.values, 
                            hole=0.6,  # Increase hole size for a larger donut effect
                            color_discrete_sequence=["rgb(128,4,0)", "rgb(23,158,14)"]
                        )

                        # Customize the chart
                        fig.update_traces(
                            textinfo="percent+label",  # Show labels and percentages
                            marker=dict(line=dict(color="white", width=2))  # Improve slice separation
                        )

                        # Adjust layout
                        fig.update_layout(
                            title="gender distribution",
                            height=250,  # Set figure height
                            width=200,   # Set figure width
                            margin=dict(t=20, b=20),  # Reduce top and bottom margins to push it up
                            legend=dict(
                                orientation="h",  # Horizontal legend
                                yanchor="top", 
                                y=-0.2,  # Position legend below the chart
                                xanchor="center", 
                                x=0.5
                            )
                        )

                        # Display in Streamlit
                        st.plotly_chart(fig, use_container_width=False)

                    with col3_:
                        # Count occurrences of "ELIGIBILITE AU DON" per "Genre"
                        # Rename eligibility values for the legend
                        legend_mapping = {
                            "eligible": "El",
                            "temporairement non-eligible": "T-N-El",
                            "définitivement non-eligible": "D-N-El"
                        }
                        df_filtered["ELIGIBILITE AU DON"] = df_filtered["ELIGIBILITE AU DON"].replace(legend_mapping)

                        # Count occurrences of "ELIGIBILITE AU DON" per "Genre"
                        eligibility_counts = df_filtered.groupby(["Genre", "ELIGIBILITE AU DON"]).size().reset_index(name="Count")

                        # Create a grouped bar chart
                        fig = px.bar(
                            eligibility_counts, 
                            x="Genre", 
                            y="Count", 
                            color="ELIGIBILITE AU DON",  # Different bars for each eligibility category
                            barmode="group",  # Grouped bars
                            color_discrete_map={
                                "El": "rgb(23,158,14)", 
                                "T-N-El": "rgb(255,165,0)", 
                                "D-N-El": "rgb(128,4,0)"
                            }
                        )

                        # Customize layout
                        fig.update_layout(
                            title="eligibility per gender",
                            xaxis_title="Gender",
                            yaxis_title="Count",
                            height=250,
                            legend_title_text="",  # Remove legend title
                            legend=dict(
                                orientation="h",  # Horizontal legend
                                yanchor="bottom", 
                                y=-0.45,  # Move legend below the plot
                                xanchor="center", 
                                x=0.5
                            ),
                            margin=dict(t=20, b=20)
                        )

                        # Display in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                    with col1_:
                        # Calculate gender proportions
                        gender_counts = df_filtered["Genre"].value_counts(normalize=True) * 100  # Convert to percentage
                        for gender,count in gender_counts.items():
                            st.markdown(
                            f"<h1 style='font-size: 30px;color:rgb(128,4,0)'>{gender}: {count:.2f}% </h1>", 
                            unsafe_allow_html=True 
                            )
            except:
                print("")
        except:
            print("")
############################################################################################################################
############################################################################################################################
    
    
    

    
############################################################################################################################
############################################################################################################################
#######################################                Eligibility                  # ######################################
############################################################################################################################
############################################################################################################################ 
    elif tab_index == 1:  # 
        with st.sidebar:
            st.sidebar.markdown("<h2 style='color: rgb(128,4,0); font-size: 20px;'>Filter Health Condition</h2>", unsafe_allow_html=True)
            X = st.session_state.df.columns
            health_conditions_columns = [x for x in X if 'raison'.lower() in x.lower()]
            df = st.session_state.df

            # Select the eligibility column
            eligibility_column = st.sidebar.multiselect("Choose the health condition:", df.columns)
            
            if eligibility_column:  # Ensure a column is selected
                eligibility_types = df[eligibility_column[0]].unique()
                selected_eligibility = st.multiselect("Filter eligibility status:", eligibility_types, default=eligibility_types[0])
            else:
                selected_eligibility = []

            # Select Health Condition Columns to see their impact on eligibility
            selected_conditions = st.multiselect("🩺 Select health conditions:", health_conditions_columns)
            
            # Straight line for the end of this filter block
            st.sidebar.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)
        
        # Apply Filters
        if eligibility_column and selected_eligibility:
            df_filtered = df[df[eligibility_column[0]].isin(selected_eligibility)]
        else:
            df_filtered = df.copy()
        
        # Display Key Statistics
        def printer(list_):
            if len(list_) == 1:
                return f"{list_[0]}"
            elif len(list_) == 2:
                return f"{list_[0]} and {list_[1]}"
            else:
                return f"{list_[0]}, {list_[1]} and {list_[2]}"

        st.subheader("Key statistics")
        col1, col2 = st.columns(2)
        col1.metric("Total donors ", len(df))
        col2.metric(f"Number of donors {printer(selected_eligibility)}", len(df_filtered))

        # Eligibility Analysis
        if selected_conditions:
            st.subheader("Eligibility analysis")
            
            # Arrange plots in rows with max 3 per row
            num_conditions = len(selected_conditions)
            num_cols = min(num_conditions, 3)
            
            rows = [selected_conditions[i:i+num_cols] for i in range(0, num_conditions, num_cols)]
            
            for row in rows:
                cols = st.columns(len(row))
                
                for i, condition in enumerate(row):
                    with cols[i]:
                        st.markdown(f"<p style='font-size: 12px; font-weight: bold;'>{condition} impact on eligibility</p>", unsafe_allow_html=True)
                        
                        # Grouping Data
                        condition_eligibility_counts = df_filtered.groupby([condition, eligibility_column[0]]).size().reset_index(name="count")
                        
                        # Ensure there is data to plot
                        if condition_eligibility_counts.empty:
                            st.warning(f"No data available for {condition}.")
                            continue
                        
                        # Display Chart with fixed bar color and height
                        fig = px.bar(
                            condition_eligibility_counts, 
                            x=condition,  # X-axis: health condition values
                            y="count",  # Y-axis: count of donors
                            color=eligibility_column[0],  # Color: Eligibility Status
                            barmode="stack"
                        )

                        # Manually set the bar color and figure height
                        fig.update_traces(marker=dict(color="rgb(128,4,0)"))
                        fig.update_layout(height=300)  # Set height to 300px

                        st.plotly_chart(fig, use_container_width=True)
############################################################################################################################
############################################################################################################################   
    
    


############################################################################################################################
############################################################################################################################
#######################################             Donor profiling        ########################################
############################################################################################################################
############################################################################################################################ 
    elif tab_index == 2:
        st.markdown("<h1 style='font-size: 45px;color:rgb(128,4,0)'>Donor profiling</h1>", unsafe_allow_html=True)

        # Sidebar - User selects the eligibility column
        eligibility_column = st.sidebar.selectbox(
            "Select Eligibility Column",
            st.session_state.df.columns,
            index=st.session_state.df.columns.tolist().index("Eligible au don") if "Eligible au don" in st.session_state.df.columns else 0
        )
        
        # Map eligibility values to numeric scale
        eligibility_mapping = {
            "eligible": 1,
            "temporairement non-eligible": 0.5,
            "définitivement non-eligible": 0
        }
        st.session_state.df[eligibility_column] = st.session_state.df[eligibility_column].map(eligibility_mapping).fillna(0.5)

        # Sidebar - Feature selection
        selected_features = st.sidebar.multiselect(
            "Select Features for Clustering",
            st.session_state.df.columns,
            default=st.session_state.df.columns[:5]  # Ensuring it selects available columns
        )

        # ✅ Fix: Handle Empty List in `printer()` Function
        def printer(list_):
            if len(list_) == 0:
                return "No eligibility selected"
            elif len(list_) == 1:
                return f"{list_[0]}"
            elif len(list_) == 2:
                return f"{list_[0]} and {list_[1]}"
            else:
                return f"{list_[0]}, {list_[1]} and {list_[2]}"

        # Get eligibility types
        eligibility_types = st.session_state.df[eligibility_column].unique() if eligibility_column else []
        
        # ✅ Fix: Prevents list index error in multiselect
        selected_eligibility = st.sidebar.multiselect(
            "Filter eligibility status:",
            eligibility_types.tolist() if len(eligibility_types) > 0 else [],
            default=[eligibility_types[0]] if len(eligibility_types) > 0 else []
        )

        if selected_features:
            df_selected = st.session_state.df[selected_features].copy()

            # Handle categorical variables
            for col in df_selected.select_dtypes(include=["object"]).columns:
                df_selected[col] = LabelEncoder().fit_transform(df_selected[col])

            # Standardize numerical features
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_selected)

            # Number of clusters selection
            n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
            st.sidebar.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True) # straight line for the end of the section

            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(df_scaled)
            st.session_state.df["Cluster"] = clusters

            # PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_scaled)
            st.session_state.df["PCA1"] = pca_result[:, 0]
            st.session_state.df["PCA2"] = pca_result[:, 1]

            # Visualization - Cluster Scatter Plot
            st.subheader("Cluster Visualization")
            fig = px.scatter(
                st.session_state.df,
                x="PCA1",
                y="PCA2",
                color=st.session_state.df["Cluster"].astype(str),
                title="Clusters Based on Selected Features"
            )
            st.plotly_chart(fig)

            # Pie Chart - Cluster Distribution
            st.subheader("Cluster Distribution")
            cluster_counts = st.session_state.df["Cluster"].value_counts()
            fig_pie = px.pie(
                cluster_counts,
                values=cluster_counts.values,
                names=cluster_counts.index.astype(str),
                title="Percentage of Donors per Cluster"
            )
            st.plotly_chart(fig_pie)

            # Bar Chart - Cluster Breakdown
            st.subheader("Cluster Breakdown by Feature")
            selected_feature_for_bar = st.selectbox("Select Feature for Breakdown", selected_features)
            if selected_feature_for_bar:
                fig_bar = px.histogram(
                    st.session_state.df,
                    x=selected_feature_for_bar,
                    color=st.session_state.df["Cluster"].astype(str),
                    barmode='group',
                    title=f"Distribution of {selected_feature_for_bar} by Cluster"
                )
                st.plotly_chart(fig_bar)

            # Ideal Donor Profiling
            st.subheader("Ideal Donor Profiling & Insights")

            # Ensure only numeric columns are considered
            numeric_cols = st.session_state.df.select_dtypes(include=["number"]).columns
            cluster_summary = st.session_state.df.groupby("Cluster")[numeric_cols].mean()

            # Display eligibility scores for all clusters
            if eligibility_column in cluster_summary.columns:
                st.write("### 📊 Eligibility Score for Each Cluster:")
                eligibility_scores = cluster_summary[eligibility_column]

                # Show scores for each cluster
                for cluster, score in eligibility_scores.items():
                    st.write(f"- **Cluster {cluster} Eligibility Score:** {round(score, 3)}")

                # Identify the best cluster
                ideal_cluster = eligibility_scores.idxmax()
                st.success(f"✅ **Cluster {ideal_cluster} has the highest eligibility score and represents the ideal donor profile!**")

                # Display detailed profiles for each cluster
                st.write("### 📌 Detailed Cluster Profiles")
                for cluster in cluster_summary.index:
                    st.write(f"#### Cluster {cluster}")
                    profile = cluster_summary.loc[cluster]

                    # Extract key features dynamically
                    avg_age = profile.get('Age', np.nan)
                    hemoglobin = profile.get('Hemoglobin', np.nan)
                    health_risk = profile.get('Health_Risk_Score', np.nan)
                    donation_frequency = profile.get('Donation_Frequency', np.nan)

                    if not np.isnan(avg_age):
                        st.write(f"- **Average Age**: {round(avg_age, 2)}")
                    if not np.isnan(hemoglobin):
                        st.write(f"- **Average Hemoglobin Level**: {round(hemoglobin, 2)}")
                    if not np.isnan(health_risk):
                        st.write(f"- **Health Risk**: {'Low' if health_risk < 0.5 else 'High'}")
                    if not np.isnan(donation_frequency):
                        st.write(f"- **Donation Frequency**: {round(donation_frequency, 2)}")

                    # Highlight the best cluster
                    if cluster == ideal_cluster:
                        st.success(f"✅ **Cluster {cluster} is the ideal donor group!**")

                # **Conclusion: Why this cluster is ideal**
                st.subheader("Final Conclusion")

                # Extract ideal cluster details
                ideal_profile = cluster_summary.loc[ideal_cluster]
                conclusion_text = f"""
                🔎 **Cluster {ideal_cluster} is identified as the best donor group because of the following characteristics:**
                - **Eligibility Score:** {round(ideal_profile[eligibility_column], 3)}
                """

                # Dynamically add features if available
                if not np.isnan(ideal_profile.get('Age', np.nan)):
                    conclusion_text += f"\n- **Average Age:** {round(ideal_profile['Age'], 2)} years"
                if not np.isnan(ideal_profile.get('Hemoglobin', np.nan)):
                    conclusion_text += f"\n- **Average Hemoglobin Level:** {round(ideal_profile['Hemoglobin'], 2)} g/dL"
                if not np.isnan(ideal_profile.get('Health_Risk_Score', np.nan)):
                    health_risk_status = "Low" if ideal_profile['Health_Risk_Score'] < 0.5 else "High"
                    conclusion_text += f"\n- **Health Risk:** {health_risk_status}"
                if not np.isnan(ideal_profile.get('Donation_Frequency', np.nan)):
                    conclusion_text += f"\n- **Donation Frequency:** {round(ideal_profile['Donation_Frequency'], 2)} times per year"

                st.success(conclusion_text)





############################################################################################################################
############################################################################################################################ 



    
    
    
    
############################################################################################################################
############################################################################################################################
#######################################             Campaign effectiveness          ########################################
############################################################################################################################
############################################################################################################################ 
    elif tab_index == 3:  
        
        st.markdown("<h1 style='font-size: 45px;color:rgb(128,4,0)'>Campaign effectiveness</h1>", unsafe_allow_html=True)
        # visualization
        # Display Dataset Info
        with st.expander("Dataset Overview"):
            st.subheader("Dataset Preview")
            st.write(st.session_state.df.head())
            
            st.subheader("Dataset Information")
            buffer = st.empty()
            df_info = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Non-Null Count': st.session_state.df.count().values,
                'Data Type': st.session_state.df.dtypes.values
            })
            st.write(df_info)
        
        df = st.session_state.df

        # Basic data cleaning
        # Handle missing values in the eligibility column
        #if st.session_state.eligibility_column in df.columns:
         #   df[st.session_state.eligibility_column] = df[st.session_state.eligibility_column].str.lower().fillna("unknown")
        
        # Convert dates to datetime format
        df[st.session_state.date_column] = pd.to_datetime(df[st.session_state.date_column], errors='coerce')
        df[st.session_state.last_donation_column] = pd.to_datetime(df[st.session_state.last_donation_column], errors='coerce')
        if st.session_state.birth_date_column in df.columns:
            df[st.session_state.birth_date_column] = pd.to_datetime(df[st.session_state.birth_date_column], errors='coerce')
            
            # Calculate age based on birth date
            df['Age'] = ((df[st.session_state.date_column] - df[st.session_state.birth_date_column]).dt.days / 365.25).astype(int)
            
            # Create age groups for analysis
        df['Age Group'] = pd.cut(
                df['Age'], 
                bins=[0, 18, 25, 35, 45, 55, 65, 100],
                labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        )
        
        # Drop rows where essential columns are missing
        df = df.dropna(subset=[st.session_state.date_column])
        
        # Extract time components for analysis
        df["Year"] = df[st.session_state.date_column].dt.year
        df["Month"] = df[st.session_state.date_column].dt.month
        df["Day"] = df[st.session_state.date_column].dt.day
        df["Weekday"] = df[st.session_state.date_column].dt.day_name()
        df["Year-Month"] = df[st.session_state.date_column].dt.to_period("M")
        
        # Allow user to filter by year

        st.sidebar.markdown("<h2 style='color: rgb(128,4,0); font-size: 20px;'>filter on campains and donors</h2>", unsafe_allow_html=True)
        available_years = sorted(df["Year"].dropna().unique().astype(int))
        selected_years = st.sidebar.multiselect("Select year(s) to analyze:", available_years, default=available_years)
        if selected_years:
            df_filtered = df[df["Year"].isin(selected_years)]
        else:
            df_filtered = df
            
        # Filter by eligibility if the column exists
        if st.session_state.eligibility_column in df.columns:
            eligibility_options = ["all"] + list(df[st.session_state.eligibility_column].dropna().unique())
            selected_eligibility = st.sidebar.selectbox("Filter by eligibility:", eligibility_options)
            if selected_eligibility != "all":
                df_filtered = df_filtered[df_filtered[st.session_state.eligibility_column] == selected_eligibility]
        ######################################################################
        # Additional filter options specific to this dataset
        if st.session_state.gender_column in df.columns:
            gender_options = ["all"] + list(df[st.session_state.gender_column].dropna().unique())
            selected_gender = st.sidebar.selectbox("Filter by gender:", gender_options)
            if selected_gender != "all":
                df_filtered = df_filtered[df_filtered[st.session_state.gender_column] == selected_gender]
                
        if st.session_state.profession_column in df.columns:
            profession_options = ["all"] + list(df[st.session_state.profession_column].dropna().unique())
            if len(profession_options) > 10:  # If too many professions, just provide "all" option
                selected_profession = st.sidebar.selectbox("Filter by profession:", ["all"])
            else:
                selected_profession = st.sidebar.selectbox("Filter by profession:", profession_options)
            if selected_profession != "all":
                df_filtered = df_filtered[df_filtered[st.session_state.profession_column] == selected_profession]
        st.session_state.df_filtered = df_filtered
        st.sidebar.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True) # straight line for the end of the section
        #####################################################################################
        
        # --- CAMPAIGN PERFORMANCE ANALYSIS ---    
        #################
                    # Create summary metrics
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            
        with ecol1:
                st.metric("Total Donors", f"{len(df_filtered):,}")
            
        with ecol2:
                if st.session_state.has_donated_before_column in df.columns:
                    previous_donors = df_filtered[df_filtered[st.session_state.has_donated_before_column].str.lower() == "oui"].shape[0]
                    previous_donors_pct = (previous_donors / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
                    st.metric("Previous Donors", f"{previous_donors_pct:.1f}%")
            
        with ecol3:
                if st.session_state.eligibility_column in df.columns:
                    eligible_donors = df_filtered[df_filtered[st.session_state.eligibility_column] == 'eligible'].shape[0]
                    eligible_percentage = (eligible_donors / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
                    st.metric("Eligible Donors", f"{eligible_percentage:.1f}%")
            
        with ecol4:
                if 'Age' in df_filtered.columns:
                    avg_age = df_filtered['Age'].mean()
                    st.metric("Average Age", f"{avg_age:.1f} years")
        #################
        st.markdown('<p style="color:rgb(128,4,0);font-size:40px;font-weight:bold;">Trend over time</p>', unsafe_allow_html=True)
        time_period = st.radio("Select time period to observe:", ["Daily", "Weekly", "Monthly"], horizontal=True)
        camp_col1,camp_col2 = st.columns([0.6,0.4])
        with camp_col1:
            if time_period == "Daily":
                # Daily analysis
                daily_counts = df_filtered.groupby(df_filtered[st.session_state.date_column].dt.date).size().reset_index(name="count")
                daily_counts.columns = ["date", "donors"]
                
                fig = px.line(
                    daily_counts, 
                    x="date", 
                    y="donors", 
                    title="Daily Donor Trends",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif time_period == "Weekly":
                # Weekly analysis
                df_filtered['week'] = df_filtered[st.session_state.date_column].dt.isocalendar().week
                df_filtered['year_week'] = df_filtered[st.session_state.date_column].dt.strftime('%Y-%U')
                weekly_counts = df_filtered.groupby('year_week').size().reset_index(name="donors")
                
                fig = px.line(
                    weekly_counts, 
                    x="year_week", 
                    y="donors", 
                    title="Weekly Donor Trends",
                    markers=True
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
            elif time_period == "Monthly":
                # Monthly analysis with seasonal decomposition if enough data
                monthly_counts = df_filtered.groupby(df_filtered[st.session_state.date_column].dt.to_period("M")).size()
                monthly_counts.index = monthly_counts.index.to_timestamp()
                
                # Convert to DataFrame for plotting
                monthly_df = monthly_counts.reset_index()
                monthly_df.columns = ["date", "donors"]
                
                fig = px.line(
                    monthly_df, 
                    x="date", 
                    y="donors", 
                    title="Monthly Donor Trends",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Try seasonal decomposition if enough data points
                if len(monthly_counts) > 12:
                    try:
                        st.subheader("Seasonal Decomposition of Monthly Donors")
                        # Fill missing months with 0 to ensure continuous time series
                        full_range = pd.date_range(
                            start=monthly_counts.index.min(),
                            end=monthly_counts.index.max(),
                            freq='MS'
                        )
                        monthly_counts = monthly_counts.reindex(full_range, fill_value=0)
                        
                        # Apply seasonal decomposition
                        result = seasonal_decompose(monthly_counts, model='additive', period=12)
                        
                        # Plot the decomposition components
                        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
                        result.observed.plot(ax=axes[0])
                        axes[0].set_title('Observed')
                        result.trend.plot(ax=axes[1])
                        axes[1].set_title('Trend')
                        result.seasonal.plot(ax=axes[2])
                        axes[2].set_title('Seasonal')
                        result.resid.plot(ax=axes[3])
                        axes[3].set_title('Residual')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Insights from seasonal decomposition
                        seasonal_pattern = result.seasonal.groupby(result.seasonal.index.month).mean()
                        peak_month = seasonal_pattern.idxmax()
                        month_name = datetime(2000, peak_month, 1).strftime('%B')
                        
                        st.markdown(f'<div class="insight-box">Based on seasonal decomposition, <b>{month_name}</b> tends to have the highest donation rates, which suggests this could be the most effective month for campaigns.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Couldn't perform seasonal decomposition: {e}")

        with camp_col2:
            # Identify top campaign dates
            daily_donations = df_filtered.groupby([df_filtered[st.session_state.date_column].dt.date]).size().reset_index(name="Donors")
            daily_donations.columns = ["Date", "Donors"]
            top_days = daily_donations.sort_values("Donors", ascending=False).head(10)
            
            fig = px.bar(
                top_days,
                x="Date",
                y="Donors",
                title="Top 10 Days with Most Donors",
                color="Donors",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p style="color:rgb(128,4,0);font-size:40px;font-weight:bold;">Demogrphic factor</p>', unsafe_allow_html=True)
        st.session_state.demographic_columns = [
                st.session_state.gender_column,
                st.session_state.profession_column,
                st.session_state.marital_status_column,
                st.session_state.profession_column,
                st.session_state.district_column,
                st.session_state.city_column,
                st.session_state.religion_column
            ]
            
        # Filter to include only columns that exist in the dataset
        st.session_state.demographic_options = [col for col in st.session_state.demographic_columns if col in df.columns]
            
        # Add age group if it exists
        if 'Age Group' in df.columns:
                st.session_state.demographic_options.append('Age Group')
            
        # Select demographic features to analyze
        selected_demos = st.multiselect(
                "Select demographic features to analyze:", 
                st.session_state.demographic_options, 
                default=[st.session_state.gender_column] if st.session_state.gender_column in st.session_state.demographic_options else []
            )
        
        ##################       
        if selected_demos:
            trend_plots = []
            bar_charts = []
            
            for index_selected, demo in enumerate(selected_demos):
                # Fill missing values
                df_filtered[demo] = df_filtered[demo].fillna("Non précisé")
                
                # Get the value counts and clean up the data
                value_counts = df_filtered[demo].value_counts()
                
                # If there are too many categories, only show the top 10
                if len(value_counts) > 10:
                    # Keep the top categories and group the rest as "Other"
                    top_categories = value_counts.nlargest(9).index.tolist()
                    df_filtered[f"{demo}_grouped"] = df_filtered[demo].apply(
                        lambda x: x if x in top_categories else "Autres"
                    )
                    plot_demo = f"{demo}_grouped"
                else:
                    plot_demo = demo
                
                # Basic demographic bar chart
                plot_data = df_filtered[plot_demo].value_counts().reset_index()
                plot_data.columns = ['category', 'count']
                
                fig_bar = px.bar(
                    plot_data,
                    x='category',
                    y='count',
                    title=f"Donors by {demo}",
                    color='count',
                    color_continuous_scale="Reds"
                )
                bar_charts.append(fig_bar)
                
                # More complex visualizations for gender and age
                if demo in [st.session_state.gender_column, 'Age Group']:
                    st.subheader(f"{demo} Trends Over Time")
                    
                    # Group by year-month and demographic
                    time_demo_data = df_filtered.groupby([df_filtered[st.session_state.date_column].dt.to_period("M"), plot_demo]).size().reset_index()
                    time_demo_data.columns = ['period', 'category', 'count']
                    time_demo_data['period'] = time_demo_data['period'].astype(str)
                    
                    # Create a stacked area chart
                    fig_trend = px.area(
                        time_demo_data,
                        x='period',
                        y='count',
                        color='category',
                        title=f"{demo} Distribution Over Time",
                        labels={'period': 'Month', 'count': 'Number of Donors', 'category': demo}
                    )
                    fig_trend.update_layout(xaxis_tickangle=-45)
                    trend_plots.append(fig_trend)
            
            # Display the trend plot at 110%
            if trend_plots:
                st.plotly_chart(trend_plots[0], use_container_width=True, key="trend_chart")
            
            # DISPLAY BAR CHARTS IN PAIRS (0.45 / 0.45) & SINGLE (0.8)
            num_bars = len(bar_charts)
            for i in range(0, num_bars, 2):
                if i + 1 < num_bars:
                    # Display two bar charts side by side
                    col1, col2 = st.columns([0.45, 0.45])
                    with col1:
                        st.plotly_chart(bar_charts[i], use_container_width=True, key=f"bar_chart_{i}")
                    with col2:
                        st.plotly_chart(bar_charts[i + 1], use_container_width=True, key=f"bar_chart_{i+1}")
                else:
                    # Last one on a separate row with 0.8 width
                    col = st.columns([0.8])[0]
                    with col:
                        st.plotly_chart(bar_charts[i], use_container_width=True, key=f"bar_chart_{i}")
            
            if len(selected_demos) >= 2:
                st.subheader("Demographic Intersection Analysis")
                demo1, demo2 = selected_demos[:2]
                
                cross_tab = pd.crosstab(df_filtered[demo1], df_filtered[demo2])
                cross_tab_long = cross_tab.reset_index().melt(id_vars=demo1, var_name=demo2, value_name='count')
                
                fig_stacked_bar = px.bar(
                    cross_tab_long,
                    x=demo1,
                    y='count',
                    color=demo2,
                    title=f"Distribution of {demo2} across {demo1}",
                    labels={demo1: demo1, 'count': 'Number of Donors', demo2: demo2}
                )
                st.plotly_chart(fig_stacked_bar, use_container_width=True)
############################################################################################################################
############################################################################################################################



     
         
    
############################################################################################################################
############################################################################################################################
#######################################             Donors retention          ##############################################
############################################################################################################################
############################################################################################################################
    elif tab_index == 4:              
            # Previous donation analysis
            st.markdown("<h2 style='font-size: 25px;color:rgb(128,4,0)'>Previous donation analysis</h2>", unsafe_allow_html=True)

            df = st.session_state.df
            df_filtered = st.session_state.df_filtered
            if st.session_state.has_donated_before_column in df.columns:
                # Clean up the response data (assuming "oui" means "yes" and anything else means "no")
                df_filtered[st.session_state.has_donated_before_column] = df_filtered[st.session_state.has_donated_before_column].fillna("non")
                df_filtered['Has_Donated_Before'] = df_filtered[st.session_state.has_donated_before_column].str.lower().apply(
                    lambda x: "Has donated before" if x == "oui" else "First-time donor"
                )
                
                # Plot as a donut chart
                donation_history = df_filtered['Has_Donated_Before'].value_counts().reset_index()
                donation_history.columns = ['status', 'count']
                
                fig_donut = px.pie(
                    donation_history,
                    values='count',
                    names='status',
                    title="Previous Donation History",
                    color='status',
                    hole=0.4,  # Donut chart
                    color_discrete_map={
                        "First-time donor": "rgb(128,4,0)",
                        "Has donated before": "rgb(23,158,14)"
                    }
                )
                fig_donut.update_layout(legend_orientation="h", legend_y=-0.2)
                
                # Previous donation history trends over time
                monthly_history = df_filtered.groupby([df_filtered[st.session_state.date_column].dt.to_period("M"), "Has_Donated_Before"]).size().reset_index()
                monthly_history.columns = ['month', 'status', 'count']
                monthly_history['month'] = monthly_history['month'].astype(str)
                
                fig_trend = px.line(
                    monthly_history,
                    x='month',
                    y='count',
                    color='status',
                    title="First-time vs Returning Donors Over Time",
                    labels={'month': 'Month', 'count': 'Number of Donors', 'status': 'Donor Status'},
                    color_discrete_map={
                        "First-time donor": "rgb(128,4,0)",
                        "Has donated before": "rgb(23,158,14)"
                    }
                )
                fig_trend.update_layout(xaxis_tickangle=-45)
                
                # Display side by side with ratio 0.4 for donut and 0.6 for trend
                col1, col2 = st.columns([0.4, 0.6])
                with col1:
                    st.plotly_chart(fig_donut, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_trend, use_container_width=True)

            # Calculate donation intervals for repeat donors
            if st.session_state.last_donation_column in df.columns:
                df_filtered["Donation_Interval"] = (df_filtered[st.session_state.date_column] - df_filtered[st.session_state.last_donation_column]).dt.days
                repeat_donors = df_filtered[df_filtered["Donation_Interval"].notna()]
                
                # Filter out unreasonable intervals (negative or extremely large)
                reasonable_intervals = repeat_donors[
                    (repeat_donors["Donation_Interval"] > 0) & 
                    (repeat_donors["Donation_Interval"] < 1000)  # Assume donations over ~3 years are data errors
                ]
                
                if not reasonable_intervals.empty:
                    st.subheader("Time Between Donations")
                    
                    # Create histogram of donation intervals
                    fig_hist = px.histogram(
                        reasonable_intervals,
                        x="Donation_Interval",
                        nbins=30,
                        title="Distribution of Time Between Donations",
                        labels={"Donation_Interval": "Days Between Donations", "count": "Number of Donors"},
                        color_discrete_sequence=["#C62828"]
                    )
                    
                    # Add median and mean lines
                    median_interval = reasonable_intervals["Donation_Interval"].median()
                    mean_interval = reasonable_intervals["Donation_Interval"].mean()
                    
                    fig_hist.add_vline(x=median_interval, line_dash="dash", line_color="black", annotation_text=f"Median: {median_interval:.0f} days")
                    fig_hist.add_vline(x=mean_interval, line_dash="dot", line_color="blue", annotation_text=f"Mean: {mean_interval:.0f} days")
                    fig_hist.add_vline(x=56, line_dash="dash", line_color="green", annotation_text="Min. Safe: 56 days")
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Display median/mean intervals
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Median Time Between Donations", f"{median_interval:.0f} days")
                    with col2:
                        st.metric("Average Time Between Donations", f"{mean_interval:.0f} days")
                    
                    # Analyze donation intervals by demographic factors
                    st.subheader("Donation Intervals by Demographic")
                    
                    # Select demographic to analyze
                    demo_options = [col for col in st.session_state.demographic_options if col in reasonable_intervals.columns]
                    if demo_options:
                        selected_demo = st.selectbox("Select demographic to analyze donation intervals:", demo_options)
                        
                        # Box plot of intervals by demographic
                        fig_box = px.box(
                            reasonable_intervals,
                            x=selected_demo,
                            y="Donation_Interval",
                            title=f"Distribution of Donation Intervals by {selected_demo}",
                            labels={selected_demo: selected_demo, "Donation_Interval": "Days Between Donations"},
                            color=selected_demo
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
############################################################################################################################
############################################################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
############################################################################################################################
############################################################################################################################
#######################################             Survey/Feedback          ###############################################
############################################################################################################################
############################################################################################################################
    elif tab_index == 5:  # 
        st.markdown("<h1 style='font-size: 45px;color:rgb(128,4,0)'>Survey/Feedback</h1>", unsafe_allow_html=True)
        st.write("Survey content goes here...")
    
############################################################################################################################
############################################################################################################################ 
    
    
    
    
    

    
############################################################################################################################
############################################################################################################################
###############################             Eligibility prediction           ###############################################
############################################################################################################################
############################################################################################################################
    
    elif tab_index == 6:  # Eligibility prediction
        st.markdown("<h1 style='font-size: 45px;color:rgb(128,4,0)'>Eligibility prediction</h1>", unsafe_allow_html=True)
        # Custom CSS to increase the width and height of the input form
        st.markdown(
            """
            <style>
            .stTextArea textarea {
                height: 500px;
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Form to input data
        with st.form(key='input_form'):
            input_data = st.text_area("Input Data (in JSON format)", "")
            submit_button = st.form_submit_button(label='Submit')

        # When the form is submitted
        if submit_button:
            try:
                # We convert the input data to a dictionary
                input_data_dict = json.loads(input_data)
                
                # We send a POST request to our Flask API lauched previously
                response = requests.post("http://127.0.0.1:5001/predict", json=input_data_dict)
                
                # Display the response
                if response.status_code == 200:
                    st.success("Request successful!")
                    st.json(response.json())
                else:
                    st.error(f"Request failed with status code {response.status_code}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")

############################################################################################################################
############################################################################################################################

























############################################################################################################################
############################################################################################################################
# Display content for each tab and update active tab when tab is selected
with tabs[0]:
    set_active_tab(tab_names[0])
    show_tab_content(0)
    
with tabs[1]:
    set_active_tab(tab_names[1])
    show_tab_content(1)
    
with tabs[2]:
    set_active_tab(tab_names[2])
    show_tab_content(2)
    
with tabs[3]:
    set_active_tab(tab_names[3])
    show_tab_content(3)
    
with tabs[4]:
    set_active_tab(tab_names[4])
    show_tab_content(4)
    
with tabs[5]:
    set_active_tab(tab_names[5])
    show_tab_content(5)
    
with tabs[6]:
    set_active_tab(tab_names[6])
    show_tab_content(6)

# Add filter action buttons
col1, col2 = st.sidebar.columns([0.45,0.45])
with col1:
    apply_button = st.button("Apply Filters")
with col2:
    reset_button = st.button("Reset Filters")

# Handle filter actions
if apply_button:
    st.sidebar.success(f"Filters applied for {st.session_state.active_tab}")
    
if reset_button:
    st.success(f"Filters reset for {st.session_state.active_tab}")

############################################################################################################################
############################################################################################################################
