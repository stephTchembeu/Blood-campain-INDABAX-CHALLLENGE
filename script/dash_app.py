# initialize an app just by running the python file with streamlit
from main_app_requirement import *
from functions import *

# set title and icon
st.set_page_config(
    page_title="GO-TEC Dashboard",
    page_icon="../images/logo.png",
    layout="wide",
    initial_sidebar_state="auto"
)
# Initialize session state for tracking the active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Donnors distribution"
# Load the logos we use
SIMPLE_LOGO = "../images/my_logo.png"
LOGO_WITH_TEXT = "../images/my_logo_with_text.png"
options = [SIMPLE_LOGO, LOGO_WITH_TEXT]

# set our logo
st.logo(options[1], icon_image=options[0])
# set the hearder in the main age as our brand image
st.image(options[1])  
# Sidebar file uploader
st.sidebar.markdown("<h2 style='color: rgb(128,4,0); font-size: 25px;'>Upload dataset</h2>", unsafe_allow_html=True) # we set the title of the upload section
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
    global printer



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
                geojson_path = "geo_file/geoBoundaries-CMR-ADM3.geojson"
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
                if 'color_mapping' not in st.session_state:
                    # Generate color mapping for arrondissements
                    st.session_state.color_mapping = {
                        arrondissement: get_gradient_color(count, st.session_state.df["Arrondissement de residence"].value_counts().min(), st.session_state.df["Arrondissement de residence"].value_counts().max())
                        for arrondissement, count in st.session_state.df["Arrondissement de residence"].value_counts().items()
                    }

                # Ajouter les départements à la carte avec un style gris par défaut et des couleurs au survol
                folium.GeoJson(
                    cameroon_geojson,
                    name="Cameroon arrodissement",
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
            set_sidebar_title("Filter Health Condition")
            health_conditions_columns = [x for x in st.session_state.df.columns if 'raison'.lower() in x.lower()]
            df = st.session_state.df

            # Select the eligibility column
            eligibility_column, selected_eligibility, selected_conditions = get_eligibility_observations_param(df,health_conditions_columns)
            # Straight line for the end of this filter block
            st.sidebar.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)
        
        # Apply Filters
        if eligibility_column and selected_eligibility:
            df_filtered = df[df[eligibility_column[0]].isin(selected_eligibility)]
        else:
            df_filtered = df.copy()

        # show titles of the tab      
        show_title("Eligibility",30)
        show_title("Key statistics",20)

            
        col1, col2 = st.columns(2)
        
        col2.metric(f"Number of donors {printer(selected_eligibility)}", len(df_filtered))
        col1.metric("Total donors ", len(df))
        # Eligibility Analysis
        if selected_conditions:
            show_title("Eligibility analysis",20)
            show_el_plots(selected_conditions,df_filtered,eligibility_column)
        
############################################################################################################################
############################################################################################################################   
    
    


############################################################################################################################
############################################################################################################################
#######################################             Donor profiling        ########################################
############################################################################################################################
############################################################################################################################ 
    elif tab_index == 2:
        st.markdown("<h1 style='font-size: 45px;color:rgb(128,4,0)'>Donor profiling</h1>", unsafe_allow_html=True)

        # Custom CSS for better styling
        st.markdown("""
        <style>
            .main-header {
                font-size: 2rem;
                color: rgb(128,4,0);
                text-align: center;
            }
            .section-header {
                font-size: 1rem;
                color: rgb(128,4,0);
                padding-top: 1rem;
            }
            .stat-box {
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
            }
            .note-box {
                background-color: #FFF3CD;
                border-left: 5px solid #FFC107;
                padding: 10px 15px;
                margin: 10px 0;
                border-radius: 0 5px 5px 0;
            }
        </style>
        """, unsafe_allow_html=True)

       
        # Sidebar section
        with st.sidebar:
            st.sidebar.markdown("<h2 style='color: rgb(128,4,0); font-size: 20px;'>Clustering filter</h2>", unsafe_allow_html=True) 
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
            st.sidebar.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)
        # Main dashboard content
        if st.session_state.df is not None:
            # Load data
            try:
                df = st.session_state.df
                # Display a sample of the raw data
                st.markdown("<h2 class='section-header'>Raw Data Sample</h2>", unsafe_allow_html=True)
                st.dataframe(df.head())
                
                # Basic data info
                st.markdown("<h2 class='section-header'>Data Overview</h2>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                    st.metric("Total Records", f"{df.shape[0]:,}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                    st.metric("Features", f"{df.shape[1]}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                    missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Missing Values", f"{missing_percentage:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Column selection
                st.markdown("<h2 class='section-header'>Column Selection</h2>", unsafe_allow_html=True)
                
                # Feature selection note
                st.markdown("""
                <div class='note-box'>
                    <strong>Important Note:</strong> Please select only relevant features for clustering. 
                    Features that don't indicate eligibility or donor characteristics should not be included 
                    as they may bias the clustering results.
                </div>
                """, unsafe_allow_html=True)
                
                # Auto-detect numeric and categorical columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

                # Let user select columns
                st.subheader("Select Numeric Features")
                st.markdown("Choose numeric features like age, hemoglobine levels, etc.")

                # User selection
                # Improved hemoglobin detection with regex
                hemoglobin_pattern = re.compile(r'hemoglobine?|taux', re.IGNORECASE)

                selected_numeric = st.multiselect(
                    "Numeric Features", 
                    options=numeric_cols,
                    default=[col for col in numeric_cols if 
                            re.search(r'age|âge|taux|hemoglobine?', col, re.IGNORECASE)]
                )
                        
                st.subheader("Select Categorical Features")
                st.markdown("Choose categorical features like gender, profession, location, etc.")
                selected_categorical = st.multiselect(
                    "Categorical Features", 
                    options=categorical_cols,
                    default=[col for col in categorical_cols if any(keyword in col.lower() for keyword in 
                                                                ['genre', 'gender', 'sexe', 'domain', 'profession', 'ville', 'city', 'location'])]
                )
                
                # Only proceed with clustering if columns are selected
                if len(selected_numeric) > 0 and len(selected_categorical) > 0:
                    # Preprocessing and clustering section
                    st.markdown("<h2 class='section-header'>Donor Profiling with Clustering</h2>", unsafe_allow_html=True)
                    
                    with st.spinner("Processing data and creating clusters..."):
                        # Select and clean data
                        df1 = df[selected_numeric + selected_categorical].copy()
                        
                        # Display missing values before cleaning
                        missing_cols = df1.columns[df1.isna().any()].tolist()
                        if missing_cols:
                            st.markdown("<div class='note-box'>", unsafe_allow_html=True)
                            st.write("Missing values detected in the following columns:")
                            for col in missing_cols:
                                missing_count = df1[col].isna().sum()
                                missing_percent = (missing_count / len(df1)) * 100
                                st.write(f"- {col}: {missing_count} values ({missing_percent:.2f}%)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Drop rows where categorical features are missing
                        df1 = df1.dropna(subset=selected_categorical)
                        
                        # Clean column values for categorical features
                        for col in selected_categorical:
                            if df1[col].dtype == 'object':
                                df1[col] = df1[col].astype(str).str.strip().str.lower()
                        
                        # Preprocessing pipeline
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='mean')),
                                    ('scaler', StandardScaler())
                                ]), selected_numeric),
                                
                                ('cat', Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
                                ]), selected_categorical)
                            ]
                        )
                        
                        # Fit and transform the data
                        X_preprocessed = preprocessor.fit_transform(df1)
                        
                        # Elbow method
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Finding Optimal Number of Clusters")
                            inertia = []
                            k_range = range(2, min(10, len(df1) // 10 + 1))  # Limit k based on data size
                            for k in k_range:
                                kmeans = KMeans(n_clusters=k, random_state=42)
                                kmeans.fit(X_preprocessed)
                                inertia.append(kmeans.inertia_)
                            
                            # Plot elbow method
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.plot(k_range, inertia, marker='o')
                            ax.set_title("Elbow Method for Optimal Clusters")
                            ax.set_xlabel("Number of clusters")
                            ax.set_ylabel("Inertia")
                            ax.grid(True)
                            st.pyplot(fig)
                            
                            st.info("The 'elbow' in the graph indicates the optimal number of clusters. You can adjust the number of clusters using the slider in the sidebar.")
                        
                        # Perform clustering with chosen k
                        kmeans = KMeans(n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_preprocessed)
                        
                        # Add cluster labels to original dataframe
                        df1['Cluster'] = clusters
                        
                        # Generate cluster summary
                        summary_dict = {}
                        
                        # Add numeric columns to summary - calculate mean
                        for col in selected_numeric:
                            summary_dict[col] = 'mean'
                        
                        # Add categorical columns to summary - get most common value
                        for col in selected_categorical:
                            summary_dict[col] = lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else "N/A"
                        
                        # Add count of donors
                        summary_dict['Cluster'] = 'count'
                        
                        # Create the summary dataframe
                        summary = df1.groupby('Cluster').agg(summary_dict)
                        
                        # Rename columns for better display
                        col_rename = {}
                        hemoglobin_pattern = re.compile(r'(taux[\s\w]*hemoglobine?)|(hemoglobine?)', re.IGNORECASE)
                        for col in selected_numeric:
                            if hemoglobin_pattern.search(col):
                                col_rename[col] = 'Avg_Haemoglobin'
                            else:
                                col_rename[col] = f'Avg_{col}'

                        for col in selected_categorical:
                            col_rename[col] = f'Most_Common_{col}'

                        col_rename['Cluster'] = 'Donor_Count'
                        
                        summary = summary.rename(columns=col_rename).reset_index()
                        
                        with col2:
                            st.subheader("Cluster Distribution")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            cluster_counts = df1['Cluster'].value_counts().sort_index()
                            
                            # Create a colormap for the clusters
                            colors = plt.cm.get_cmap('tab10', n_clusters)
                            cluster_colors = [colors(i) for i in range(n_clusters)]
                            
                            ax.bar(cluster_counts.index, cluster_counts.values, color=cluster_colors)
                            ax.set_title("Number of Donors per Cluster")
                            ax.set_xlabel("Cluster")
                            ax.set_ylabel("Number of Donors")
                            for i, v in enumerate(cluster_counts.values):
                                ax.text(i, v + 5, str(v), ha='center')
                            st.pyplot(fig)
                        
                        # Display cluster summary
                        st.subheader("Cluster Profiles")
                        st.dataframe(summary)
                        
                        # Identifying ideal donor cluster
                        st.markdown("<h2 class='section-header'>Identifying Ideal Donor Profile</h2>", unsafe_allow_html=True)
                        
                        # Check if there's a hemoglobin column
                        hemoglobin_col = None
                        for col in selected_numeric:
                            if 'Avg_Haemoglobin' in summary.columns:
                                ideal_hemoglobin_cluster = summary['Avg_Haemoglobin'].idxmax()
                                break
                        
                        # Find ideal donor cluster based on hemoglobin and other metrics
                        if hemoglobin_col:
                            ideal_hemoglobin_cluster = summary[f'Avg_{hemoglobin_col}'].idxmax()
                            
                            st.markdown(f"""
                            <div class='note-box'>
                                <strong>Ideal Donor Profile (Cluster {ideal_hemoglobin_cluster}):</strong><br>
                                This cluster has the highest average hemoglobin level ({summary.loc[ideal_hemoglobin_cluster, f'Avg_{hemoglobin_col}']:.2f}) 
                                among all clusters, suggesting these donors may be optimal for blood donation campaigns.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display ideal donor characteristics
                            st.subheader(f"Characteristics of Ideal Donors (Cluster {ideal_hemoglobin_cluster})")
                            
                            ideal_cluster_data = summary.loc[ideal_hemoglobin_cluster].to_dict()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Numeric Characteristics:**")
                                for col in selected_numeric:
                                    avg_col = f'Avg_{col}'
                                    st.markdown(f"- Average {col}: {ideal_cluster_data[avg_col]:.2f}")
                            
                            with col2:
                                st.markdown("**Categorical Characteristics:**")
                                for col in selected_categorical:
                                    most_common_col = f'Most_Common_{col}'
                                    value = ideal_cluster_data[most_common_col]
                                    if isinstance(value, str):
                                        value = value.title()
                                    st.markdown(f"- {col}: {value}")
                        
                        # Visualize cluster characteristics for numeric features
                        if len(selected_numeric) > 0:
                            st.subheader("Cluster Characteristics - Numeric Features")
                            
                            # Create a figure for the numeric features comparison
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Create a grouped bar chart for numeric features
                            bar_width = 0.8 / len(selected_numeric)
                            x = np.arange(n_clusters)
                            
                            for i, col in enumerate(selected_numeric):
                                avg_col = f'Avg_{col}'
                                position = x + (i - len(selected_numeric)/2 + 0.5) * bar_width
                                ax.bar(position, summary[avg_col], bar_width, label=col, color=plt.cm.tab10(i))
                            
                            ax.set_xlabel('Cluster')
                            ax.set_xticks(x)
                            ax.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
                            ax.set_title('Average Values by Cluster')
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            st.pyplot(fig)
                        
                        # Create a correlation heatmap between numeric features
                        if len(selected_numeric) > 1:
                            st.subheader("Correlation Between Numeric Features")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            corr = df1[selected_numeric].corr()
                            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                            st.pyplot(fig)
                        
                        # Detailed profile for each cluster
                        st.subheader("Detailed Donor Profiles by Cluster")
                        
                        for i in range(n_clusters):
                            with st.expander(f"Cluster {i} - {summary.loc[i, 'Donor_Count']} donors"):
                                cluster_data = df1[df1['Cluster'] == i]
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**Most Common Characteristics:**")
                                    
                                    # Display categorical features
                                    for col in selected_categorical:
                                        most_common = summary.loc[i, f'Most_Common_{col}']
                                        if isinstance(most_common, str):
                                            most_common = most_common.title()
                                        st.markdown(f"- {col}: {most_common}")
                                    
                                    # Display numeric features
                                    for col in selected_numeric:
                                        avg_value = summary.loc[i, f'Avg_{col}']
                                        st.markdown(f"- Average {col}: {avg_value:.2f}")
                                    
                                with col2:
                                    # Create a simple visualization for this cluster
                                    if len(selected_numeric) > 0:
                                        # Pick first numeric column for visualization
                                        viz_col = selected_numeric[0]
                                        
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        
                                        # Distribution within cluster
                                        sns.histplot(cluster_data[viz_col], kde=True, ax=ax)
                                        ax.set_title(f"{viz_col} Distribution in Cluster {i}")
                                        ax.set_xlabel(viz_col)
                                        ax.set_ylabel("Count")
                                        
                                        st.pyplot(fig)
                        
                        # Allow downloading the clustered data
                        csv = df1.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="donor_clusters.csv">Download Clustered Data</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Campaign recommendations based on clusters
                        st.markdown("<h2 class='section-header'>Campaign Recommendations</h2>", unsafe_allow_html=True)
                        
                        # Generate recommendations based on cluster analysis
                        for i in range(n_clusters):
                            with st.expander(f"Targeting Strategy for Cluster {i}"):
                                st.markdown("**Donor Profile:**")
                                
                                # Demographic info
                                demo_info = []
                                for col in selected_categorical:
                                    value = summary.loc[i, f'Most_Common_{col}']
                                    if isinstance(value, str):
                                        value = value.title()
                                    demo_info.append(f"{col}: {value}")
                                
                                st.markdown(f"- Demographics: {', '.join(demo_info)}")
                                
                                # Age and health info
                                health_info = []
                                for col in selected_numeric:
                                    health_info.append(f"Average {col}: {summary.loc[i, f'Avg_{col}']:.2f}")
                                
                                st.markdown(f"- Health Metrics: {', '.join(health_info)}")
                                
                                # Targeting recommendations
                                st.markdown("**Recommended Targeting Strategy:**")
                                
                                # Get most common location and profession if available
                                location_col = next((col for col in selected_categorical if any(loc in col.lower() for loc in ['ville', 'city', 'location'])), None)
                                profession_col = next((col for col in selected_categorical if any(prof in col.lower() for prof in ['domain', 'profession', 'occupation'])), None)
                                
                                if location_col:
                                    location = summary.loc[i, f'Most_Common_{location_col}']
                                    if isinstance(location, str):
                                        location = location.title()
                                    st.markdown(f"- Focus campaign efforts in {location} area")
                                
                                if profession_col:
                                    profession = summary.loc[i, f'Most_Common_{profession_col}']
                                    if isinstance(profession, str):
                                        profession = profession.title()
                                    st.markdown(f"- Target individuals in the {profession} sector")
                                
                                # Specific recommendation based on cluster size
                                cluster_size = summary.loc[i, 'Donor_Count']
                                total_donors = df1.shape[0]
                                percentage = (cluster_size / total_donors) * 100
                                
                                if percentage > 30:
                                    st.markdown(f"- This is a large segment ({percentage:.1f}% of donors). Consider broad marketing campaigns.")
                                elif percentage > 10:
                                    st.markdown(f"- This is a medium segment ({percentage:.1f}% of donors). Use targeted marketing campaigns.")
                                else:
                                    st.markdown(f"- This is a small segment ({percentage:.1f}% of donors). Focus on personalized outreach.")
                        
                else:
                    st.warning("Please select at least one numeric and one categorical feature to proceed with clustering.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please make sure your CSV file has the required format and try again.")

        else:
            # Display instructions when no file is uploaded
            st.info("Please upload a CSV file containing donor data to begin analysis.")
            
            st.markdown("""
            ## Expected Data Format
            
            Your CSV file should contain:
            
            **Numeric features** such as:
            - Age
            - Hemoglobin levels (important for blood donation eligibility)
            - Number of previous donations
            
            **Categorical features** such as:
            - Gender/Genre
            - Professional domain/occupation
            - City/Location
            
            ## Important Note
            
            <div class='note-box'>
            Features that don't show eligibility or don't represent donor characteristics should not be included 
            in the clustering analysis, as they may bias the results and lead to inaccurate donor profiles.
            </div>
            
            ## Dashboard Features
            
            Once you upload your data, this dashboard will:
            1. Allow you to select relevant features for clustering
            2. Profile and segment donors using clustering techniques
            3. Identify characteristics of ideal donors (based on hemoglobin levels and other factors)
            4. Generate visual insights about donor segments
            5. Provide detailed profiles and targeting strategies for each donor cluster
            """, unsafe_allow_html=True)
############################################################################################################################
############################################################################################################################ 



    
    
    
    
#########################################################################
###########              Campaign effectiveness            ##############
######################################################################### 
    elif tab_index == 3:  
        show_title("Campaign effectiveness",30)
        
        # visualization
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
        set_sidebar_title("filter on campains and donors")
        available_years = sorted(df["Year"].unique().astype(int))
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
##########################################################################
##########################################################################



     
         
    
#########################################################################
###########            Previous donation analysis          ##############
#########################################################################
    elif tab_index == 4:              
            # Previous donation analysis
            show_title("Previous donation analysis",30)
            # rename the data and filtered data in the session_state
            df = st.session_state.df
            df_filtered = st.session_state.df_filtered
            # plot the pie and trend of the first time donor and none first time donors
            show_recurrent_pie_trend(df,df_filtered)
            # Calculate donation intervals for repeat donors
            donation_intervale(df,df_filtered)
##########################################################################
##########################################################################
    
        
#########################################################################
###########                 Survey/Feedback               ###############
#########################################################################
    elif tab_index == 5:  
        show_title("Survey/Feedback",30)
        
#########################################################################
#########################################################################
    
#########################################################################
###########              Eligibility prediction           ###############
#########################################################################
    elif tab_index == 6:
        # title of the tab section
        show_title("Eligibility prediction",30)
        # Custom CSS to increase the width and height of the input form
        st.markdown(
            """
            <style>
            .stTextArea textarea {
                height: 200px;
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # Show the formular to get the data fro prediction 
        show_pred_form()
        # When the form is submitted 
        try_pred()
#########################################################################
#########################################################################

#########################################################################
#########################################################################
# Display content for each tab and update active tab when tab is selected#
try:
    #print("")
    with tabs[0]:
        set_active_tab(tab_names[0])
        show_tab_content(0)

    try:
        with tabs[1]:
            set_active_tab(tab_names[1])
            show_tab_content(1)
    except IndexError:
        with tabs[1]:
            indication_message("Ensure that you have fill the filter for at least the eligibility to get the informations.") 


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
except AttributeError:
    st.warning("Ensure that you have import the dataset before continue please !")

with tabs[6]:
    set_active_tab(tab_names[6])
    show_tab_content(6)
#################################
#################################
