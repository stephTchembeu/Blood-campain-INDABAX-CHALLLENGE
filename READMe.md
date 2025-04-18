![logo](images/my_logo_with_text.png)

# Blood Campaign Dataset Preprocessing Documentation

## **Overview**

This document outlines the preprocessing steps applied to the **Blood Campaign Dataset** to ensure data quality and consistency, particularly focusing on the column different columns of the blood donation campaign dataset

## **A-Preprocessing for Data to be used for Campaign Dashboard**

## I-Date de remplissage column

## **a-Libraries Used**

-Pandas
-Json

## **b-Data Cleaning Steps**

### 1. **Loading the Dataset**

- The Excel file was imported using `pandas` with `sheet_name=None` to load all sheets.
- Example code:
  ```python
  blood_campaign = pd.read_excel("path/to/Challenge dataset.xlsx", sheet_name=None)
  ```

### 2. **Date Column Preprocessing**

- **Parsing Dates**:  
  The `Date de remplissage de la fiche` column was converted to a datetime format. Dates in ambiguous formats (e.g., `m/d/yyyy` with incorrect years) were standardized.
- **Year Correction**:
  - Dates with years **> 2019** (e.g., 2020) or **between 2010–2018** were adjusted to **2019**, assuming data entry errors.
  - Example adjustment logic:
    ```python
    def adjust_year(date):
        if (date.year > 2019) or (2010 < date.year < 2019):
            return date.replace(year=2019)
        return date
    ```
- **Invalid Date Removal**:  
  4 rows with dates that could not be reliably inferred (e.g., years outside 2010–2020) were dropped from the dataset.

### 3. **Validation**

- Post-processing, all dates in `Date de remplissage de la fiche` were confirmed to belong to **2019**.
- Final check:
  ```python
  filtered_dates = sheet_2019[sheet_2019['Date de remplissage de la fiche'].dt.year != 2019]
  filtered_dates.shape  # Output: (0, 39)
  ```

## **Key Changes**

- **Rows Removed**: 4 invalid entries.
- **Date Format**: Standardized to `YYYY-MM-DD`.
- **Assumptions**:
  - Years in `2010–2018` or `>2019` were treated as typos and corrected to 2019.
  - Dates with ambiguous formats (e.g., `d/m/yyyy` vs. `m/d/yyyy`) were parsed with day-first logic.

## **II-Date de naissance (Date of Birth) Preprocessing**

### **1. Handling Invalid Birth Years**

- **Issue**: Some entries had implausible future years (e.g., `2089`, `2092`), likely due to data entry errors (e.g., `1989` typed as `2089`).
- **Fix**:
  - Years `2089` and `2092` were corrected to `1989` and `1992`, respectively.
  - Code logic:
    ```python
    def fix_birthdate_year(date):
        if date.year > 2019:
            if date.year == 2089:
                return date.replace(year=1989)
            elif date.year == 2092:
                return date.replace(year=1992)
        return date
    sheet_2019["Date de naissance"] = sheet_2019["Date de naissance"].apply(fix_birthdate_year)
    ```

### **2. Filtering Underage Donors**

- **Issue**: Blood donors must be at least **18 years old**. Entries with calculated age `<18` were deemed invalid.
- **Fix**:
  - Rows with `Age = 2019 - Birth Year < 18` were removed.
  - Code logic:
    ```python
    sheet_2019 = sheet_2019[2019 - sheet_2019["Date de naissance"].dt.year >= 18]
    ```
- **Rows Removed**: Entries violating the age threshold were dropped (exact count dependent on data).

### **3. Validation**

- **Distinct Years**: Post-correction, birth years were confirmed to fall within plausible ranges (e.g., no future years).
- **Age Check**: All remaining donors are `≥18 years old` as of 2019.

---

## **Key Assumptions**

- Future years `2089` and `2092` were assumed to be typos for `1989` and `1992`.
- Age was calculated as `2019 - Birth Year`; no exact birthdates were used for precision.

## **III-NIveau d'Etude, Genre and Situation Matrimonial**

These columns were not touched as the carried the required information and correct filling techniques. Assumptions for Niveau d'Etude couldnot be extracted from Profession because we don't know the level of education the person got the job with and so they were features for these columns were not touched

## **IV-Profession**

Here a profession mapping was created to map profession with right spellings
In this portion we encountered series of
1-Similar words written in different ways e.g Etudiant and Etudiant(e)
2-Words wrongly spelled e.g Logisticien and Logiticien
3-Words which meant the same or performed same duties but written in different ways e.g 'Agent de securite and vigil'
4-words with different spacings e.g RAS and R.A.S

```
    profession_mapping={
    'conducteur': 'Chauffeur',
    'comptable financier':'Comptable_financier',
    'non precise':'Pas_precise',
    'hotellier':"Hotelier",
    'chaufeur': 'Chauffeur'
    'r a s':'Pas_precise',
    }
```

The profession mapping in the json file was used to handle these cases and new column was created known as domain which was used to group these professions into domains so it can be used for clustering applications

## \*\*V-Arrondissement and Quartier de Residence

These two had different json files which including Arrondissement mapping and Quarter mapping. These were used to map words which mean same but may have different spellings or mispelling to be map to the right word.

````quarter_mapping={
    'logbaba': 'Logbaba',
    'ndogpassi 2': 'Ngodpassi II',
    'dakar': 'Dakar',
    'ngangue':'Ngangue',
    'douala': 'Pas_precise',
    'bependa':'Bepanda',
    'bepanda':'Bepanda',

    ```

-Also a new column was created called town. This is because in manipulating the dataset, we noticed that some rather filled towns and not arrondissement.

````

cleaned['Ville'] = cleaned['Arrondissement de résidence'].map(new_column)

```
-Also in places where the Arrondissement had non precise but we the quarter was given, we used the quarter to be able to put the in the right town mapping

```

# Mapping dictionary for Quartier -> Ville

quarter_to_ville = {
'mbanga': 'Mbanga',
'nkolmesseng': 'Yaounde',
'bangapongo': 'Mbanga-Pongo',
'mbangopongo': 'Mbanga-Pongo'
}

## \*\*Poids and Taille

These two columns were removed as they were not needed to provide responses to the questions the dashboard was required to answer

## **Taux 'Haemoglobine**

Here the preprocessing steps applied were

1-Correcting wrong inputs
2-Manually correcting values which were strings e.g '13.2' to '13.2
3-Removal of unit of measurement which were found in some hemoglobine values 13,1g/dl
4-Some values had '.' while others had ',' as seperator indicating decimal point and this was corrected to '.'

#### Assumptions

We noticed a very high hemglobine value of 123.1 and we assumed this was a recording mistake and as such we assumed the person wanted to with 12.31 and so the value was replaced with 12.31

# B-Preprocessing Techniques for Each Question

## 1-Donor Profiling

This document outlines the data preprocessing and transformation steps applied to enable the interactive dashboard visualizations.

---

## **Key Preprocessing Techniques**

### a. **Geospatial Data Integration**

- **Purpose**: Map donor distribution across arrondissements
- **Steps**:
  - Loaded Cameroon ADM3 GeoJSON boundaries
  - Merged donor counts with geographical features using `shapeName` as key
  - Added `num_candidates` property to GeoJSON features for choropleth mapping
- **Tools**: `geojson`, `folium.GeoJson`

### b. **Temporal Data Formatting**

- **Purpose**: Enable time-series analysis
- **Steps**:
  - Converted `Date de remplissage de la fiche` to datetime format
  - Extracted month/year components using `dt.strftime("%Y-%m")`
  - Aggregated counts per month for time-series visualization
- **Tools**: `pandas.to_datetime`, `dt.strftime`

### c. **Interactive Filter Preparation**

- **Purpose**: Support map-driven filtering
- **Steps**:
  - Maintained session state for selected arrondissement
  - Created boolean mask for filtering data:  
    `arr_selected = df["Arrondissement de residence"] == selected_dept`
  - Generated filtered DataFrame for drill-down visualizations
- **Tools**: `st.session_state`, boolean indexing

### d. **Categorical Data Encoding**

- **Purpose**: Enable demographic analysis
- **Steps**:
  - Mapped eligibility categories to abbreviated labels:
    ```python
    {"eligible": "El", "temporairement...": "T-N-El", ...}
    ```
  - Calculated gender proportions using `value_counts(normalize=True)`
  - Created grouped aggregates for eligibility-gender analysis
- **Tools**: `pd.value_counts`, `groupby`

### e. **Visual Encoding Preprocessing**

- **Purpose**: Enhance visual perception
- **Steps**:
  - Generated gradient color mapping based on donor density
  - Created color-discrete maps for categorical variables:
    ```python
    {"El": "rgb(23,158,14)", "T-N-El": "rgb(255,165,0)", ...}
    ```
  - Formatted temporal axes using month/year labels
- **Tools**: `plotly.express`, custom color functions

## \*\*2-Eligibility criteria

## Overview

The preprocessing pipeline ensures that:

- **Relevant columns are extracted** based on the keyword `"raison"` to identify health conditions.
- **Eligibility parameters and conditions are dynamically extracted** based on user input.
- **The dataset is filtered** according to user-selected eligibility values.
- **The original data remains unchanged** by applying filters on a copy of the DataFrame.

### 1. Loading Dataset into Session State

The raw data is loaded and stored in `st.session_state.df` so that all parts of the dashboard can access the same updated DataFrame.

### 2. Extracting Health Condition Columns

To focus on the health-related criteria influencing donor eligibility, the dashboard extracts columns that include the word `"raison"` (ignoring case):

```python
health_conditions_columns = [x for x in st.session_state.df.columns if 'raison'.lower() in x.lower()]

```

### 3. Extracting Eligibility and Condition Parameters

A helper function, get_eligibility_observations_param(), is utilized to extract key filtering parameters from the DataFrame:

    eligibility_column: The column that indicates donor eligibility status.

    selected_eligibility: The selected eligibility values (for example, “Eligible” or “Not Eligible”).

    selected_conditions: The specific health condition reasons chosen for deeper analysis.

This function dynamically sets the filtering criteria based on the available dataset and user inputs.

### 4. Filtering the Dataset

After extracting the necessary parameters, the dataset is filtered based on the user’s selection. If an eligibility filter is applied, only the rows matching the selected values are retained; otherwise, the full dataset is maintained:

### 5. Preservation of Original Data

To enable flexible and multiple analyses, the dashboard ensures the original DataFrame (st.session_state.df) remains unmodified. All filtering is performed on a separate copy (df_filtered), ensuring data integrity and preserving the complete raw data for reference and future operations.

## \*\*3-Donor Profiling

This document details the preprocessing steps carried out in the Donor Profiling section before performing clustering analysis. The preprocessing pipeline ensures that the selected features are properly cleaned, transformed, and ready for the clustering algorithm.

---

## 1. Feature Selection

- **Numeric Features:**  
  The dashboard allows users to select numeric columns (e.g., age, hemoglobin levels) through a multi-select widget.
- **Categorical Features:**  
  Users can also select categorical columns (e.g., gender, occupation, city) to include in the profiling analysis.

A copy of the original dataset is created containing only the selected features. This is to ensure that further processing and cleaning are done only on the data relevant to the clustering task.

---

## 2. Data Cleaning & Preparation

The following steps are taken to ensure data quality:

- **Missing Value Detection:**  
  The preprocessing step detects columns with missing values and displays them to the user. Although this information is shown for awareness, further cleaning is performed automatically.

- **Dropping Incomplete Rows:**  
  To maintain consistency, rows with missing values in the categorical features are dropped. This ensures that the categorical data used in clustering is complete.

- **Cleaning Categorical Data:**  
  Categorical columns are cleaned by:
  - Converting values to strings.
  - Stripping any extra whitespace.
  - Converting text to lowercase to maintain consistency.

---

## 3. Preprocessing Pipeline Construction

A preprocessing pipeline is built using scikit-learn’s `ColumnTransformer` along with specific pipelines for numeric and categorical data:

- **Numeric Pipeline:**

  - **Imputation:** Missing numeric values are replaced using the mean value.
  - **Scaling:** Data is standardized using `StandardScaler` to ensure that numeric features contribute equally to the clustering.

- **Categorical Pipeline:**
  - **Imputation:** Missing categorical values are filled in with the most frequent value in each column.
  - **One-Hot Encoding:** Categorical features are then converted into numerical representations via one-hot encoding, with the `drop='first'` parameter to avoid multicollinearity.

## Code Snippet

```python
# Select and copy only the relevant features from the original DataFrame
df1 = df[selected_numeric + selected_categorical].copy()

# Detect and display missing values for awareness
missing_cols = df1.columns[df1.isna().any()].tolist()
if missing_cols:
    st.markdown("<div class='note-box'>", unsafe_allow_html=True)
    st.write("Missing values detected in the following columns:")
    for col in missing_cols:
        missing_count = df1[col].isna().sum()
        missing_percent = (missing_count / len(df1)) * 100
        st.write(f"- {col}: {missing_count} values ({missing_percent:.2f}%)")
    st.markdown("</div>", unsafe_allow_html=True)

# Drop rows with missing values in categorical features
df1 = df1.dropna(subset=selected_categorical)

# Clean categorical feature values: strip whitespace and convert to lowercase
for col in selected_categorical:
    if df1[col].dtype == 'object':
        df1[col] = df1[col].astype(str).str.strip().str.lower()

# Construct the preprocessing pipeline for numeric and categorical data
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

# Fit the preprocessor and transform the selected data to produce the preprocessed feature matrix
X_preprocessed = preprocessor.fit_transform(df1)

```

After executing these steps, the resulting preprocessed data (X_preprocessed) is clean, standardized, and ready for effective donor clustering and profiling.

## 4.Campaign effectiveness Preprocessing

This document describes the preprocessing steps performed in the Campaign Effectiveness section of the dashboard. These steps ensure that the dataset is properly formatted, enriched with date and age-related features, and filtered for accurate campaign performance analysis.

## Overview

Before analyzing campaign performance, the dataset undergoes several preprocessing operations to:

- Convert date-related columns into the proper datetime format.
- Calculate donor age using the birth date.
- Create age groups for demographic segmentation.
- Extract time components to facilitate trend analysis.
- Filter the dataset based on user-selected criteria (e.g., year, eligibility, gender, and profession).

These preparatory transformations enhance the quality and consistency of data used for further performance analysis.

## Preprocessing Steps

### 1. Date Conversion

- **Convert Dates:**  
  The campaign analysis requires consistent date formats. Date-related columns (e.g., campaign dates and last donation dates) are converted from string to `datetime` objects while handling any conversion errors. For example:

  ```python
  df[st.session_state.date_column] = pd.to_datetime(df[st.session_state.date_column], errors='coerce')
  df[st.session_state.last_donation_column] = pd.to_datetime(df[st.session_state.last_donation_column], errors='coerce')

  ```

### 2. Age Grouping

Inorder to support demographic segmentation, the age values are grouped into defined bins:

```
df['Age Group'] = pd.cut(
    df['Age'],
    bins=[0, 18, 25, 35, 45, 55, 65, 100],
    labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
)

```

###3. Time Feature Extraction
Rows without a valid campaign date are removed to ensure accurate time-based analysis:

```
df = df.dropna(subset=[st.session_state.date_column])

```

Later, time components were extracted and new columns were created from the campaign date to assist with temporal trend analysis:
-Year, Month, Day: Simple extractions of year, month, and day for granular analysis.
-Weekday:Extracts the name of the day (e.g., Monday, Tuesday).
-Year-Month Period:Groups data into periods for monthly trend visualizations.

```
df["Year"] = df[st.session_state.date_column].dt.year
df["Month"] = df[st.session_state.date_column].dt.month
df["Day"] = df[st.session_state.date_column].dt.day
df["Weekday"] = df[st.session_state.date_column].dt.day_name()
df["Year-Month"] = df[st.session_state.date_column].dt.to_period("M")

```

Additional filter were included which included filtering by year, elibility and other demographic factors

## 5.Previous donation analysis/Donor retention preprocessing

This section explains the preprocessing and data preparation steps for the Previous Donation Analysis tab. The goal is to distinguish between first-time donors and repeat donors and calculate key donation interval metrics.

## Overview

The Previous Donation Analysis uses two primary data sources:

- **Original Dataset (`df`):** Contains all donor records.
- **Filtered Dataset (`df_filtered`):** A subset of `df` based on prior user-defined filters (e.g., year, eligibility, gender, etc.) applied in an earlier step.

Key tasks include:

- Visualizing the distribution of first-time donors versus repeat donors.
- Analyzing the trend of recurring donations.
- Calculating the intervals between donations for repeat donors.

These operations rely on data already cleaned and filtered in earlier steps.

## Preprocessing Steps

### 1. Data Retrieval

- **Retrieve the DataFrames:**  
  The analysis uses the original dataset (`df`) and the filtered dataset (`df_filtered`), both stored in the session state. This ensures consistency across different tabs:
  ```python
  df = st.session_state.df
  df_filtered = st.session_state.df_filtered
  ```

### 2. Visualization and Analysis Preparation

Visualizing Donor Categories:The function show_recurrent_pie_trend(df, df_filtered) is invoked to create:
-A pie chart breaking down first-time versus repeat donors.
-A trend plot indicating the donation patterns over time for these groups.

Donation Interval Calculation:The function donation_intervale(df, df_filtered) is used to calculate the time intervals between donations for repeat donors. This helps in understanding donor retention and frequency.

# Prerequisites

Upload the project, open it in a terminal go into the folder script and run:

```sh
streamlit run dash_app.py
```

the dashboard show in your web navigator is a fully interactive dashboard made to be intuitive for the user.</br>
It is compose of two parts the main part which is showing you information from data and the sidebar used to give you
more access to what you may want.</br>
The central part contain the navigatio to naviguate amoung different otions and under the navigation you have the part to see the different informations.

![interface](images/readme_imgs/dashboard_view.png)

## Upload the dataset.

To ensure you have upload the dataset on which you want to extract information you get a customise message which indicates that first we need to load the dataset.</br>
To upload the dataset you will need to go in the sidebar and click on the upload button to load your <b style="font-size:15px;">.csv</b> file.</br>
After the data been load you will start get information in the main part of the dashboard.</br>
Each of the tab in the nav contains specifique information on each on the proposing question. As tab we get the :

- <b style="font-size:15px;">Donors distributions</b> : which show the distribution of the candidates over the different arrondissement in the dataset.
- <b style="font-size:15px;">The Eligibility</b> : this tab give you information on the different proportion of candidate according to different heath condition which impat the eligibility.
- <b style="font-size:15px;">Donors profiling</b>: here we study the different profile of each donos and build some cluster base on similar information.
- <b style="font-size:15px;">campaign effectiveness</b>: here we give you insights on the different ongoing facts all over the donation campaign duration.
- <b style="font-size:15px;">Eligibility prediction</b>: in this last section you can get the prediction of some one from who you have collected data.
  ![naviguation](images/readme_imgs/nav.png)

# Donors distribution

Putting the cursor on the map gives the number of donors in each department and increasing or tapping on the map outputs the distribution of selected quarter in the map showing eligibility and non-eligibility.
![donors_interface](images/readme_imgs/donors.png)

# Eligibility

Moving to the eligibility tab gives information about the number of donors. But firstly you should select the ELIGIBLE AU DON in the side-bar
You also have a filter which gives us the opportunity to move between the two unique option of multiple selected values the one of ELIGIBLE AU DON.
and two other for the health condition When you choose the health conditions for temporary or definitely non eligible it will display non eligible donors will display the number of non eligible per in total number of non eligible
![donors_interface](images/readme_imgs/eligibility.png)

# Ideal Donor Profile

Here we still choose variable related to eligibility and select features we want to include in our clusters. the cluster with the highest number of eligible donors is considered and later eligibility score computer. The average age of the best donors is outputted which indicates the best donors related to our clusters

# Campaign effectiveness

This portion analyses the monthly, weekly and daily distributions of blood donations and also identifies the top ten donation periods.

# Donor Retention

This portion reveals to us how often those who have given blood come back for blood donation. The plots which are interactive give insights based on the different features

# Model Prediction API

This part contains the code to run a machine learning model and making predictions using a Flask API. The API allows users to send data and receive predictions on the eligibility condition of the person who data are passed here

## Prerequisites to make prediction

- Python 3.10 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE.git
   cd Blood-campain-INDABAX-CHALLLENGE
   ```

2. **Create a virtual environment** (optional but recommended):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required libraries**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Download the Textblob corpora for Sentiment Analysis**

   ```sh
   python -m textblob.download_corpora
   ```

## Training the Model

1. **Prepare your data**: Ensure the data ml_data_2019.xlsx is in the folder.

2. **Train the model**: Run the training script to train the model and save it along with the preprocessor and label encoder.

   ```sh
   python train.py
   ```

   This script will:

   - Load the data
   - Preprocess the data
   - Train the model
   - Save the model, preprocessor, and label encoder to a file named [model_pipeline.pkl](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/model_pipeline.pkl)

## Running the Flask API

1. **Start the Flask server**:

   ```sh
   python app.py
   ```

   This will start the Flask server on `http://127.0.0.1:5001`.

## Making Predictions on the Web App

1. **Donor Status Prediction**: Go to the "Eligibity prediction" tab to make prediction on the Web App.

2. In the Input Data form, enter your data in a JSON format following the format below, you can just copy and paste it, and change the value of features.

   ```json
   {
     "ID": "DONOR_2788",
     "Age": 28,
     "Horodateur": "24:18.6",
     "Niveau_d'etude": "Universitaire",
     "Genre_": "Homme",
     "Taille_": null,
     "Poids": null,
     "Situation_Matrimoniale_(SM)": "Célibataire",
     "Profession_": "Etudiant (e)",
     "Arrondissement_de_résidence_": "Douala 3",
     "Quartier_de_Résidence_": "Ndogbon",
     "Nationalité_": "Camerounaise",
     "Religion_": "Chretien (Catholique)",
     "A-t-il_(elle)_déjà_donné_le_sang_": "Non",
     "Si_oui_preciser_la_date_du_dernier_don._": null,
     "Taux_d’hémoglobine_": null,
     "Raison_indisponibilité__[Est_sous_anti-biothérapie__]": null,
     "Raison_indisponibilité__[Taux_d’hémoglobine_bas_]": null,
     "Raison_indisponibilité__[date_de_dernier_Don_<_3_mois_]": null,
     "Raison_indisponibilité__[IST_récente_(Exclu_VIH,_Hbs,_Hcv)]": null,
     "Date_de_dernières_règles_(DDR)__": null,
     "Raison_de_l’indisponibilité_de_la_femme_[La_DDR_est_mauvais_si_<14_jour_avant_le_don]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Allaitement_]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[A_accoucher_ces_6_derniers_mois__]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Interruption_de_grossesse__ces_06_derniers_mois]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[est_enceinte_]": null,
     "Autre_raisons,__preciser_": null,
     "Sélectionner_\"ok\"_pour_envoyer_": null,
     "Raison_de_non-eligibilité_totale__[Antécédent_de_transfusion]": null,
     "Raison_de_non-eligibilité_totale__[Porteur(HIV,hbs,hcv)]": null,
     "Raison_de_non-eligibilité_totale__[Opéré]": null,
     "Raison_de_non-eligibilité_totale__[Drepanocytaire]": null,
     "Raison_de_non-eligibilité_totale__[Diabétique]": null,
     "Raison_de_non-eligibilité_totale__[Hypertendus]": null,
     "Raison_de_non-eligibilité_totale__[Asthmatiques]": null,
     "Raison_de_non-eligibilité_totale__[Cardiaque]": null,
     "Raison_de_non-eligibilité_totale__[Tatoué]": null,
     "Raison_de_non-eligibilité_totale__[Scarifié]": null,
     "Si_autres_raison_préciser_": null
   }
   ```

3. Click on the "Submit" button to get the prediction.

## Making Predictions using Postman

1. **Open Postman**: If you don't have Postman installed, you can download it from [Postman's official website](https://www.postman.com/downloads/).

2. **Create a new request**:

   - Click on the "New" button, then select "Request".
   - Name your request (e.g., "Model Prediction") and save it to a collection if you have one, or create a new collection.

3. **Set up the request**:

   - Set the request type to `POST`.
   - Enter the URL for your Flask API endpoint: `http://127.0.0.1:5001/predict`.

4. **Set the request headers**:

   - Click on the "Headers" tab.
   - Add a new header with the key `Content-Type` and the value `application/json`.

5. **Set the request body**:

   - Click on the "Body" tab.
   - Select the "raw" radio button.
   - Ensure the format is set to `JSON` (application/json).
   - Enter the JSON data that your model expects. For example:

   ```json
   {
     "ID": "DONOR_2788",
     "Age": 28,
     "Horodateur": "24:18.6",
     "Niveau_d'etude": "Universitaire",
     "Genre_": "Homme",
     "Taille_": null,
     "Poids": null,
     "Situation_Matrimoniale_(SM)": "Célibataire",
     "Profession_": "Etudiant (e)",
     "Arrondissement_de_résidence_": "Douala 3",
     "Quartier_de_Résidence_": "Ndogbon",
     "Nationalité_": "Camerounaise",
     "Religion_": "Chretien (Catholique)",
     "A-t-il_(elle)_déjà_donné_le_sang_": "Non",
     "Si_oui_preciser_la_date_du_dernier_don._": null,
     "Taux_d’hémoglobine_": null,
     "Raison_indisponibilité__[Est_sous_anti-biothérapie__]": null,
     "Raison_indisponibilité__[Taux_d’hémoglobine_bas_]": null,
     "Raison_indisponibilité__[date_de_dernier_Don_<_3_mois_]": null,
     "Raison_indisponibilité__[IST_récente_(Exclu_VIH,_Hbs,_Hcv)]": null,
     "Date_de_dernières_règles_(DDR)__": null,
     "Raison_de_l’indisponibilité_de_la_femme_[La_DDR_est_mauvais_si_<14_jour_avant_le_don]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Allaitement_]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[A_accoucher_ces_6_derniers_mois__]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Interruption_de_grossesse__ces_06_derniers_mois]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[est_enceinte_]": null,
     "Autre_raisons,__preciser_": null,
     "Sélectionner_\"ok\"_pour_envoyer_": null,
     "Raison_de_non-eligibilité_totale__[Antécédent_de_transfusion]": null,
     "Raison_de_non-eligibilité_totale__[Porteur(HIV,hbs,hcv)]": null,
     "Raison_de_non-eligibilité_totale__[Opéré]": null,
     "Raison_de_non-eligibilité_totale__[Drepanocytaire]": null,
     "Raison_de_non-eligibilité_totale__[Diabétique]": null,
     "Raison_de_non-eligibilité_totale__[Hypertendus]": null,
     "Raison_de_non-eligibilité_totale__[Asthmatiques]": null,
     "Raison_de_non-eligibilité_totale__[Cardiaque]": null,
     "Raison_de_non-eligibilité_totale__[Tatoué]": null,
     "Raison_de_non-eligibilité_totale__[Scarifié]": null,
     "Si_autres_raison_préciser_": null
   }
   ```

   The json above is an example that should give you a prediction "Eligible".
   You can replace the values to see what you got.

   Example of json that should give you "Temporairement Non-eligible"

   ```json
   {
     "ID": "DONOR_2788",
     "Age": 23,
     "Horodateur": "44:19.0",
     "Niveau_d'etude": "Universitaire",
     "Genre_": "Homme",
     "Taille_": null,
     "Poids": null,
     "Situation_Matrimoniale_(SM)": "Célibataire",
     "Profession_": "Sans Emplois",
     "Arrondissement_de_résidence_": "R A S",
     "Quartier_de_Résidence_": "R A S",
     "Nationalité_": "R A S",
     "Religion_": "Chretien (Catholique)",
     "A-t-il_(elle)_déjà_donné_le_sang_": "Oui",
     "Si_oui_preciser_la_date_du_dernier_don._": "1/1/2017 0:00",
     "Taux_d’hémoglobine_": "10.8",
     "Raison_indisponibilité__[Est_sous_anti-biothérapie__]": null,
     "Raison_indisponibilité__[Taux_d’hémoglobine_bas_]": null,
     "Raison_indisponibilité__[date_de_dernier_Don_<_3_mois_]": null,
     "Raison_indisponibilité__[IST_récente_(Exclu_VIH,_Hbs,_Hcv)]": null,
     "Date_de_dernières_règles_(DDR)__": null,
     "Raison_de_l’indisponibilité_de_la_femme_[La_DDR_est_mauvais_si_<14_jour_avant_le_don]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Allaitement_]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[A_accoucher_ces_6_derniers_mois__]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Interruption_de_grossesse__ces_06_derniers_mois]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[est_enceinte_]": null,
     "Autre_raisons,__preciser_": null,
     "Sélectionner_\"ok\"_pour_envoyer_": null,
     "Raison_de_non-eligibilité_totale__[Antécédent_de_transfusion]": "Non",
     "Raison_de_non-eligibilité_totale__[Porteur(HIV,hbs,hcv)]": "Non",
     "Raison_de_non-eligibilité_totale__[Opéré]": "Oui",
     "Raison_de_non-eligibilité_totale__[Drepanocytaire]": "Non",
     "Raison_de_non-eligibilité_totale__[Diabétique]": "Non",
     "Raison_de_non-eligibilité_totale__[Hypertendus]": "Non",
     "Raison_de_non-eligibilité_totale__[Asthmatiques]": "Non",
     "Raison_de_non-eligibilité_totale__[Cardiaque]": "Non",
     "Raison_de_non-eligibilité_totale__[Tatoué]": "Non",
     "Raison_de_non-eligibilité_totale__[Scarifié]": "Non",
     "Si_autres_raison_préciser_": null
   }
   ```

   Example of json that should give you "Définitivement non-eligible"

   ```json
   {
     "ID": "DONOR_2788",
     "Age": 46,
     "Horodateur": "02:58.2",
     "Niveau_d'etude": "Pas Précisé",
     "Genre_": "Homme",
     "Taille_": null,
     "Poids": null,
     "Situation_Matrimoniale_(SM)": "Marié (e)",
     "Profession_": "ELECTRICIEN",
     "Arrondissement_de_résidence_": "Douala 3",
     "Quartier_de_Résidence_": "NDOGPOSSI",
     "Nationalité_": "Camerounaise",
     "Religion_": "Chretien (Catholique)",
     "A-t-il_(elle)_déjà_donné_le_sang_": "Non",
     "Si_oui_preciser_la_date_du_dernier_don._": "1/1/0001",
     "Taux_d’hémoglobine_": null,
     "Raison_indisponibilité__[Est_sous_anti-biothérapie__]": null,
     "Raison_indisponibilité__[Taux_d’hémoglobine_bas_]": "Oui",
     "Raison_indisponibilité__[date_de_dernier_Don_<_3_mois_]": null,
     "Raison_indisponibilité__[IST_récente_(Exclu_VIH,_Hbs,_Hcv)]": null,
     "Date_de_dernières_règles_(DDR)__": null,
     "Raison_de_l’indisponibilité_de_la_femme_[La_DDR_est_mauvais_si_<14_jour_avant_le_don]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Allaitement_]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[A_accoucher_ces_6_derniers_mois__]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[Interruption_de_grossesse__ces_06_derniers_mois]": null,
     "Raison_de_l’indisponibilité_de_la_femme_[est_enceinte_]": null,
     "Autre_raisons,__preciser_": null,
     "Sélectionner_\"ok\"_pour_envoyer_": "Ok",
     "Raison_de_non-eligibilité_totale__[Antécédent_de_transfusion]": null,
     "Raison_de_non-eligibilité_totale__[Porteur(HIV,hbs,hcv)]": null,
     "Raison_de_non-eligibilité_totale__[Opéré]": null,
     "Raison_de_non-eligibilité_totale__[Drepanocytaire]": null,
     "Raison_de_non-eligibilité_totale__[Diabétique]": null,
     "Raison_de_non-eligibilité_totale__[Hypertendus]": null,
     "Raison_de_non-eligibilité_totale__[Asthmatiques]": null,
     "Raison_de_non-eligibilité_totale__[Cardiaque]": null,
     "Raison_de_non-eligibilité_totale__[Tatoué]": null,
     "Raison_de_non-eligibilité_totale__[Scarifié]": null,
     "Si_autres_raison_préciser_": null
   }
   ```

6. **Send the request**:
   - Click the "Send" button.
   - You should see the response from your Flask API in the "Response" section at the bottom of the Postman window.

## File Structure

- [app.py](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/app.py): Flask application to serve the model and make predictions.
- [preprocess.py](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/preprocess.py): Contains preprocessing functions to prepare the data for the model.
- `train.py`: Script to train the model and save the model, preprocessor, and label encoder.
- [requirements.txt](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/requirements.txt): List of required Python libraries.
- [professions_mapping.json](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/professions_mapping.json),
  [arrondissements_mapping.json](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/arrondissements_mapping.json)
  ,[quarters_mapping.json](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/quarters_mapping.json)
  , [religion_mapping.json](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/religion_mapping.json)
  , [nationality_mapping.json](https://github.com/stephTchembeu/Blood-campain-INDABAX-CHALLLENGE/model_training/nationality_mapping.json):
  JSON files containing mappings for standardizing column values.

Troubleshooting
If you encounter any issues, check the console output for error messages and ensure that all dependencies are installed correctly. You can also add print statements in the code to debug and identify any problematic areas.

License
This project is licensed under the MIT License.
This README file provides a step-by-step guide on how to set up the environment, run the dashboard and a Flask API, and make predictions using Postman. It also includes an example of the expected JSON data format and troubleshooting tips.
