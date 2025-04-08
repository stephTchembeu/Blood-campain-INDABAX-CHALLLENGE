# Blood Campaign Dataset Preprocessing Documentation
## **Overview**
This document outlines the preprocessing steps applied to the **Blood Campaign Dataset** to ensure data quality and consistency, particularly focusing on the  column different columns of the blood donation campaign dataset

## **Preprocessing**

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
3-Words which meant the same or performed same duties but written in different ways e.g 'Agent de securite  and vigil'
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

## **V-Arrondissement and Quartier de Residence
These two had different json files which including Arrondissement mapping and Quarter mapping. These were used to map words which mean same but may have different spellings or mispelling to be map to the right word. 
```quarter_mapping={
    'logbaba': 'Logbaba',
    'ndogpassi 2': 'Ngodpassi II',
    'dakar': 'Dakar',
    'ngangue':'Ngangue',
    'douala': 'Pas_precise',
    'bependa':'Bepanda',
    'bepanda':'Bepanda',

    ```
    
-Also a new column was created called town. This is because in manipulating the dataset, we noticed that some rather filled towns and not arrondissement.

```
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
```

##**Poids and Taille
These two columns were removed as they were not needed to provide responses to the questions the dashboard was required to answer


##**Taux 'Haemoglobine**

Here the preprocessing steps applied were

1-Correcting wrong inputs
2-Manually correcting values which were strings e.g '13.2' to '13.2
3-Removal of unit of measurement which were found in some hemoglobine values 13,1g/dl
4-Some values had '.' while others had ',' as seperator indicating decimal point and this was corrected to '.'

#### Assumptions
We noticed a very high hemglobine value of 123.1 and we assumed this was a recording mistake and as such we assumed the person wanted to with 12.31 and so the value was replaced with 12.31


