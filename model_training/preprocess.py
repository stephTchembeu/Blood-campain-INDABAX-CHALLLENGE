# preprocess.py
import json
import pandas as pd

# Function to standardize values in specified columns
def standardize_column_values(df, column, mapping):
    return df[column].replace(mapping)

# Load mappings from JSON files
with open('professions_mapping.json', 'r') as json_file:
    professions_mapping = json.load(json_file)

with open('arrondissements_mapping.json', 'r') as json_file:
    arrondissements_mapping = json.load(json_file)

with open('quarters_mapping.json', 'r') as json_file:
    quarters_mapping = json.load(json_file)

with open('religion_mapping.json', 'r') as json_file:
    religion_mapping = json.load(json_file)

with open('nationality_mapping.json', 'r') as json_file:
    nationality_mapping = json.load(json_file)

def preprocessing_data(df):

    df["Niveau_d'etude"] = standardize_column_values(df, "Niveau_d'etude", {
    'Aucun': 'Pas Précisé'
    })

    df['Profession_'] = standardize_column_values(df, 'Profession_', professions_mapping)

    df['Arrondissement_de_résidence_'] = standardize_column_values(df, 'Arrondissement_de_résidence_', arrondissements_mapping)

    df['Quartier_de_Résidence_'] = standardize_column_values(df, 'Quartier_de_Résidence_', quarters_mapping)

    df['Religion_'] = standardize_column_values(df, 'Religion_', religion_mapping)

    df['Nationalité_'] = standardize_column_values(df, 'Nationalité_', nationality_mapping)




    # Replace values in 'Si_autres_raison_préciser_'
    df['Si_autres_raison_préciser_'] = df['Si_autres_raison_préciser_'].replace({
        'EU UNE ENDOSCOPIE ( FIBROSCOPIE,  GASTROSCOPIE, COLOSCOPIE .......)': 'EU UNE ENDOSCOPIE',
        'DROGUES': 'Consommation de drogue',
        'PAS D INFORMATION SUR SON DOSSIER': 'Aucune information',
        'ETE TRAITE PAR ACUPUNCTURE': 'TRAITE PAR ACUPUNCTURE'
    })

    # Replace values in 'Autre_raisons,__preciser_'
    df['Autre_raisons,__preciser_'] = df['Autre_raisons,__preciser_'].replace({
        'RAISON NON PRECISEE': 'Aucune',
        'RAS': 'Aucune',
        'PAS DE RAISON SPECIFIQUES': 'Aucune',
        'Rapport non protégé et changement de partenaire': 'Rapport non protege',
        'RAPPORT NON PROTEGER': 'Rapport non protege',
        'RAPPORT NON PROTEGE': 'Rapport non protege',
        'RAPPORTS NON PROTEGES': 'Rapport non protege',
        'Changé de partenaire et eu des rapports non protégé': 'Rapport non protege',
        'CONSOMMATION DE DROGUES': 'Consommation de drogue',
        'Eu à consommer de la cocaïne et d autre drogues': 'Consommation de drogue'
    })

    # Drop the specified columns
    df.drop(columns=['ID', 'Horodateur', 'Sélectionner_"ok"_pour_envoyer_', 'Date_de_dernières_règles_(DDR)__'], inplace=True)

    # Convert 'Si_oui_preciser_la_date_du_dernier_don._' to datetime and extract year, month, week, day
    df['Si_oui_preciser_la_date_du_dernier_don._'] = pd.to_datetime(df['Si_oui_preciser_la_date_du_dernier_don._'], errors='coerce')
    df['Year'] = df['Si_oui_preciser_la_date_du_dernier_don._'].dt.year
    df['Month'] = df['Si_oui_preciser_la_date_du_dernier_don._'].dt.month
    df['Week'] = df['Si_oui_preciser_la_date_du_dernier_don._'].dt.isocalendar().week
    df['Day'] = df['Si_oui_preciser_la_date_du_dernier_don._'].dt.day


    df.drop(columns=['Si_oui_preciser_la_date_du_dernier_don._'], inplace=True)

    # Replace ',' with '.' for decimal consistency
    df['Taux_d’hémoglobine_'] = df['Taux_d’hémoglobine_'].astype(str).str.replace(',', '.')

    # Remove non-numeric characters (e.g., 'g/dl')
    df['Taux_d’hémoglobine_'] = df['Taux_d’hémoglobine_'].str.extract(r'([\d\.]+)')

    df['Year'] = df['Year'].astype(float)
    df['Month'] = df['Month'].astype(float)
    df['Week'] = df['Week'].astype(float)
    df['Day'] = df['Day'].astype(float)
    df['Taux_d’hémoglobine_'] = df['Taux_d’hémoglobine_'].astype(float)

    cols = ['Raison_indisponibilité__[Est_sous_anti-biothérapie__]',
        'Raison_indisponibilité__[Taux_d’hémoglobine_bas_]',
        'Raison_indisponibilité__[date_de_dernier_Don_<_3_mois_]',
        'Raison_indisponibilité__[IST_récente_(Exclu_VIH,_Hbs,_Hcv)]',
        'Raison_de_l’indisponibilité_de_la_femme_[La_DDR_est_mauvais_si_<14_jour_avant_le_don]',
        'Raison_de_l’indisponibilité_de_la_femme_[Allaitement_]',
        'Raison_de_l’indisponibilité_de_la_femme_[A_accoucher_ces_6_derniers_mois__]',
        'Raison_de_l’indisponibilité_de_la_femme_[Interruption_de_grossesse__ces_06_derniers_mois]',
        'Raison_de_l’indisponibilité_de_la_femme_[est_enceinte_]',
        'Autre_raisons,__preciser_', 'Si_autres_raison_préciser_']

    df = df.astype({col: 'str' for col in cols})
    # Handle missing values
    #df.fillna("Others", inplace=True)

    return df

