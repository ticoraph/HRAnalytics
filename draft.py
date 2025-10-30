col_onehot = ["poste", "domaine_etude", "statut_marital", "departement"]
col_ordinal = ["frequence_deplacement"]

# Définition du préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), make_column_selector(dtype_include='number')),
        ("onehot", OneHotEncoder(dtype=bool), col_onehot), # drop="first",
        ("ordinal", OrdinalEncoder(), col_ordinal)
    ],
    remainder='passthrough'  # Garde les autres colonnes
)

# Pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Application
X_transformed = pipeline.fit_transform(fc)

# --- Récupérer les noms de colonnes ---
# Colonnes numériques scalées
num_cols = pipeline.named_steps["preprocessor"].named_transformers_["num"].feature_names_in_

# Colonnes OneHot
ohe = pipeline.named_steps["preprocessor"].named_transformers_["onehot"]
onehot_cols = ohe.get_feature_names_out(col_onehot)

# Toutes les colonnes dans l'ordre
all_columns = list(num_cols) + list(onehot_cols) + col_ordinal

# Ajouter les colonnes "passthrough" (celles non transformées)
remaining_cols = [col for col in fc.columns if col not in list(num_cols) + col_onehot + col_ordinal]
all_columns += remaining_cols

# --- Refaire un DataFrame propre ---
fc_transformed = pd.DataFrame(
    X_transformed,
    columns=all_columns,
    index=fc.index
)

fc = fc_transformed.sort_index(axis=1)