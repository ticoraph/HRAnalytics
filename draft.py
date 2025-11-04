def compute_shap_values(pipeline, X_test, model_name):
    """
    Calcule les valeurs SHAP pour un modèle tree-based.

    Args:
        pipeline: Pipeline entraîné contenant scaler et modèle
        X_test: Features de test
        model_name: Nom du modèle

    Returns:
        dict: Dictionnaire contenant les valeurs SHAP et métadonnées
              ou None si le modèle n'est pas compatible
    """
    trained_model = pipeline.named_steps['model']
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

    # SHAP uniquement pour modèles tree-based
    tree_based_models = (XGBClassifier, CatBoostClassifier)

    if not isinstance(trained_model, tree_based_models):
        print(f"⚠ Modèle ignoré (non compatible avec TreeExplainer): {type(trained_model).__name__}")
        return None

    # Calcul des valeurs SHAP
    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Récupérer les noms de features
    feature_names_in = (
        trained_model.feature_names_in_
        if hasattr(trained_model, 'feature_names_in_')
        else X_test.columns.tolist()
    )

    # Créer un DataFrame avec les bonnes colonnes pour le plot
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=feature_names_in,
        index=X_test.index
    )

    # Métadonnées SHAP
    shap_metadata = {
        'model': model_name,
        'shap_values': shap_values,
        'X_test_scaled': X_test_scaled_df,  # DataFrame avec noms de colonnes
        'feature_names': feature_names_in,
        'explainer': explainer
    }

    return shap_metadata


def _plot_shap(all_shap_metadata):
    """
    Plot des valeurs SHAP pour les modèles tree-based.

    Args:
        all_shap_metadata: Liste de dictionnaires contenant les métadonnées SHAP
    """
    for meta in all_shap_metadata:
        # Skip si le modèle n'est pas compatible
        if meta is None:
            continue

        model_name = meta['model']
        shap_values = meta['shap_values']
        X_test_scaled = meta['X_test_scaled']

        # Gérer le cas multiclasse vs binaire
        if isinstance(shap_values, list):
            # Multiclasse : on plot pour chaque classe
            for class_idx, shap_vals_class in enumerate(shap_values):
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_vals_class,
                    X_test_scaled,
                    show=False,
                    plot_type="dot"
                )
                plt.title(
                    f'SHAP Summary Plot - {model_name} (Classe {class_idx})',
                    fontsize=14,
                    fontweight='bold'
                )
                plt.tight_layout()

                filename = f'images/classification_shap_summary_{model_name}_class{class_idx}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ SHAP summary plot sauvegardé: {filename}")

            # Plot global (toutes classes confondues)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test_scaled,
                show=False,
                plot_type="dot"
            )
            plt.title(
                f'SHAP Summary Plot - {model_name} (Toutes classes)',
                fontsize=14,
                fontweight='bold'
            )
            plt.tight_layout()

            filename = f'images/classification_shap_summary_{model_name}_all.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ SHAP summary plot sauvegardé: {filename}")

        else:
            # Binaire : un seul plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test_scaled,
                show=False,
                plot_type="dot"
            )
            plt.title(
                f'SHAP Summary Plot - {model_name}',
                fontsize=14,
                fontweight='bold'
            )
            plt.tight_layout()

            filename = f'images/classification_shap_summary_{model_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ SHAP summary plot sauvegardé: {filename}")


# Dans votre boucle principale, après avoir collecté tous les métadonnées :
print("\n>>> Génération des plots SHAP...")
_plot_shap(all_shap_metadata)


################################

# === Créer les tranches d'expérience totale ===
bins = [0, 5, 10, 15, 20, 100]
labels = ['0-5 ans', '5-10 ans', '10-15 ans', '15-20 ans', '20+ ans']
fc['tranche_experience_totale'] = pd.cut(fc['annees_experience_totale'], bins=bins, labels=labels)

# === Conversion Oui/Non vers 1/0 si besoin ===
if fc['a_quitte_l_entreprise'].dtype == 'object':
    fc['a_quitte_l_entreprise_num'] = fc['a_quitte_l_entreprise'].map({'Oui': 1, 'Non': 0})
else:
    fc['a_quitte_l_entreprise_num'] = fc['a_quitte_l_entreprise']

# === Calcul des stats par poste et tranche d'expérience totale ===
stats_poste_exp = (
    fc.groupby(['poste', 'tranche_experience_totale'], observed=True)['revenu_mensuel']
    .agg(mean='mean', std='std', count='count')
    .round(0)
    .reset_index()
)

# Calcul du coefficient de variation (CV)
stats_poste_exp['cv'] = (stats_poste_exp['std'] / stats_poste_exp['mean'] * 100).round(1)

# === Calcul du nombre de démissions par poste et tranche ===
demissions_poste_exp = (
    fc.groupby(['poste', 'tranche_experience_totale'], observed=True)['a_quitte_l_entreprise_num']
    .sum()
    .reset_index()
)

# === Fusion des deux jeux de données ===
merged = pd.merge(stats_poste_exp, demissions_poste_exp,
                  on=['poste', 'tranche_experience_totale'],
                  how='left')

# === Ajouter une colonne de couleur selon les seuils du CV ===
def get_color(cv):
    if cv < 15:
        return 'green'
    elif cv < 30:
        return 'orange'
    else:
        return 'red'

merged['couleur_cv'] = merged['cv'].apply(get_color)

# === Visualisation ===
postes = merged['poste'].unique()
fig, axes = plt.subplots(1, len(postes), figsize=(6 * len(postes), 5), sharey=True)

if len(postes) == 1:
    axes = [axes]

for ax, poste in zip(axes, postes):
    data_poste = merged[merged['poste'] == poste]

    sns.barplot(
        data=data_poste,
        x='tranche_experience_totale',
        y='cv',
        hue='couleur_cv',
        palette={'green': 'green', 'orange': 'orange', 'red': 'red'},
        dodge=False,
        legend=False,
        ax=ax,
        edgecolor='black'
    )

    # Ajouter les valeurs sur les barres
    for i, val in enumerate(data_poste['cv']):
        ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_title(f'Poste : {poste}', fontweight='bold', fontsize=13)
    ax.set_xlabel('Tranche d’expérience totale', fontweight='bold')
    ax.set_ylabel('CV Salaire (%)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=15, color='orange', linestyle='--', alpha=0.6, linewidth=2)
    ax.axhline(y=30, color='red', linestyle='--', alpha=0.6, linewidth=2)

plt.suptitle('Disparité salariale par poste et expérience totale', fontweight='bold', fontsize=16, y=1.03)
plt.tight_layout()
plt.savefig('images/disparite_cv_poste_experience_totale.png', dpi=300, bbox_inches='tight')
plt.show()

# Nettoyage
fc.drop(['tranche_experience_totale', 'a_quitte_l_entreprise_num'], axis=1, inplace=True)

######################

# Créer des tranches
bins = [18, 30, 40, 50, 60]
labels = ['18-30 ans', '30-40 ans', '40-50 ans', '50-60 ans']
fc['tranche_age'] = pd.cut(fc['age'], bins=bins, labels=labels)

# Convertir 'Oui'/'Non' en 1/0 si nécessaire
if fc['a_quitte_l_entreprise'].dtype == 'object':
    fc['a_quitte_l_entreprise_num'] = fc['a_quitte_l_entreprise'].map({'Oui': 1, 'Non': 0})
else:
    fc['a_quitte_l_entreprise_num'] = fc['a_quitte_l_entreprise']

# Calculer le taux de départ par combinaison de variables
taux_depart = fc.groupby(['tranche_age', 'genre', 'statut_marital'], observed=True).agg({
    'a_quitte_l_entreprise_num': ['sum', 'count']
})
taux_depart.columns = ['departs', 'total']
taux_depart['taux'] = (taux_depart['departs'] / taux_depart['total'] * 100).round(1)
taux_depart = taux_depart.reset_index()

# Créer un pivot pour la heatmap (exemple: Age x Genre, avec statut_marital en facettes)
statuts = sorted(fc['statut_marital'].unique())
fig, axes = plt.subplots(1, len(statuts), figsize=(18, 5))

# Gérer le cas d'un seul statut
if len(statuts) == 1:
    axes = [axes]

for i, statut in enumerate(statuts):
    data_statut = taux_depart[taux_depart['statut_marital'] == statut]
    pivot = data_statut.pivot(index='tranche_age', columns='genre', values='taux')

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=axes[i], cbar_kws={'label': 'Taux de départ (%)'},
                vmin=0, vmax=taux_depart['taux'].max())

    axes[i].set_title(f'Statut marital: {statut}', fontweight='bold', fontsize=12)
    axes[i].set_xlabel('Genre', fontweight='bold')
    axes[i].set_ylabel('Tranche d\'âge', fontweight='bold')

plt.suptitle('Taux de départ par statut', fontweight='bold', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Nettoyer
fc.drop(['a_quitte_l_entreprise_num'], axis=1, inplace=True)


subset = fc[
    (fc['genre'] == 'M') &
    (fc['tranche_age'] == '50-60 ans') &
    (fc['statut_marital'] == 'Celibataire')
]
subset.head()
print(f"{len(subset)} employés trouvés.")
subset['a_quitte_l_entreprise'].value_counts(normalize=True) * 100

fc.drop((['tranche_age']), axis=1, inplace=True)


#############################



# Indice composite normalisé - créer un score global (0 à 100) qui combine tout
scaler = MinMaxScaler()

fc[['age_norm', 'revenu_mensuel_norm', 'annees_experience_totale_norm', 'annees_dans_l_entreprise_norm']] = scaler.fit_transform(
    fc[['age', 'revenu_mensuel', 'annees_experience_totale', 'annees_dans_l_entreprise']]
)

# pondérations à ajuster selon ton objectif
fc['score_salaire'] = ((
    0.1 * fc['age_norm'] +
    0.4 * fc['revenu_mensuel_norm'] +
    0.3 * fc['annees_experience_totale_norm'] +
    0.2 * fc['annees_dans_l_entreprise_norm']
) * 100).round(2)

fc.drop(['age_norm', 'revenu_mensuel_norm','annees_experience_totale_norm', 'annees_dans_l_entreprise_norm'], axis=1, inplace=True)