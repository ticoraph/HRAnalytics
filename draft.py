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