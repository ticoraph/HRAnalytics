import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np


def save_results(results_df, detailed_reports, output_prefix='classification_results',
                 models_data=None):
    """
    Sauvegarde les résultats dans des fichiers CSV et génère des graphiques.

    Args:
        results_df: DataFrame avec tous les résultats
        detailed_reports: Liste des rapports détaillés
        output_prefix: Préfixe pour les fichiers de sortie
        models_data: Liste de dict avec {
            'model': nom du modèle,
            'y_true': vraies étiquettes,
            'y_pred_proba': probabilités prédites (si disponible),
            'y_pred': prédictions
        }
    """
    # 1. Fichier principal avec toutes les métriques
    results_df.to_csv(f'exports/{output_prefix}_summary.csv', index=False)
    print(f"\n✓ Résumé sauvegardé: {output_prefix}_summary.csv")

    # 2. Fichier avec les rapports détaillés de classification
    detailed_data = []
    for item in detailed_reports:
        model = item['model']
        report = item['report']

        # Extraction des métriques par classe
        for class_label, metrics in report.items():
            if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                detailed_data.append({
                    'model': model,
                    'class': class_label,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                })

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(f'exports/{output_prefix}_by_class.csv', index=False)
    print(f"✓ Résultats par classe sauvegardés: {output_prefix}_by_class.csv")

    # 3. Fichier avec les matrices de confusion
    cm_data = []
    for item in detailed_reports:
        cm_data.append({
            'model': item['model'],
            'confusion_matrix': str(item['confusion_matrix'].tolist())
        })

    cm_df = pd.DataFrame(cm_data)
    cm_df.to_csv(f'exports/{output_prefix}_confusion_matrices.csv', index=False)
    print(f"✓ Matrices de confusion sauvegardées: {output_prefix}_confusion_matrices.csv")

    # 4. Fichier spécifique pour l'analyse d'overfitting
    overfitting_data = results_df[['model', 'method', 'train_accuracy', 'test_accuracy',
                                   'accuracy_gap', 'train_f1_macro', 'test_f1_macro',
                                   'f1_gap', 'overfitting']].copy()
    overfitting_data.to_csv(f'exports/{output_prefix}_overfitting_analysis.csv', index=False)
    print(f"✓ Analyse d'overfitting sauvegardée: {output_prefix}_overfitting_analysis.csv")

    # 5. Génération des graphiques ROC et Précision-Rappel
    if models_data is not None:
        print("\n📈 Génération des graphiques...")
        _plot_roc_curves(models_data, output_prefix)
        _plot_precision_recall_curves(models_data, output_prefix)

    print("\n" + "=" * 70)
    print("TOUS LES RÉSULTATS ONT ÉTÉ SAUVEGARDÉS")
    print("=" * 70)

    # Affichage d'un résumé de l'overfitting
    print("\n📊 RÉSUMÉ DE L'OVERFITTING:")
    print("-" * 70)
    for _, row in overfitting_data.iterrows():
        status = "⚠️  OVERFITTING DÉTECTÉ" if row['overfitting'] == 'OUI' else "✓  Pas d'overfitting"
        print(f"{row['model']:25s} ({row['method']:17s}): {status}")
        print(f"  → Écart accuracy: {row['accuracy_gap']:+.4f} | Écart F1: {row['f1_gap']:+.4f}")
    print("-" * 70)


def _plot_roc_curves(models_data, output_prefix):
    """
    Génère et sauvegarde les courbes ROC pour tous les modèles.
    """
    # Déterminer le nombre de classes
    y_true_sample = models_data[0]['y_true']
    classes = np.unique(y_true_sample)
    n_classes = len(classes)

    if n_classes == 2:
        # Classification binaire
        _plot_binary_roc_curves(models_data, output_prefix)
    else:
        # Classification multi-classes
        _plot_multiclass_roc_curves(models_data, output_prefix, classes)


def _plot_binary_roc_curves(models_data, output_prefix):
    """
    Courbes ROC pour classification binaire.
    """
    plt.figure(figsize=(10, 8))

    for data in models_data:
        model_name = data['model']
        y_true = data['y_true']

        # Vérifier si les probabilités sont disponibles
        if 'y_pred_proba' in data and data['y_pred_proba'] is not None:
            y_score = data['y_pred_proba']
            # Si format (n_samples, n_classes), prendre la colonne de la classe positive
            if len(y_score.shape) > 1:
                y_score = y_score[:, 1]
        else:
            print(f"⚠️  Pas de probabilités pour {model_name}, utilisation des prédictions")
            y_score = data['y_pred']

        # Calculer la courbe ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

    # Ligne diagonale (classificateur aléatoire)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Aléatoire (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs', fontsize=12)
    plt.title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f'exports/{output_prefix}_roc_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Courbes ROC sauvegardées: {output_prefix}_roc_curves.png")


def _plot_multiclass_roc_curves(models_data, output_prefix, classes):
    """
    Courbes ROC pour classification multi-classes (One-vs-Rest).
    """
    n_classes = len(classes)
    n_models = len(models_data)

    # Créer une figure avec sous-graphiques
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for idx, data in enumerate(models_data):
        ax = axes[idx]
        model_name = data['model']
        y_true = data['y_true']

        # Binariser les labels
        y_true_bin = label_binarize(y_true, classes=classes)

        if 'y_pred_proba' in data and data['y_pred_proba'] is not None:
            y_score = data['y_pred_proba']
        else:
            print(f"⚠️  Pas de probabilités pour {model_name}, graphique ROC limité")
            continue

        # Calculer ROC pour chaque classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], lw=2,
                    label=f'Classe {classes[i]} (AUC = {roc_auc[i]:.3f})')

        # Calculer micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, lw=3, linestyle='--',
                label=f'Micro-avg (AUC = {roc_auc_micro:.3f})', color='navy')

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux de Faux Positifs', fontsize=11)
        ax.set_ylabel('Taux de Vrais Positifs', fontsize=11)
        ax.set_title(f'ROC - {model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'exports/{output_prefix}_roc_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Courbes ROC sauvegardées: {output_prefix}_roc_curves.png")


def _plot_precision_recall_curves(models_data, output_prefix):
    """
    Génère et sauvegarde les courbes Précision-Rappel pour tous les modèles.
    """
    y_true_sample = models_data[0]['y_true']
    classes = np.unique(y_true_sample)
    n_classes = len(classes)

    if n_classes == 2:
        _plot_binary_pr_curves(models_data, output_prefix)
    else:
        _plot_multiclass_pr_curves(models_data, output_prefix, classes)


def _plot_binary_pr_curves(models_data, output_prefix):
    """
    Courbes Précision-Rappel pour classification binaire.
    """
    plt.figure(figsize=(10, 8))

    for data in models_data:
        model_name = data['model']
        y_true = data['y_true']

        if 'y_pred_proba' in data and data['y_pred_proba'] is not None:
            y_score = data['y_pred_proba']
            if len(y_score.shape) > 1:
                y_score = y_score[:, 1]
        else:
            print(f"⚠️  Pas de probabilités pour {model_name}, utilisation des prédictions")
            y_score = data['y_pred']

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)

        plt.plot(recall, precision, lw=2,
                 label=f'{model_name} (AP = {avg_precision:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Rappel', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.title('Courbes Précision-Rappel - Comparaison des Modèles',
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f'exports/{output_prefix}_precision_recall_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Courbes Précision-Rappel sauvegardées: {output_prefix}_precision_recall_curves.png")


def _plot_multiclass_pr_curves(models_data, output_prefix, classes):
    """
    Courbes Précision-Rappel pour classification multi-classes.
    """
    n_classes = len(classes)
    n_models = len(models_data)

    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for idx, data in enumerate(models_data):
        ax = axes[idx]
        model_name = data['model']
        y_true = data['y_true']

        y_true_bin = label_binarize(y_true, classes=classes)

        if 'y_pred_proba' in data and data['y_pred_proba'] is not None:
            y_score = data['y_pred_proba']
        else:
            print(f"⚠️  Pas de probabilités pour {model_name}, graphique PR limité")
            continue

        # Calculer PR pour chaque classe
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i],
                                                          y_score[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, lw=2,
                    label=f'Classe {classes[i]} (AP = {avg_precision:.3f})')

        # Micro-average
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_true_bin.ravel(), y_score.ravel())
        avg_precision_micro = average_precision_score(y_true_bin, y_score,
                                                      average='micro')
        ax.plot(recall_micro, precision_micro, lw=3, linestyle='--',
                label=f'Micro-avg (AP = {avg_precision_micro:.3f})', color='navy')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Rappel', fontsize=11)
        ax.set_ylabel('Précision', fontsize=11)
        ax.set_title(f'Précision-Rappel - {model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'exports/{output_prefix}_precision_recall_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Courbes Précision-Rappel sauvegardées: {output_prefix}_precision_recall_curves.png")