import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Bidirectional, Input, Concatenate, LayerNormalization,
                                   GRU, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_curve, auc, confusion_matrix, classification_report,
                           precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle
import os
import time
from datetime import datetime
import shap
from scipy import stats
from itertools import combinations
import matplotlib.ticker as ticker

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('logs', exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

parkinson_colors = ['#3A539B', '#F4D03F', '#E74C3C']
parkinson_cmap = LinearSegmentedColormap.from_list('parkinson_cmap', parkinson_colors)
parkinson_palette = parkinson_colors  # Separate palette for barplots

def create_advanced_features(dataset):
    df = dataset.copy()
    df['voice_instability'] = df['MDVP:Jitter(%)'] * df['MDVP:Shimmer']
    df['frequency_variations'] = df['MDVP:Fhi(Hz)'] / df['MDVP:Flo(Hz)']
    df['amplitude_perturbation'] = (df['Shimmer:APQ3'] + df['Shimmer:APQ5']) / 2
    df['combined_jitter'] = df[['MDVP:Jitter(%)', 'MDVP:RAP', 'MDVP:PPQ']].mean(axis=1)
    df['combined_shimmer'] = df[['MDVP:Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ5']].mean(axis=1)
    df['nonlinear_complexity'] = df['RPDE'] * df['D2']
    df['signal_stability'] = df['HNR'] / df['NHR']
    df['shimmer_to_jitter_ratio'] = df['combined_shimmer'] / (df['combined_jitter'] + 1e-6)
    df['perturbation_index'] = (df['MDVP:RAP'] + df['Shimmer:APQ3']) / 2
    df['vocal_fold_stability'] = df['HNR'] / (df['MDVP:Jitter(%)'] + 1e-6)
    df['frequency_range'] = df['MDVP:Fhi(Hz)'] - df['MDVP:Flo(Hz)']
    df['frequency_stability'] = df['MDVP:Fo(Hz)'] / (df['frequency_range'] + 1e-6)
    df['combined_perturbation'] = df['combined_jitter'] * df['combined_shimmer']
    df['noise_complexity'] = df['NHR'] * df['RPDE']
    df['jitter_shimmer_product'] = df['MDVP:Jitter(%)'] * df['MDVP:Shimmer']
    df['pitch_range_ratio'] = df['MDVP:Fhi(Hz)'] / (df['MDVP:Flo(Hz)'] + 1e-6)
    df['harmonic_to_noise_stability'] = df['HNR'] / (df['NHR'] + 1e-6)
    df['complexity_index'] = (df['RPDE'] + df['DFA'] + df['D2'] + df['PPE']) / 4
    return df 

def visualize_dataset_composition(y):
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y.flatten()).value_counts()
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=parkinson_colors[:2])
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 1, str(v), ha='center')
    plt.title('Class Distribution in Parkinson\'s Disease Dataset', fontsize=15)
    plt.xlabel('Class (0: Healthy, 1: Parkinson\'s)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks([0, 1], ['Healthy', 'Parkinson\'s'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_correlation(dataset):
    plt.figure(figsize=(20, 16))
    original_features = dataset.drop('status', axis=1).columns.tolist()[:22]
    corr_matrix = dataset[original_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap=parkinson_cmap, annot=False, 
                center=0, square=True, linewidths=.5)
    plt.title('Correlation Matrix of Original Features', fontsize=16)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('visualizations/original_features_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(24, 20))
    engineered_features = dataset.drop('status', axis=1).columns.tolist()[22:]
    if engineered_features:
        corr_matrix = dataset[engineered_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap=parkinson_cmap, annot=False, 
                    center=0, square=True, linewidths=.5)
        plt.title('Correlation Matrix of Engineered Features', fontsize=16)
        plt.xticks(fontsize=10, rotation=45, ha='right')
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('visualizations/engineered_features_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.figure(figsize=(12, 10))
    target_corr = dataset.corr()['status'].drop('status').sort_values(ascending=False)
    top_corr = pd.concat([target_corr.head(15), target_corr.tail(15)])
    sns.barplot(x=top_corr.values, y=top_corr.index, palette=parkinson_palette)
    plt.title('Features with Highest Correlation to Parkinson\'s Disease', fontsize=15)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/feature_target_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_distributions(dataset):
    target_corr = dataset.corr()['status'].drop('status').abs().sort_values(ascending=False)
    top_features = target_corr.head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        sns.histplot(data=dataset, x=feature, hue='status', 
                     bins=15, kde=True, ax=ax, palette=parkinson_colors[:2],
                     hue_order=[0, 1], element="step")
        ax.set_title(f'Distribution of {feature}', fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(['Healthy', 'Parkinson\'s'])
        
    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        sns.violinplot(data=dataset, x='status', y=feature, ax=ax, palette=parkinson_colors[:2])
        ax.set_title(f'Violin Plot of {feature}', fontsize=12)
        ax.set_xlabel('Class (0: Healthy, 1: Parkinson\'s)', fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Healthy', 'Parkinson\'s'])
        
    plt.tight_layout()
    plt.savefig('visualizations/feature_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def build_attention_lstm_block(inputs, units):
    lstm = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    lstm = LayerNormalization()(lstm)
    attention = MultiHeadAttention(num_heads=4, key_dim=units)(lstm, lstm)
    attention = Dropout(0.2)(attention)
    x = LayerNormalization()(attention + lstm)
    lstm2 = Bidirectional(LSTM(units//2, return_sequences=True))(x)
    lstm2 = LayerNormalization()(lstm2)
    return Concatenate()([x, lstm2])

def build_balanced_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_block1 = build_attention_lstm_block(inputs, 128)
    lstm_block2 = build_attention_lstm_block(inputs, 64)
    gru_path = Bidirectional(GRU(64, return_sequences=True))(inputs)
    gru_path = LayerNormalization()(gru_path)
    gru_attention = MultiHeadAttention(num_heads=2, key_dim=32)(gru_path, gru_path)
    combined = Concatenate()([lstm_block1, lstm_block2, gru_attention])
    x = Bidirectional(LSTM(32))(combined)
    x = LayerNormalization()(x)
    x = Dense(48, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(24, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_simple_bilstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_gru_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Bidirectional(GRU(32))(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    attention = Dropout(0.1)(attention)
    attention = LayerNormalization()(attention + inputs)
    attention2 = MultiHeadAttention(num_heads=4, key_dim=32)(attention, attention)
    attention2 = Dropout(0.1)(attention2)
    attention2 = LayerNormalization()(attention2 + attention)
    x = Dense(64, activation='relu')(attention2)
    x = Dropout(0.2)(x)
    x = Dense(input_shape[-1])(x)
    x = LayerNormalization()(x + attention2)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model 

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = tf.keras.backend.epsilon()
        pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
        pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
        return -tf.keras.backend.sum(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1)) - \
               tf.keras.backend.sum((1 - alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(1. - pt_0))
    return focal_loss_fixed

def plot_learning_curves(history, fold, model_name):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy (Fold {fold+1})', fontsize=12)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss (Fold {fold+1})', fontsize=12)
    plt.ylabel('Loss', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title(f'Model F1 Score (Fold {fold+1})', fontsize=12)
    plt.ylabel('F1 Score', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_fold{fold+1}_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(all_true_labels, all_predictions, model_name, fold=None):
    y_true = np.array([t[1] for t in all_true_labels])
    y_pred = np.array([p[1] for p in all_predictions])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color=parkinson_colors[0], lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', markersize=8, 
             color=parkinson_colors[2], label=f'Optimal threshold: {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    title = f'Receiver Operating Characteristic'
    if fold is not None:
        title += f' (Fold {fold+1})'
    
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(linestyle='--', alpha=0.7)
    
    filename = f'visualizations/{model_name}'
    if fold is not None:
        filename += f'_fold{fold+1}'
    filename += '_roc_curve.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_threshold, roc_auc

def plot_precision_recall_curve(all_true_labels, all_predictions, model_name, fold=None):
    y_true = np.array([t[1] for t in all_true_labels])
    y_pred = np.array([p[1] for p in all_predictions])
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-7)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color=parkinson_colors[0], lw=2, 
             label=f'Precision-Recall curve (AP = {pr_auc:.4f})')
    
    plt.plot(recall[best_f1_idx], precision[best_f1_idx], 'o', markersize=8, 
             color=parkinson_colors[2], 
             label=f'Best F1 threshold: {best_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    
    title = 'Precision-Recall Curve'
    if fold is not None:
        title += f' (Fold {fold+1})'
    
    plt.title(title, fontsize=15)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(linestyle='--', alpha=0.7)
    
    filename = f'visualizations/{model_name}'
    if fold is not None:
        filename += f'_fold{fold+1}'
    filename += '_pr_curve.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_threshold, pr_auc

def plot_confusion_matrix(y_true, y_pred, model_name, fold=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=parkinson_cmap, cbar=False)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    
    title = 'Confusion Matrix'
    if fold is not None:
        title += f' (Fold {fold+1})'
    
    plt.title(title, fontsize=15)
    plt.xticks([0.5, 1.5], ['Healthy', 'Parkinson\'s'])
    plt.yticks([0.5, 1.5], ['Healthy', 'Parkinson\'s'])
    
    filename = f'visualizations/{model_name}'
    if fold is not None:
        filename += f'_fold{fold+1}'
    filename += '_confusion_matrix.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def load_and_preprocess():
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE", "status"
    ]
    
    dataset = pd.read_csv("data/data.csv", names=features)
    
    with open('reports/dataset_info.txt', 'w') as f:
        f.write(f"Oxford Parkinson's Disease Dataset Information\n")
        f.write(f"==========================================\n\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Healthy subjects: {len(dataset[dataset['status'] == 0])}\n")
        f.write(f"Parkinson's patients: {len(dataset[dataset['status'] == 1])}\n\n")
        f.write(f"Features: {len(features) - 1}\n\n")
        f.write("Feature Description:\n")
        f.write("------------------\n")
        for feature in features[:-1]:
            f.write(f"- {feature}\n")
        f.write("\nBasic Statistics:\n")
        f.write("----------------\n")
        f.write(str(dataset.describe()))
    
    healthy = dataset[dataset['status'] == 0].drop('status', axis=1)
    parkinsons = dataset[dataset['status'] == 1].drop('status', axis=1)
    
    with open('reports/class_statistics.txt', 'w') as f:
        f.write(f"Statistics by Class\n")
        f.write(f"==================\n\n")
        f.write(f"Healthy Subjects Statistics:\n")
        f.write(f"--------------------------\n")
        f.write(str(healthy.describe()))
        f.write(f"\n\nParkinson's Patients Statistics:\n")
        f.write(f"-------------------------------\n")
        f.write(str(parkinsons.describe()))
    
    dataset = create_advanced_features(dataset)
    
    with open('reports/advanced_features_info.txt', 'w') as f:
        f.write(f"Advanced Features Information\n")
        f.write(f"===========================\n\n")
        original_features = features[:-1]
        advanced_features = [col for col in dataset.columns if col not in features and col != 'status']
        f.write(f"Original features: {len(original_features)}\n")
        f.write(f"Advanced features: {len(advanced_features)}\n")
        f.write(f"Total features: {len(original_features) + len(advanced_features)}\n\n")
        f.write("Advanced Features Description:\n")
        f.write("---------------------------\n")
        for feature in advanced_features:
            f.write(f"- {feature}\n")
        f.write("\nAdvanced Features Statistics:\n")
        f.write("---------------------------\n")
        f.write(str(dataset[advanced_features].describe()))
    
    visualize_dataset_composition(dataset['status'].values)
    visualize_feature_correlation(dataset)
    visualize_feature_distributions(dataset)
    
    y = dataset['status'].values
    X = dataset.drop('status', axis=1).values
    feature_names = dataset.drop('status', axis=1).columns.tolist()
    
    return X, y, feature_names 

def train_and_evaluate_model(X, y, feature_names, model_builder, model_name, use_synthetic=False, n_synthetic=100):
    y = y.reshape(-1, 1)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    metrics_list = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auc': []
    }
    
    all_predictions = []
    all_true_labels = []
    models = []
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs(f'models/{model_name}', exist_ok=True)
    os.makedirs(f'visualizations/{model_name}', exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if use_synthetic:
            print(f"Generating {n_synthetic} synthetic samples using GAN...")
            synthetic_X, synthetic_y = generate_synthetic_data_with_gan(X_train, y_train, n_synthetic=n_synthetic)
            X_train = np.vstack([X_train, synthetic_X])
            y_train = np.vstack([y_train, synthetic_y])
        
        X_train_reshaped = X_train.reshape(-1, 1, X_train.shape[1])
        X_val_reshaped = X_val.reshape(-1, 1, X_val.shape[1])
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.ravel()), y=y_train.ravel())
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        model = model_builder((1, X_train.shape[1]))
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.F1Score(
                    name='f1_score',
                    threshold=0.5,
                    average='micro'
                )
            ]
        )
        
        callbacks = [
            ModelCheckpoint(
                filepath=f'models/{model_name}/fold{fold+1}_best_model.h5',
                monitor='val_f1_score',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_f1_score',
                patience=20,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'logs/{model_name}/fold{fold+1}',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=200,
            batch_size=16,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        plot_learning_curves(history, fold, model_name)
        
        eval_results = model.evaluate(X_val_reshaped, y_val, verbose=0)
        metrics_list['accuracy'].append(eval_results[1])
        metrics_list['auc'].append(eval_results[2])
        metrics_list['precision'].append(eval_results[3])
        metrics_list['recall'].append(eval_results[4])
        metrics_list['f1_score'].append(eval_results[5])
        
        models.append(model)
        
        fold_preds = model.predict(X_val_reshaped).flatten()
        
        all_predictions.extend(list(zip(val_idx, fold_preds)))
        all_true_labels.extend(list(zip(val_idx, y_val.flatten())))
        
        optimal_threshold, _ = plot_roc_curves(
            [(i, y_val[j, 0]) for i, j in enumerate(range(len(y_val)))],
            [(i, fold_preds[j]) for i, j in enumerate(range(len(fold_preds)))],
            model_name, fold
        )
        
        best_f1_threshold, _ = plot_precision_recall_curve(
            [(i, y_val[j, 0]) for i, j in enumerate(range(len(y_val)))],
            [(i, fold_preds[j]) for i, j in enumerate(range(len(fold_preds)))],
            model_name, fold
        )
        
        y_pred_binary = (fold_preds > optimal_threshold).astype(int)
        plot_confusion_matrix(y_val.flatten(), y_pred_binary, model_name, fold)
        
        if fold == 0:
            feature_importance = visualize_feature_importance(model, X_scaled, feature_names)
            feature_importance.to_csv(f'results/{model_name}_feature_importance.csv', index=False)
    
    all_predictions.sort(key=lambda x: x[0])
    all_true_labels.sort(key=lambda x: x[0])
    y_pred = np.array([p[1] for p in all_predictions])
    y_true = np.array([t[1] for t in all_true_labels])
    
    thresholds = np.linspace(0.3, 0.7, 40)
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(float)
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()
        precision_metric.update_state(y_true, y_pred_binary)
        recall_metric.update_state(y_true, y_pred_binary)
        precision = precision_metric.result().numpy()
        recall = recall_metric.result().numpy()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    plot_roc_curves(all_true_labels, all_predictions, model_name)
    plot_precision_recall_curve(all_true_labels, all_predictions, model_name)
    
    y_pred_binary = (y_pred > best_threshold).astype(int)
    plot_confusion_matrix(y_true.flatten(), y_pred_binary, model_name)
    
    print(f"\nOptimal threshold found: {best_threshold:.3f}")
    print(f"At this threshold - Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    print(f"\nAverage metrics across folds:")
    print(f"Accuracy: {np.mean(metrics_list['accuracy']):.4f} ± {np.std(metrics_list['accuracy']):.4f}")
    print(f"AUC: {np.mean(metrics_list['auc']):.4f} ± {np.std(metrics_list['auc']):.4f}")
    print(f"Precision: {np.mean(metrics_list['precision']):.4f} ± {np.std(metrics_list['precision']):.4f}")
    print(f"Recall: {np.mean(metrics_list['recall']):.4f} ± {np.std(metrics_list['recall']):.4f}")
    print(f"F1-Score: {np.mean(metrics_list['f1_score']):.4f} ± {np.std(metrics_list['f1_score']):.4f}")
    
    best_model_idx = np.argmax(metrics_list['f1_score'])
    best_model_fold = best_model_idx + 1
    
    best_model_state = {
        'model_weights': models[best_model_idx].get_weights(),
        'metrics': {k: v[best_model_idx] for k, v in metrics_list.items()},
        'best_threshold': best_threshold,
        'feature_names': feature_names,
        'scaler': scaler
    }
    
    with open(f'models/{model_name}/best_model_state.pkl', 'wb') as f:
        pickle.dump(best_model_state, f)
    
    return models[best_model_idx], best_model_state 

if __name__ == "__main__":
    X, y, feature_names = load_and_preprocess()
    
    model_builders = {
        'balanced_model': build_balanced_model,
        'simple_bilstm': build_simple_bilstm_model,
        'gru_model': build_gru_model,
        'transformer_model': build_transformer_model
    }
    
    results = {}
    
    for model_name, model_builder in model_builders.items():
        model, model_state = train_and_evaluate_model(
            X, y, feature_names, model_builder, model_name,
            use_synthetic=True, n_synthetic=100
        )
        results[model_name] = model_state['metrics']
    
    perform_statistical_analysis(results) 