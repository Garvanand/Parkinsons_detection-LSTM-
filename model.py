import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Bidirectional, Input, Concatenate, LayerNormalization,
                                   GRU, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os

os.makedirs('models', exist_ok=True)

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
    
    return df

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

def load_and_preprocess():
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE", "status"
    ]
    
    dataset = pd.read_csv("data/data.csv", names=features)
    dataset = create_advanced_features(dataset)
    y = dataset['status'].values
    X = dataset.drop('status', axis=1).values
    return X, y

def train_and_pickle_model():
    X, y = load_and_preprocess()
    y = y.reshape(-1, 1)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    auc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    models = []
    all_predictions = []
    all_true_labels = []
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train_reshaped = X_train.reshape(-1, 1, X_train.shape[1])
        X_val_reshaped = X_val.reshape(-1, 1, X_val.shape[1])
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.ravel()), y=y_train.ravel())
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        model = build_balanced_model((1, X_train.shape[1]))
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
        
        eval_results = model.evaluate(X_val_reshaped, y_val, verbose=0)
        scores.append(eval_results[1])
        auc_scores.append(eval_results[2])
        precision_scores.append(eval_results[3])
        recall_scores.append(eval_results[4])
        f1_scores.append(eval_results[5])
        models.append(model)
        
        fold_preds = model.predict(X_val_reshaped).flatten()
        all_predictions.extend(list(zip(val_idx, fold_preds)))
        all_true_labels.extend(list(zip(val_idx, y_val.flatten())))
    
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
    
    print(f"\nOptimal threshold found: {best_threshold:.3f}")
    print(f"At this threshold - Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    print(f"\nAverage metrics across folds:")
    print(f"Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    best_model_idx = np.argmax(f1_scores)
    best_model_state = {
        'model_weights': models[best_model_idx].get_weights(),
        'model_config': models[best_model_idx].get_config(),
        'scaler': scaler,
        'threshold': best_threshold,
        'metrics': {
            'accuracy': scores[best_model_idx],
            'auc': auc_scores[best_model_idx],
            'precision': precision_scores[best_model_idx],
            'recall': recall_scores[best_model_idx],
            'f1_score': f1_scores[best_model_idx]
        }
    }
    
    with open('models/best_parkinsons_model.pkl', 'wb') as f:
        pickle.dump(best_model_state, f)
    
    return best_model_state

if __name__ == "__main__":
    best_model_state = train_and_pickle_model()
    print(f"Best model metrics:")
    print(f"  Accuracy: {best_model_state['metrics']['accuracy']:.4f}")
    print(f"  AUC: {best_model_state['metrics']['auc']:.4f}")
    print(f"  Precision: {best_model_state['metrics']['precision']:.4f}")
    print(f"  Recall: {best_model_state['metrics']['recall']:.4f}")
    print(f"  F1-Score: {best_model_state['metrics']['f1_score']:.4f}")
    print(f"  Using threshold: {best_model_state['threshold']:.3f}")