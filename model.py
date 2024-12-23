import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Bidirectional, Input, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
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
    
    return df
def build_refined_model(input_shape):
    """
    Simplified and more robust model architecture with better regularization
    """
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM path with reduced complexity
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.5)(x)
    
    # Second LSTM layer
    x = Bidirectional(LSTM(32))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Dense layers with strong regularization
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Final layer with sigmoid activation
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
def build_attention_lstm_block(inputs, units):
    x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    attention = MultiHeadAttention(num_heads=4, key_dim=units)(x, x)
    x = LayerNormalization()(attention + x)
    return x

def build_ensemble_model(input_shape):
    inputs = Input(shape=input_shape)
    
    lstm_block1 = build_attention_lstm_block(inputs, 256)
    lstm_block2 = build_attention_lstm_block(lstm_block1, 128)
    lstm_block3 = build_attention_lstm_block(lstm_block2, 64)
    
    gru_path = Bidirectional(GRU(128, return_sequences=True))(inputs)
    gru_path = LayerNormalization()(gru_path)
    
    combined = Concatenate()([lstm_block3, gru_path])
    
    x = Bidirectional(LSTM(32))(combined)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(32, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

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
    # Load and preprocess data (assuming same data loading function as before)
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE", "status"
    ]
    
    dataset = pd.read_csv("data/data.csv", names=features)
    
    # Split features and target
    y = dataset['status'].values
    X = dataset.drop('status', axis=1).values
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = []
    scores = []
    predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Reshape for LSTM
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        
        # Build and compile model
        model = build_refined_model((1, X_train_scaled.shape[1]))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), 
                    tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Enhanced callbacks
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, 
                             min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_loss', patience=20, 
                         restore_best_weights=True, verbose=1)
        ]
        
        # Balanced class weights
        class_weights = {0: 1.0, 
                        1: float(len(y_train[y_train == 0])) / len(y_train[y_train == 1])}
        
        # Train with larger batch size and more epochs
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate and store results
        eval_results = model.evaluate(X_val_reshaped, y_val, verbose=0)
        scores.append(eval_results[1])
        models.append(model)
        
        # Store predictions for this fold
        fold_preds = model.predict(X_val_reshaped)
        predictions.extend(list(zip(val_idx, fold_preds)))
        
        print(f"\nFold {fold + 1} Results:")
        print(f"Accuracy: {eval_results[1]:.4f}")
        print(f"AUC: {eval_results[2]:.4f}")
        print(f"Precision: {eval_results[3]:.4f}")
        print(f"Recall: {eval_results[4]:.4f}")
        
        # Save model state
        model_state = {
            'model_weights': model.get_weights(),
            'model_config': model.get_config(),
            'scaler': scaler,
            'metrics': {
                'accuracy': eval_results[1],
                'auc': eval_results[2],
                'precision': eval_results[3],
                'recall': eval_results[4]
            },
            'threshold': 0.5  # Default threshold
        }
        
        # Save fold model
        with open(f'models/parkinsons_model_fold_{fold+1}.pkl', 'wb') as f:
            pickle.dump(model_state, f)
    
    # Calculate optimal threshold using all predictions
    predictions.sort(key=lambda x: x[0])  # Sort by original index
    all_preds = np.array([p[1] for p in predictions])
    
    # Find optimal threshold using validation results
    thresholds = np.linspace(0.3, 0.7, 40)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        pred_labels = (all_preds > threshold).astype(int)
        f1 = tf.keras.metrics.F1Score()(y, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nOptimal threshold found: {best_threshold:.3f}")
    print(f"Average Accuracy across folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Save best model with optimal threshold
    best_model_idx = np.argmax(scores)
    best_model_state = {
        'model_weights': models[best_model_idx].get_weights(),
        'model_config': models[best_model_idx].get_config(),
        'scaler': scaler,
        'metrics': {
            'accuracy': scores[best_model_idx]
        },
        'threshold': best_threshold
    }
    
    with open('models/best_parkinsons_model.pkl', 'wb') as f:
        pickle.dump(best_model_state, f)
    
    return best_model_state


if __name__ == "__main__":
    best_model_state = train_and_pickle_model()
    print(f"Best model accuracy: {best_model_state['metrics']['accuracy']:.4f}")
    print(f"Using threshold: {best_model_state['threshold']:.3f}")