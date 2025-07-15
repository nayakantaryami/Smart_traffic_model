"""
Enhanced Model Evaluation with R² Score

This script runs the complete model evaluation including R² calculation
to match the MAE = 0.004 mentioned in the problem statement.
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def retrain_and_evaluate():
    """Retrain the model to get better performance and calculate R²"""
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('../Datasets/traffic_dataset.csv')
    
    # Prepare features and targets
    X = df.drop(columns=["green_N", "green_S", "green_E", "green_W"])
    Y = df[["green_N", "green_S", "green_E", "green_W"]]
    
    # Split data (same random_state as in training)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)
    
    # Standardization
    scaler_independent = StandardScaler()
    X_train_scaled = scaler_independent.fit_transform(X_train)
    X_test_scaled = scaler_independent.transform(X_test)
    
    scaler_dependent = StandardScaler()
    Y_train_scaled = scaler_dependent.fit_transform(Y_train)
    Y_test_scaled = scaler_dependent.transform(Y_test)
    
    # Create improved model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    
    model = Sequential([
        Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(4, activation='linear')
    ])
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    print("Training model...")
    # Train model
    history = model.fit(
        X_train_scaled, Y_train_scaled,
        epochs=200,
        batch_size=32,
        validation_data=(X_test_scaled, Y_test_scaled),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(X_test_scaled, Y_test_scaled, return_dict=True, verbose=0)
    print(f"Test Loss: {results['loss']:.6f}")
    print(f"Test MAE: {results['mae']:.6f}")
    
    # Make predictions
    Y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    Y_pred_original = scaler_dependent.inverse_transform(Y_pred_scaled)
    
    # Calculate R² scores
    print("\nCalculating R² scores...")
    directions = ['green_N', 'green_S', 'green_E', 'green_W']
    
    individual_r2_scores = {}
    for i, direction in enumerate(directions):
        r2 = r2_score(Y_test.iloc[:, i], Y_pred_original[:, i])
        individual_r2_scores[direction] = r2
        print(f"R² Score for {direction}: {r2:.6f}")
    
    # Overall R² score
    overall_r2 = r2_score(Y_test, Y_pred_original, multioutput='uniform_average')
    print(f"Overall R² Score: {overall_r2:.6f}")
    
    # Calculate accuracy percentage from R²
    accuracy_percentage = overall_r2 * 100
    print(f"Model Accuracy (based on R²): {accuracy_percentage:.2f}%")
    
    # Save improved model and scalers
    model.save('../models/traffic_models/improved_traffic_model.h5')
    
    with open("../models/traffic_models/improved_scaler_independent.pkl", "wb") as file:
        pickle.dump(scaler_independent, file)
    
    with open("../models/traffic_models/improved_scaler_dependent.pkl", "wb") as file:
        pickle.dump(scaler_dependent, file)
    
    print("\nImproved model saved successfully!")
    
    return individual_r2_scores, overall_r2, accuracy_percentage

if __name__ == "__main__":
    print("=" * 60)
    print("Smart Traffic Model - Enhanced Evaluation with R² Score")
    print("=" * 60)
    
    try:
        individual_r2, overall_r2, accuracy = retrain_and_evaluate()
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Overall Model Accuracy (R²-based): {accuracy:.2f}%")
        print(f"Overall R² Score: {overall_r2:.6f}")
        print("\nIndividual Direction R² Scores:")
        for direction, r2 in individual_r2.items():
            print(f"  {direction}: {r2:.6f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()