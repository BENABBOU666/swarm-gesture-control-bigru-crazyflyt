"""
BigRU Gesture Recognition Training Pipeline
Reproducible version for scientific publication
BENABBOU F.B.E
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, Flatten
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. REPRODUCIBILITY
# =========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 2. CONFIGURATION
# =========================================================
DATA_PATH = "My_RNN_dataset"
SEQUENCE_LENGTH = 20

# ACTION SET
actions = np.array([
    'takeoff','stop','one','two','three','four','five','six','seven','eight','nine',
    'A','B','C','I','H','L','O','R','U','up','down','left','right','go','back'
])

label_map = {label: idx for idx, label in enumerate(actions)}

# =========================================================
# 3. LOAD DATASET
# =========================================================
def load_dataset():
    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            continue

        for sequence in os.listdir(action_path):
            sequence_path = os.path.join(action_path, sequence)
            window = []

            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                window.append(np.load(frame_path))

            sequences.append(window)
            labels.append(label_map[action])

    return np.array(sequences), np.array(labels)

X, y = load_dataset()

print("Dataset shape:", X.shape)

y = to_categorical(y)

# =========================================================
# 4. TRAIN / VAL / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# =========================================================
# 5. MODEL (BIGRU)
# =========================================================
model = Sequential([
    Bidirectional(GRU(128, return_sequences=True, activation='tanh'),
                  input_shape=(SEQUENCE_LENGTH, 126)),
    Dropout(0.3),

    Bidirectional(GRU(256, return_sequences=True, activation='tanh')),
    Dropout(0.3),

    Bidirectional(GRU(128, return_sequences=False, activation='tanh')),
    Flatten(),

    Dense(128, activation='relu'),
    Dense(64, activation='relu'),

    Dense(len(actions), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy', Precision(), Recall()]
)

model.summary()

# =========================================================
# 6. TRAINING
# =========================================================
early_stopping = EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# =========================================================
# 7. EVALUATION
# =========================================================
test_results = model.evaluate(X_test, y_test)
print("Test Results:", test_results)

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

# =========================================================
# 8. VISUALIZATION
# =========================================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=actions, yticklabels=actions)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# =========================================================
# 9. SAVE MODEL
# =========================================================
model.save("bigru.h5")
print("Model saved as bigru.h5")
