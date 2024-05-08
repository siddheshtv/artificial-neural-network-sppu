from keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = models.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss='binary_crossentropy'
)

model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Accuracy: {test_accuracy}")