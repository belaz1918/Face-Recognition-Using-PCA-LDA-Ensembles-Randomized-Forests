import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=100)
train_pca = pca.fit_transform(train_images)
test_pca = pca.transform(test_images)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf_clf.fit(train_pca, train_labels)

# Predictions
predictions = rf_clf.predict(test_pca)

# Evaluate
accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
