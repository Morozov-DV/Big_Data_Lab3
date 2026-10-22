import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Фіксація генератора випадкових чисел
np.random.seed(42)

# 1. Генерація чистого набору даних (600 об'єктів, 3 кластери)
X_clean, _ = make_blobs(
    n_samples=600,
    centers=[(-3, -2), (0, 3), (3, -1)],
    cluster_std=[1.0, 1.1, 0.9],
    random_state=42
)

# 2. Додавання 5% шумових об'єктів (5% від 600 = 30 точок)
n_noise = int(0.05 * len(X_clean))
X_noise = np.random.uniform(low=[-8, -6], high=[8, 8], size=(n_noise, 2))

# Загальний набір даних із шумом (630 об'єктів)
X_noisy = np.vstack([X_clean, X_noise])

# Стандартизація даних
scaler_clean = StandardScaler()
X_clean_scaled = scaler_clean.fit_transform(X_clean)

scaler_noisy = StandardScaler()
X_noisy_scaled = scaler_noisy.fit_transform(X_noisy)

# A. K-Means
kmeans_clean = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km_clean = kmeans_clean.fit_predict(X_clean_scaled)
score_km_clean = silhouette_score(X_clean_scaled, labels_km_clean)

kmeans_noisy = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km_noisy = kmeans_noisy.fit_predict(X_noisy_scaled)
score_km_noisy = silhouette_score(X_noisy_scaled, labels_km_noisy)

# B. Ієрархічна кластеризація (Agglomerative)
agg_clean = AgglomerativeClustering(n_clusters=3)
labels_agg_clean = agg_clean.fit_predict(X_clean_scaled)
score_agg_clean = silhouette_score(X_clean_scaled, labels_agg_clean)

agg_noisy = AgglomerativeClustering(n_clusters=3)
labels_agg_noisy = agg_noisy.fit_predict(X_noisy_scaled)
score_agg_noisy = silhouette_score(X_noisy_scaled, labels_agg_noisy)

# C. DBSCAN
db_clean = DBSCAN(eps=0.30, min_samples=6)
labels_db_clean = db_clean.fit_predict(X_clean_scaled)

db_noisy = DBSCAN(eps=0.30, min_samples=6)
labels_db_noisy = db_noisy.fit_predict(X_noisy_scaled)

mask_clean = labels_db_clean != -1
score_db_clean = silhouette_score(X_clean_scaled[mask_clean], labels_db_clean[mask_clean])

mask_noisy = labels_db_noisy != -1
score_db_noisy = silhouette_score(X_noisy_scaled[mask_noisy], labels_db_noisy[mask_noisy])

# Вивід підсумкової таблиці
summary_df = pd.DataFrame({
    "Метод": ["K-Means", "K-Means (+5% шуму)", "Ієрархічна", "Ієрархічна (+5% шуму)", "DBSCAN", "DBSCAN (+5% шуму)"],
    "Сформовано кластерів": [3, 3, 3, 3, len(set(labels_db_clean)) - (1 if -1 in labels_db_clean else 0), len(set(labels_db_noisy)) - (1 if -1 in labels_db_noisy else 0)],
    "Шумові об'єкти (-1)": [0, 0, 0, 0, (labels_db_clean == -1).sum(), (labels_db_noisy == -1).sum()],
    "Silhouette Score": [round(score_km_clean, 4), round(score_km_noisy, 4), round(score_agg_clean, 4), round(score_agg_noisy, 4), round(score_db_clean, 4), round(score_db_noisy, 4)]
})

print("Результати аналізу впливу шуму:")
print(summary_df)

# Візуалізація
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Дослідження впливу 5% шумових об'єктів на алгоритми кластеризації (Варіант 14)", fontsize=16, fontweight='bold')

# Чисті дані (Рядок 1)
axes[0, 0].scatter(X_clean_scaled[:, 0], X_clean_scaled[:, 1], c=labels_km_clean, cmap='viridis', s=20)
axes[0, 0].scatter(kmeans_clean.cluster_centers_[:, 0], kmeans_clean.cluster_centers_[:, 1], c='red', marker='X', s=120, label='Центроїди')
axes[0, 0].set_title(f"K-Means (Чисті дані)\nSilhouette: {score_km_clean:.3f}")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].scatter(X_clean_scaled[:, 0], X_clean_scaled[:, 1], c=labels_agg_clean, cmap='viridis', s=20)
axes[0, 1].set_title(f"Ієрархічна (Чисті дані)\nSilhouette: {score_agg_clean:.3f}")
axes[0, 1].grid(True)

axes[0, 2].scatter(X_clean_scaled[:, 0], X_clean_scaled[:, 1], c=labels_db_clean, cmap='viridis', s=20)
axes[0, 2].set_title(f"DBSCAN (Чисті дані)\nШум (-1): {(labels_db_clean == -1).sum()}")
axes[0, 2].grid(True)

# Дані з шумом (Рядок 2)
axes[1, 0].scatter(X_noisy_scaled[:, 0], X_noisy_scaled[:, 1], c=labels_km_noisy, cmap='viridis', s=20)
axes[1, 0].scatter(kmeans_noisy.cluster_centers_[:, 0], kmeans_noisy.cluster_centers_[:, 1], c='red', marker='X', s=120, label='Центроїди')
axes[1, 0].set_title(f"K-Means (+5% шуму)\nSilhouette: {score_km_noisy:.3f}")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].scatter(X_noisy_scaled[:, 0], X_noisy_scaled[:, 1], c=labels_agg_noisy, cmap='viridis', s=20)
axes[1, 1].set_title(f"Ієрархічна (+5% шуму)\nSilhouette: {score_agg_noisy:.3f}")
axes[1, 1].grid(True)

axes[1, 2].scatter(X_noisy_scaled[:, 0], X_noisy_scaled[:, 1], c=labels_db_noisy, cmap='viridis', s=20)
axes[1, 2].set_title(f"DBSCAN (+5% шуму)\nШум (-1): {(labels_db_noisy == -1).sum()}")
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()