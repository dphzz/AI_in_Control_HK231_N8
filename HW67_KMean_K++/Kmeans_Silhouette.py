import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def Normalization(X):
    return (X-X.min())/(X.max()-X.min())

def kmeans_plusplus_initialization(X, k):
    # Chọn một điểm dữ liệu ngẫu nhiên làm tâm đầu tiên
    centroids = [X[np.random.randint(X.shape[0])]]
    
    # Lặp lại quá trình chọn tâm cho đến khi chúng ta đã chọn đủ k tâm
    while len(centroids) < k:
        # Tính toán khoảng cách từ mỗi điểm dữ liệu đến tâm gần nhất đã chọn
        distances = [min([euclidean_distance(x, c) for c in centroids]) for x in X]
        
        # Chọn điểm dữ liệu mới làm tâm dựa trên xác suất tỉ lệ khoảng cách
        probabilities = distances / np.sum(distances)
        next_centroid = X[np.random.choice(X.shape[0], p=probabilities)]
        
        centroids.append(next_centroid)

    return np.array(centroids)

# Hàm phân chia dữ liệu vào các cụm
def assign_clusters(X, centroids):
    num_clusters = len(centroids)
    clusters = [[] for _ in range(num_clusters)]
    cluster_labels_list=[]
    for x in X:
        # Tìm cụm gần nhất cho điểm dữ liệu x
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        cluster_labels_list.append(cluster_index)
        # Thêm điểm dữ liệu vào cụm tương ứng
        clusters[cluster_index].append(x)
    cluster_labels=np.array(cluster_labels_list)
    return clusters,cluster_labels

# Hàm cập nhật lại các tâm cụm
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) > 0:
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
    return new_centroids

# Hàm tính Silhouette Score cho một tập dữ liệu và một phân cụm
def silhouette_score(X, cluster_labels):
    num_samples = len(X)
    silhouette_scores = []

    for i in range(num_samples):
        a = 0  # Khoảng cách trung bình nội bộ
        b = float('inf')  # Khoảng cách trung bình ngoại bộ

        for j in range(num_samples):
            if i == j:
                continue

            if cluster_labels[i] == cluster_labels[j]:
                a += euclidean_distance(X[i], X[j])
            else:
                dist = euclidean_distance(X[i], X[j])
                if dist < b:
                    b = dist

        a /= max(np.sum(cluster_labels == cluster_labels[i]) - 1, 1)
        silhouette_score_i = (b - a) / max(a, b)
        silhouette_scores.append(silhouette_score_i)

    return np.mean(silhouette_scores)

# X = np.loadtxt('du_lieu_ngau_nhien.csv', delimiter=',')
dataframe=pd.read_excel('Mall_Custormer.xlsx')
X0=dataframe.values[:,3]
X1=dataframe.values[:,4]
X0=Normalization(X0)
X1=Normalization(X1)
X = np.column_stack((X0, X1))
colors = ['r', 'g', 'b','y','c','m','k','gray','turquoise','orangered','lawngreen']  # Màu cho từng cụm
markers = ['o', 's', 'D','v','^','<','>','H','d','p','1']  # Dấu cho từng cụm
# Số lần lặp để cập nhật tâm cụm (có thể điều chỉnh)
num_iterations = 10
num_centroid_init=3
# Dãy số lượng cụm k để thử
k_values = range(2, 11)
silhouette_scores = []
k_plot=range(2,len(k_values)*num_centroid_init+min(k_values))
# Tính Silhouette Score cho từng số lượng cụm k
for k in k_values:
# for i in range(num_centroid_init):
    # Khởi tạo các tâm theo phương pháp K-means++
    centroids = kmeans_plusplus_initialization(X, k)
    for iteration in range(num_iterations):
        # Phân chia dữ liệu vào các cụm
        clusters,cluster_labels = assign_clusters(X, centroids)
        # Cập nhật lại các tâm cụm
        cluster_centers = update_centroids(clusters)

    silhouette_score_avg_i = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_score_avg_i)

# Vẽ biểu đồ Silhouette Score
fig_scores = plt.figure() 
ax_scores = fig_scores.add_subplot(111)
ax_scores.plot(k_values, silhouette_scores, marker='o')
ax_scores.set_xlabel('Số lượng cụm (k)')
ax_scores.set_ylabel('Silhouette Score')
plt.title('Biểu đồ Silhouette Score')
plt.grid()
plt.show()

# fig_scores = plt.figure() 
# ax_scores = fig_scores.add_subplot(121)
# ax_scores.plot(k_plot, silhouette_scores, marker='o')
# ax_scores.set_xlabel('Số lượng cụm (k)')
# ax_scores.set_ylabel('Silhouette Score')
# plt.title('Biểu đồ Silhouette Score')
# plt.grid()
# k=(np.argmax(np.array(silhouette_scores))%len(k_values))+min(k_values)

# centroids = kmeans_plusplus_initialization(X,k)
# ax_result = fig_scores.add_subplot(122)

# for iteration in range(num_iterations):
#     # Phân chia dữ liệu vào các cụm
#     clusters,cluster_labels = assign_clusters(X, centroids)
#     # Cập nhật lại các tâm cụm
#     centroids = update_centroids(clusters)

# for i, cluster in enumerate(clusters):
#     cluster = np.array(cluster)
#     ax_result.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], marker=markers[i], label=f'Cluster {i + 1}')

# centroids = np.array(centroids)
# for i in range(k):
#     ax_result.scatter(centroids[i][0],centroids[i][1],100,c=colors[i],marker='X',label=f'Centroids {i + 1}')
# ax_result.legend()
# ax_result.set_xlabel('X-axis')
# ax_result.set_ylabel('Y-axis')
# plt.title('K-means Clustering')
# plt.show()