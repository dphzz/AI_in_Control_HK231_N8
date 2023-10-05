import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Hàm tính Inertia (Sum of Squared Errors) cho một tập dữ liệu và một phân cụm
def inertia(X, cluster_centers, cluster_labels):
    num_samples = len(X)
    sum_of_squared_errors = 0

    for i in range(num_samples):
        dist_to_center = euclidean_distance(X[i], cluster_centers[cluster_labels[i]])
        sum_of_squared_errors += dist_to_center**2

    return sum_of_squared_errors

# X = np.loadtxt('du_lieu_ngau_nhien.csv', delimiter=',')
dataframe=pd.read_excel('Mall_Custormer.xlsx')
X0=dataframe.values[:,3]
X1=dataframe.values[:,4]

# fig = plt.figure() 
# ax0 = fig.add_subplot(121)

# Vẽ biểu đồ 3D
# ax0.scatter(X0, X1, c='r', marker='o')

# # Đặt nhãn cho các trục
# ax0.set_xlabel('Annual Income')
# ax0.set_ylabel('Spending Score')
# plt.title('Raw Data')

# X0=Normalization(X0)
# X1=Normalization(X1)

# ax1 = fig.add_subplot(122)

# # Vẽ biểu đồ 3D
# ax1.scatter(X0, X1, c='r', marker='o')

# # Đặt nhãn cho các trục
# ax1.set_xlabel('Annual Income')
# ax1.set_ylabel('Spending Score')
# plt.title('Normalization Data')
# plt.show()

X = np.column_stack((X0, X1))

# Dãy số lượng cụm k để thử
k_values = range(2, 11)
inertia_values = []

for k in k_values:
    # Khởi tạo các tâm theo phương pháp K-means++
    centroids = kmeans_plusplus_initialization(X, k)
    # Số lần lặp để cập nhật tâm cụm (có thể điều chỉnh)
    num_iterations = 10

    for iteration in range(num_iterations):
        # Phân chia dữ liệu vào các cụm
        clusters,cluster_labels = assign_clusters(X, centroids)
        
        # Cập nhật lại các tâm cụm
        cluster_centers = update_centroids(clusters)

    inertia_score = inertia(X, cluster_centers, cluster_labels)
    inertia_values.append(inertia_score)

plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('Inertia')
plt.title('Biểu đồ Elbow Method')
plt.grid()
plt.show()