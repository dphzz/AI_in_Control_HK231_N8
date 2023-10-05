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

    for x in X:
        # Tìm cụm gần nhất cho điểm dữ liệu x
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        
        # Thêm điểm dữ liệu vào cụm tương ứng
        clusters[cluster_index].append(x)

    return clusters

# Hàm cập nhật lại các tâm cụm
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) > 0:
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
    return new_centroids

dataframe=pd.read_excel('Mall_Custormer.xlsx')
X0=dataframe.values[:,3]
X1=dataframe.values[:,4]
X0=Normalization(X0)
X1=Normalization(X1)
X = np.column_stack((X0, X1))
# Số lượng cụm (clusters)
k = 7

# Khởi tạo các tâm theo phương pháp K-means++
centroids = kmeans_plusplus_initialization(X, k)
# Số lần lặp để cập nhật tâm cụm (có thể điều chỉnh)
num_iterations = 10
colors = ['r', 'g', 'b','y','c','m','k','gray','turquoise','orangered','lawngreen']  # Màu cho từng cụm
markers = ['o', 's', 'D','v','^','<','>','H','d','p','1']  # Dấu cho từng cụm

    

for iteration in range(num_iterations):
    # Phân chia dữ liệu vào các cụm
    clusters = assign_clusters(X, centroids)
    
    # Cập nhật lại các tâm cụm
    centroids = update_centroids(clusters)
    
    # fig_result = plt.figure() 
    # ax_result = fig_result.add_subplot(111)

    # for i, cluster in enumerate(clusters):
    #     cluster = np.array(cluster)
    #     ax_result.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], marker=markers[i], label=f'Cluster {i + 1}')
    #     plt.legend()
    #     plt.title('K-means Clustering')
    #     plt.xlabel('Annual Income')
    #     plt.ylabel('Spending Score')

    # centroids = np.array(centroids)
    # for i in range(k):
    #     ax_result.scatter(centroids[i][0],centroids[i][1],100,c=colors[i],marker='X',label=f'Centroids {i + 1}')
    #     plt.legend()
    #     plt.title('K-means Clustering')
    #     plt.xlabel('Annual Income')
    #     plt.ylabel('Spending Score')

    # plt.pause(0.5)  # Tạm dừng 0.5 giây để người dùng có thời gian quan sát
    # plt.draw()  # Vẽ lại biểu đồ
    
    # # Đợi sự kiện nhấn phím 'n' để chuyển tiếp
    # while True:
    #     if plt.waitforbuttonpress(timeout=0.1):
    #         break
# Hiển thị dữ liệu và các tâm cụm

for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], marker=markers[i], label=f'Cluster {i + 1}')

centroids = np.array(centroids)
for i in range(k):
    plt.scatter(centroids[i][0],centroids[i][1],100,c=colors[i],marker='X',label=f'Centroids {i + 1}')
plt.legend()
plt.title('K-means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()