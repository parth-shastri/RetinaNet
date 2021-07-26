import numpy as np
from sklearn.cluster import KMeans

clusters = []
data = [
    5, 4.8, 4.8, 4.6, 4.6, 4.4, 4.4, 4.2, 4.2, 4,
    5, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1,
    5, 4.9, 4.9, 4.8, 4.8, 4.7, 4.7, 4.6, 4.6, 4.5,
    5, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1,
    5, 4.8, 4.8, 4.6, 4.6, 4.4, 4.4, 4.2, 4.2, 4,
]

data = np.array(data).reshape(-1, 1)
print(data.shape)

avg_pressure = np.average(data)
variance = np.var(data, ddof=4)

kmeans = KMeans(n_clusters=5).fit(data)

# print(c_data)
print(kmeans.labels_.reshape(5, -1))
data = data.reshape(5, -1)

labels_mask = kmeans.labels_.reshape(5, -1)
for i in range(5):
    clusters.append(data[labels_mask == i])
    if (np.abs(kmeans.cluster_centers_[i] - data[labels_mask == i]) <= 0.1).all():
        print(f"The cluster {i} satisfies the given cond")

print(clusters)
print(f"The average pressure is {avg_pressure}\nThe variance of the data is {variance}")

# while len(data) > 0:
# 	clusters.append([])
# 	a = max(data)
#
# 	for i in data:
# 		if i >= (max(data) - 0.1):
# 			clusters[len(clusters)-1].append(i)
#
# 			data.remove(i)
# 			#print(data)
# 	print(clusters)
