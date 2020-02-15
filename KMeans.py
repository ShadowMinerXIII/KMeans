import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

f = open("D:\Master work\AI\pima2.txt", "r")
lines = f.readlines()
f.close();
features = []
targets = []
row = 0
for line in lines:
    nums = line.split("\t")
    features.append([])
    numFeatureCols = len(nums)

    for col in range(0, numFeatureCols):
        features[row].append(nums[col])
       
    row += 1

features = np.array(features)
targets = np.array(targets)

for i in range(0, 100):
    if len(features[i]) != 112:
        print(i)
        break

numFeatureRows, numFeatureCols = np.shape(features)
le = LabelEncoder()

for i in range(0, numFeatureCols):
    column = features[:, i]
    le.fit(column)
    column = le.transform(column)
    features[:, i] = column

features = np.transpose(features)
targets = np.transpose(targets)

statement = "np.matrix(list(zip("
for i in range(0, numFeatureCols):
    statement += "features[" + str(i) + "], "
statement = statement[0:len(statement) - 2]
statement += ")))"
print(statement)
X = eval(statement)

numClusters = 5
kmeans = KMeans(n_clusters=numClusters).fit(X)
labels = kmeans.labels_

labelIndices = [None] * numClusters
f1 = [None] * numClusters
f2 = [None] * numClusters
f3 = [None] * numClusters

for i in range(0, numClusters):
    labelIndices[i] = np.where(labels == i)[0].tolist()
    f1[i] = [features[6][j] for j in labelIndices[i]]
    f2[i] = [features[7][j] for j in labelIndices[i]]
    f3[i] = [features[3][j] for j in labelIndices[i]]

labels = kmeans.labels_
f = open("testData.txt", "w")
for l in labels:
    f.write(str(l) + "\n")
f.close()
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
fig = plt.figure()

for i in range(0, numClusters):
    plt.plot(f1[i], f2[i], colours[i] + 'o')

leg = plt.legend(range(0, numClusters))
leg.draggable(True)
fig3D = plt.figure()
ax = fig3D.gca(projection='3d')

for i in range(0, numClusters):
    f1[i] = [float(i) for i in f1[i]]
    f2[i] = [float(i) for i in f2[i]]
    f3[i] = [float(i) for i in f3[i]]
    ax.scatter(np.array(f1[i]), np.array(f2[i]), np.array(f3[i]), c=colours[i])

ax.set_xlabel('polarity')
ax.set_ylabel('subjectivity')
ax.set_zlabel('no. long words')

plt.show()