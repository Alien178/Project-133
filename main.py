from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("main.csv")

planet_masses = df["Mass"]
planet_radiuses = df["Radius"]

x = []

for index, planet_mass in enumerate(planet_masses):
  temp_list = [
               planet_radiuses[index],
               planet_mass
  ]

  x.append(temp_list)

wcss = []

for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
  kmeans.fit(x)

  wcss.append(kmeans.inertia_)

plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), wcss, marker = "o", color = "red")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()