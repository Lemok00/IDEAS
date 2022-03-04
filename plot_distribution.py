import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

real_structure_path = f'../dataset/Statistics/structures_Bedroom.npy'
fake_structure_path = f'PAMI_results/generate_PAMI_samples_forRetrieve/PAMI_Bedroom_N=1_lambdaREC=10/sigma=1_delta=0.5/np_structure.npy'
STANDARDLIZE = True


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


real_structures = np.load(real_structure_path)[:1000]
fake_structures = np.load(fake_structure_path)[:1000]

if STANDARDLIZE:
    real_structures = standardization(real_structures)
    fake_structures = standardization(fake_structures)

real_labels = np.ones(shape=(real_structures.shape[0]), dtype=np.int8)
fake_labels = np.zeros(shape=(fake_structures.shape[0]), dtype=np.int8)

structures = np.concatenate([real_structures, fake_structures], axis=0)
labels = np.concatenate([real_labels, fake_labels], axis=0)

tsne = TSNE(n_components=2, init='pca', random_state=0)

result = tsne.fit_transform(structures)

fig = plot_embedding(result, labels, 'Distribution')
plt.show()
plt.savefig('distribution.png')


for i in range(fake_structures.shape[0]):
    # print(real_structures.shape)
    # print(np.expand_dims(fake_structures[i],0).shape)
    fake_structure = np.expand_dims(fake_structures[i], 0)
    distance = np.sum(np.square(real_structures - fake_structure), axis=1)
    #print(real_structures[0:2])
    #print(fake_structure)
    #print((real_structures - fake_structure)[0:2])
    # print(distance)
    min_distance_index = np.argmin(distance)
    print(min_distance_index)
