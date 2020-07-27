import os 
import logging 

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class APP:
    def __init__(self):
        self.model = None 

    def clustering(self, x, clustering_type='kmeans', **kwargs):
        assert clustering_type in ('kmeans', 'gmm')
        if clustering_type == 'kmeans':
            return self.kmeans_clustering(x, **kwargs)
        else:
            raise NotImplementedError

    def kmeans_clustering(self, x, max_clusters=20):
        return _kmeans_clustering(x, max_clusters)

    def clustering_with_path_and_code(self, path_list, x, clustering_type, **kwargs):
        cluster_labels = self.clustering(x, clustering_type, **kwargs)

        res = dict()

        for i, label in enumerate(cluster_labels):
            path = path_list[i]

            if label not in res:
                res[label] = [path]
            else:
                res[label].append(path)

        return res 

    def save_images_by_clusters(self, dir_path, res, num_imgs=5, output_size=(416, 416)):
        assert num_imgs >= 1
        logging.info('[APP] save images by res')

        for key_ in res.keys():
            logging.info('[APP] process cluster : {}'.format(key_))
            pathes = res.get(key_)

            for i, path in enumerate(pathes):
                logging.debug('[APP] num : {}'.format())
                img = Image.open(path).resize(output_size)
                img_path = 'cluster_' + str(key_) + '_' + str(i) + '.jpg'
                save_path = os.path.join(dir_path, img_path)
                img.save(save_path)

        return None
 
    def encode_images(self, batch_images, model_type='VAE'):
        output = _encode_images(self.model, batch_images)

        if model_type == 'NaiveAE':
            enc, _ = output
        elif model_type == 'VAE':
            enc, _, _, _ = output
        else:
            raise NotImplementedError
        return enc

    def encode_images_from_pathes(self, path_list, batch_size, model_type):
        dataset = MuraDataset()



def _argmax(list_):
    return max(range(len(list_)), key=list_.__getitem__)


def _kmeans_clustering(x, max_clusters=20):
    clusters = list(range(1, max_clusters))
    scores = []
    
    for n_clusters in clusters:
        clusterer = KMeans(n_clusters, random_state=427)
        cluster_labels = clusterer.fit_predict(x)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(x, cluster_labels)
        print(
            "For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg
        )
        scores.append(silhouette_avg)

    best_clusters = clusters[_argmax(scores)]

    print('best number of clusters is : {}'.format(best_clusters))

    # fit best n_clusters
    clusterer = KMeans(n_clusters, random_state=427) 
    cluster_labels = clusterer.fit_predict(x)

    return cluster_labels

def _encode_images(model, batch_images):
    return model(batch_images)