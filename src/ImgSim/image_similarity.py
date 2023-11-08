import torch
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from kmeans_pytorch import kmeans
from PIL import Image


class Img2Vec:
    """
    Class for embedding dataset of image files into vectors using Pytorch
    standard neural networks.

    Parameters:
    -----------
    model_name: str object specifying neural network architecture to utilise.
        Must align to the naming convention specified on Pytorch documentation:
        https://pytorch.org/vision/main/models.html#classification
        For supported model architectures see self.embed_dict below
    weights: str object specifying the pretrained weights to load into model.
        Only weights supported by Pytorch torchvision library can be accessed.
        Current functionality reverts to DEFAULT weights if no specified.

    See also:
    -----------
    Img2Vec.embed_dataset(): embed passed images as feature vectors
    Img2Vec.save_dataset(): save embedded dataset to file for future loading
    Img2Vec.load_dataset(): load previously embedded dataset of feature vectors
    Img2Vec.similar_image(): pass target image and return most similar image(s)
    Img2Vec.cluster_dataset(): group embedded images into specified n clusters

    Example:
    -----------

    ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
    ImgSim.embed_dataset('[EXAMPLE PATH TO DIRECTORY OF IMAGES]')

    ImgSim.save_dataset('[OUTPUT PATH FOR SAVING EMBEDDEDINGS]')

    ImgSim.similar_images('[EXAMPLE PATH TO TARGET IMAGE]')

    ImgSim.cluster_dataset(nclusters=6, display=True)
    """

    def __init__(self, model_name, weights="DEFAULT"):
        # dictionary defining the supported NN architectures
        self.embed_dict = {
            "resnet50": self.obtain_children,
            "vgg19": self.obtain_classifier,
            "efficientnet_b0": self.obtain_classifier,
        }

        # assign class attributes
        self.architecture = self.validate_model(model_name)
        self.weights = weights
        self.transform = self.assign_transform(weights)
        self.device = self.set_device()
        self.model = self.initiate_model()
        self.embed = self.assign_layer()
        self.dataset = {}
        self.image_clusters = {}
        self.cluster_centers = {}

    def validate_model(self, model_name):
        if model_name not in self.embed_dict.keys():
            raise ValueError(f"The model {model_name} is not supported")
        else:
            return model_name

    def assign_transform(self, weights):
        weights_dict = {
            "resnet50": models.ResNet50_Weights,
            "vgg19": models.VGG19_Weights,
            "efficientnet_b0": models.EfficientNet_B0_Weights,
        }

        # try load preprocess from torchvision else assign default
        try:
            w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess = weights.transforms()
        except Exception:
            preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        return preprocess

    def set_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        return device

    def initiate_model(self):
        m = getattr(
            models, self.architecture
        )  # equ to assigning m as models.resnet50()
        model = m(weights=self.weights)  # equ to models.resnet50(weights=...)
        model.to(self.device)

        return model.eval()

    def assign_layer(self):
        model_embed = self.embed_dict[self.architecture]()

        return model_embed

    def obtain_children(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])

        return model_embed

    def obtain_classifier(self):
        self.model.classifier = self.model.classifier[:-1]

        return self.model

    def directory_to_list(self, dir):
        ext = (".png", ".jpg", ".jpeg")

        d = os.listdir(dir)
        source_list = [os.path.join(dir, f) for f in d if os.path.splitext(f)[1] in ext]

        return source_list

    def validate_source(self, source):
        # convert source format into standard list of file paths
        if isinstance(source, list):
            source_list = [f for f in source if os.path.isfile(f)]
        elif os.path.isdir(source):
            source_list = self.directory_to_list(source)
        elif os.path.isfile(source):
            source_list = [source]
        else:
            raise ValueError('"source" expected as file, list or directory.')

        return source_list


    def process_individual_image(self, images_folder, output_folder):
        """
        Processes all images in a given folder and saves their embeddings.

        Parameters:
        -----------
        images_folder: str
            The folder containing images to process.
        output_folder: str
            The folder where the embeddings will be saved.
        """
        # Process each image in the folder
        for image_file in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_file)
            if os.path.isfile(image_path):
                try:
                    # Get the embedding for the individual image
                    embedding = self.embed_image(image_path)

                    # Create a unique filename for the embedding
                    filename = os.path.splitext(os.path.basename(image_path))[0] + '_embedding.pt'

                    # Save the embedding to the specified folder
                    embedding_path = os.path.join(output_folder, filename)
                    torch.save(embedding, embedding_path)

                    print(f"Processed and saved embedding for {image_file}")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")


    def embed_image(self, img):
        # load and preprocess image
        img = Image.open(img).convert("RGB")
        img_trans = self.transform(img)

        # store computational graph on GPU if available
        if self.device == "cuda:0":
            img_trans = img_trans.cuda()

        img_trans = img_trans.unsqueeze(0)

        return img_trans

    def embed_dataset(self, source):
        # convert source to appropriate format
        self.files = self.validate_source(source)

        for file in self.files:
            vector = self.embed_image(file)
            self.dataset[str(file)] = vector

        return

    def similar_images(self, target_file, n=None):
        """
        Function for comparing target image to embedded image dataset

        Parameters:
        -----------
        target_file: str specifying the path of target image to compare
            with the saved feature embedding dataset
        n: int specifying the top n most similar images to return
        """

        target_vector = self.embed_image(target_file)

        # initiate computation of consine similarity
        cosine = nn.CosineSimilarity(dim=1)

        # iteratively store similarity of stored images to target image
        sim_dict = {}
        for k, v in self.dataset.items():
            sim = cosine(v, target_vector)[0].item()
            sim_dict[k] = sim

        # sort based on decreasing similarity
        items = sim_dict.items()
        sim_dict = {k: v for k, v in sorted(items, key=lambda i: i[1], reverse=True)}

        # cut to defined top n similar images
        if n is not None:
            sim_dict = dict(list(sim_dict.items())[: int(n)])

        self.output_images(sim_dict, target_file)

        return sim_dict

    
    def similar_images_df(self, target_file, embeddings_dir, output_csv_dir=None, n=None):
        """
        Modified function for comparing target image to individually stored
        embeddings dataset.

        Parameters:
        -----------
        target_file: str
            The path of the target image to compare with the saved feature embedding dataset.
        embeddings_dir: str
            The directory where the embeddings are stored.
        output_csv: str, optional
            Path to save the resulting DataFrame as CSV.
        n: int, optional
            The top n most similar images to return. If None, returns all.
        """

        # Embed the target image
        target_path = os.path.join(embeddings_dir, target_file)
        target_vector = torch.load(target_path)

        # Initialize the cosine similarity function
        cosine = nn.CosineSimilarity(dim=1)

        # Load embeddings and compute similarity
        sim_dict = {}
        for embedding_file in os.listdir(embeddings_dir):
            if embedding_file == target_file:
                continue
            else:
                embedding_path = os.path.join(embeddings_dir, embedding_file)
                image_embedding = torch.load(embedding_path)
                image_name = os.path.splitext(embedding_file)[0].replace('_embedding', '') #os.path.splitext split the file path into (root, extion) pairt
                sim = cosine(image_embedding, target_vector)[0].item() #.item() converts single-item tensor to the value itself
                sim_dict[image_name] = sim

        # Sort based on decreasing similarity
        sorted_sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))

        # Cut to defined top n similar images if n is specified
        if n is not None:
            sorted_sim_dict = dict(list(sorted_sim_dict.items())[:int(n)])

        # Convert the similarity dictionary to a DataFrame
        df_similarity = pd.DataFrame.from_dict(sorted_sim_dict, orient='index', columns=['Similarity'])

        # Set the column and index names based on the image filenames
        df_similarity.index.name = 'Image'
        df_similarity.reset_index(inplace=True)

        # Save the DataFrame to a CSV if output path is provided
        if output_csv_dir:
            output_path = os.path.join(output_csv_dir, image_name + ".csv")
            df_similarity.to_csv(output_path, index=False)

        # Optionally, you could return the DataFrame if needed
        return df_similarity

    def output_images(self, similar, target):
        self.display_img(target, "original")

        for k, v in similar.items():
            self.display_img(k, "similarity:" + str(v))

        return

    def display_img(self, path, title):
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.title(title)
        plt.show()

        return

    def save_dataset(self, path):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        path: str specifying the output folder to save the tensors to
        """

        # convert embeddings to dictionary
        data = {"model": self.architecture, "embeddings": self.dataset}

        torch.save(
            data, os.path.join(path, "tensors.pt")
        )  # need to update functionality for naming convention

        return

    def load_dataset(self, source):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        source: str specifying tensor.pt file to load previous embeddings
        """

        data = torch.load(source)

        # assess that embedding nn matches currently initiated nn
        if data["model"] == self.architecture:
            self.dataset = data["embeddings"]
        else:
            raise AttributeError(
                f'NN architecture "{self.architecture}" does not match the '
                + f'"{data["model"]}" model used to generate saved embeddings.'
                + " Re-initiate Img2Vec with correct architecture and reload."
            )

        return

    def plot_list(self, img_list, cluster_num):
        fig, axes = plt.subplots(math.ceil(len(img_list) / 2), 2)
        fig.suptitle(f"Cluster: {str(cluster_num)}")
        [ax.axis("off") for ax in axes.ravel()]

        for img, ax in zip(img_list, axes.ravel()):
            ax.imshow(Image.open(img))

        fig.tight_layout()

        return

    def display_clusters(self):
        for num in self.cluster_centers.keys():
            # print(f'Displaying cluster: {str(cluster_num)}')

            img_list = [k for k, v in self.image_clusters.items() if v == num]
            self.plot_list(img_list, num)

        return

    def cluster_dataset(self, nclusters, dist="euclidean", display=False):
        vecs = torch.stack(list(self.dataset.values())).squeeze()
        imgs = list(self.dataset.keys())
        np.random.seed(100)

        cluster_ids_x, cluster_centers = kmeans(
            X=vecs, num_clusters=nclusters, distance=dist, device=self.device
        )

        # assign clusters to images
        self.image_clusters = dict(zip(imgs, cluster_ids_x.tolist()))

        # store cluster centres
        cluster_num = list(range(0, len(cluster_centers)))
        self.cluster_centers = dict(zip(cluster_num, cluster_centers.tolist()))

        if display:
            self.display_clusters()

        return
