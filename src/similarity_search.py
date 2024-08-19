import logging
from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import torch
from PIL import Image
from typing import List

logger = logging.getLogger("uvicorn")


class Similarity:
    """
    Load model images from images/database. Subdirectory should be in format supercategory__category
    Example usage
    from PIL import Image
    img = Image.open("cut_.png")

    sm = Similarity()
    scores, vector_id, examples = sm.get_neighbors(img)
    # scores now contains the L2 distances - i.e. the lower, the better
    # examples is dict{image_file_path: ['', '', ...], image: [PIL.Image, PIL.Image ....],
                        labels:[1, 1, 2], embeddings:[[0.921], [0.321], ...]}
    """

    def __init__(self, external_metadata: List):
        self.external_metadata = external_metadata
        self.extractor = AutoFeatureExtractor.from_pretrained(
            "models/vit_base_224_21k",
            
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            "models/vit_base_224_21k",
            device_map=self.device)
        # move model to gpu
        self.model = self.model.to(self.device)
        self.dataset = load_dataset(
            "imagefolder", data_dir="src/img_index/smartcart_images", drop_labels=False
        )
        # move the dataset to cuda
        # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
        #self.dataset.to("cuda:0")
        self.dataset_with_embeddings = self.add_emb_faiss_index()
        
        

    def extract_embeddings(self, image):
        #image = Image.open(image).convert("RGB")
        #logger.info(f"image: {image}")
        # prevent ValueError: Unable to infer channel dimension format
        image = image.convert("RGB")
        image_pp = self.extractor(image, return_tensors="pt")
        image_pp.to(self.device)
        # NB! The model and image are moved to cuda
        # The results cannot go directly to numpy
        # -> we need to move them to .cpu() first to convert them to numpy
        features = self.model(**image_pp).last_hidden_state[:, 0].detach().cpu().numpy()

        return features.squeeze()

    def get_neighbors(self, query_image, top_k=1):
        """
        :param query_image: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x14FE9BCA0>
        :param top_k:
        :return:
        """
        # NB! The self.extractor cannot be moved to GPU, it runs on CPU
        # hence it's results need to be moved to GPU via .to("cuda")
        # e.g. https://www.reddit.com/r/learnmachinelearning/comments/wsn0lr/how_do_i_change_this_code_to_make_it_run_on_my/
        query_image = torch.from_numpy(query_image).float().to(self.device)
        qi_embedding = self.model(**self.extractor(query_image, return_tensors="pt").to(self.device))
        qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().cpu().numpy().squeeze()
        # TODO decide which search will be used get_nearest_examples or search
        # TODO vector_id is the same as category_id i.e. =label[0]
        scores, retrieved_examples = self.dataset_with_embeddings[
            "train"
        ].get_nearest_examples("embeddings", qi_embedding, k=top_k)

        return scores, retrieved_examples

    def add_emb_faiss_index(self):
        # TODO: decide if folder enhancement is necessary
        dataset_with_embeddings = self.dataset.map(
            lambda example, idx: {
                "embeddings": self.extract_embeddings(example["image"]),
                "image_path": self.external_metadata[idx]["image_path"],
                #"additional_info": self.external_metadata[idx]["additional_info"],
            },
            with_indices=True
        )

        if self.device == "cuda":
            # device (`Union[int, List[int]]`, *optional*):
            # If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
            # If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            dataset_with_embeddings["train"].add_faiss_index(column="embeddings", device=-1)
        else:
            dataset_with_embeddings["train"].add_faiss_index(column="embeddings")

        return dataset_with_embeddings


"""
1. Enhanced the dataset except with embeddings, also with parent folder name in which is the file. The parent folder is category.
2. Different folder structure of COCO &         
    self.dataset = load_dataset(
            "imagefolder", data_dir=settings.images_database, drop_labels=False
        )

"""
