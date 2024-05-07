import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json

from tqdm import tqdm

from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms

from transformers import AutoTokenizer, ViTImageProcessor

import os

class MSCOCO_train_dataset(Dataset):
    def __init__(self, dataset_folder='/home/minkyu4506/Dataset/MSCOCO', image_size = (256, 256), clip_name = "openai/clip-vit-base-patch32", node_rank=0):
        
        self.dataset_folder = dataset_folder

        self.tokenizer = AutoTokenizer.from_pretrained(clip_name)

        with open('./materials/mscoco_train_name_list_larger_than_256.json', 'r') as f:
            self.image_name_list = json.load(f)
        with open('./materials/mscoco_train_caption_list_larger_than_256.json', 'r') as f:
            self.caption_list = json.load(f)

        self.transform = transforms.Compose(
            [transforms.RandomCrop(image_size), transforms.ToTensor()])
       
    
    def __len__(self): 
        return len(self.image_name_list)
    
    def load_image(self, image_path) :
        
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def __getitem__(self, idx): 
        
        img_name = self.image_name_list[idx]

        img = self.load_image(f'{self.dataset_folder}/train2014/{img_name}')
        caption = self.caption_list[idx]

        tokenized_output = self.tokenizer(caption, padding="max_length", max_length=38, truncation=True, return_tensors="pt")
        
        token = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']

        if len(token.size()) != 1 :
            token = token.squeeze(0)
        
        if len(attention_mask.size()) != 1 :
            attention_mask = attention_mask.squeeze(0)
        
        return img, token, attention_mask