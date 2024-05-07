import os, json

from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms
from transformers import AutoProcessor

class Image_Cap_pair_dataset(Dataset):
    def __init__(self, image_dataset_folder:str, path_caption_json:str):

        self.image_dataset_folder = image_dataset_folder

        img_name_list = os.listdir(image_dataset_folder)
        img_name_list.sort()

        self.img_name_list = []

        with open(path_caption_json, 'r') as f:
            self.img_cap_dict = json.load(f)

        for image_name in img_name_list:
            if ('.png' in image_name.lower()) or ('.jpg' in image_name.lower()) or ('.jpeg' in image_name.lower()):
                self.img_name_list.append(image_name)

        self.image_processor = transforms.ToTensor()

    def __len__(self): 
        return len(self.img_name_list)
    
    def load_image(self, image_path) :
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return self.image_processor(image)

    def __getitem__(self, idx): 

        image_path = f'{self.image_dataset_folder}/{self.img_name_list[idx]}'
        image = self.load_image(image_path)
        caption = self.img_cap_dict[self.img_name_list[idx]]
        
        return self.img_name_list[idx], image, caption