import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
from torchvision import transforms

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

class MVTecAT(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train",memory_number = 15):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.待检测工件的名称
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        self.test_transform = transforms.Compose([])
        self.test_transform.transforms.append(transforms.ToTensor())
        
        # find test images
        if self.mode == "train":
            self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.image_names.sort()
            self.image_names = self.image_names[memory_number:]
            #print(self.image_names)
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            #self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            train_transform = transforms.Compose([])
            train_transform.transforms.append(transforms.Resize(size, Image.ANTIALIAS))
            self.imgs = Parallel(n_jobs=10)(delayed(lambda file: train_transform(Image.open(file).convert("RGB")))(file) for file in self.image_names)
            print(f"loaded {len(self.imgs)} images")
        else:
            #test mode
            self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            self.image_names.sort()
            #self.imagemask_names = list((self.root_dir / defect_name / "ground_truth").glob(str(Path("*") / "*.png")))
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img



        else:
            filename = self.image_names[idx]

            try:
                temp = self.image_names[idx]
                temp = str(temp)
                list_path = temp.split('/')
                list_path[-3] = 'ground_truth'
                Path_mask = ''
                for ii in list_path:
                    Path_mask += ii
                    Path_mask += "/"
                Path_mask = Path_mask[0:-5]
                Path_mask += '_mask.png'
                img_mask = Image.open(Path_mask).convert('L')
                img_mask = img_mask.resize((int(self.size), int(self.size)))
                img_mask = self.test_transform(img_mask)
            except:
                img_mask = torch.zeros((1,256,256))


            # print(img_mask.shape)
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good",img_mask
