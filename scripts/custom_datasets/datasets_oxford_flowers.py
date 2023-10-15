import os.path
import tarfile

import torchvision.datasets
import torchvision.datasets.folder
import torchvision.datasets.utils


class OxfordFlowers102(torchvision.datasets.VisionDataset):
    """
    Dataset class for the OxfordFlowers102 dataset
    (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
    
    This class is PyTorch/torchvision compatible.
    
    The `root` dir must contain the raw files `images.tar` and `lists.tar`,
    which are available from the URL above. They can also be downloaded
    automatically by setting `download=True`.
    """
    
    sources = (
        {
            "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
            "md5": "52808999861908f626f3c1f4e79d11fa",
            "filename": "102flowers.tgz",
            "extracted_filenames": ("102flowers",),
        },
        {
            "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
            "md5": "e0620be6f572b9609742df49c70aed4d",
            "filename": "imagelabels.mat",
            "extracted_filenames": (),
        },
        {
            "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
            "md5": "a5357ecc9cb78c4bef273ce3793fc85c",
            "filename": "setid.mat",
            "extracted_filenames": (),
        },
    )
    
    def __init__(self, root, train=True, transforms=None, transform=None, target_transform=None, download=False, loader=torchvision.datasets.folder.default_loader):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.loader = loader
        
        self.train = train
        
        if download:
            self.download()
        
        self.unique_class_names = self.read_unique_class_names()
        self.image_filenames, self.targets = self.read_filenames_and_targets()

    @staticmethod
    def read_image_labels_from_mat(filename):
        """
        Reads a file list from a .mat file as formatted in this dataset.
        Requires SciPy.
        """
        import scipy.io
        mat = scipy.io.loadmat(filename)
        return mat['labels'][0,:]
    
    @staticmethod
    def read_setid_from_mat(filename):
        """
        Reads a file list from a .mat file as formatted in this dataset.
        Requires SciPy.
        """
        import scipy.io
        mat = scipy.io.loadmat(filename)
        return mat
    
    def read_unique_class_names(self):
        """
        Load the class names from the file list and return all unique values
        as a tuple.
        """
        img_labels = self.read_image_labels_from_mat(filename=os.path.join(self.root, "imagelabels.mat"))
        class_names = set()
        for label in img_labels:
            class_names.add(label)
        return tuple(sorted(class_names))

    def get_class_index_from_class_name(self, class_name):
        return self.unique_class_names.index(class_name)

    def read_filenames_and_targets(self):
        """
        Read all image filenames from the dataset's list files. Varies whether
        `self.train` is `True` or `False`.
        """
        setid = self.read_setid_from_mat(filename=os.path.join(self.root, 'setid.mat'))
        img_label = self.read_image_labels_from_mat(filename=os.path.join(self.root, "imagelabels.mat"))
        if self.train:
            dict_keys = ["trnid", "valid"]
            image_count = 2040 # 1020+1020
        else:
            dict_keys = ["tstid"]
            image_count = 6149
        
        # image_filenames = "image_00001.jpg"
        image_filenames = tuple()
        targets = tuple()
        for dict_key in dict_keys:
            image_filenames += tuple(os.path.join(self.root, 'jpg', f"image_{img_id:05}.jpg") for img_id in setid[dict_key][0])
            targets += tuple(self.get_class_index_from_class_name(img_label[img_id-1]) for img_id in setid[dict_key][0])
        
        assert len(image_filenames) == image_count
        assert len(targets) == image_count

        return (image_filenames, targets)
    
    def __getitem__(self, index):
        """
        Return the `index`-th sample of the dataset, which is a tuple
        `(image, target)`.
        """
        image = self.loader(self.image_filenames[index])
        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return (image, target)
    
    def __len__(self):
        """
        Returns the number of images in this dataset. Varies whether
        `self.train` is `True` or `False`.
        """
        return len(self.image_filenames)
    
    def download(self):
        """
        Download and extract the neccessary source files into the root
        directory.
        """
        for source in self.sources:
            full_filename = os.path.join(self.root, source["filename"])
            
            # download
            torchvision.datasets.utils.download_url(url=source["url"], root=self.root, filename=source["filename"], md5=source["md5"])
                
            # extract
            if not all(os.path.exists(os.path.join(self.root, extracted_filename)) for extracted_filename in source["extracted_filenames"]):
                print("Extracting '{}' to '{}'".format(source["filename"], self.root))
                with tarfile.open(full_filename, "r") as tar:
                    tar.extractall(path=self.root)
            else:
                print("File '{}' was already extracted, skipped extraction".format(source["filename"]))
