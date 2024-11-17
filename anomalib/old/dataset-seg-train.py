# Import the datamodule
from anomalib.data import Folder

# Create the datamodule
datamodule = Folder(
    name="hazelnut_toy",
    root="datasets/gear_training_data",
    normal_dir="good",
    abnormal_dir="collective_bad",
    #mask_dir="mask/crack",
    normal_split_ratio=0.2,
)

# Setup the datamodule
datamodule.setup()
# Access the datasets
train_dataset = datamodule.train_data
val_dataset = datamodule.val_data
test_dataset = datamodule.test_data

# Access the dataloaders
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()


i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

print(train_data["image"].shape)
# torch.Size([32, 3, 256, 256])

print(train_data["mask"].shape)
# torch.Size([32, 256, 256])

# Import the model and engine
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Create the model and engine
model = Patchcore()
engine = Engine()

# Train a Patchcore model on the given datamodule
engine.train(datamodule=datamodule, model=model)

