# Import the datamodule
from anomalib.data import Folder

# Create the datamodule
datamodule = Folder(
    name="gears",
    root="datasets/gear_training_data",
    normal_dir="good",
    abnormal_dir="bad",
    task="classification",
)

# Setup the datamodule
datamodule.setup()


# Import the model and engine
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Create the model and engine
model = Patchcore()
engine = Engine(task="classification")

# Train a Patchcore model on the given datamodule
engine.train(datamodule=datamodule, model=model)
