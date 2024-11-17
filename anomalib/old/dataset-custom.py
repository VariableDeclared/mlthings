# Import the datamodule
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode

# Create the datamodule
datamodule = Folder(
    name="gears",
    root="datasets/gear_training_data",
    normal_dir="good",
    test_split_mode=TestSplitMode.SYNTHETIC,
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
