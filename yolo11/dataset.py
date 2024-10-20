
from roboflow import Roboflow
rf = Roboflow(api_key="I6oD3j0nzuOWcKftvyTg")
project = rf.workspace("registrationplates").project("reg_plates")
version = project.version(2)
dataset = version.download("yolov11")
                
