from roboflow import Roboflow
rf = Roboflow(api_key="GJIAFTIpnofM5VjzHCb3")
project=rf.workspace("tomahhhto").project("person-tracking-cctv-gffyk")
version=project.version(1)
dataset=version.download("yolov8")
                