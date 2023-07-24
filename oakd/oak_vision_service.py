from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Union, cast

import blobconverter
import depthai
from PIL import Image
from typing_extensions import Self
from viam.components.component_base import ComponentBase
from viam.logging import logging
from viam.media.video import RawImage
from viam.module.types import Reconfigurable, Stoppable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.types import Model
from viam.services.vision import Vision
from viam.utils import ValueTypes

import numpy as np

from oakd.oak_camera_factory import OakCameraFactory

# LOGGER: logging.Logger = logging.getLogger(__name__)

class OakVisionService(Vision, Reconfigurable, Stoppable):
    MODEL: ClassVar[Model] = Model.from_string("oak:vision:d")

    CameraFactory: ClassVar[OakCameraFactory]

    mxid: str

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        """Construct a new instance of the OakVisionService

        Args:
            
        """
        # "model_path": str
        # "label_path": str
        # "input_size_width_px": int
        # "input_size_height_px": int
        # "shaves": int
        # "data_type": str
        # "model_type": "openvino|caffe|tensorflow|onnx|zoo|raw"
        # "mxid": "cam0"
        ## model specific settings
        # "zoo_type": str
        # "tf_optimizer_params": arr[str]
        # "caffe_proto": str
        # "openvino_ir_xml": str
        # "openvino_ir_bin": str
        # "raw_config_path": str
        # "raw_name": str
       

        model = cls(config.name)
        print("creating new oak vision service")

        if "mxid" in config.attributes.fields:
            model.mxid = config.attributes.fields["mxid"].string_value
        else:
            raise ValueError("mxid is required")

        model_path = None
        if "model_path" in config.attributes.fields:
            model_path = config.attributes.fields["model_path"].string_value
        
        label_path = None
        if "label_path" in config.attributes.fields:
            label_path = config.attributes.fields["label_path"].string_value
        
        input_size_width_px = None
        if "input_size_width_px" in config.attributes.fields:
            input_size_width_px = int(config.attributes.fields["input_size_width_px"].number_value)
        
        input_size_height_px = None
        if "input_size_height_px" in config.attributes.fields:
            input_size_height_px = int(config.attributes.fields["input_size_height_px"].number_value)
        
        shaves = None
        if "shaves" in config.attributes.fields:
            shaves = int(config.attributes.fields["shaves"].number_value)
        
        data_type = None
        if "data_type" in config.attributes.fields:
            data_type = config.attributes.fields["data_type"].string_value
        
        model_type = None
        if "model_type" in config.attributes.fields:
            model_type = config.attributes.fields["model_type"].string_value
        
        zoo_type = None
        if "zoo_type" in config.attributes.fields:
            zoo_type = config.attributes.fields["zoo_type"].string_value
        
        tf_optimizer_params = list()
        if "tf_optimizer_params" in config.attributes.fields:
            for t in config.attributes.fields["tf_optimizer_params"].list_value.values:
                tf_optimizer_params.append(t.string_value)
        
        caffe_proto = None
        if "caffe_proto" in config.attributes.fields:
            caffe_proto = config.attributes.fields["caffe_proto"].string_value

        openvino_xml = None
        if "openvino_xml" in config.attributes.fields:
            openvino_xml = config.attributes.fields["openvino_xml"].string_value

        openvino_bin = None
        if "openvino_bin" in config.attributes.fields:
            openvino_bin = config.attributes.fields["openvino_bin"].string_value

        raw_config_path = None
        if "raw_config_path" in config.attributes.fields:
            raw_config_path = config.attributes.fields["raw_config_path"].string_value

        raw_name = None
        if "raw_name" in config.attributes.fields:
            raw_name = config.attributes.fields["raw_name"].string_value

        blob:Path
        mt = "" if model_type is None else model_type.lower()
        if mt == "zoo":
            blob = blobconverter.from_zoo(name=model_path, shaves=6, zoo_type=zoo_type)
        elif mt == "caffe":
            blob = blobconverter.from_caffe(proto=caffe_proto, model=model_path, data_type=data_type, shaves=shaves)
        elif mt == "tf":
            blob = blobconverter.from_tf(frozen_pb=model_path, data_type=data_type, shaves=shaves, optimizer_params=tf_optimizer_params)
        elif mt == "onnx":
            blob = blobconverter.from_onnx(model=model_path, data_type=data_type, shaves=shaves)
        elif mt == "openvino":
            blob = blobconverter.from_openvino(xml=openvino_xml, bin=openvino_bin, data_type=data_type, shaves=shaves)
        elif mt == "raw":
            blob = blobconverter.from_config(name=raw_name, path=raw_config_path, data_type=data_type, shaves=shaves)
        else:
            raise ValueError("invalid model_type")

        print("creating camera")
        model.CameraFactory.createDevice(model.mxid)
        print("adding neural net to pipeline")
        model.CameraFactory.addNeuralNetToPipeline(model.mxid, blob)

        print("build vision service complete")
        return model

    def frameNorm(self, bbox):
        normVals = [416, 416, 416, 416]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    def get_detections_internal(self) -> List[Detection]:
        print("called get detections!")
        detections:List[Detection] = []
        current_camera = self.CameraFactory.getDevice(self.mxid)
        if current_camera is None:
            raise ValueError("cannot get detections on non-existent camera")
        
        queue = self.CameraFactory.getNnQueue(self.mxid)
        if queue is None:
            print("NN queue not found")
            return detections
        in_nn = cast(depthai.ImgDetections, queue.get())
        if in_nn is not None:
            print("found detections: ", len(in_nn.detections))
            for d in in_nn.detections:
                f = self.frameNorm((d.xmin, d.ymin, d.xmax, d.ymax))
                print("bounds: ", f, d.confidence)
                detections.append(Detection(x_min=int(f[0]), x_max=int(f[1]), y_min=int(f[2]), y_max=int(f[3]), confidence=d.confidence))
        else:
            print("no detections found!")
        return detections
    
    async def get_detections_from_camera(self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[Detection]:
        return self.get_detections_internal()

    async def get_detections(self, image: Union[Image.Image, RawImage], *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[Detection]:
        return self.get_detections_internal()
    
    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[Classification]:
        raise NotImplementedError()
    
    async def get_classifications(self, image: Union[Image.Image, RawImage], count: int, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[Classification]:
        raise NotImplementedError()
    
    async def get_object_point_clouds(self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None) -> List[PointCloudObject]:
        raise NotImplementedError()
    
    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None) -> Mapping[str, ValueTypes]:
        raise NotImplementedError()
    
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ComponentBase]):
        # todo...
        pass

    def stop(self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs):
        print("Stop called in service")
        return super().stop(extra=extra, timeout=timeout, **kwargs)
