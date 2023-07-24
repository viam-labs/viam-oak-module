import asyncio
import time
from timeit import default_timer as timer
from typing import Any, ClassVar, Mapping, Optional, Tuple, Union

import cv2
import PIL.Image as img
from depthai import Device, ImgFrame
from typing_extensions import Self
from viam.components.camera import (Camera, DistortionParameters,
                                    IntrinsicParameters, RawImage)
from viam.components.component_base import ComponentBase
from viam.logging import logging
from viam.module.types import Reconfigurable, Stoppable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model

from oakd.camera_thread import CameraThread
from oakd.oak_camera_factory import OakCameraFactory

# LOGGER: logging.Logger = logging.getLogger(__name__)

class OakCamera(Camera, Reconfigurable, Stoppable):
    MODEL: ClassVar[Model] = Model.from_string("oak:camera:d")

    CameraFactory: ClassVar[OakCameraFactory]
    cameraProperties: Camera.Properties
    worker: CameraThread

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        """Construct a new instance of the OakVisionService

        Args:
            
        """
        model = cls(config.name)
        print("creating new Oak camera")
        print("creating new Oak camera")

        mxid = None
        if "mxid" in config.attributes.fields:
            mxid = config.attributes.fields["mxid"].string_value
        
        if mxid is None:
            raise ValueError("mxid is required")
        
        print("creating camera")
        OakCamera.CameraFactory.createDevice(mxid)
        print("adding neural net to pipeline")
        OakCamera.CameraFactory.addRgbCameraToPipeline(mxid)

        model.cameraProperties = Camera.Properties(
            supports_pcd=False,
            distortion_parameters=None,
            intrinsic_parameters=None
        )
        print("creating camera thread for ", mxid)
        model.worker = CameraThread(mxid, model.CameraFactory)
        print("starting camera thread for ", mxid)
        model.worker.start()

        print("build oak camera complete")
        return model

    async def get_image(self, mime_type: str = "", *, timeout: Optional[float] = None, **kwargs) -> Union[img.Image, RawImage, None]:
        while True:
            image = self.worker.get_image()
            if image is not None:
                return image
            await asyncio.sleep(0)

    async def get_point_cloud(self, *, timeout: Optional[float] = None, **kwargs) -> Tuple[bytes, str]:
        raise NotImplemented()

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Camera.Properties:
        return self.cameraProperties
        
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ComponentBase]):
        print("reconfigure called...")
        pass
    
    def stop(self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs):
        print("Stop called in camera")
        while self.worker.running:
            self.worker.stop()
            time.sleep(1)
        return super().stop(extra=extra, timeout=timeout, **kwargs)
