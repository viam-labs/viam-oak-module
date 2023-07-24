import asyncio
import sys
from typing import Any, Mapping, Union

from viam.logging import logging
from viam.module.types import Stoppable
from viam.module.module import Module
from oakd.oak_camera import OakCamera
from oakd.oak_vision_service import OakVisionService
from oakd.oak_camera_factory import OakCameraFactory
from viam.resource.registry import Registry, ResourceCreatorRegistration, ResourceRegistration

async def main(address: str):
    """This function creates and starts a new oak module. It adds support for a Vision Service and a Camera Component.

    Args:
        address (str): The address to serve the module on
    """
    print("initializing oak module")
    cameraFactory = OakCameraFactory()
    OakVisionService.CameraFactory = cameraFactory
    OakCamera.CameraFactory = cameraFactory

    print("adding creators")
    Registry.register_resource_creator(OakVisionService.SUBTYPE, OakVisionService.MODEL, ResourceCreatorRegistration(OakVisionService.new))
    Registry.register_resource_creator(OakCamera.SUBTYPE, OakCamera.MODEL, ResourceCreatorRegistration(OakCamera.new))

    module = Module(address, log_level=logging.DEBUG)
    print("registering types")
    module.add_model_from_registry(OakVisionService.SUBTYPE, OakVisionService.MODEL)
    module.add_model_from_registry(OakCamera.SUBTYPE, OakCamera.MODEL)
    
    print("starting module")
    await module.start()

def stop(self, *, extra: Union[Mapping[str, Any], None], timeout: Union[float, None], **kwargs):
    print("Called stop in main.py")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Need socket path as command line argument")

    asyncio.run(main(sys.argv[1]))
