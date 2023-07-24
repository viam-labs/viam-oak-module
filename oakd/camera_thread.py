import time
from threading import Lock, Thread
from typing import ClassVar, Union, cast

import cv2
import PIL.Image as img
from depthai import DataOutputQueue, Device, ImgFrame
from viam.logging import logging

from oakd.oak_camera_factory import OakCameraFactory

# LOGGER: logging.Logger = logging.getLogger(__name__)

class CameraThread(Thread):
    running:bool
    current_image: Union[img.Image, None] = None
    camera_factory: OakCameraFactory
    stop_request: bool

    def __init__(self, mxid:str, camera_factory:OakCameraFactory) -> None:
        """

        """
        print("init new thread for ", mxid)
        if camera_factory is None:
            raise ValueError("camera_factory cannot be none")
        if mxid is None or mxid == "":
            raise ValueError("mxid must be set")
        
        self.running = False
        self.camera_factory = camera_factory
        self.mxid = mxid
        self.lock = Lock()
        self.stop_request = False
        super().__init__()

    def get_image(self):
        """

        :return:
        """
        # return self.current_image
        if self.current_image is not None:
            return self.current_image.copy()
        else:
            return None
    
    def stop(self):
        self.stop_request = True
    
    def run(self) -> None:
        print("Starting camera thread for ", self.mxid)
        self.lock.acquire()
        try:
            if self.running:
                raise RuntimeError("Camera thread is already running")
            self.running = True
        finally:
            self.lock.release()
        print("Camera thread now running", self.mxid)
        try:
            while not self.stop_request:
                try:
                    queue = self.camera_factory.getRbgQueue(self.mxid)
                    if queue is not None:
                        try:
                            while not queue.has() and queue.isClosed() == False:
                                time.sleep(0.00001)
                        except RuntimeError:
                            # this is kind of expected when reconfiguring queues?
                            continue

                        if queue.isClosed():
                            print("queue is closed")
                            continue
                        
                        frame = cast(ImgFrame, queue.get())
                        c = frame.getCvFrame()
                        cvFrame = cv2.cvtColor(cast(cv2.Mat, c), cv2.COLOR_BGR2RGB)
                        current_image = self.current_image
                        self.current_image = img.fromarray(cvFrame)
                        if current_image is not None:
                            try:
                                current_image.close()
                            except:
                                #eat it
                                pass
                except:
                    print("thread crash")
        finally:
            print("thread exiting")
            self.running = False
