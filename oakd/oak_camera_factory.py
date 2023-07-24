import os
from pathlib import Path
from threading import Lock
from typing import ClassVar, Union

import depthai as dai
from viam.logging import logging

# LOGGER: logging.Logger = logging.getLogger(__name__)

class PipelineSettings:
    blob: Union[Path, None] = None
    cameraSockets: list[dai.CameraBoardSocket] = [dai.CameraBoardSocket.CENTER]

    def build(self) -> dai.Pipeline:
        print("building pipeline")
        pipeline = dai.Pipeline()
        print("including RBG in pipeline")
        color_camera = pipeline.create(dai.node.ColorCamera)
        color_camera.setPreviewSize(416, 416)
        color_camera.setBoardSocket(self.cameraSockets[0])
        color_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color_camera.setInterleaved(False)
        color_camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        color_camera.setFps(20)
        
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName('rgb')
        color_camera.preview.link(xout_rgb.input)
            
        if self.blob is not None:
            print("including neural net in pipeline: ", self.blob)
            detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)

            detection_nn.setBlobPath(self.blob)
            detection_nn.setNumClasses(80)
            detection_nn.setConfidenceThreshold(0.8)
            detection_nn.setNumInferenceThreads(1)
            detection_nn.setCoordinateSize(4)
            detection_nn.setIouThreshold(0.5)
            detection_nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
            detection_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
            color_camera.preview.link(detection_nn.input)
            detection_nn.input.setBlocking(False)
            
            xout_nn = pipeline.createXLinkOut()
            xout_nn.setStreamName('nn')
            detection_nn.out.link(xout_nn.input)

        print("pipeline build complete")
        return pipeline
    

class OakCameraFactory:
    camera_lock: Lock = Lock()
    cameras: dict[str,dai.Device] = {}
    pipelines: dict[str, PipelineSettings] = {}
    rgbQueues: dict[str, dai.DataOutputQueue] = {}
    nnQueues: dict[str, dai.DataOutputQueue] = {}

    def getDevice(self, mxid:str) -> Union[dai.Device,None]:
        return self.cameras.get(mxid)
    
    def getRbgQueue(self, mxid:str) -> Union[dai.DataOutputQueue, None]:
        self.camera_lock.acquire()
        try:
            return self.rgbQueues.get(mxid, None)
        finally:
            self.camera_lock.release()
    
    def getNnQueue(self, mxid:str) -> Union[dai.DataOutputQueue, None]:
        self.camera_lock.acquire()
        try:
            return self.nnQueues.get(mxid, None)
        finally:
            self.camera_lock.release()
    
    def addRgbCameraToPipeline(self, mxid:str):
        self.camera_lock.acquire()
        print("adding rbg camera to pipeline")
        try:
            if self.pipelines[mxid] is None:
                self.pipelines[mxid] = PipelineSettings()
            self.pipelines[mxid].cameraSockets = [dai.CameraBoardSocket.CENTER]

            with self.cameras[mxid]:
                print("closing existing camera ", mxid)
            
            print("building new camera ", mxid)
            camera = dai.Device(self.pipelines[mxid].build(), dai.DeviceInfo(mxid))
            self.cameras[mxid] = camera
            
            existing_queues = camera.getOutputQueueNames()
            if "nn" in existing_queues and mxid in self.nnQueues:
                print("there is an existing nn queue, closing ", mxid)
                self.nnQueues[mxid].close()
                self.nnQueues[mxid] = camera.getOutputQueue("nn", maxSize=4, blocking=False)
            if "rgb" in existing_queues and mxid in self.rgbQueues:
                print("there is an existing rbg queue, closing ", mxid)
                self.rgbQueues[mxid].close()
            
            self.rgbQueues[mxid] = camera.getOutputQueue("rgb", maxSize=4, blocking=False)
        finally:
            self.camera_lock.release()

    def addNeuralNetToPipeline(self, mxid:str, blob):
        self.camera_lock.acquire()
        print("adding neural net to pipeline")
        try:
            if self.pipelines[mxid] is None:
                self.pipelines[mxid] = PipelineSettings()
            if os.path.exists(blob) == False:
                raise FileNotFoundError("blob not found", blob)
            self.pipelines[mxid].blob = blob

            with self.cameras[mxid]:
                print("closing existing camera ", mxid)
            
            print("building new camera ", mxid)
            camera = dai.Device(self.pipelines[mxid].build(), dai.DeviceInfo(mxid))
            self.cameras[mxid] = camera
            
            existing_queues = camera.getOutputQueueNames()
            if "nn" in existing_queues and mxid in self.nnQueues:
                print("there is an existing nn queue, closing ", mxid)
                self.nnQueues[mxid].close()
                
            if "rgb" in existing_queues and mxid in self.rgbQueues:
                print("there is an existing rbg queue, closing ", mxid)
                self.rgbQueues[mxid].close()
                self.rgbQueues[mxid] = camera.getOutputQueue("rgb", maxSize=4, blocking=False)

            self.nnQueues[mxid] = camera.getOutputQueue("nn", maxSize=4, blocking=False)
        finally:
            self.camera_lock.release()
    
    def createDevice(self, mxid:str) -> dai.Device:
        self.camera_lock.acquire()
        try:
            camera = self.cameras.get(mxid)
            if camera is None:
                print("creating new camera")
                
                if self.pipelines.get(mxid, None) == None:
                    self.pipelines[mxid] = PipelineSettings()
                deviceInfo = dai.DeviceInfo(mxid)
                camera = dai.Device(self.pipelines[mxid].build(), deviceInfo)
                self.cameras[mxid] = camera
            else:
                print("found existing camera")

        finally:
            self.camera_lock.release()

        return camera
    
