# Sample Oak D Module

This sample module provides access to both the video stream as well as a custom Vision Service for running ML models on the camera hardware itself, instead of on the main CPU of the host computer. This proof of concept is provided "as-is". It is not feature complete but is a great starting point for integrating with the Oak cameras.

You can choose to run either the Vision Service, or the Camera, or both, depending on your needs. You will need to manually configure the module as well as the components in app.viam.com in order to use them. Please see below for an example configuration.

```json
{
  "services": [
    {
      "name": "myVision",
      "type": "vision",
      "attributes": {
        "mxid": "18443010016B060F00",
        "model_path": "yolo-v3-tiny-tf",
        "model_type": "zoo"
      },
      "model": "oak:vision:d"
    }
  ],
  "modules": [
    {
      "executable_path": "/home/viam/dev/oakd/module.sh",
      "name": "oakd"
    }
  ],
  "components": [
    {
      "depends_on": [],
      "name": "board",
      "type": "board",
      "model": "jetson",
      "attributes": {
        "spis": [],
        "i2cs": []
      }
    },
    {
      "model": "oak:camera:d",
      "attributes": {
        "mxid": "18443010016B060F00"
      },
      "depends_on": [],
      "name": "myCamera",
      "type": "camera"
    },
    {
      "depends_on": [
        "myCamera"
      ],
      "name": "myDetections",
      "type": "camera",
      "model": "transform",
      "attributes": {
        "source": "myCamera",
        "pipeline": [
          {
            "type": "detections",
            "attributes": {
              "detector_name": "myVision",
              "confidence_threshold": 0.5
            }
          }
        ]
      }
    }
  ]
}
```

The `services` section defines the Vision Services to use, one per OakD Camera (note there is no option to choose the Left/Right/Center cameras, the sample implementation only uses the Center camera). Note that you must provide the `mxid` of the camera in order to use this. The `mxid` can be obtained from the camera itself.

The `components` section adds 2 components. The first one is a `camera` of model `oak:camera:d` that gives access the the RGB camera stream. The `transform` camera is to show the detections from the Vision Service overlayed on the RBG stream from the `oak:camera:d`. As with the Vision Service, the `mxid` is required.

If you need to extend this code to provide access to the left or right cameras on the Oak D, you will either need to directly modify the `oak_camera_factory.py` to select the right module, or update the configuration so the desired camera(s) can be passed into the `oak_camera_factory`

## Starting The Module

This is a [Modular Resource](https://docs.viam.com/program/extend/modular-resources/) for the Viam RDK. To add this to your robot you must:

1. Unzip on the robot
2. Install the Python dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Add a new Module in app.viam.com under the Modules tab and point to the `module.sh` file
4. Add the desired camera(s) and/or Vision Service(s) to your robot's config. The Model string is
   1. Camera: `oak:camera:d`
   2. Vision Service: `oak:vision:d`
5. Save the configuration

Once you are satisfied with the functionality of the module, please go to `main.py` and change line 28 from:

```python
module = Module(address, log_level=logging.DEBUG)
```

To

```python
module = Module(address, log_level=logging.INFO)
```
This will reduce the number of log messages.
