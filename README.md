# YoloV7 with customization

custmize yolov7 such as Sinkhorn loss optimize and so on.
and base yolov7 for various purpose.


# version to convert as onnx




# convert to CoreML 

<b>to include "class labels" to model</b>

<img src="https://user-images.githubusercontent.com/48679574/200152880-9e9d5557-b2d6-4418-8774-63e96d02dd45.png" width="800" height="300"/>

```python
COREML_CLASS_LABELS = ["trafficlight","stop", "speedlimit","crosswalk"]
# add "classifier_config" argument to model
classifier_config = ct.ClassifierConfig(COREML_CLASS_LABELS)
ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])], 
           classifier_config=classifier_config)
```

## experiment ： Sinkhorn use for OTA loss

# References
- [CoreML API References of Classifiers](https://coremltools.readme.io/docs/classifiers)

