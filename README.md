# YoloV7 with customization

```
# version
- python3.7
- pytorch 1.8.1+cu101
- torchvison 0.9.1+cu101
- torchaudio '0.8.1'
```


custmize yolov7 such as Sinkhorn loss optimize and so on. and base yolov7 for various purpose.


# version to convert as onnx

```sh
# version for onnx 
- onnx '1.12.0'
- onnxruntime '1.13.1'
- onnxsim(simplify) '0.4.8'
# coreml tool version
- coremltools '6.0'
```

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

## experiment result ： Sinkhorn use for OTA loss
optimize loss matrix(as cost-matrix) by Sinkhorn as follows:




# References
- [CoreML API References of Classifiers](https://coremltools.readme.io/docs/classifiers)

