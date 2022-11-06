# YoloV7 with customization












# convert to CoreML 

<b>to include "class labels" to model</b>

<img src="https://user-images.githubusercontent.com/48679574/200152880-9e9d5557-b2d6-4418-8774-63e96d02dd45.png" width="800" height="200"/>


```python
COREML_CLASS_LABELS = ["trafficlight","stop", "speedlimit","crosswalk"]
# add "classifier_config" argument to model
classifier_config = ct.ClassifierConfig(COREML_CLASS_LABELS)
ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])], 
           classifier_config=classifier_config)
```



# References
- [CoreML API References of Classifiers](https://coremltools.readme.io/docs/classifiers)

