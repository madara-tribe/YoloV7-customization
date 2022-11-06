# YoloV7 with customization












# convert to CoreML 

<b>to include "class labels" to model</b>
```python
COREML_CLASS_LABELS = ["trafficlight","stop", "speedlimit","crosswalk"]
# add "classifier_config" argument to model
classifier_config = ct.ClassifierConfig(COREML_CLASS_LABELS)
ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])], 
           classifier_config=classifier_config)
```



# References
- [CoreML API References of Classifiers](https://coremltools.readme.io/docs/classifiers)

