#/bin/sh
unzip archive.zip
mkdir -p images/train annotations/train
python3 annotate.py
mkdir images/val annotations/val
rm annotations/*.xml
mv annotations labels
mv images/train/road2*.png images/val/
#mv images/train/road3*.png images/test/
mv labels/train/road2*.txt labels/val/
#mv labels/train/road3*.txt labels/test/
ls images/train |wc -l && ls labels/train |wc -l
ls images/val |wc -l && ls labels/val |wc -l  

