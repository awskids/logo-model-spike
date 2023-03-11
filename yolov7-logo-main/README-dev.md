# attempt to use

## Flickr scripts need fixing

doesn't download data to the proper place

```bash
sh data/getFlickr.sh
```

/data/flickr_logos_27_dataset.tar.gz

## Convert annotations

maybe this worked, but validation failed

```bash
python src/convert_annotations.py --dataset flickr27 --plot

Traceback (most recent call last):
  File "/Users/trollin/localdevelopment/AWSKids/repos/yolov7-logo/src/convert_annotations.py", line 246, in <module>
    main()
  File "/Users/trollin/localdevelopment/AWSKids/repos/yolov7-logo/src/convert_annotations.py", line 242, in main
    plot_bounding_box(image, annotation_list)
TypeError: plot_bounding_box() missing 1 required positional argument: 'class_id_to_name_mapping'
```

## Training

### setup for GPU

python src/yolov7/train.py --img-size 640 --cfg src/cfg/training/yolov7.yaml --hyp data/hyp.scratch.yaml --batch 2 --epoch 300 --data data/logo_data_flickr.yaml --weights src/yolov7_training.pt --workers 2 --name yolo_logo_det --device 0

### try on CPU

python src/yolov7/train.py --img-size 640 --cfg src/cfg/training/yolov7.yaml --hyp data/hyp.scratch.yaml --batch 2 --epoch 300 --data data/logo_data_flickr.yaml --weights src/yolov7_training.pt --workers 2 --name yolo_logo_det --device cpu

```bash
train: New cache created: data/labels/train.cache
Traceback (most recent call last):
  File "/Users/trollin/localdevelopment/AWSKids/repos/yolov7-logo/src/yolov7/train.py", line 616, in <module>
    train(hyp, opt, device, tb_writer)
  File "/Users/trollin/localdevelopment/AWSKids/repos/yolov7-logo/src/yolov7/train.py", line 245, in train
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
  File "/Users/trollin/localdevelopment/AWSKids/repos/yolov7-logo/src/yolov7/utils/datasets.py", line 69, in create_dataloader
    dataset = LoadImagesAndLabels(path, imgsz, batch_size,
  File "/Users/trollin/localdevelopment/AWSKids/repos/yolov7-logo/src/yolov7/utils/datasets.py", line 418, in __init__
    bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
  File "/Users/trollin/localdevelopment/AWSKids/repos/yolov7-logo/.venv/lib/python3.10/site-packages/numpy/__init__.py", line 305, in __getattr__
    raise AttributeError(__former_attrs__[attr])
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
```

### try downgrading numpy < 1.20.0

did not end well.  couldn't get numpy installed.

### pip install "numpy<1.24.0"

```bash
pip install "numpy<1.24.0"
```

This worked!

### run a quick test

```bash
python src/yolov7/train.py --img-size 320 --cfg src/cfg/training/yolov7.yaml --hyp data/hyp.scratch.yaml --batch 2 --epoch 5 --data data/logo_data_flickr.yaml --weights src/yolov7_training.pt --workers 2 --name yolo_logo_det --device cpu

AssertionError: Torch not compiled with CUDA enabled
```
