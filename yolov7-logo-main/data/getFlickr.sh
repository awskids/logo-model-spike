#!/bin/bash
wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz -P ./data/
tar -xvzf ./data/flickr_logos_27_dataset.tar.gz --directory ./data/
tar -xvzf ./data/flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz -C ./data/flickr_logos_27_dataset/
