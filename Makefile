do_bert-qa:
	cd bert-qa && isc train config.isc

do_tiny_training: 
	cd tiny_training && isc train config.isc

do_tv-classification:
	cd tv-classification && isc train config.isc

do_tv-segmentation:
	cd tv-segmentation && isc train config.isc

do_tv-detection:
	cd tv-detection && isc train config.isc

# do_tv-efficientnetv2l-cls:
# 	cd tv-classification && isc train effnetv2l-config.isc

# do_timm-efficientnet_v2l_cls:
# 	cd pytorch-image-models && isc train effnetv2l-config.isc

do_nerf:
	cd nerf_ddp && isc train config.isc