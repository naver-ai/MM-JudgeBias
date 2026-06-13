# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

SOURCES = {
    "doc-vqa": {
        "mode": "auto", "insecure": True,
        "url": "https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_images.tar.gz",
        "archive": "spdocvqa_images.tar.gz", "image_root": "spdocvqa_images.tar.gz",
    },
    "infographicVQA": {
        "mode": "auto", "insecure": True,
        "url": "https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_images.tar.gz",
        "archive": "infographicsvqa_images.tar.gz", "image_root": "infographicsvqa_images.tar.gz",
    },
    "mmcode": {
        "mode": "hf",
        "url": "https://huggingface.co/datasets/likaixin/MMCode",
    },
    "SUN397": {
        "mode": "hf",
        "url": "https://vision.princeton.edu/projects/2010/SUN/  (mirror: 1aurent/SUN397)",
    },
    "art-bench": {
        "mode": "auto",
        "url": "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
        "archive": "artbench-10-imagefolder-split.tar",
        "image_root": "artbench-10-imagefolder-split.tar",
    },
    "Places365": {
        "mode": "auto", 
        "url": "https://data.csail.mit.edu/places/places365/train_256_places365standard.tar",
        "archive": "places365_train_256.tar", "image_root": "places365_train_256.tar",
    },
    "GeoQA+": {
        "mode": "manual", 
        "pk": True, "unpack": True,
        "url": "https://github.com/SCNU203/GeoQA-Plus  (Google Drive, author permission)",
        "archive": ["GeoQA+.rar"], "image_root": "geoqa_plus",
    },
    "vcr-qa": {
        "mode": "manual", 
        "url": "https://visualcommonsense.com/",
        "archive": "vcr1images.zip",
    },
    "UniGeo": {
        "mode": "manual", "pk": True,
        "url": "https://github.com/chen-judge/UniGeo  (Google Drive, calculation subset)",
        "archive": ["calculation_train.pk", "calculation_val.pk", "calculation_test.pk"],
        "image_root": "unigeo",
    },
}
