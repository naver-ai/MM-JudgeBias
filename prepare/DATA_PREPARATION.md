# Data preparation

> Prepare the MM-JudgeBias data for evaluation by running the steps below in order.

## 1. Load the dataset

Pulls the benchmark from [HuggingFace](https://huggingface.co/datasets/naver-ai/MM-JudgeBias).
```python
from datasets import load_dataset
ds = load_dataset("naver-ai/MM-JudgeBias", split="test")
```

## 2. Provide the manual sources

The three datasets below are license/Drive gated — download them yourself into `dataset/`.
A later step extracts the needed samples from the data you prepared.

| dataset | file | where |
|---|---|---|
| VCR | `vcr1images.zip` | https://visualcommonsense.com/ (license agreement required) |
| GeoQA+ | `GeoQA+.rar` | https://github.com/SCNU203/GeoQA-Plus (author permission) |
| UniGeo | `calculation_{train,val,test}.pk` | https://github.com/chen-judge/UniGeo (Drive, *calculation* subset) |

## 3. Reconstruct

Running `scripts/prepare_dataset.py` automatically downloads the remaining sources
(`SUN397`, `ArtBench`, `DocVQA`, `InfographicVQA`, `MMCode`, `Places365`),
then extracts the exact samples MM-JudgeBias uses from every prepared source and
writes them into `images/`.

```bash
python scripts/prepare_dataset.py            # writes images/
python scripts/prepare_dataset.py --only SUN397
python scripts/prepare_dataset.py --list     # per-dataset coverage
```

## Source benchmarks

The source datasets MM-JudgeBias is built on, with their original links. Image
copyright belongs to the original providers — please review the terms at each
source before use.

| source | link |
|---|---|
| COCO 2017 | https://cocodataset.org/ |
| A-OKVQA | https://github.com/allenai/aokvqa |
| CoSyn-400K | https://huggingface.co/datasets/allenai/CoSyn-400K |
| MMMU-Pro | https://mmmu-benchmark.github.io |
| TextVQA | https://textvqa.org/ |
| ScienceQA | https://scienceqa.github.io |
| MathVision | https://mathllm.github.io/mathvision |
| MathVista | https://mathvista.github.io |
| TabRecSet | https://github.com/MaxKinny/TabRecSet |
| RICO | https://interactionmining.org/rico |
| ScreenSpot / SeeClick | https://github.com/njucckevin/SeeClick |
| Plot2Code | https://github.com/TencentARC/Plot2Code |
| MMTab | https://github.com/spursgozmy/table-llava |
| Geometry3K | https://github.com/lupantech/InterGPS |
| ChartBench | https://chartbench.github.io |
| ChartQAPro | https://github.com/vis-nlp/ChartQAPro |
| AI2D | https://prior.allenai.org/projects/diagram-understanding |
| MMEval | https://github.com/MCEVAL/MMCoder |
| TextbookQA (TQA) | https://prior.allenai.org/projects/tqa |
| VQA-RAD | https://huggingface.co/datasets/flaviagiammarino/vqa-rad |
| DocVQA | https://rrc.cvc.uab.es/?ch=17 |
| InfographicVQA | https://www.docvqa.org/datasets/infographicvqa |
| MMCode | https://github.com/likaixin2000/MMCode |
| ArtBench | https://github.com/liaopeiyuan/artbench |
| SUN397* | https://3dvision.princeton.edu/projects/2010/SUN/ |
| Places365 | http://places2.csail.mit.edu/ |
| VCR | https://visualcommonsense.com/ |
| GeoQA+ | https://github.com/SCNU203/GeoQA-Plus |
| UniGeo | https://github.com/chen-judge/UniGeo |

\* Since the original download paths were invalid, we referred to https://huggingface.co/datasets/1aurent/SUN397.