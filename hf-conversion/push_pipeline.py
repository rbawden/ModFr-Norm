#!/usr/bin/python
from pipeline import NormalisationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSeq2SeqLM, pipeline

PIPELINE_REGISTRY.register_pipeline(
    "modern_french_normalisation",
    pipeline_class=NormalisationPipeline,
    pt_model=AutoModelForSeq2SeqLM,
    default={"pt": ("rbawden/modern_french_normalisation", "main")}
)

classifier = pipeline("modern_french_normalisation", model="rbawden/modern_french_normalisation", trust_remote_code=True)


from huggingface_hub import Repository

repo = Repository("modern_french_normalisation", clone_from="rbawden/modern_french_normalisation")
classifier.save_pretrained("modern_french_normalisation")
repo.push_to_hub()
