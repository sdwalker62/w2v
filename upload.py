from huggingface_hub import HfApi

api = HfApi()

# api.create_repo(
#     repo_id="sigil-ml/PreTokenizedWikiEn", repo_type="dataset", exist_ok=True
# )
# api.upload_large_folder(
#     folder_path="./data",
#     repo_id="sigil-ml/PreTokenizedWikiEn",
#     repo_type="dataset",
#     num_workers=48,
# )


# api.create_repo(
#     repo_id="sigil_ml/WikipediaBPETokenizer", repo_type="model", exist_ok=True
# )

api.upload_folder(
    repo_id="sigil_ml/WikipediaBPETokenizer",
    repo_type="model",
    folder_path="./models/tokenizer",
)
