import os
import shutil

from huggingface_hub import snapshot_download
import kagglehub

MODEL_ID = "FremyCompany/BioLORD-2023"
MODEL_DIR = "FremyCompany/BioLORD-2023"

# A-Z Medicine Dataset of India: ~254k brands WITH explicit salt columns
# (short_composition1 / short_composition2). The old 1mg dataset had only
# product titles, so compositions had to be guessed from the name.
DATASET_ID = "shudhanshusingh/az-medicine-dataset-of-india"
DATASET_DIR = "shudhanshusingh/az-medicine-dataset-of-india"
def get_model(model_id=MODEL_ID, model_dir=MODEL_DIR):
    # Treat the model as "present" only if the folder exists AND has weights
    weights_present = os.path.isdir(model_dir) and any(
        f.endswith((".safetensors", ".bin")) for f in os.listdir(model_dir)
    )

    if weights_present:
        print(f"Model already present at {model_dir}, skipping download.")
    else:
        print(f"Model not found. Downloading {model_id} ...")
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
        )
        print(f"Downloaded to {model_dir}")

    return model_dir

def get_dataset(dataset_id=DATASET_ID, dataset_dir=DATASET_DIR):
    dataset_present = os.path.isdir(dataset_dir) and any(
        f.endswith((".csv", ".json")) for f in os.listdir(dataset_dir)
    )
    if dataset_present:
        print(f"Dataset already present at {dataset_dir}, skipping download.")
    else:
        print(f"Dataset not found. Downloading {dataset_id} ...")
        # kagglehub downloads into its own cache; mirror the files into the repo.
        cached = kagglehub.dataset_download(dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        for fname in os.listdir(cached):
            src = os.path.join(cached, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dataset_dir, fname))
        print(f"Downloaded to {dataset_dir}")
    return dataset_dir

if __name__ == "__main__":
    model_path = get_model()
    dataset_path = get_dataset()
    print("Model path:", model_path)
    print("Dataset path:", dataset_path)







