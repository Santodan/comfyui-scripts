from huggingface_hub import HfApi, login
import os

def main():
    # 🔑 Replace with your own Hugging Face token
    HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

    login(token=HF_TOKEN)
    api = HfApi()

    repo_id = "BigDannyPt/fp8e5m2-models"
    repo_type = "model"

    # 👉 Ask user for local file
    local_file_path = input("Enter full path to the file on your computer: ").strip()
    if not os.path.isfile(local_file_path):
        print("❌ Error: file not found.")
        return

    # 👉 Ask user for repo folder
    folder_in_repo = input("Enter folder path inside the repo (e.g., Chroma or Chroma/subfolder): ").strip()
    file_name = os.path.basename(local_file_path)

    # Full repo path
    path_in_repo = f"{folder_in_repo}/{file_name}" if folder_in_repo else file_name

    # 🚀 Upload
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=HF_TOKEN
    )

    print(f"✅ Done: uploaded {file_name} to {repo_id}/{path_in_repo}")

if __name__ == "__main__":
    main()
