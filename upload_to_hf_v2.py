import os
import re
import argparse
from huggingface_hub import HfApi, login, whoami, create_repo
from huggingface_hub.errors import HfHubHTTPError

def upload_to_huggingface(token: str = None):
    """
    Authenticates the user using a token, lists repositories, allows creating a new one,
    and uploads a file to a specified repository and folder.
    """
    # --- Authentication ---
    try:
        # Use the provided token to log in non-interactively
        if token:
            print("Attempting to log in with provided token...")
            login(token=token)
        else:
            # If no token is provided via arg or env var, fall back to default behavior
            # which might include a cached token or an interactive prompt.
            print("No token provided directly, checking cached credentials or prompting for login...")
            login()

        user_info = whoami()
        username = user_info['name']
        print(f"✅ Successfully logged in as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed. Please check your token. Error: {e}")
        return

    api = HfApi()

    # --- Repository Selection ---
    while True:
        try:
            print("\nFetching your repositories...")
            models = api.list_models(author=username)
            datasets = api.list_datasets(author=username)
            spaces = api.list_spaces(author=username)
            
            repos = sorted([repo.id for repo in models] + [repo.id for repo in datasets] + [repo.id for repo in spaces])

            if not repos:
                print("You have no repositories.")
            else:
                print("\nYour repositories:")
                for i, repo_id in enumerate(repos):
                    print(f"{i + 1}. {repo_id}")

            print("\nN. Create a new private repository")

            # --- User Choice for Repository ---
            repo_choice_input = input("\nEnter the number of the repository or 'N' to create a new one: ").strip().upper()

            if repo_choice_input == 'N':
                while True:
                    new_repo_name = input("Enter a name for your new private repository (e.g., 'my-cool-dataset'): ").strip()
                    if not re.match("^[a-zA-Z0-9-._]+$", new_repo_name):
                         print("Invalid name. Use only letters, numbers, and '-', '.', '_'")
                         continue
                    try:
                        print(f"Creating new private repository '{new_repo_name}'...")
                        repo_url = create_repo(repo_id=new_repo_name, private=True, repo_type='dataset')
                        print(f"✅ Successfully created new private repository: {repo_url.repo_id}")
                        selected_repo = repo_url.repo_id
                        break
                    except HfHubHTTPError as e:
                        if "You already created this repo" in str(e):
                            print(f"Error: Repository '{new_repo_name}' already exists.")
                            retry = input("Would you like to try a different name? (y/n): ").strip().lower()
                            if retry != 'y': return
                        else:
                            print(f"❌ An unexpected error occurred: {e}")
                            return
                break
            
            else:
                try:
                    repo_choice_index = int(repo_choice_input) - 1
                    if 0 <= repo_choice_index < len(repos):
                        selected_repo = repos[repo_choice_index]
                        break
                    else:
                        print("Invalid number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'N'.")

        except Exception as e:
            print(f"❌ An error occurred while fetching repositories: {e}")
            return

    # --- Folder and File Path ---
    folder_path_in_repo = input(f"\nEnter the folder path in '{selected_repo}' (e.g., 'data/', press Enter for root): ").strip()
    while True:
        local_file_path = input("\nEnter the full path of the file to upload: ").strip()
        if os.path.isfile(local_file_path):
            break
        else:
            print("File not found. Please enter a valid file path.")

    # --- File Upload ---
    file_name = os.path.basename(local_file_path)
    path_in_repo = f"{folder_path_in_repo}/{file_name}" if folder_path_in_repo else file_name
    
    print(f"\nPreparing to upload...")
    print(f"  - Local file:          '{local_file_path}'")
    print(f"  - Target repository:   '{selected_repo}'")
    print(f"  - Path in repository:  '{path_in_repo}'")

    if input("\nProceed with upload? (y/n): ").strip().lower() == 'y':
        try:
            print("Uploading...")
            api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=path_in_repo,
                repo_id=selected_repo,
                commit_message=f"Upload {file_name}"
            )
            print(f"\n✅ File uploaded successfully!")
            print(f"View it at: https://huggingface.co/{selected_repo}/tree/main")
        except Exception as e:
            print(f"\n❌ An error occurred during upload: {e}")
    else:
        print("\nUpload cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to your Hugging Face repository.")
    parser.add_argument("--token", type=str, help="Your Hugging Face API token.")
    args = parser.parse_args()

    # Prioritize the token source: command-line arg > environment variable
    hf_token = args.token or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    upload_to_huggingface(token=hf_token)