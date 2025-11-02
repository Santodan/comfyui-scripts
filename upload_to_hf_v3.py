#pip install huggingface_hub prompt_toolkit
import os
import argparse
import glob
from huggingface_hub import HfApi, login, whoami, create_repo
from huggingface_hub.errors import HfHubHTTPError

# --- Enhanced Input Handling with prompt_toolkit ---
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import PathCompleter
    PROMPT_TOOLKIT_AVAILABLE = True
    print("✅ 'prompt_toolkit' found. Advanced tab completion is enabled.")
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    print("⚠️ 'prompt_toolkit' not found. Falling back to basic input.")
    print("   For a better experience, run: pip install prompt_toolkit")


def get_upload_paths():
    """
    Prompts the user for local file(s)/folder(s) using advanced completion
    and expands wildcards (globbing).
    """
    completer = PathCompleter()
    session = PromptSession(completer=completer) if PROMPT_TOOLKIT_AVAILABLE else None

    while True:
        try:
            prompt_message = "\nEnter local path(s) using spaces or wildcards (*): "
            if session:
                input_str = session.prompt(prompt_message)
            else:
                input_str = input(prompt_message)
            
            if not input_str: continue

            expanded_paths = []
            for part in input_str.strip().split():
                expanded_part = os.path.expanduser(part)
                matches = glob.glob(expanded_part)
                if matches:
                    expanded_paths.extend(matches)
                elif os.path.exists(expanded_part):
                    expanded_paths.append(expanded_part)

            if expanded_paths: return expanded_paths
            else: print(f"❌ No files or folders found matching pattern: {input_str}")

        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user."); return []


def main(token: str = None):
    """Main function to handle authentication, repo selection, and upload."""
    try:
        if token: login(token=token)
        else: login()
        username = whoami()['name']
        print(f"✅ Successfully logged in as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed. Please check your token. Error: {e}"); return

    api = HfApi()

    # --- Repository Selection ---
    selected_repo = None
    while not selected_repo:
        try:
            print("\nFetching your repositories...")
            models = api.list_models(author=username)
            datasets = api.list_datasets(author=username)
            spaces = api.list_spaces(author=username)
            repos = sorted([repo.id for repo in models] + [repo.id for repo in datasets] + [repo.id for repo in spaces])

            if repos:
                for i, repo_id in enumerate(repos): print(f"{i + 1}. {repo_id}")
            else:
                print("You have no repositories.")

            print("\nN. Create a new repository")
            choice = input("Enter number, 'N' to create, or 'Q' to quit: ").strip().upper()

            if choice == 'Q': return
            elif choice == 'N':
                repo_name = input("Enter a name for the new repository: ").strip()
                if not repo_name:
                    print("Repository name cannot be empty."); continue
                
                # --- NEW: Ask for repository privacy ---
                privacy_choice = input("Make this repository private? [y/n] (default: n): ").strip().lower()
                is_private = (privacy_choice == 'y')
                
                # --- NEW: Ask for repository type ---
                type_choice = input("Select repo type [1: Model, 2: Dataset, 3: Space] (default: 2): ").strip()
                repo_type = 'model' if type_choice == '1' else 'space' if type_choice == '3' else 'dataset'

                try:
                    print(f"Creating new { 'private' if is_private else 'public' } {repo_type} repository: '{repo_name}'...")
                    repo_url = create_repo(repo_id=repo_name, private=is_private, repo_type=repo_type)
                    selected_repo = repo_url.repo_id
                    print(f"✅ Successfully created '{selected_repo}'")
                except HfHubHTTPError as e:
                    print(f"❌ Error creating repository: {e}")
            
            else:
                try:
                    choice_index = int(choice) - 1
                    if 0 <= choice_index < len(repos): selected_repo = repos[choice_index]
                    else: print("Invalid number.")
                except (ValueError, IndexError): print("Invalid selection.")
        except Exception as e:
            print(f"❌ An error occurred: {e}"); return

    # --- Upload Logic ---
    print(f"\nSelected repository: {selected_repo}")
    local_paths = get_upload_paths()
    
    if not local_paths:
        print("No paths provided for upload. Exiting."); return

    dest_folder_in_repo = input(f"\nEnter destination folder in '{selected_repo}' (press Enter for root): ").strip()

    print("\n--- UPLOAD SUMMARY ---")
    print(f"  - Target repository:   '{selected_repo}'")
    print(f"  - Destination folder:  '{dest_folder_in_repo or 'root'}'")
    print("  - Items to upload:")
    for path in local_paths: print(f"    - {path} ({'Folder' if os.path.isdir(path) else 'File'})")
    print("----------------------")

    if input("\nProceed with upload? (y/n): ").strip().lower() != 'y':
        print("\nUpload cancelled."); return

    for path in local_paths:
        item_name = os.path.basename(path.rstrip('/\\'))
        path_in_repo = os.path.join(dest_folder_in_repo, item_name) if dest_folder_in_repo else item_name
        
        try:
            if os.path.isfile(path):
                print(f"\nUploading FILE '{path}' to '{path_in_repo}'...")
                api.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo, repo_id=selected_repo)
            elif os.path.isdir(path):
                print(f"\nUploading FOLDER '{path}' to '{path_in_repo}'...")
                api.upload_folder(folder_path=path, path_in_repo=path_in_repo, repo_id=selected_repo)
            print(f"  ✅ Successfully uploaded {item_name}")
        except Exception as e:
            print(f"  ❌ FAILED to upload {item_name}. Error: {e}")

    print(f"\n🚀 All operations complete!")
    print(f"View your repository at: https://huggingface.co/{selected_repo}/tree/main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files/folders to your Hugging Face repo.")
    parser.add_argument("--token", help="Your Hugging Face API token.")
    args = parser.parse_args()
    hf_token = args.token or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    try:
        main(token=hf_token)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
