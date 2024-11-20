import subprocess


def add_modified_files():
    try:
        # Get the list of modified files in the staging area
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        # Get the output (list of files)
        files = result.stdout.strip().split("\n")

        if not files or files == [""]:
            print("No files to add.")
            return

        # Add the modified files back to the staging area
        subprocess.run(["git", "add"] + files, check=True)
        print(f"Added files to staging: {', '.join(files)}")

    except subprocess.CalledProcessError as e:
        print(f"Error while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    add_modified_files()
