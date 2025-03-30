#!/usr/bin/env python3
"""
Git Automation Script for Trading Strategy Development

This script automates common Git operations based on a configuration set at the top.
It handles branching, committing, merging, tagging, and pushing to GitHub.

Usage:
    Simply run the script: python git_auto.py
"""

import os
import sys
import subprocess
import datetime

###################
# CONFIGURATION   #
###################

# Repository settings
REPO_PATH = "."  # Current directory by default

# Operation mode (choose one)
# "commit" - Just commit changes to current branch
# "feature" - Create a new feature branch, commit to it
# "complete" - Finish a feature branch and merge to main
# "release" - Tag and push a new version
# "abandon" - Abandon a branch and start over from main
# "revert" - Go back to a previous state in your current branch
# "checkpoint" - Create a recovery point you can return to later
# "recover" - Recover to a previous tag or commit
MODE = "commit"

# Branch settings
BRANCH_NAME = "feature-branch"  # Used for feature, complete, and abandon modes
CREATE_NEW_BRANCH = True  # Set to False to use existing branch
NEW_BRANCH_NAME = "new-attempt"  # Used when abandoning and starting over

# Commit settings
COMMIT_MESSAGE = "Added trade analysis improvements and new analysis scripts " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
AUTO_STAGE_ALL = False  # We'll use custom staging logic
FILES_TO_STAGE = ["*.py", "README.md"]  # Only used for specific patterns
INCLUDE_ROOT_FILES = True  # Include files in the root directory
EXCLUDE_SUBFOLDERS = True  # Exclude all subfolders by default
INCLUDED_FOLDERS = ["Full_Send"]  # Subfolders to include

# Tag and recovery settings
CREATE_TAG = True
TAG_NAME = "v1.0"  # Used for release and checkpoint modes
TAG_MESSAGE = "Version 1.0 release"

# Revert and recovery settings
REVERT_TYPE = "commit"  # "commit" to undo last commit, "hard" to go back to specific commit
COMMIT_HASH = ""  # For hard revert, specify commit hash to return to
STEPS_BACK = 1  # For commit revert, how many commits to go back
RECOVERY_POINT = "pre-experiment"  # Tag or commit to recover to

# Push settings
PUSH_TO_REMOTE = True
REMOTE_NAME = "origin"
MAIN_BRANCH = "main"  # or "master" depending on your repository

# Safety settings
CONFIRM_BEFORE_PUSH = True  # Set to False to push without confirmation


###################
# IMPLEMENTATION  #
###################

class GitCommand:
    """Helper class to run Git commands"""

    @staticmethod
    def run(command, show_output=True):
        """Run a git command and return the output"""
        try:
            # Add git to the beginning of the command
            if not command.startswith("git "):
                command = "git " + command

            # Run the command
            process = subprocess.run(command, shell=True, check=True,
                                     capture_output=True, text=True)

            if show_output and process.stdout:
                print(f"Command output: {process.stdout}")

            return process.stdout.strip()

        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {command}")
            print(f"Error message: {e.stderr}")
            sys.exit(1)

    @staticmethod
    def check_status():
        """Check if there are changes to commit"""
        return GitCommand.run("git status --porcelain", show_output=False)

    @staticmethod
    def current_branch():
        """Get the name of the current branch"""
        return GitCommand.run("git rev-parse --abbrev-ref HEAD", show_output=False)


def setup_repo():
    """Set up the repository and change to the specified directory"""
    if REPO_PATH != ".":
        try:
            os.chdir(REPO_PATH)
            print(f"Changed to directory: {REPO_PATH}")
        except FileNotFoundError:
            print(f"Error: Directory {REPO_PATH} not found.")
            sys.exit(1)

    # Check if we're in a git repository
    try:
        GitCommand.run("git rev-parse --is-inside-work-tree", show_output=False)
    except subprocess.CalledProcessError:
        print("Error: Not a git repository. Please run this script from within a git repository.")
        sys.exit(1)


def commit_changes():
    """Stage and commit changes"""
    # Check if there are changes to commit
    if not GitCommand.check_status():
        print("No changes to commit.")
        return False

    # Stage files based on configuration
    if AUTO_STAGE_ALL:
        print("Staging all changes...")
        GitCommand.run("git add .")
    else:
        # Custom staging logic to include/exclude folders
        if INCLUDE_ROOT_FILES:
            # Stage all files in the root directory (not subfolders)
            print("Staging files in root directory...")

            # Use find command to list only root files
            try:
                # The command below lists files in the current directory, not in subdirectories
                # MacOS/Linux version
                import subprocess
                result = subprocess.run("find . -maxdepth 1 -type f | grep -v '^\./\.'",
                                        shell=True, capture_output=True, text=True)
                files = result.stdout.strip().split('\n')

                # Add each file individually
                for file in files:
                    if file and not file.startswith('./.'):  # Skip hidden files
                        print(f"Staging: {file}")
                        GitCommand.run(f"git add \"{file}\"")
            except Exception as e:
                print(f"Error finding root files: {e}")
                print("Falling back to pattern matching...")
                # Fallback to pattern matching if find command fails
                for file_pattern in FILES_TO_STAGE:
                    print(f"Staging files matching: {file_pattern}")
                    GitCommand.run(f"git add {file_pattern}")

        # Include specified subfolders
        if INCLUDED_FOLDERS:
            for folder in INCLUDED_FOLDERS:
                if os.path.exists(folder) and os.path.isdir(folder):
                    print(f"Staging included folder: {folder}")
                    GitCommand.run(f"git add \"{folder}/\"")
                else:
                    print(f"Warning: Included folder '{folder}' not found.")

    # Commit changes
    print(f"Committing with message: {COMMIT_MESSAGE}")
    GitCommand.run(f"git commit -m \"{COMMIT_MESSAGE}\"")
    return True


def create_feature_branch():
    """Create and switch to a new feature branch"""
    # Check if the branch already exists
    branches = GitCommand.run("git branch", show_output=False)

    if f" {BRANCH_NAME}" in branches or f"* {BRANCH_NAME}" in branches:
        if CREATE_NEW_BRANCH:
            print(f"Warning: Branch {BRANCH_NAME} already exists.")
            response = input(f"Do you want to use the existing branch? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)

        # Switch to the branch
        current = GitCommand.current_branch()
        if current != BRANCH_NAME:
            print(f"Switching to branch: {BRANCH_NAME}")
            GitCommand.run(f"git checkout {BRANCH_NAME}")
    else:
        # Create and switch to the new branch
        print(f"Creating and switching to new branch: {BRANCH_NAME}")
        GitCommand.run(f"git checkout -b {BRANCH_NAME}")


def complete_feature():
    """Complete a feature branch by merging it into main"""
    # Check current branch
    current = GitCommand.current_branch()

    # Commit any pending changes
    pending_changes = commit_changes()

    if current != BRANCH_NAME:
        if current == MAIN_BRANCH:
            print(f"You are on {MAIN_BRANCH}, not on a feature branch.")
            response = input(f"Do you want to merge {BRANCH_NAME} into {MAIN_BRANCH}? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        else:
            print(f"You are on {current}, not on {BRANCH_NAME}.")
            response = input(f"Do you want to switch to {BRANCH_NAME}? (y/n): ")
            if response.lower() == 'y':
                GitCommand.run(f"git checkout {BRANCH_NAME}")
            else:
                sys.exit(1)

    # Switch to main branch
    print(f"Switching to {MAIN_BRANCH} branch...")
    GitCommand.run(f"git checkout {MAIN_BRANCH}")

    # Pull latest changes
    print(f"Pulling latest changes from remote {REMOTE_NAME}/{MAIN_BRANCH}...")
    try:
        GitCommand.run(f"git pull {REMOTE_NAME} {MAIN_BRANCH}")
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to pull from {REMOTE_NAME}/{MAIN_BRANCH}.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Merge the feature branch
    print(f"Merging {BRANCH_NAME} into {MAIN_BRANCH}...")
    GitCommand.run(f"git merge {BRANCH_NAME}")

    # Delete the feature branch if successful
    response = input(f"Delete the feature branch {BRANCH_NAME}? (y/n): ")
    if response.lower() == 'y':
        GitCommand.run(f"git branch -d {BRANCH_NAME}")


def create_release():
    """Create a tag for release"""
    if not CREATE_TAG:
        print("Skipping tag creation (CREATE_TAG is False).")
        return

    # Check if tag already exists
    tags = GitCommand.run("git tag", show_output=False)
    if TAG_NAME in tags.split():
        print(f"Warning: Tag {TAG_NAME} already exists.")
        response = input("Do you want to create the tag anyway? This will overwrite the existing tag. (y/n): ")
        if response.lower() != 'y':
            return
        # Delete the existing tag
        GitCommand.run(f"git tag -d {TAG_NAME}")

    # Create the tag
    print(f"Creating tag {TAG_NAME} with message: {TAG_MESSAGE}")
    GitCommand.run(f"git tag -a {TAG_NAME} -m \"{TAG_MESSAGE}\"")


def push_changes():
    """Push changes to GitHub"""
    if not PUSH_TO_REMOTE:
        print("Skipping push to remote (PUSH_TO_REMOTE is False).")
        return

    # Confirm before pushing
    if CONFIRM_BEFORE_PUSH:
        response = input(f"Push changes to {REMOTE_NAME}/{GitCommand.current_branch()}? (y/n): ")
        if response.lower() != 'y':
            print("Push aborted.")
            return

    # Push changes
    current = GitCommand.current_branch()
    print(f"Pushing to {REMOTE_NAME}/{current}...")
    GitCommand.run(f"git push {REMOTE_NAME} {current}")

    # Push tags if in release mode
    if MODE == "release" and CREATE_TAG:
        print(f"Pushing tag {TAG_NAME}...")
        GitCommand.run(f"git push {REMOTE_NAME} {TAG_NAME}")


def abandon_branch():
    """Abandon a branch and start over from main"""
    current = GitCommand.current_branch()

    if current == MAIN_BRANCH:
        print(f"You are already on {MAIN_BRANCH}.")
        response = input(f"Do you want to delete branch {BRANCH_NAME} and create {NEW_BRANCH_NAME}? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        # Switch to main
        print(f"Switching to {MAIN_BRANCH}...")
        GitCommand.run(f"git checkout {MAIN_BRANCH}")

        # Confirm branch deletion
        response = input(f"Delete branch {BRANCH_NAME}? (y/n): ")
        if response.lower() == 'y':
            print(f"Deleting branch {BRANCH_NAME}...")
            try:
                GitCommand.run(f"git branch -d {BRANCH_NAME}")
            except subprocess.CalledProcessError:
                print(f"Branch {BRANCH_NAME} has unmerged changes.")
                force_delete = input("Force delete? This will PERMANENTLY LOSE your changes! (y/n): ")
                if force_delete.lower() == 'y':
                    GitCommand.run(f"git branch -D {BRANCH_NAME}")

    # Pull latest changes
    print(f"Pulling latest changes from {REMOTE_NAME}/{MAIN_BRANCH}...")
    try:
        GitCommand.run(f"git pull {REMOTE_NAME} {MAIN_BRANCH}")
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to pull from {REMOTE_NAME}/{MAIN_BRANCH}.")

    # Create new branch
    print(f"Creating new branch {NEW_BRANCH_NAME}...")
    GitCommand.run(f"git checkout -b {NEW_BRANCH_NAME}")
    print(f"Successfully created new branch {NEW_BRANCH_NAME} from {MAIN_BRANCH}.")


def revert_changes():
    """Revert to a previous state"""
    if REVERT_TYPE == "commit":
        # Soft reset - keep changes but undo commit
        print(f"Reverting the last {STEPS_BACK} commit(s) (keeping changes)...")
        GitCommand.run(f"git reset --soft HEAD~{STEPS_BACK}")
        print("Changes have been unstaged but preserved in your working directory.")

    elif REVERT_TYPE == "hard":
        if not COMMIT_HASH:
            # List recent commits if hash not specified
            print("Showing recent commits. Please specify COMMIT_HASH in the config section:")
            GitCommand.run("git log --oneline -n 10")
            return

        # Hard reset - discard all changes
        print(f"WARNING: Hard reverting to commit {COMMIT_HASH}...")
        print("This will PERMANENTLY DISCARD all changes after this commit!")
        confirm = input("Are you sure? This cannot be undone! (type 'yes' to confirm): ")
        if confirm.lower() == 'yes':
            GitCommand.run(f"git reset --hard {COMMIT_HASH}")
            print(f"Successfully reset to commit {COMMIT_HASH}.")
        else:
            print("Hard reset aborted.")
    else:
        print(f"Error: Unknown revert type '{REVERT_TYPE}'. Use 'commit' or 'hard'.")


def create_checkpoint():
    """Create a recovery checkpoint using tags"""
    # Create the tag as a recovery point
    print(f"Creating recovery checkpoint: {TAG_NAME}")
    GitCommand.run(f"git tag -a {TAG_NAME} -m \"{TAG_MESSAGE}\"")

    # Optionally push the tag
    if PUSH_TO_REMOTE:
        push_response = input(f"Push recovery checkpoint {TAG_NAME} to remote? (y/n): ")
        if push_response.lower() == 'y':
            GitCommand.run(f"git push {REMOTE_NAME} {TAG_NAME}")

    print(f"Recovery checkpoint {TAG_NAME} created. Use MODE='recover' to return to this point later.")


def recover_to_point():
    """Recover to a previous tag or commit"""
    if not RECOVERY_POINT:
        # List available tags and recent commits
        print("Available recovery points:")
        print("\nTags:")
        GitCommand.run("git tag")
        print("\nRecent commits:")
        GitCommand.run("git log --oneline -n 10")
        return

    # Check if we're recovering to a tag or commit
    print(f"Recovering to checkpoint: {RECOVERY_POINT}")
    print("WARNING: This will put you in 'detached HEAD' state.")

    # Switch to the recovery point
    GitCommand.run(f"git checkout {RECOVERY_POINT}")

    # Create a recovery branch
    branch_name = f"recovery-from-{RECOVERY_POINT}"
    create_branch = input(f"Create new branch '{branch_name}' from this recovery point? (y/n): ")
    if create_branch.lower() == 'y':
        GitCommand.run(f"git checkout -b {branch_name}")
        print(f"Created and switched to new branch: {branch_name}")
    else:
        print("You are now in 'detached HEAD' state.")
        print("If you want to make changes, you should create a new branch.")


def main():
    """Main function to run the script"""
    print("=== Git Automation Script for Trading Strategy Development ===")

    # Set up the repository
    setup_repo()

    # Check the current repository status
    print("Current git status:")
    GitCommand.run("git status")

    # Perform actions based on mode
    if MODE == "commit":
        print("\n--- Simple Commit Mode ---")
        commit_changes()

    elif MODE == "feature":
        print("\n--- Feature Branch Mode ---")
        create_feature_branch()
        commit_changes()

    elif MODE == "complete":
        print("\n--- Complete Feature Mode ---")
        complete_feature()

    elif MODE == "release":
        print("\n--- Release Mode ---")
        current = GitCommand.current_branch()
        if current != MAIN_BRANCH:
            print(f"Warning: You are on branch {current}, not {MAIN_BRANCH}.")
            response = input(f"Switch to {MAIN_BRANCH} before releasing? (y/n): ")
            if response.lower() == 'y':
                GitCommand.run(f"git checkout {MAIN_BRANCH}")

        commit_changes()
        create_release()

    elif MODE == "abandon":
        print("\n--- Abandon Branch Mode ---")
        abandon_branch()

    elif MODE == "revert":
        print("\n--- Revert Changes Mode ---")
        revert_changes()

    elif MODE == "checkpoint":
        print("\n--- Create Checkpoint Mode ---")
        commit_changes()
        create_checkpoint()

    elif MODE == "recover":
        print("\n--- Recover to Checkpoint Mode ---")
        recover_to_point()

    else:
        print(f"Error: Unknown mode '{MODE}'. Please check your configuration.")
        sys.exit(1)

    # Push changes if configured (only for certain modes)
    if MODE in ["commit", "feature", "complete", "release"]:
        push_changes()

    print("\nGit automation completed successfully.")


if __name__ == "__main__":
    main()