"""
Quick validation test for the fraud detection project.
Tests that all files are present and code is syntactically correct.
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg):
    print(f"{RED}✗{RESET} {msg}")

def print_info(msg):
    print(f"{YELLOW}ℹ{RESET} {msg}")


def check_project_structure():
    """Verify all expected directories and files exist."""
    print("\n" + "="*60)
    print("PROJECT STRUCTURE VALIDATION")
    print("="*60)

    required_dirs = [
        "src/ingestion",
        "src/etl",
        "src/models",
        "src/utils",
        "data/raw",
        "data/processed",
        "data/curated",
        "config",
        "artifacts/models",
        ".github/workflows",
    ]

    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "Dockerfile",
        "config/config.yaml",
        "src/ingestion/batch_ingest.py",
        "src/etl/bronze_to_silver.py",
        "src/etl/feature_engineering.py",
        "src/models/train.py",
        "src/utils/spark_session.py",
        "src/utils/logger.py",
    ]

    all_good = True

    print("\nChecking directories...")
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Missing directory: {dir_path}")
            all_good = False

    print("\nChecking files...")
    for file_path in required_files:
        if os.path.isfile(file_path):
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"Missing file: {file_path}")
            all_good = False

    return all_good


def check_data():
    """Check if sample data exists."""
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)

    data_file = "data/raw/sample_transactions.csv"

    if os.path.isfile(data_file):
        file_size = os.path.getsize(data_file) / 1024  # KB
        print_success(f"Sample data found: {data_file} ({file_size:.2f} KB)")

        # Count lines
        with open(data_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print_info(f"  Records (including header): {line_count}")
        return True
    else:
        print_error(f"Sample data not found: {data_file}")
        print_info("  Run: python scripts/generate_sample_data.py")
        return False


def check_python_syntax():
    """Verify Python files have valid syntax."""
    print("\n" + "="*60)
    print("PYTHON SYNTAX VALIDATION")
    print("="*60)

    python_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    all_valid = True
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file, 'exec')
            print_success(f"Valid syntax: {py_file}")
        except SyntaxError as e:
            print_error(f"Syntax error in {py_file}: {e}")
            all_valid = False

    return all_valid


def check_git():
    """Check git repository status."""
    print("\n" + "="*60)
    print("GIT REPOSITORY VALIDATION")
    print("="*60)

    if os.path.isdir(".git"):
        print_success("Git repository initialized")

        # Check remote
        import subprocess
        try:
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True,
                text=True,
                check=True
            )
            if "silpa-das-analytics" in result.stdout:
                print_success("Remote repository configured")
                print_info("  " + result.stdout.split('\n')[0])
            else:
                print_error("Remote not configured correctly")
                return False

            # Check branches
            result = subprocess.run(
                ["git", "branch", "-a"],
                capture_output=True,
                text=True,
                check=True
            )
            branch_count = len([b for b in result.stdout.split('\n') if b.strip()])
            print_success(f"Branches: {branch_count}")

            return True

        except subprocess.CalledProcessError:
            print_error("Git command failed")
            return False
    else:
        print_error("Not a git repository")
        return False


def check_readme():
    """Validate README content."""
    print("\n" + "="*60)
    print("README VALIDATION")
    print("="*60)

    readme_path = "README.md"
    if not os.path.isfile(readme_path):
        print_error("README.md not found")
        return False

    with open(readme_path, 'r') as f:
        content = f.read()

    checks = {
        "Title present": "Fraud Detection" in content,
        "Architecture diagram": "Architecture" in content,
        "Technology stack": "Technology Stack" in content,
        "Quick start guide": "Quick Start" in content,
        "Model performance": "Model Performance" in content,
    }

    all_good = True
    for check_name, result in checks.items():
        if result:
            print_success(check_name)
        else:
            print_error(check_name)
            all_good = False

    word_count = len(content.split())
    print_info(f"  Word count: {word_count}")

    return all_good


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("🛡️  FRAUD DETECTION ML PIPELINE - VALIDATION SUITE")
    print("="*60)

    results = {
        "Project Structure": check_project_structure(),
        "Data": check_data(),
        "Python Syntax": check_python_syntax(),
        "Git Repository": check_git(),
        "README": check_readme(),
    }

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print(f"{GREEN}🎉 ALL CHECKS PASSED! Project is ready.{RESET}")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run ingestion: python src/ingestion/batch_ingest.py")
        print("  3. View on GitHub: https://github.com/silpa-das-analytics/fraud-detection-ml-pipeline")
    else:
        print(f"{RED}⚠️  Some checks failed. Please review errors above.{RESET}")

    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
