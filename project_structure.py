# Create the project directory structure and first batch of files
import os

# Create directory structure
directories = [
    'backend',
    'backend/agents',
    'backend/models',
    'backend/utils', 
    'backend/data',
    'backend/config',
    'frontend',
    'tests',
    'docs'
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

print("\nFirst batch of files (1-5):")
print("1. README.md")
print("2. requirements.txt") 
print("3. backend/main.py")
print("4. backend/config/settings.py")
print("5. backend/models/data_models.py")