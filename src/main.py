import subprocess
import sys

def run_script(script_name):
    """Run a Python script"""
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print(f"Error running {script_name}")
        return False

def main():
    print("🚀 Starting Travel Planner AI System")
    
    # Step 1: Install dependencies
    print("1. Installing dependencies...")
    if not run_script("install_dependencies.py"):
        print("Please install dependencies manually")
        return
    
    # Step 2: Preprocess data
    print("2. Preprocessing data...")
    if not run_script("data_preprocessing.py"):
        print("Data preprocessing failed")
        return
    
    # Step 3: Train model
    print("3. Training model...")
    if not run_script("model_training.py"):
        print("Model training failed, using base model")
    
    # Step 4: Build RAG system
    print("4. Building RAG system...")
    if not run_script("rag_system.py"):
        print("RAG system building failed")
        return
    
    # Step 5: Run the application
    print("5. Starting application...")
    run_script("run_app.py")

if __name__ == "__main__":
    main()
