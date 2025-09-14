import subprocess
import time
import requests
import threading
from pyngrok import ngrok

def run_streamlit():
    """Run Streamlit app"""
    try:
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"])
    except Exception as e:
        print(f"Error running Streamlit: {e}")

def check_app_status():
    """Check if Streamlit app is running"""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        return response.status_code == 200
    except:
        return False

def setup_ngrok_with_auth():
    """Setup ngrok tunnel with authentication"""
    try:
        # استخدام token المصادقة الخاص بك
        ngrok.set_auth_token("32QCuqces3fFsY2swBD36T0CNBk_5riNg8ZVXmpNt9HGxJ6Wi")
        
        # Create tunnel
        public_url = ngrok.connect(8501, "http")
        print(f"✅ Streamlit is running at: {public_url}")
        return public_url
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
        return "http://localhost:8501"

if __name__ == "__main__":
    print("Starting Travel Planner Application...")
    
    # Start Streamlit in background
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Wait for Streamlit to start
    print("Waiting for Streamlit to start...")
    for i in range(10):
        if check_app_status():
            print("✅ Streamlit app is running!")
            break
        print(f"Waiting... ({i+1}/10)")
        time.sleep(2)
    else:
        print("❌ Streamlit app failed to start")
        exit(1)
    
    # Setup ngrok with authentication
    public_url = setup_ngrok_with_auth()
    
    print(f"Application is accessible at: {public_url}")
    print("Press Ctrl+C to stop the application")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        ngrok.kill()
