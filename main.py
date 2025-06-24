#main.py
#BuiltWithLove by @papitx0
import subprocess
import sys

def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "entrypoint.py"])

if __name__ == "__main__":
    run_streamlit()