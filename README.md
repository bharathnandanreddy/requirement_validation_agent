python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt


gcloud auth application-default login

export GOOGLE_API_KEY="AIzaSyDG8Azu4oGygUqs2-A_QECBDKIH9fSWkdg"

python3 process_unstructured_requirements.py