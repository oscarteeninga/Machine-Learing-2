python3 -m venv env
source env/bin/activate
python3 -m pip install-r requirements.txt
python3 -m ipykernel install --user --name=env
jupyter notebook
