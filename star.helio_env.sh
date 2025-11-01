chmod +x scripts/install.sh
./scripts/install.sh
source helio_env/bin/activate
python scripts/setup_apis.py
python app/main.py
