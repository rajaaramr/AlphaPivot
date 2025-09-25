# login_launcher.py
from utils.kite_utils import load_config, generate_login_url

config = load_config()
api_key = config["kite"]["api_key"]

url = generate_login_url(api_key)
print("🔗 Click this URL and login to Zerodha:", url)
