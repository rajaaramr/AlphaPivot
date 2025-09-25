# utils/token_generator.py
"""
Kite Connect Token Generator for the AlphaPivot Trading System.

This script exchanges a one-time request token for a permanent access token
and saves it to the `config.ini` file. This is a command-line utility
that should be run manually during the initial setup of the system.
"""
from __future__ import annotations

import sys
import argparse
import configparser
from kiteconnect import KiteConnect

DEFAULT_INI_PATH = "config.ini"

def main():
    """
    Main function to handle the token exchange process.

    It parses command-line arguments, reads the configuration, exchanges
    the request token for an access token, and writes the new token back
    to the configuration file.
    """
    parser = argparse.ArgumentParser(description="Exchange Kite request_token for access_token")
    parser.add_argument("--request-token", required=True, help="One-time request_token from Kite login redirect")
    parser.add_argument("--ini", default=DEFAULT_INI_PATH, help=f"Path to the configuration file (default: {DEFAULT_INI_PATH})")
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    if not cfg.read(args.ini):
        sys.exit(f"❌ Could not read INI file: {args.ini}")

    try:
        api_key = cfg.get("kite", "api_key")
        api_secret = cfg.get("kite", "api_secret")
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        sys.exit(f"❌ Missing [kite] section or required keys (api_key, api_secret) in INI file: {e}")

    if "your_api_key" in api_key or "your_api_secret" in api_secret:
        sys.exit("❌ Please replace the placeholder API key and secret in your config file before generating a token.")

    try:
        kite = KiteConnect(api_key=api_key)
        session = kite.generate_session(args.request_token, api_secret=api_secret)
        access_token = session["access_token"]
    except Exception as e:
        sys.exit(f"❌ Token exchange failed: {e}")

    cfg.set("kite", "access_token", access_token)
    try:
        with open(args.ini, "w") as f:
            cfg.write(f)
    except IOError as e:
        sys.exit(f"❌ Failed to write access token to {args.ini}: {e}")

    print(f"✅ Access token has been successfully updated in {args.ini}.")
    print(f"   New token: {access_token[:8]}...")

if __name__ == "__main__":
    main()