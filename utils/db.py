# utils/db.py
"""
Centralized PostgreSQL/TimescaleDB connection helper.

This module provides a standardized way to connect to the PostgreSQL database,
ensuring that all connections are configured correctly and use UTC as the
timezone. It is designed to work with a context manager for safe and
reliable database connections.
"""
from __future__ import annotations

import os
import psycopg2
from psycopg2 import sql
from typing import Optional

from .configs import get_db_config

def get_db_connection(*, autocommit: bool = False, config_path: str = "config.ini"):
    """
    Return a psycopg2 connection to PostgreSQL/TimescaleDB.

    This function provides a robust and centralized way to establish a database
    connection. It supports configuration from both environment variables
    (DATABASE_URL) and the main `config.ini` file.

    - Prefers the DATABASE_URL environment variable if set.
    - Otherwise, uses the [postgres] section from the `config.ini` file.
    - Sets the session timezone to UTC to ensure consistent timestamp handling.
    - Applies the search_path if it is provided in the configuration.
    - Can be used as a context manager (e.g., `with get_db_connection() as conn:`).

    Args:
        autocommit: If True, the connection will be in autocommit mode.
        config_path: The path to the configuration file.

    Returns:
        A psycopg2 connection object.

    Raises:
        RuntimeError: If the database configuration is missing or if the
                      connection fails.
    """
    dsn = os.environ.get("DATABASE_URL")
    params = {}
    if not dsn:
        try:
            params = get_db_config(path=config_path)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Database configuration error: {e}") from e

        required = ("host", "port", "dbname", "user", "password")
        if not all(params.get(k) for k in required):
            raise RuntimeError(
                "PostgreSQL config is missing required fields. "
                "Ensure [postgres] section in config.ini is complete or set DATABASE_URL."
            )

    try:
        if dsn:
            conn = psycopg2.connect(dsn)
        else:
            conn = psycopg2.connect(
                host=params["host"],
                port=params["port"],
                dbname=params["dbname"],
                user=params["user"],
                password=params["password"],
                sslmode=params.get("sslmode"),
            )

        conn.autocommit = autocommit

        # Ensure UTC timestamps across the session
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC';")
            sp = params.get("search_path")
            if sp:
                cur.execute(sql.SQL("SET search_path TO {};").format(sql.SQL(sp)))

        return conn

    except Exception as e:
        raise RuntimeError(f"Failed to connect to PostgreSQL: {e}") from e