"""
Thin connection wrapper around psycopg2.

Kept deliberately simple (no ORM) so that the SQL itself stays visible
and readable -- the point of this project is to demonstrate SQL/Postgres
skills, not to hide them behind an abstraction layer.
"""

import psycopg2
from pgvector.psycopg2 import register_vector
from contextlib import contextmanager

from src.config import config


@contextmanager
def get_connection():
    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
