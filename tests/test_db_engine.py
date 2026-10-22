"""The application engines check a pooled connection before using it.

Without this, the first request after a database restart used a connection the server had closed
and failed with "connection is closed" (a 500).
"""

from app.db.engine import engine, sync_engine


def test_both_engines_ping_a_connection_before_using_it():
    assert engine.pool._pre_ping is True
    assert sync_engine.pool._pre_ping is True
