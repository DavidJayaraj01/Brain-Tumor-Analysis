"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    os.environ["TESTING"] = "true"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing"
    yield
    os.environ.pop("TESTING", None)
