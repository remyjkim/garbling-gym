# ABOUTME: Pytest configuration and shared fixtures for all tests
# ABOUTME: Loads .env so OPENROUTER_API_KEY and other vars are available in live tests

from dotenv import load_dotenv

load_dotenv()
