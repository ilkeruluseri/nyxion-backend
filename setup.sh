#!/bin/bash

# Create base folders
mkdir -p app/models app/routes app/services app/utils tests

# Create files
touch pyproject.toml uv.lock
touch app/__init__.py app/main.py app/config.py app/db.py
touch app/models/__init__.py app/models/user.py
touch app/routes/__init__.py app/routes/user.py
touch app/services/__init__.py app/services/user_service.py
touch app/utils/__init__.py app/utils/auth.py
touch tests/test_basic.py

echo "✅ Project structure created successfully in nyxion-backend/"
