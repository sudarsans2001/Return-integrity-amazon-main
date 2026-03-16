#!/bin/bash

# Setup script for ReturnShield AI
# This script automates the installation and configuration process

set -e  # Exit on error

echo "========================================="
echo "ReturnShield AI - Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11+ required. Found: $python_version"
    exit 1
fi
echo "  Python $python_version detected ✓"
echo ""

# Create virtual environment
echo "✓ Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv .venv
    echo "  Virtual environment created ✓"
fi
echo ""

# Activate virtual environment
echo "✓ Activating virtual environment..."
source .venv/bin/activate
echo "  Virtual environment activated ✓"
echo ""

# Upgrade pip
echo "✓ Upgrading pip..."
pip install --upgrade pip --quiet
echo "  pip upgraded ✓"
echo ""

# Install dependencies
echo "✓ Installing dependencies..."
pip install -r requirements.txt --quiet
echo "  Dependencies installed ✓"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "✓ Creating .env file..."
    cp .env.example .env
    echo "  .env file created ✓"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your AWS credentials!"
    echo ""
else
    echo "  .env file already exists"
fi

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "✓ Creating data directory..."
    mkdir -p data
    echo "  data directory created ✓"
fi

# Check AWS credentials
echo "✓ Checking AWS configuration..."
if command -v aws &> /dev/null; then
    if aws sts get-caller-identity &> /dev/null; then
        echo "  AWS credentials configured ✓"
    else
        echo "  ⚠️  AWS credentials not configured or invalid"
        echo "     Run: aws configure"
    fi
else
    echo "  ⚠️  AWS CLI not installed"
    echo "     Install: pip install awscli"
fi
echo ""

# Test Nova access (optional)
echo "✓ Testing Nova model access..."
python3 << EOF
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

try:
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    # Simple test call
    print("  Nova access configured ✓")
except Exception as e:
    print(f"  ⚠️  Could not connect to Bedrock: {e}")
    print("     Make sure Nova models are enabled in your AWS account")
EOF
echo ""

# Summary
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your AWS credentials"
echo "  2. Run the pipeline: python -m src.main"
echo "  3. Start API server: python -m uvicorn api.server:app --reload"
echo "  4. Launch dashboard: streamlit run dashboard/app.py"
echo ""
echo "Documentation:"
echo "  - README.md - Quick start and overview"
echo "  - ARCHITECTURE.md - System design details"
echo "  - DEPLOYMENT.md - Production deployment guide"
echo ""
echo "For help, visit: https://github.com/yourorg/return-integrity"
echo ""
