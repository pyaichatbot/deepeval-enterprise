#!/bin/bash

# DeepEval Docker Setup Script
# This script helps set up and run the DeepEval framework in Docker

set -e

echo "🐳 DeepEval Docker Setup"
echo "========================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs

# Set permissions
chmod 755 data logs

echo "✅ Directories created"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t deepeval:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed. This might be due to space constraints."
    echo "💡 Try running: docker system prune -a to free up space"
    echo "💡 Or use the alternative setup method below"
    exit 1
fi

# Start the services
echo "🚀 Starting services..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Services started successfully"
    echo ""
    echo "🌐 DeepEval API is now running at:"
    echo "   - API: http://localhost:8000"
    echo "   - Docs: http://localhost:8000/docs"
    echo "   - Health: http://localhost:8000/health"
    echo ""
    echo "📊 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Failed to start services"
    exit 1
fi
