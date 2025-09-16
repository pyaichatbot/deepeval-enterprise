# DeepEval Docker Setup Guide

This guide provides instructions for running the DeepEval framework using Docker.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- At least 4GB of free disk space

## Quick Start

### Option 1: Automated Setup

Run the setup script:

```bash
./docker-setup.sh
```

This script will:
- Check Docker installation
- Create necessary directories
- Build the Docker image
- Start the services

### Option 2: Manual Setup

1. **Create directories:**
   ```bash
   mkdir -p data logs
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t deepeval:latest .
   ```

3. **Start the services:**
   ```bash
   docker-compose up -d
   ```

## Accessing the Application

Once the services are running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **API Endpoints**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## Configuration

### Environment Variables

You can customize the configuration by setting environment variables in the `docker-compose.yml` file:

```yaml
environment:
  - DEEPEVAL_DATABASE_URL=sqlite:///app/data/deepeval.db
  - DEEPEVAL_LOG_LEVEL=INFO
  - DEEPEVAL_DEBUG=false
  - DEEPEVAL_MAX_WORKERS=10
  - DEEPEVAL_TIMEOUT=300
  - DEEPEVAL_PARALLEL_EXECUTION=true
  - DEEPEVAL_SAVE_RESULTS=true
  - DEEPEVAL_ENABLE_CACHE=true
  - DEEPEVAL_ENABLE_SECURITY=true
  - DEEPEVAL_MAX_CONCURRENT_EVALUATIONS=5
```

### Database Options

#### SQLite (Default)
The default configuration uses SQLite, which is suitable for development and small deployments.

#### PostgreSQL (Production)
For production deployments, uncomment the PostgreSQL service in `docker-compose.yml`:

```yaml
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: deepeval
    POSTGRES_USER: deepeval
    POSTGRES_PASSWORD: deepeval_password
  ports:
    - "5432:5432"
  volumes:
    - postgres_data:/var/lib/postgresql/data
  restart: unless-stopped
```

Then update the database URL:
```yaml
environment:
  - DEEPEVAL_DATABASE_URL=postgresql://deepeval:deepeval_password@postgres:5432/deepeval
```

### Redis Caching (Optional)

For improved performance, you can enable Redis caching by uncommenting the Redis service:

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  restart: unless-stopped
  command: redis-server --appendonly yes
```

## Management Commands

### View Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f deepeval
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### Update Services
```bash
docker-compose pull
docker-compose up -d
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: This will delete all data)
docker-compose down -v

# Remove images
docker rmi deepeval:latest
```

## Troubleshooting

### Build Failures

If the Docker build fails due to space constraints:

1. **Free up Docker space:**
   ```bash
   docker system prune -a
   ```

2. **Use minimal requirements:**
   The Dockerfile uses `requirements-minimal.txt` which includes only essential dependencies.

3. **Build without cache:**
   ```bash
   docker build --no-cache -t deepeval:latest .
   ```

### Port Conflicts

If port 8000 is already in use, modify the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Permission Issues

If you encounter permission issues with volumes:

```bash
sudo chown -R $USER:$USER data logs
```

### Health Check Failures

If the health check fails:

1. Check if the service is running:
   ```bash
   docker-compose ps
   ```

2. Check the logs:
   ```bash
   docker-compose logs deepeval
   ```

3. Test the health endpoint manually:
   ```bash
   curl http://localhost:8000/health
   ```

## Development

### Running in Development Mode

For development with live reloading:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Accessing the Container

To access the running container:

```bash
docker-compose exec deepeval bash
```

### Running Tests

To run tests inside the container:

```bash
docker-compose exec deepeval python -m pytest
```

## Production Deployment

### Security Considerations

1. **Change default passwords** for PostgreSQL and Redis
2. **Use environment files** for sensitive configuration
3. **Enable HTTPS** with a reverse proxy (nginx/traefik)
4. **Set up proper logging** and monitoring
5. **Use secrets management** for API keys

### Scaling

To scale the application:

```bash
docker-compose up -d --scale deepeval=3
```

### Monitoring

Consider adding monitoring services:

```yaml
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Support

For issues and questions:

1. Check the logs: `docker-compose logs -f`
2. Verify configuration in `docker-compose.yml`
3. Test individual components
4. Check Docker and Docker Compose versions

## File Structure

```
.
├── Dockerfile                 # Main Docker image definition
├── docker-compose.yml        # Docker Compose configuration
├── docker-setup.sh          # Automated setup script
├── requirements-minimal.txt  # Minimal Python dependencies
├── .dockerignore            # Files to ignore during build
├── data/                    # Persistent data directory
└── logs/                    # Log files directory
```
