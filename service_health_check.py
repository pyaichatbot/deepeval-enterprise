#!/usr/bin/env python3
"""
Service Health Check Script for Docker FastAPI Services

This script checks if all your Docker services are running and healthy
before running the RAG evaluation.
"""

import asyncio
import aiohttp
import sys
from typing import Dict, List, Tuple
import time

# Service configuration
SERVICES = {
    "pdf_extractor": {
        "url": "http://localhost:8001",
        "health_endpoint": "/health",
        "required": True
    },
    "code_search": {
        "url": "http://localhost:8002", 
        "health_endpoint": "/health",
        "required": True
    },
    "document_retriever": {
        "url": "http://localhost:8003",
        "health_endpoint": "/health", 
        "required": True
    },
    "rag_pipeline": {
        "url": "http://localhost:8004",
        "health_endpoint": "/health",
        "required": True
    }
}

class ServiceHealthChecker:
    """Check health of Docker FastAPI services"""
    
    def __init__(self, services: Dict[str, Dict], timeout: int = 10):
        self.services = services
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_service_health(self, service_name: str, service_config: Dict) -> Tuple[bool, str]:
        """Check health of a single service"""
        url = f"{service_config['url']}{service_config['health_endpoint']}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return True, "Healthy"
                else:
                    return False, f"HTTP {response.status}"
        except asyncio.TimeoutError:
            return False, "Timeout"
        except aiohttp.ClientConnectorError:
            return False, "Connection refused"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    async def check_all_services(self) -> Dict[str, Tuple[bool, str]]:
        """Check health of all services"""
        results = {}
        
        print("🔍 Checking service health...")
        print("-" * 50)
        
        for service_name, service_config in self.services.items():
            print(f"Checking {service_name}...", end=" ")
            
            is_healthy, message = await self.check_service_health(service_name, service_config)
            results[service_name] = (is_healthy, message)
            
            if is_healthy:
                print("✅ Healthy")
            else:
                print(f"❌ {message}")
        
        return results
    
    def print_summary(self, results: Dict[str, Tuple[bool, str]]):
        """Print health check summary"""
        print("\n📊 Health Check Summary:")
        print("-" * 50)
        
        healthy_count = 0
        total_count = len(results)
        
        for service_name, (is_healthy, message) in results.items():
            status = "✅ Healthy" if is_healthy else f"❌ {message}"
            required = "Required" if self.services[service_name]["required"] else "Optional"
            print(f"{service_name:20} | {status:15} | {required}")
            
            if is_healthy:
                healthy_count += 1
        
        print("-" * 50)
        print(f"Healthy: {healthy_count}/{total_count}")
        
        # Check if all required services are healthy
        required_services = [name for name, config in self.services.items() if config["required"]]
        required_healthy = all(results[name][0] for name in required_services)
        
        if required_healthy:
            print("🎉 All required services are healthy!")
            return True
        else:
            print("⚠️  Some required services are not healthy!")
            return False

async def wait_for_services(max_wait_time: int = 300, check_interval: int = 10):
    """Wait for services to become healthy"""
    print(f"⏳ Waiting for services to become healthy (max {max_wait_time}s)...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        async with ServiceHealthChecker(SERVICES) as checker:
            results = await checker.check_all_services()
            all_healthy = checker.print_summary(results)
            
            if all_healthy:
                return True
            
            print(f"\n⏳ Waiting {check_interval}s before next check...")
            await asyncio.sleep(check_interval)
    
    print("❌ Timeout waiting for services to become healthy!")
    return False

async def test_service_endpoints():
    """Test actual service endpoints with sample data"""
    print("\n🧪 Testing service endpoints...")
    print("-" * 50)
    
    test_cases = [
        {
            "service": "pdf_extractor",
            "url": "http://localhost:8001/extract",
            "data": {"pdf_path": "test.pdf"}
        },
        {
            "service": "code_search", 
            "url": "http://localhost:8002/search",
            "data": {"query": "test query"}
        },
        {
            "service": "document_retriever",
            "url": "http://localhost:8003/retrieve", 
            "data": {"query": "test query", "k": 5}
        },
        {
            "service": "rag_pipeline",
            "url": "http://localhost:8004/rag",
            "data": {"query": "test query"}
        }
    ]
    
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for test_case in test_cases:
            service_name = test_case["service"]
            url = test_case["url"]
            data = test_case["data"]
            
            print(f"Testing {service_name}...", end=" ")
            
            try:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print("✅ Working")
                    else:
                        print(f"❌ HTTP {response.status}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

async def main():
    """Main health check function"""
    print("🏥 Docker Service Health Check")
    print("=" * 50)
    
    # Check if services are healthy
    async with ServiceHealthChecker(SERVICES) as checker:
        results = await checker.check_all_services()
        all_healthy = checker.print_summary(results)
    
    if not all_healthy:
        print("\n❌ Not all required services are healthy!")
        print("\nTroubleshooting steps:")
        print("1. Make sure Docker is running")
        print("2. Start your services: docker-compose up -d")
        print("3. Check service logs: docker-compose logs <service-name>")
        print("4. Verify ports are not in use: netstat -tulpn | grep :800")
        
        # Ask if user wants to wait for services
        try:
            response = input("\nWould you like to wait for services to become healthy? (y/n): ")
            if response.lower() in ['y', 'yes']:
                success = await wait_for_services()
                if not success:
                    sys.exit(1)
            else:
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(1)
    
    # Test service endpoints
    await test_service_endpoints()
    
    print("\n🎉 All services are ready for evaluation!")
    print("\nYou can now run: python rag_evaluation_example.py")

if __name__ == "__main__":
    asyncio.run(main())
