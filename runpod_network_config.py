#!/usr/bin/env python3
"""
RunPod Network Configuration and Port Management
Handles port configuration, proxy compatibility, and networking setup for RunPod deployment
"""

import os
import json
import socket
import logging
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class RunPodNetworkConfig:
    """Manages network configuration for RunPod deployment"""
    
    def __init__(self):
        self.workspace_dir = Path("/workspace")
        self.config_dir = self.workspace_dir / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # Default port configuration
        self.port_config = {
            "primary_port": 7860,      # Main Gradio interface
            "health_port": 8080,       # Health check endpoint
            "api_port": 8000,          # REST API (if needed)
            "websocket_port": 7860,    # WebSocket (same as primary for RunPod compatibility)
            "monitoring_port": 9090    # Monitoring dashboard (internal)
        }
        
        # RunPod specific settings
        self.runpod_settings = {
            "proxy_compatible": True,
            "cors_enabled": True,
            "websocket_upgrade": True,
            "max_connections": 100,
            "timeout_seconds": 300
        }
    
    def check_port_availability(self, port: int, host: str = "0.0.0.0") -> bool:
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = sock.bind((host, port))
                logger.info(f"Port {port} is available")
                return True
        except OSError as e:
            logger.warning(f"Port {port} is not available: {e}")
            return False
    
    def find_available_port(self, start_port: int, end_port: int = None) -> Optional[int]:
        """Find an available port in the given range"""
        if end_port is None:
            end_port = start_port + 100
        
        for port in range(start_port, end_port + 1):
            if self.check_port_availability(port):
                return port
        
        return None
    
    def configure_ports(self) -> Dict[str, int]:
        """Configure and validate all required ports"""
        logger.info("Configuring network ports...")
        
        configured_ports = {}
        
        # Check primary port (7860 - RunPod standard)
        if self.check_port_availability(self.port_config["primary_port"]):
            configured_ports["primary"] = self.port_config["primary_port"]
        else:
            # Find alternative port
            alt_port = self.find_available_port(7861, 7900)
            if alt_port:
                configured_ports["primary"] = alt_port
                logger.warning(f"Primary port 7860 not available, using {alt_port}")
            else:
                raise RuntimeError("No available ports found for primary service")
        
        # Check health port
        if self.check_port_availability(self.port_config["health_port"]):
            configured_ports["health"] = self.port_config["health_port"]
        else:
            alt_port = self.find_available_port(8081, 8100)
            if alt_port:
                configured_ports["health"] = alt_port
                logger.warning(f"Health port 8080 not available, using {alt_port}")
            else:
                logger.warning("No available port for health service")
                configured_ports["health"] = None
        
        # API port (optional)
        if self.check_port_availability(self.port_config["api_port"]):
            configured_ports["api"] = self.port_config["api_port"]
        else:
            configured_ports["api"] = None
            logger.info("API port not configured (optional)")
        
        return configured_ports
    
    def create_nginx_config(self, ports: Dict[str, int]) -> str:
        """Create nginx configuration for reverse proxy (if needed)"""
        nginx_config = f"""
# Nginx configuration for FireRedTTS2 on RunPod
upstream fireredtts2_app {{
    server 127.0.0.1:{ports['primary']};
}}

upstream health_service {{
    server 127.0.0.1:{ports['health']};
}}

server {{
    listen 80;
    server_name _;
    
    # Main application
    location / {{
        proxy_pass http://fireredtts2_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering off;
        proxy_request_buffering off;
    }}
    
    # Health check endpoint
    location /health {{
        proxy_pass http://health_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Quick health checks
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
    }}
    
    # Static files (if any)
    location /static/ {{
        alias /workspace/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }}
}}
"""
        return nginx_config
    
    def create_runpod_proxy_config(self, ports: Dict[str, int]) -> Dict[str, any]:
        """Create RunPod proxy configuration"""
        proxy_config = {
            "version": "1.0",
            "services": {
                "main": {
                    "port": ports["primary"],
                    "protocol": "http",
                    "path": "/",
                    "websocket": True,
                    "cors": True,
                    "timeout": 300
                }
            }
        }
        
        if ports.get("health"):
            proxy_config["services"]["health"] = {
                "port": ports["health"],
                "protocol": "http",
                "path": "/health",
                "websocket": False,
                "cors": True,
                "timeout": 30
            }
        
        return proxy_config
    
    def setup_firewall_rules(self, ports: Dict[str, int]) -> bool:
        """Setup firewall rules for the configured ports"""
        logger.info("Configuring firewall rules...")
        
        try:
            # Check if ufw is available
            result = subprocess.run(["which", "ufw"], capture_output=True)
            if result.returncode != 0:
                logger.info("UFW not available, skipping firewall configuration")
                return True
            
            # Allow configured ports
            for service, port in ports.items():
                if port:
                    cmd = f"ufw allow {port}/tcp"
                    result = subprocess.run(cmd.split(), capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info(f"Allowed port {port} for {service}")
                    else:
                        logger.warning(f"Failed to configure firewall for port {port}: {result.stderr}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Firewall configuration failed: {e}")
            return False
    
    def create_environment_file(self, ports: Dict[str, int]) -> None:
        """Create environment file with port configuration"""
        env_vars = {
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "GRADIO_SERVER_PORT": str(ports["primary"]),
            "HEALTH_CHECK_PORT": str(ports.get("health", 8080)),
            "API_PORT": str(ports.get("api", 8000)),
            "RUNPOD_PROXY_COMPATIBLE": "true",
            "CORS_ENABLED": "true",
            "WEBSOCKET_ENABLED": "true"
        }
        
        env_file = self.workspace_dir / ".env.network"
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Network environment file created: {env_file}")
    
    def test_network_connectivity(self, ports: Dict[str, int]) -> Dict[str, bool]:
        """Test network connectivity for configured ports"""
        logger.info("Testing network connectivity...")
        
        connectivity_results = {}
        
        for service, port in ports.items():
            if port is None:
                connectivity_results[service] = False
                continue
            
            try:
                # Test local connectivity
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    result = sock.connect_ex(("127.0.0.1", port))
                    connectivity_results[service] = (result == 0)
                    
                if connectivity_results[service]:
                    logger.info(f"✓ {service} service connectivity test passed (port {port})")
                else:
                    logger.warning(f"✗ {service} service connectivity test failed (port {port})")
                    
            except Exception as e:
                logger.error(f"Connectivity test failed for {service}: {e}")
                connectivity_results[service] = False
        
        return connectivity_results
    
    def generate_access_urls(self, ports: Dict[str, int], runpod_id: str = None) -> Dict[str, str]:
        """Generate access URLs for RunPod deployment"""
        urls = {}
        
        if runpod_id:
            # RunPod proxy URLs
            base_url = f"https://{runpod_id}-{{port}}.proxy.runpod.net"
            urls["main_app"] = base_url.format(port=ports["primary"])
            
            if ports.get("health"):
                urls["health_check"] = f"{base_url.format(port=ports['health'])}/health"
            
            if ports.get("api"):
                urls["api"] = base_url.format(port=ports["api"])
        else:
            # Local URLs (for testing)
            urls["main_app"] = f"http://localhost:{ports['primary']}"
            
            if ports.get("health"):
                urls["health_check"] = f"http://localhost:{ports['health']}/health"
            
            if ports.get("api"):
                urls["api"] = f"http://localhost:{ports['api']}"
        
        return urls
    
    def save_network_config(self, ports: Dict[str, int], urls: Dict[str, str]) -> None:
        """Save complete network configuration to file"""
        config = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "ports": ports,
            "urls": urls,
            "runpod_settings": self.runpod_settings,
            "environment_variables": {
                "GRADIO_SERVER_NAME": "0.0.0.0",
                "GRADIO_SERVER_PORT": str(ports["primary"]),
                "HEALTH_CHECK_PORT": str(ports.get("health", 8080)),
                "CORS_ENABLED": "true",
                "WEBSOCKET_ENABLED": "true"
            }
        }
        
        config_file = self.config_dir / "network_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Network configuration saved to: {config_file}")
    
    def setup_complete_network_config(self, runpod_id: str = None) -> Dict[str, any]:
        """Complete network configuration setup"""
        logger.info("Setting up complete network configuration...")
        
        try:
            # Configure ports
            ports = self.configure_ports()
            logger.info(f"Configured ports: {ports}")
            
            # Setup firewall (optional)
            self.setup_firewall_rules(ports)
            
            # Create environment file
            self.create_environment_file(ports)
            
            # Generate access URLs
            urls = self.generate_access_urls(ports, runpod_id)
            logger.info(f"Access URLs: {urls}")
            
            # Save configuration
            self.save_network_config(ports, urls)
            
            # Create RunPod proxy config
            proxy_config = self.create_runpod_proxy_config(ports)
            
            result = {
                "success": True,
                "ports": ports,
                "urls": urls,
                "proxy_config": proxy_config,
                "message": "Network configuration completed successfully"
            }
            
            logger.info("Network configuration setup completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Network configuration setup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Network configuration setup failed"
            }

def main():
    """Main function for network configuration"""
    logging.basicConfig(level=logging.INFO)
    
    network_config = RunPodNetworkConfig()
    
    # Get RunPod ID from environment if available
    runpod_id = os.environ.get("RUNPOD_POD_ID")
    
    result = network_config.setup_complete_network_config(runpod_id)
    
    if result["success"]:
        print("Network Configuration Summary:")
        print("=" * 40)
        print(f"Primary Port: {result['ports']['primary']}")
        print(f"Health Port: {result['ports'].get('health', 'Not configured')}")
        print(f"API Port: {result['ports'].get('api', 'Not configured')}")
        print("\nAccess URLs:")
        for name, url in result['urls'].items():
            print(f"  {name}: {url}")
        print("=" * 40)
        return 0
    else:
        print(f"Network configuration failed: {result['message']}")
        return 1

if __name__ == "__main__":
    exit(main())