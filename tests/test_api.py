"""Tests for the FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from plantpilot.main import app
from plantpilot.schemas import IncidentCreate, InvestigationCreate
from plantpilot.config import reset_config
from plantpilot.storage import reset_database


@pytest.fixture
def client():
    """Create a test client."""
    reset_config()
    reset_database()
    return TestClient(app)


def test_root_endpoint(client):
    """Test GET / returns application info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == "PlantPilot"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_health_endpoint(client):
    """Test GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "ok"
    assert "provider" not in data  # Not checked by default


def test_health_endpoint_with_provider_check(client):
    """Test GET /health?check_provider=true."""
    response = client.get("/health?check_provider=true")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    # Provider check may succeed or fail depending on API key validity


def test_create_incident(client):
    """Test POST /api/v1/incidents creates an incident."""
    payload = {
        "description": "Production line stopped unexpectedly",
        "severity": "high",
        "category": "operational",
        "metadata": {"station": "A"},
    }
    response = client.post("/api/v1/incidents", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["description"] == payload["description"]
    assert data["severity"] == payload["severity"]
    assert data["category"] == payload["category"]
    assert data["status"] == "pending"
    assert "id" in data
    assert data["id"].startswith("inc_")


def test_create_incident_validation(client):
    """Test POST /api/v1/incidents validates input."""
    payload = {
        "description": "",  # Too short
        "severity": "high",
    }
    response = client.post("/api/v1/incidents", json=payload)
    assert response.status_code == 422  # Validation error


def test_get_incident(client):
    """Test GET /api/v1/incidents/{id} retrieves an incident."""
    # Create an incident first
    payload = {
        "description": "Temperature sensor anomaly",
        "severity": "medium",
    }
    create_response = client.post("/api/v1/incidents", json=payload)
    incident_id = create_response.json()["id"]

    # Retrieve it
    response = client.get(f"/api/v1/incidents/{incident_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == incident_id
    assert data["description"] == payload["description"]


def test_get_incident_not_found(client):
    """Test GET /api/v1/incidents/{id} returns 404 for unknown ID."""
    response = client.get("/api/v1/incidents/inc_nonexistent")
    assert response.status_code == 404


def test_list_incidents(client):
    """Test GET /api/v1/incidents lists incidents."""
    import uuid
    unique_suffix = uuid.uuid4().hex[:8]
    # Create 3 incidents with unique descriptions
    for i in range(3):
        payload = {
            "description": f"Test incident {unique_suffix} #{i}",
            "severity": "low",
        }
        client.post("/api/v1/incidents", json=payload)

    # List them
    response = client.get("/api/v1/incidents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # At least 3 incidents with our suffix should be present
    matching = [d for d in data if f"Test incident {unique_suffix}" in d["description"]]
    assert len(matching) >= 3  # Each created incident should be in the list


def test_list_incidents_pagination(client):
    """Test GET /api/v1/incidents supports pagination."""
    # Create 5 incidents
    for i in range(5):
        payload = {
            "description": f"Incident {i}",
            "severity": "low",
        }
        client.post("/api/v1/incidents", json=payload)

    # Get first page
    response = client.get("/api/v1/incidents?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # Get second page
    response = client.get("/api/v1/incidents?limit=2&offset=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_list_incidents_validation(client):
    """Test GET /api/v1/incidents validates pagination parameters."""
    response = client.get("/api/v1/incidents?limit=0")
    assert response.status_code == 400

    response = client.get("/api/v1/incidents?limit=300")
    assert response.status_code == 400

    response = client.get("/api/v1/incidents?offset=-1")
    assert response.status_code == 400
