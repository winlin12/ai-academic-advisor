from app.services.ollama_client import is_local_model_endpoint


def test_local_model_endpoint_allows_loopback_private_and_tailscale():
    assert is_local_model_endpoint("http://localhost:11434")
    assert is_local_model_endpoint("http://127.0.0.1:11434")
    assert is_local_model_endpoint("http://192.168.1.20:11434")
    assert is_local_model_endpoint("http://10.0.0.5:11434")
    assert is_local_model_endpoint("http://100.105.151.35:11434")
    assert is_local_model_endpoint("http://ollama:11434")


def test_local_model_endpoint_rejects_public_cloud_hosts():
    assert not is_local_model_endpoint("https://api.openai.com/v1")
    assert not is_local_model_endpoint("https://generativelanguage.googleapis.com")
    assert not is_local_model_endpoint("ftp://localhost:11434")
