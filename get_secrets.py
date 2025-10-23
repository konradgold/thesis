from infisical_sdk import InfisicalSDKClient
import dotenv
import os
# Load variables from .env file

dotenv.load_dotenv(".env_infisical")

# Access environment variables
client_id = os.environ.get("INFISICAL_CLIENT_ID")
client_secret = os.environ.get("INFISICAL_CLIENT_SECRET")

# Initialize client
client = InfisicalSDKClient(
    host="https://eu.infisical.com",
)

# Authenticate using environment variables
client.auth.universal_auth.login(
    client_id=client_id,
    client_secret=client_secret
)

secrets = client.secrets.list_secrets(
    environment_slug="dev",
    project_id="c4429065-4651-4f74-8726-e26b53ea43bb",
    secret_path="/")

    # Write secrets to a fresh .env file
with open('.env', 'w') as env_file:
    for secret in secrets.secrets:
        if hasattr(secret, 'secretKey') and hasattr(secret, 'secretValue'):
            # Write each secret in KEY=VALUE format
            env_file.write(f"{secret.secretKey}={secret.secretValue}\n")

print(f"Successfully wrote {len(secrets.secrets)} secrets to .env file")
