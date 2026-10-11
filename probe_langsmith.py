import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
run = client.create_run(
    name="probe",
    run_type="chain",
    inputs={},
    project_name="boreas-agent",
)
print("created:", run.id)
client.update_run(
    run_id=run.id,
    end_time=datetime.now(timezone.utc),
)
print("updated ok")
