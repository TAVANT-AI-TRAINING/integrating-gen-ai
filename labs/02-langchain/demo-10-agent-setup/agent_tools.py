import json
import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


# --- Step 1: Simulate internal order database ---
ORDER_DATABASE = {
    "ABC-123": {"status": "Shipped", "estimated_delivery": "2024-10-25", "carrier": "FedEx"},
    "DEF-456": {"status": "Processing", "estimated_delivery": "2024-10-28", "carrier": "UPS"},
    "GHI-789": {"status": "Delivered", "estimated_delivery": "2024-10-22", "carrier": "FedEx"},
}

# --- Step 2: Define internal Tool ---
@tool
def get_order_status(order_id: str) -> str:
    """
    Use this tool to retrieve the current status and details of a specific order.
    The input must be a valid order ID string. It returns a JSON string with
    the order's status, carrier, and estimated delivery date.
    If the order ID is not found, it returns an error message.
    """
    order_info = ORDER_DATABASE.get(order_id)
    if order_info:
        return json.dumps(order_info)
    else:
        return json.dumps({"error": "Order ID not found in the system."})
    
# --- Step 3: Define the external Tool ---
@tool
def get_user_location(dummy_input: str = "fetch_location") -> str:
    """
    Fetches the user's geographical location based on their IP address.
    (dummy_input is unused; required for compatibility with ZeroShot agent.)
    """
    APIIP_API_KEY = os.getenv("APIIP_API_KEY")
    if not APIIP_API_KEY:
        return json.dumps({"error": "API key for apiip.net is missing or invalid."})

    url = f"http://apiip.net/api/check?accessKey={APIIP_API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        location_info = {
            "ip": data.get("ip"),
            "city": data.get("city"),
            "region": data.get("regionName"),
            "country": data.get("countryName"),
        }
        return json.dumps(location_info)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Network error fetching location: {e}"})
