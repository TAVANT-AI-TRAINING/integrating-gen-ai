"""
External API Tool Demo - Geolocation

A FastAPI application demonstrating LangChain tool integration with external APIs.
Shows how to create tools that fetch data from external services.
"""

import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.tools import tool
import uvicorn

# --- Step 1: Load Environment Variables ---
load_dotenv()
APIIP_API_KEY = os.getenv("APIIP_API_KEY")
if not APIIP_API_KEY:
    raise ValueError("APIIP_API_KEY is not set")


# --- Step 2: Define Response Models ---
class LocationInfo(BaseModel):
    """Location information from IP address"""
    ip: str
    city: str
    region: str
    country: str


class LocationResponse(BaseModel):
    """Response model for location tool"""
    tool_name: str
    description: str
    result: LocationInfo


class UserLocationToolResponse(BaseModel):
    """Response model for raw tool result"""
    tool_name: str
    description: str
    result: dict


class APIInfo(BaseModel):
    """Response model for the root endpoint"""
    title: str
    description: str
    version: str
    endpoints: list[str]


# --- Step 3: Initialize FastAPI App ---
app = FastAPI(
    title="LangChain Geolocation Tool",
    version="1.0.0",
    description="Demonstrates LangChain tool integration with external APIs"
)

# --- Step 4: Define the Tool ---
@tool
def get_user_location() -> str:
    """
    Fetches the user's geographical location based on their IP address.
    Use this tool when a user's country, city, or region is needed to answer a question.
    This tool does not require any input.
    """
    if not APIIP_API_KEY:
        return json.dumps({"error": "API key for apiip.net is not configured."})

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


# --- Step 5: Root Endpoint ---
@app.get("/", response_model=APIInfo)
async def root():
    """API information and available endpoints"""
    return APIInfo(
        title="LangChain Geolocation Tool",
        description="Demonstrates LangChain tool integration with external APIs",
        version="1.0.0",
        endpoints=["/", "/get_location", "/docs", "/openapi.json"]
    )


# --- Step 6: Test Tool via FastAPI Endpoint ---
@app.get("/get_location", response_model=UserLocationToolResponse)
async def get_location():
    """
    Invoke the get_user_location tool and return location information.
    
    Returns:
        UserLocationToolResponse containing tool metadata and location data
    """
    try:
        result = get_user_location.invoke({})
        result_dict = json.loads(result)
        return UserLocationToolResponse(
            tool_name=get_user_location.name,
            description=get_user_location.description,
            result=result_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
