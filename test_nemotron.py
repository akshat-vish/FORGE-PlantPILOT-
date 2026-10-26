
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise RuntimeError(
        "NVIDIA_API_KEY not found. Check your .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

# Simulated factory telemetry.
# These are deliberately fictional test readings.
FACTORY_DATA = {
    "station_a": {
        "name": "Conveyor Station A",
        "conveyor_speed_pct": 70,
        "normal_speed_pct": 100,
        "output_units_per_hour": 80,
        "normal_output_units_per_hour": 100,
    },
    "station_b": {
        "name": "Motor Station B",
        "motor_temperature_c": 88,
        "normal_temperature_c": 65,
        "output_units_per_hour": 80,
        "normal_output_units_per_hour": 100,
        "motor_current_a": None,
    },
    "station_c": {
        "name": "Assembly Station C",
        "output_units_per_hour": 82,
        "normal_output_units_per_hour": 100,
    },
}


# This is the tool Nemotron is allowed to call.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_station_readings",
            "description": (
                "Retrieve actual simulated telemetry for a factory "
                "station. Use this before making claims about its "
                "sensor readings. Valid station IDs: station_a, "
                "station_b, station_c."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station_id": {
                        "type": "string",
                        "enum": [
                            "station_a",
                            "station_b",
                            "station_c",
                        ],
                        "description": "The station to inspect.",
                    }
                },
                "required": ["station_id"],
                "additionalProperties": False,
            },
        },
    }
]


def get_station_readings(station_id):
    """Return data from our simulated factory."""
    if station_id not in FACTORY_DATA:
        return {"error": "Unknown station."}

    return FACTORY_DATA[station_id]


def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are PlantPilot, a human-supervised industrial "
                "operations AI copilot. Investigate factory problems "
                "using available tools. Never invent sensor readings. "
                "Distinguish observations from hypotheses. Explain "
                "uncertainty and recommend the next useful check. "
                "Do not claim a root cause is proven without sufficient "
                "evidence. Never initiate physical actions."
            ),
        },
        {
            "role": "user",
            "content": (
                "The factory is producing below its target. Investigate "
                "the situation. Inspect the relevant stations using "
                "your available tool, compare their readings with normal "
                "values, identify plausible explanations, and state "
                "what additional evidence is needed. Do not assume that "
                "an abnormal reading proves the root cause."
            ),
        },
    ]

    print("PlantPilot is investigating the simulated factory...\n")

    # Allow multiple investigation steps, with a safety limit.
    for step in range(6):
        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-super-vl-120b-a12b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=2000,
        )

        assistant_message = response.choices[0].message
        print("Finish reason:", response.choices[0].finish_reason)

        # Preserve the assistant response, including any tool calls.
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        tool_calls = assistant_message.tool_calls or []

        # If the model has finished investigating, show its answer.
        if not tool_calls:
            print("=== PLANTPILOT INVESTIGATION REPORT ===\n")
            print(assistant_message.content or "(No text response)")
            print("\n=== INVESTIGATION COMPLETE ===")
            return

        # Execute only tools that we explicitly implemented.
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "get_station_readings":
                    result = get_station_readings(
                        arguments.get("station_id", "")
                    )
                else:
                    result = {"error": "Tool not implemented."}

            except (json.JSONDecodeError, TypeError):
                result = {"error": "Invalid tool arguments."}

            print(f"Tool called: {function_name}")
            print(f"Actual tool result: {json.dumps(result)}\n")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    print(
        "Investigation stopped at the step limit. "
        "The model did not finish within the allowed number of steps."
    )


if __name__ == "__main__":
    main()
