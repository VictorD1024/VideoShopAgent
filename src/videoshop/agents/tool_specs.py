from __future__ import annotations

FINAL_ACTION_TOOL = "submit_final_action"

ACTION_TYPES = [
    "delay_recommendation",
    "show_product_card",
    "show_coupon",
    "switch_to_substitute",
    "show_explanation",
]


def video_shop_tool_specs() -> list[dict]:
    """Return OpenAI-compatible tool schemas for VideoShopEnv."""
    return [
        _function_tool(
            "retrieve_candidates",
            "Retrieve candidate products for the current video and user context.",
            {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        _function_tool(
            "rank_products",
            "Rank candidate products by relevance, conversion potential, and risk.",
            {
                "type": "object",
                "properties": {
                    "candidate_product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Product IDs to rank. If omitted, all visible candidates are ranked.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        _function_tool(
            "get_coupon",
            "Check whether a product has a valid coupon for the current user and commerce context.",
            _product_id_schema(),
        ),
        _function_tool(
            "find_substitute",
            "Find a cheaper, lower-risk, in-stock substitute for a product.",
            _product_id_schema(),
        ),
        _function_tool(
            "explain_recommendation",
            "Get grounded evidence for explaining why a product is recommended.",
            _product_id_schema(),
        ),
        _function_tool(
            FINAL_ACTION_TOOL,
            "Submit the final action for this VideoShopEnv step after any needed tool calls.",
            {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ACTION_TYPES},
                    "product_id": {"type": ["string", "null"]},
                    "reason": {"type": "string", "minLength": 1},
                    "reasoning_summary": {
                        "type": "object",
                        "description": "A concise, explicit audit summary of the decision. Do not include hidden chain-of-thought.",
                        "properties": {
                            "observation_facts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Facts read from the observation that affected the decision.",
                            },
                            "evidence_used": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tool results or observed evidence used for the final action.",
                            },
                            "candidate_comparison": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Short comparisons among candidate products or actions.",
                            },
                            "rejected_options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Unsafe, unsupported, or lower-value options that were rejected.",
                            },
                            "decision_rule": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["observation_facts", "evidence_used", "decision_rule"],
                        "additionalProperties": False,
                    },
                },
                "required": ["action_type", "product_id", "reason"],
                "additionalProperties": False,
            },
        ),
    ]


def _product_id_schema() -> dict:
    return {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
        "additionalProperties": False,
    }


def _function_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
