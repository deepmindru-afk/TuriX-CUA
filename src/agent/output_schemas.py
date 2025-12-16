class OutputSchemas:
    """
    A container for all model output schemas and their associated response formats.
    """

    ACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "array",
                "minItems": 0,
                "default": [ { "wait": {} } ], 
                "items": {
                    "type": "object",
                    "properties": {
                    # ----- task finished -----
                    "done": {"type": "object",
                        "properties": {"text": {"type": "string"}}},

                    # ----- typing -----
                    "input_text": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]
                    },

                    # ----- open app -----
                    "open_app": {
                        "type": "object",
                        "properties": {"app_name": {"type": "string"}},
                        "required": ["app_name"]
                    },

                    # ----- AppleScript -----
                    "run_apple_script": {
                        "type": "object",
                        "properties": {"script": {"type": "string"}},
                        "required": ["script"]
                    },

                    # ----- hotkeys -----
                    "Hotkey": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"]
                    },
                    "multi_Hotkey": {
                        "type": "object",
                        "properties": {
                            "key1": {"type": "string"},
                            "key2": {"type": "string"},
                            "key3": {"type": "string"},
                        },
                        "required": ["key1", "key2"]
                    },

                    # ----- clicks -----
                    "RightSingle": {
                        "type": "object",
                        "properties": {
                            "position": {
                                "type": "array",
                                "items": {"type": "number"},
                            }
                        },
                        "required": ["position"]
                    },
                    "Click": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["position"]
                    },

                    # ----- drag -----
                    "Drag": {
                        "type": "object",
                        "properties": {
                            "position1": {"type": "array", "items": {"type": "number"}},
                            "position2": {"type": "array", "items": {"type": "number"}},
                        },
                        "required": ["position1", "position2"]
                    },

                    # ----- move mouse -----
                    "move_mouse": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["position"]
                    },

                    # ----- scrolling -----
                    "scroll_up": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "array", "items": {"type": "number"}},
                            "dx": {"type": "number"},
                            "dy": {"type": "number"},
                        },
                        "required": ["position"]
                    },
                    "scroll_down": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "array", "items": {"type": "number"}},
                            "dx": {"type": "number"},
                            "dy": {"type": "number"},
                        },
                        "required": ["position"]
                    },

                    # ----- memory + wait -----
                    "record_info": {"type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]
                        },
                    "wait": {"type": "object",
                        "properties": {"text": {"type": "string"}}},
                },

                    }
                },

        },
        "required": [
            "action"
        ]
    }

    ACTION_RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_action_output",
            "strict": True,
            "schema": ACTION_SCHEMA
        }
    }

    BRAIN_SCHEMA = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"}
                },
                "required": ["analysis"]
            },
            "current_state": {
                "type": "object",
                "properties": {
                    "step_evaluate": {"type": "string"},
                    "ask_human": {"type": "string"},
                    "next_goal": {"type": "string"},
                },
                "required": [
                    "step_evaluate",
                    "ask_human",
                    "next_goal"
                ]
            },
        },
        "required": ["analysis","current_state"]
    }

    BRAIN_RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_state_output",
            "strict": True,
            "schema": BRAIN_SCHEMA
        }
    }

    MEMORY_SCHEMA = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"}
        },
        "required": ["summary"]
    }

    MEMORY_RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "memory_output",
            "strict": True,
            "schema": MEMORY_SCHEMA
        }
    }

    PLANNER_SCHEMA = {
        "type": "object",
        "properties": {
            "step_by_step_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {
                            "type": "string",
                            "pattern": "^Step \\d+$"
                        },
                        "description": {
                            "type": "string"
                        }
                    },
                    "required": ["step_id", "description"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["step_by_step_plan"],
        "additionalProperties": False
    }

    PLANNER_RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "planner_output",
            "strict": True,
            "schema": PLANNER_SCHEMA
        }
    }