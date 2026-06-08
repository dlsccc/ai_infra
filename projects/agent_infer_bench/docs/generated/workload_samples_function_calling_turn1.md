# Context Compiler Workload Samples

Config: `E:\ai_infra\projects\agent_infer_bench\configs\context_compiler\realistic_mock.yaml`
Scenario: `function_calling`
Turn: `1`

说明：这份文件只展示自动生成的请求样例，不会调用模型。
同一个 scenario/turn 下对比不同 variant，可以直观看到稳定内容、动态字段、observation 的位置变化。

## context_compiler_no_compression

Request ID: `context_compiler_realistic_ablation_cfg00_000_function_calling_context_compiler_no_compression_turn_01`
Prompt chars: `3606`
Prefix overlap ratio: `0.3122`
Compiler: `context_compiler_mvp`

### Segment Order

- `system/system` stability=`static` scope=`global` words=`25`
- `tool_schema/tools` stability=`static` scope=`agent_type` words=`32`
- `tool_call/tool_call_00` stability=`session_dynamic` scope=`session` words=`4`
- `current_query/user_01` stability=`turn_dynamic` scope=`turn` words=`27`
- `observation/observation_00` stability=`turn_dynamic` scope=`turn` words=`90`
- `runtime/runtime_header` stability=`ephemeral` scope=`request` words=`4`

### Prompt

```text
[SYSTEM:system]
You are a reliable function-calling assistant. Choose exactly one tool and return valid JSON with required arguments. Never invent tool names or omit required fields.

[TOOL_SCHEMA:tools]
Tools:
{"description":"Find available flights for a route and date.","name":"search_flights","parameters":{"properties":{"date":{"type":"string"},"destination":{"type":"string"},"origin":{"type":"string"}},"required":["origin","destination","date"],"type":"object"}}
{"description":"Retrieve weather forecast for a city and date.","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"date":{"type":"string"}},"required":["city","date"],"type":"object"}}
{"description":"Estimate hotel budget for a city and number of nights.","name":"estimate_hotel_budget","parameters":{"properties":{"city":{"type":"string"},"nights":{"type":"integer"}},"required":["city","nights"],"type":"object"}}
{"description":"Persist a travel itinerary record.","name":"save_itinerary","parameters":{"properties":{"summary":{"type":"string"},"traveler":{"type":"string"}},"required":["traveler","summary"],"type":"object"}}

[TOOL_CALL:tool_call_00]
Assistant tool call: {"name":"search_flights","arguments":{"origin":"SHA","destination":"SIN","date":"2026-07-10"}}

[CURRENT_QUERY:user_01]
User request: coordinate a business trip from Shanghai to Singapore, check weather, estimate hotel budget, and store the itinerary. Current turn=1; choose the next valid tool call.

[OBSERVATION:observation_00]
tool_result: search_flights
status: success
required_fields_seen: origin,destination,date
candidate_flight: MU543 SHA->SIN departs 09:20 arrives 14:55
fare_usd: 280
policy_note: business fare must be below 500 USD unless approved.
api_json_tail:
flight_api_json_turn_0_0 flight_api_json_turn_0_1 flight_api_json_turn_0_2 flight_api_json_turn_0_3 flight_api_json_turn_0_4 flight_api_json_turn_0_5 flight_api_json_turn_0_6 flight_api_json_turn_0_7 flight_api_json_turn_0_8 flight_api_json_turn_0_9 flight_api_json_turn_0_10 flight_api_json_turn_0_11 flight_api_json_turn_0_12 flight_api_json_turn_0_13 flight_api_json_turn_0_14 flight_api_json_turn_0_15 flight_api_json_turn_0_16 flight_api_json_turn_0_17 flight_api_json_turn_0_18 flight_api_json_turn_0_19 flight_api_json_turn_0_20 flight_api_json_turn_0_21 flight_api_json_turn_0_22 flight_api_json_turn_0_23 flight_api_json_turn_0_24 flight_api_json_turn_0_25 flight_api_json_turn_0_26 flight_api_json_turn_0_27 flight_api_json_turn_0_28 flight_api_json_turn_0_29 flight_api_json_turn_0_30 flight_api_json_turn_0_31 flight_api_json_turn_0_32 flight_api_json_turn_0_33 flight_api_json_turn_0_34 flight_api_json_turn_0_35 flight_api_json_turn_0_36 flight_api_json_turn_0_37 flight_api_json_turn_0_38 flight_api_json_turn_0_39 flight_api_json_turn_0_40 flight_api_json_turn_0_41 flight_api_json_turn_0_42 flight_api_json_turn_0_43 flight_api_json_turn_0_44 flight_api_json_turn_0_45 flight_api_json_turn_0_46 flight_api_json_turn_0_47 flight_api_json_turn_0_48 flight_api_json_turn_0_49 flight_api_json_turn_0_50 flight_api_json_turn_0_51 flight_api_json_turn_0_52 flight_api_json_turn_0_53 flight_api_json_turn_0_54 flight_api_json_turn_0_55 flight_api_json_turn_0_56 flight_api_json_turn_0_57 flight_api_json_turn_0_58 flight_api_json_turn_0_59 flight_api_json_turn_0_60 flight_api_json_turn_0_61 flight_api_json_turn_0_62 flight_api_json_turn_0_63

[RUNTIME:runtime_header]
request_time=2026-06-08T09:01:00Z
session_id=context_compiler_realistic_ablation_cfg00_000
retry_count=1
trace_nonce=context_compiler_realistic_ablation_cfg00_000-01
Assistant:
```

## context_compiler_with_observation_compression

Request ID: `context_compiler_realistic_ablation_cfg00_000_function_calling_context_compiler_with_observation_compression_turn_01`
Prompt chars: `2287`
Prefix overlap ratio: `0.4338`
Compiler: `context_compiler_mvp`

### Segment Order

- `system/system` stability=`static` scope=`global` words=`25`
- `tool_schema/tools` stability=`static` scope=`agent_type` words=`32`
- `tool_call/tool_call_00` stability=`session_dynamic` scope=`session` words=`4`
- `current_query/user_01` stability=`turn_dynamic` scope=`turn` words=`27`
- `observation/observation_00` stability=`turn_dynamic` scope=`turn` words=`37`
- `runtime/runtime_header` stability=`ephemeral` scope=`request` words=`4`

### Prompt

```text
[SYSTEM:system]
You are a reliable function-calling assistant. Choose exactly one tool and return valid JSON with required arguments. Never invent tool names or omit required fields.

[TOOL_SCHEMA:tools]
Tools:
{"description":"Find available flights for a route and date.","name":"search_flights","parameters":{"properties":{"date":{"type":"string"},"destination":{"type":"string"},"origin":{"type":"string"}},"required":["origin","destination","date"],"type":"object"}}
{"description":"Retrieve weather forecast for a city and date.","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"date":{"type":"string"}},"required":["city","date"],"type":"object"}}
{"description":"Estimate hotel budget for a city and number of nights.","name":"estimate_hotel_budget","parameters":{"properties":{"city":{"type":"string"},"nights":{"type":"integer"}},"required":["city","nights"],"type":"object"}}
{"description":"Persist a travel itinerary record.","name":"save_itinerary","parameters":{"properties":{"summary":{"type":"string"},"traveler":{"type":"string"}},"required":["traveler","summary"],"type":"object"}}

[TOOL_CALL:tool_call_00]
Assistant tool call: {"name":"search_flights","arguments":{"origin":"SHA","destination":"SIN","date":"2026-07-10"}}

[CURRENT_QUERY:user_01]
User request: coordinate a business trip from Shanghai to Singapore, check weather, estimate hotel budget, and store the itinerary. Current turn=1; choose the next valid tool call.

[OBSERVATION:observation_00]
tool_result: search_flights status: success required_fields_seen: origin,destination,date candidate_flight: MU543 SHA->SIN departs 09:20 arrives 14:55 fare_usd: 280 policy_note: business fare must be
[... observation compressed; recoverable at obs://context_compiler_realistic_ablation_cfg00_000/function_calling/00 ...]
flight_api_json_turn_0_54 flight_api_json_turn_0_55 flight_api_json_turn_0_56 flight_api_json_turn_0_57 flight_api_json_turn_0_58 flight_api_json_turn_0_59 flight_api_json_turn_0_60 flight_api_json_turn_0_61 flight_api_json_turn_0_62 flight_api_json_turn_0_63

[RUNTIME:runtime_header]
request_time=2026-06-08T09:01:00Z
session_id=context_compiler_realistic_ablation_cfg00_000
retry_count=1
trace_nonce=context_compiler_realistic_ablation_cfg00_000-01
Assistant:
```

## original_bad_layout

Request ID: `context_compiler_realistic_ablation_cfg00_000_function_calling_original_bad_layout_turn_01`
Prompt chars: `3674`
Prefix overlap ratio: `0.0039`
Compiler: `none`

### Segment Order

- `runtime/runtime_header` stability=`ephemeral` scope=`request` words=`4`
- `current_query/user_01` stability=`turn_dynamic` scope=`turn` words=`27`
- `tool_schema/tools` stability=`static` scope=`agent_type` words=`100`
- `tool_call/tool_call_00` stability=`session_dynamic` scope=`session` words=`4`
- `observation/observation_00` stability=`turn_dynamic` scope=`turn` words=`90`
- `system/system` stability=`static` scope=`global` words=`25`

### Prompt

```text
[RUNTIME:runtime_header]
request_time=2026-06-08T09:01:00Z
session_id=context_compiler_realistic_ablation_cfg00_000
retry_count=1
trace_nonce=context_compiler_realistic_ablation_cfg00_000-01

[CURRENT_QUERY:user_01]
User request: coordinate a business trip from Shanghai to Singapore, check weather, estimate hotel budget, and store the itinerary. Current turn=1; choose the next valid tool call.

[TOOL_SCHEMA:tools]
Tools:
{"name": "get_weather", "description": "Retrieve weather forecast for a city and date.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}}, "required": ["city", "date"]}}
{"name": "estimate_hotel_budget", "description": "Estimate hotel budget for a city and number of nights.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "nights": {"type": "integer"}}, "required": ["city", "nights"]}}
{"name": "save_itinerary", "description": "Persist a travel itinerary record.", "parameters": {"type": "object", "properties": {"traveler": {"type": "string"}, "summary": {"type": "string"}}, "required": ["traveler", "summary"]}}
{"name": "search_flights", "description": "Find available flights for a route and date.", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}

[TOOL_CALL:tool_call_00]
Assistant tool call: {"name":"search_flights","arguments":{"origin":"SHA","destination":"SIN","date":"2026-07-10"}}

[OBSERVATION:observation_00]
tool_result: search_flights
status: success
required_fields_seen: origin,destination,date
candidate_flight: MU543 SHA->SIN departs 09:20 arrives 14:55
fare_usd: 280
policy_note: business fare must be below 500 USD unless approved.
api_json_tail:
flight_api_json_turn_0_0 flight_api_json_turn_0_1 flight_api_json_turn_0_2 flight_api_json_turn_0_3 flight_api_json_turn_0_4 flight_api_json_turn_0_5 flight_api_json_turn_0_6 flight_api_json_turn_0_7 flight_api_json_turn_0_8 flight_api_json_turn_0_9 flight_api_json_turn_0_10 flight_api_json_turn_0_11 flight_api_json_turn_0_12 flight_api_json_turn_0_13 flight_api_json_turn_0_14 flight_api_json_turn_0_15 flight_api_json_turn_0_16 flight_api_json_turn_0_17 flight_api_json_turn_0_18 flight_api_json_turn_0_19 flight_api_json_turn_0_20 flight_api_json_turn_0_21 flight_api_json_turn_0_22 flight_api_json_turn_0_23 flight_api_json_turn_0_24 flight_api_json_turn_0_25 flight_api_json_turn_0_26 flight_api_json_turn_0_27 flight_api_json_turn_0_28 flight_api_json_turn_0_29 flight_api_json_turn_0_30 flight_api_json_turn_0_31 flight_api_json_turn_0_32 flight_api_json_turn_0_33 flight_api_json_turn_0_34 flight_api_json_turn_0_35 flight_api_json_turn_0_36 flight_api_json_turn_0_37 flight_api_json_turn_0_38 flight_api_json_turn_0_39 flight_api_json_turn_0_40 flight_api_json_turn_0_41 flight_api_json_turn_0_42 flight_api_json_turn_0_43 flight_api_json_turn_0_44 flight_api_json_turn_0_45 flight_api_json_turn_0_46 flight_api_json_turn_0_47 flight_api_json_turn_0_48 flight_api_json_turn_0_49 flight_api_json_turn_0_50 flight_api_json_turn_0_51 flight_api_json_turn_0_52 flight_api_json_turn_0_53 flight_api_json_turn_0_54 flight_api_json_turn_0_55 flight_api_json_turn_0_56 flight_api_json_turn_0_57 flight_api_json_turn_0_58 flight_api_json_turn_0_59 flight_api_json_turn_0_60 flight_api_json_turn_0_61 flight_api_json_turn_0_62 flight_api_json_turn_0_63

[SYSTEM:system]
You are a reliable function-calling assistant. Choose exactly one tool and return valid JSON with required arguments. Never invent tool names or omit required fields.
Assistant:
```
