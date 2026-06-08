# Context Compiler Workload Samples

Config: `E:\ai_infra\projects\agent_infer_bench\configs\context_compiler\realistic_mock.yaml`
Scenario: `coding_agent`
Turn: `1`

说明：这份文件只展示自动生成的请求样例，不会调用模型。
同一个 scenario/turn 下对比不同 variant，可以直观看到稳定内容、动态字段、observation 的位置变化。

## context_compiler_no_compression

Request ID: `context_compiler_realistic_ablation_cfg01_000_coding_agent_context_compiler_no_compression_turn_01`
Prompt chars: `4114`
Prefix overlap ratio: `0.2698`
Compiler: `context_compiler_mvp`

### Segment Order

- `system/system` stability=`static` scope=`global` words=`31`
- `tool_schema/tools` stability=`static` scope=`agent_type` words=`25`
- `tool_call/tool_call_00` stability=`session_dynamic` scope=`session` words=`6`
- `current_query/user_01` stability=`turn_dynamic` scope=`turn` words=`20`
- `observation/observation_00` stability=`turn_dynamic` scope=`turn` words=`122`
- `runtime/runtime_header` stability=`ephemeral` scope=`request` words=`4`

### Prompt

```text
[SYSTEM:system]
You are a careful coding repair agent. Inspect repository evidence, call tools when needed, and propose minimal patches. Preserve file paths, failing tests, line numbers, error messages, and command exit codes.

[TOOL_SCHEMA:tools]
Tools:
{"description":"Read a file from the repository.","name":"read_file","parameters":{"properties":{"path":{"type":"string"},"start_line":{"type":"integer"}},"required":["path"],"type":"object"}}
{"description":"Search repository text for a query.","name":"search_repo","parameters":{"properties":{"query":{"type":"string"}},"required":["query"],"type":"object"}}
{"description":"Run a test command in the sandbox.","name":"run_tests","parameters":{"properties":{"command":{"type":"string"},"cwd":{"type":"string"}},"required":["command"],"type":"object"}}
{"description":"Apply a unified diff patch.","name":"apply_patch","parameters":{"properties":{"diff":{"type":"string"}},"required":["diff"],"type":"object"}}

[TOOL_CALL:tool_call_00]
Assistant tool call: {"name":"run_tests","arguments":{"command":"pytest tests/test_parser.py::test_nested_call -q","cwd":"/workspace/repo"}}

[CURRENT_QUERY:user_01]
Issue: pytest fails after recent parser changes. Find the root cause and decide the next tool call. Current focus turn=1.

[OBSERVATION:observation_00]
Command: pytest tests/test_parser.py::test_nested_call -q
cwd: /workspace/repo
exit_code: 1
failing_test: tests/test_parser.py::test_nested_call
file_path: src/parser.py
line_number: 128
error_type: AssertionError
root_cause_hint: parser drops nested function-call arguments after comma token.
stderr_tail:
pytest_stderr_turn_0_0 pytest_stderr_turn_0_1 pytest_stderr_turn_0_2 pytest_stderr_turn_0_3 pytest_stderr_turn_0_4 pytest_stderr_turn_0_5 pytest_stderr_turn_0_6 pytest_stderr_turn_0_7 pytest_stderr_turn_0_8 pytest_stderr_turn_0_9 pytest_stderr_turn_0_10 pytest_stderr_turn_0_11 pytest_stderr_turn_0_12 pytest_stderr_turn_0_13 pytest_stderr_turn_0_14 pytest_stderr_turn_0_15 pytest_stderr_turn_0_16 pytest_stderr_turn_0_17 pytest_stderr_turn_0_18 pytest_stderr_turn_0_19 pytest_stderr_turn_0_20 pytest_stderr_turn_0_21 pytest_stderr_turn_0_22 pytest_stderr_turn_0_23 pytest_stderr_turn_0_24 pytest_stderr_turn_0_25 pytest_stderr_turn_0_26 pytest_stderr_turn_0_27 pytest_stderr_turn_0_28 pytest_stderr_turn_0_29 pytest_stderr_turn_0_30 pytest_stderr_turn_0_31 pytest_stderr_turn_0_32 pytest_stderr_turn_0_33 pytest_stderr_turn_0_34 pytest_stderr_turn_0_35 pytest_stderr_turn_0_36 pytest_stderr_turn_0_37 pytest_stderr_turn_0_38 pytest_stderr_turn_0_39 pytest_stderr_turn_0_40 pytest_stderr_turn_0_41 pytest_stderr_turn_0_42 pytest_stderr_turn_0_43 pytest_stderr_turn_0_44 pytest_stderr_turn_0_45 pytest_stderr_turn_0_46 pytest_stderr_turn_0_47 pytest_stderr_turn_0_48 pytest_stderr_turn_0_49 pytest_stderr_turn_0_50 pytest_stderr_turn_0_51 pytest_stderr_turn_0_52 pytest_stderr_turn_0_53 pytest_stderr_turn_0_54 pytest_stderr_turn_0_55 pytest_stderr_turn_0_56 pytest_stderr_turn_0_57 pytest_stderr_turn_0_58 pytest_stderr_turn_0_59 pytest_stderr_turn_0_60 pytest_stderr_turn_0_61 pytest_stderr_turn_0_62 pytest_stderr_turn_0_63 pytest_stderr_turn_0_64 pytest_stderr_turn_0_65 pytest_stderr_turn_0_66 pytest_stderr_turn_0_67 pytest_stderr_turn_0_68 pytest_stderr_turn_0_69 pytest_stderr_turn_0_70 pytest_stderr_turn_0_71 pytest_stderr_turn_0_72 pytest_stderr_turn_0_73 pytest_stderr_turn_0_74 pytest_stderr_turn_0_75 pytest_stderr_turn_0_76 pytest_stderr_turn_0_77 pytest_stderr_turn_0_78 pytest_stderr_turn_0_79 pytest_stderr_turn_0_80 pytest_stderr_turn_0_81 pytest_stderr_turn_0_82 pytest_stderr_turn_0_83 pytest_stderr_turn_0_84 pytest_stderr_turn_0_85 pytest_stderr_turn_0_86 pytest_stderr_turn_0_87 pytest_stderr_turn_0_88 pytest_stderr_turn_0_89 pytest_stderr_turn_0_90 pytest_stderr_turn_0_91 pytest_stderr_turn_0_92 pytest_stderr_turn_0_93 pytest_stderr_turn_0_94 pytest_stderr_turn_0_95

[RUNTIME:runtime_header]
request_time=2026-06-08T09:01:00Z
session_id=context_compiler_realistic_ablation_cfg01_000
retry_count=1
trace_nonce=context_compiler_realistic_ablation_cfg01_000-01
Assistant:
```

## context_compiler_with_observation_compression

Request ID: `context_compiler_realistic_ablation_cfg01_000_coding_agent_context_compiler_with_observation_compression_turn_01`
Prompt chars: `2212`
Prefix overlap ratio: `0.4234`
Compiler: `context_compiler_mvp`

### Segment Order

- `system/system` stability=`static` scope=`global` words=`31`
- `tool_schema/tools` stability=`static` scope=`agent_type` words=`25`
- `tool_call/tool_call_00` stability=`session_dynamic` scope=`session` words=`6`
- `current_query/user_01` stability=`turn_dynamic` scope=`turn` words=`20`
- `observation/observation_00` stability=`turn_dynamic` scope=`turn` words=`44`
- `runtime/runtime_header` stability=`ephemeral` scope=`request` words=`4`

### Prompt

```text
[SYSTEM:system]
You are a careful coding repair agent. Inspect repository evidence, call tools when needed, and propose minimal patches. Preserve file paths, failing tests, line numbers, error messages, and command exit codes.

[TOOL_SCHEMA:tools]
Tools:
{"description":"Read a file from the repository.","name":"read_file","parameters":{"properties":{"path":{"type":"string"},"start_line":{"type":"integer"}},"required":["path"],"type":"object"}}
{"description":"Search repository text for a query.","name":"search_repo","parameters":{"properties":{"query":{"type":"string"}},"required":["query"],"type":"object"}}
{"description":"Run a test command in the sandbox.","name":"run_tests","parameters":{"properties":{"command":{"type":"string"},"cwd":{"type":"string"}},"required":["command"],"type":"object"}}
{"description":"Apply a unified diff patch.","name":"apply_patch","parameters":{"properties":{"diff":{"type":"string"}},"required":["diff"],"type":"object"}}

[TOOL_CALL:tool_call_00]
Assistant tool call: {"name":"run_tests","arguments":{"command":"pytest tests/test_parser.py::test_nested_call -q","cwd":"/workspace/repo"}}

[CURRENT_QUERY:user_01]
Issue: pytest fails after recent parser changes. Find the root cause and decide the next tool call. Current focus turn=1.

[OBSERVATION:observation_00]
Command: pytest tests/test_parser.py::test_nested_call -q cwd: /workspace/repo exit_code: 1 failing_test: tests/test_parser.py::test_nested_call file_path: src/parser.py line_number: 128 error_type: AssertionError root_cause_hint: parser drops nested function-call arguments after comma token.
[... observation compressed; recoverable at obs://context_compiler_realistic_ablation_cfg01_000/coding_agent/00 ...]
pytest_stderr_turn_0_84 pytest_stderr_turn_0_85 pytest_stderr_turn_0_86 pytest_stderr_turn_0_87 pytest_stderr_turn_0_88 pytest_stderr_turn_0_89 pytest_stderr_turn_0_90 pytest_stderr_turn_0_91 pytest_stderr_turn_0_92 pytest_stderr_turn_0_93 pytest_stderr_turn_0_94 pytest_stderr_turn_0_95

[RUNTIME:runtime_header]
request_time=2026-06-08T09:01:00Z
session_id=context_compiler_realistic_ablation_cfg01_000
retry_count=1
trace_nonce=context_compiler_realistic_ablation_cfg01_000-01
Assistant:
```

## original_bad_layout

Request ID: `context_compiler_realistic_ablation_cfg01_000_coding_agent_original_bad_layout_turn_01`
Prompt chars: `4168`
Prefix overlap ratio: `0.0037`
Compiler: `none`

### Segment Order

- `runtime/runtime_header` stability=`ephemeral` scope=`request` words=`4`
- `current_query/user_01` stability=`turn_dynamic` scope=`turn` words=`20`
- `tool_schema/tools` stability=`static` scope=`agent_type` words=`79`
- `tool_call/tool_call_00` stability=`session_dynamic` scope=`session` words=`6`
- `observation/observation_00` stability=`turn_dynamic` scope=`turn` words=`122`
- `system/system` stability=`static` scope=`global` words=`31`

### Prompt

```text
[RUNTIME:runtime_header]
request_time=2026-06-08T09:01:00Z
session_id=context_compiler_realistic_ablation_cfg01_000
retry_count=1
trace_nonce=context_compiler_realistic_ablation_cfg01_000-01

[CURRENT_QUERY:user_01]
Issue: pytest fails after recent parser changes. Find the root cause and decide the next tool call. Current focus turn=1.

[TOOL_SCHEMA:tools]
Tools:
{"name": "search_repo", "description": "Search repository text for a query.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
{"name": "run_tests", "description": "Run a test command in the sandbox.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "cwd": {"type": "string"}}, "required": ["command"]}}
{"name": "apply_patch", "description": "Apply a unified diff patch.", "parameters": {"type": "object", "properties": {"diff": {"type": "string"}}, "required": ["diff"]}}
{"name": "read_file", "description": "Read a file from the repository.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}}, "required": ["path"]}}

[TOOL_CALL:tool_call_00]
Assistant tool call: {"name":"run_tests","arguments":{"command":"pytest tests/test_parser.py::test_nested_call -q","cwd":"/workspace/repo"}}

[OBSERVATION:observation_00]
Command: pytest tests/test_parser.py::test_nested_call -q
cwd: /workspace/repo
exit_code: 1
failing_test: tests/test_parser.py::test_nested_call
file_path: src/parser.py
line_number: 128
error_type: AssertionError
root_cause_hint: parser drops nested function-call arguments after comma token.
stderr_tail:
pytest_stderr_turn_0_0 pytest_stderr_turn_0_1 pytest_stderr_turn_0_2 pytest_stderr_turn_0_3 pytest_stderr_turn_0_4 pytest_stderr_turn_0_5 pytest_stderr_turn_0_6 pytest_stderr_turn_0_7 pytest_stderr_turn_0_8 pytest_stderr_turn_0_9 pytest_stderr_turn_0_10 pytest_stderr_turn_0_11 pytest_stderr_turn_0_12 pytest_stderr_turn_0_13 pytest_stderr_turn_0_14 pytest_stderr_turn_0_15 pytest_stderr_turn_0_16 pytest_stderr_turn_0_17 pytest_stderr_turn_0_18 pytest_stderr_turn_0_19 pytest_stderr_turn_0_20 pytest_stderr_turn_0_21 pytest_stderr_turn_0_22 pytest_stderr_turn_0_23 pytest_stderr_turn_0_24 pytest_stderr_turn_0_25 pytest_stderr_turn_0_26 pytest_stderr_turn_0_27 pytest_stderr_turn_0_28 pytest_stderr_turn_0_29 pytest_stderr_turn_0_30 pytest_stderr_turn_0_31 pytest_stderr_turn_0_32 pytest_stderr_turn_0_33 pytest_stderr_turn_0_34 pytest_stderr_turn_0_35 pytest_stderr_turn_0_36 pytest_stderr_turn_0_37 pytest_stderr_turn_0_38 pytest_stderr_turn_0_39 pytest_stderr_turn_0_40 pytest_stderr_turn_0_41 pytest_stderr_turn_0_42 pytest_stderr_turn_0_43 pytest_stderr_turn_0_44 pytest_stderr_turn_0_45 pytest_stderr_turn_0_46 pytest_stderr_turn_0_47 pytest_stderr_turn_0_48 pytest_stderr_turn_0_49 pytest_stderr_turn_0_50 pytest_stderr_turn_0_51 pytest_stderr_turn_0_52 pytest_stderr_turn_0_53 pytest_stderr_turn_0_54 pytest_stderr_turn_0_55 pytest_stderr_turn_0_56 pytest_stderr_turn_0_57 pytest_stderr_turn_0_58 pytest_stderr_turn_0_59 pytest_stderr_turn_0_60 pytest_stderr_turn_0_61 pytest_stderr_turn_0_62 pytest_stderr_turn_0_63 pytest_stderr_turn_0_64 pytest_stderr_turn_0_65 pytest_stderr_turn_0_66 pytest_stderr_turn_0_67 pytest_stderr_turn_0_68 pytest_stderr_turn_0_69 pytest_stderr_turn_0_70 pytest_stderr_turn_0_71 pytest_stderr_turn_0_72 pytest_stderr_turn_0_73 pytest_stderr_turn_0_74 pytest_stderr_turn_0_75 pytest_stderr_turn_0_76 pytest_stderr_turn_0_77 pytest_stderr_turn_0_78 pytest_stderr_turn_0_79 pytest_stderr_turn_0_80 pytest_stderr_turn_0_81 pytest_stderr_turn_0_82 pytest_stderr_turn_0_83 pytest_stderr_turn_0_84 pytest_stderr_turn_0_85 pytest_stderr_turn_0_86 pytest_stderr_turn_0_87 pytest_stderr_turn_0_88 pytest_stderr_turn_0_89 pytest_stderr_turn_0_90 pytest_stderr_turn_0_91 pytest_stderr_turn_0_92 pytest_stderr_turn_0_93 pytest_stderr_turn_0_94 pytest_stderr_turn_0_95

[SYSTEM:system]
You are a careful coding repair agent. Inspect repository evidence, call tools when needed, and propose minimal patches. Preserve file paths, failing tests, line numbers, error messages, and command exit codes.
Assistant:
```
