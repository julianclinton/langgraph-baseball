# langgraph-baseball

From online course: https://apps.cognitiveclass.ai/learning/course/course-v1:IBMSkillsNetwork+GPXX04UIEN+v1/home


## Requirements
- `uv`
- `ollama`
  - suggest pulling the model being used e.g. `ollama pull llama3.2`, `ollama pull granite4:3b`

## Install dependencies

```
uv sync
```

## Run the code

You will need 3 terminals:
- one running `ollama serve` (may already be running)
- one running the LLM of choice e.g. `ollama run llama3.2`
- one running LlangGraph `uv run main.p`
