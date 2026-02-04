# LLM Setup and Testing Guide

## Prerequisites

1. **Install Dependencies with LLM Support**
   ```bash
   uv sync --extra llm
   ```

   This installs:
   - `pydantic-ai-slim[openai]>=0.0.14` - LLM agent framework
   - `python-dotenv>=1.0.0` - Environment variable loading

2. **Configure API Key**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

   The CLI automatically loads `.env` when running commands.

## Testing LLM Integration

### Quick Test
```bash
# Run 2 rounds with LLM agents
uv run gg run --rounds 2 --llm --name "llm_test"
```

Expected output:
```
[Sender] Using pydantic-ai with openai:gpt-4o-mini
[Receiver] Using pydantic-ai with openai:gpt-4o-mini
```

### Without API Key
If `OPENAI_API_KEY` is not set, the system gracefully falls back to heuristic agents:
```
[Sender] No API key, using heuristic agent (demonstrates same economics)
[Receiver] No API key, using heuristic agent (demonstrates same economics)
```

The heuristic agents implement rational Bayesian reasoning, so the game runs correctly without LLM calls.

## LLM Configuration Options

### Using Different Models

```bash
# Use GPT-4
gg run --llm --llm-model gpt-4o

# Use GPT-3.5 Turbo
gg run --llm --llm-model gpt-3.5-turbo
```

### Configuration File

**YAML Config** (`experiments/yaml/llm_config.yaml`):
```yaml
name: llm_experiment
num_rounds: 20
use_llm: true
llm_model: gpt-4o-mini
```

**Python Config** (`experiments/python/llm_config.py`):
```python
from garbling_gym.core.config import GameConfig

config = GameConfig(
    num_rounds=20,
    use_llm=True,
    llm_model="gpt-4o-mini"
)
```

## Architecture

### Agent Hierarchy

```
Agent (base.py)
  └── LLMAgent (llm.py)
        ├── SenderAgent (sender.py)
        └── ReceiverAgent (receiver.py)
```

### LLM Integration Flow

1. **Initialization** (`LLMAgent.__init__`):
   - Checks for `OPENAI_API_KEY` environment variable
   - Attempts to import `pydantic_ai`
   - Sets `use_api = True` if successful
   - Falls back to heuristic behavior if unavailable

2. **API Calls** (`LLMAgent._call_llm`):
   - Creates `pydantic_ai.Agent` with structured output type
   - Sends system prompt + user prompt
   - Returns validated Pydantic model
   - Handles errors gracefully with fallback

3. **Fallback Behavior**:
   - `SenderAgent`: Heuristic strategy selection based on quality
   - `ReceiverAgent`: Bayesian decision-making based on signal

## Testing

### Unit Tests
```bash
# Run all tests (excludes LLM tests by default)
uv run pytest -v

# Run specific test
uv run pytest tests/core/test_agents.py -v
```

### E2E Test with LLM
```bash
# Test LLM integration end-to-end
uv run gg run --rounds 5 --llm --name "e2e_test"

# Verify results
uv run gg visualize e2e_test
```

### Batch Experiment with LLM
```bash
# Create experiment config
cat > experiments/yaml/llm_sweep.yaml << EOF
experiments:
  - name: no_llm_baseline
    rounds: 10
    llm: false

  - name: llm_both
    rounds: 10
    llm: true
    llm_roles: [sender, receiver]
EOF

# Run experiment
uv run gg experiment experiments/yaml/llm_sweep.yaml
```

## Troubleshooting

### "pydantic-ai not available"
**Cause**: LLM dependencies not installed

**Solution**:
```bash
uv sync --extra llm
```

### API Errors
**Common errors**:
- `Unknown keyword arguments: 'result_type'` → Fixed in v1.52.0
- `'AgentRunResult' object has no attribute 'data'` → Use `result.output`
- `The api_key client option must be set` → Check `.env` file

**Debug steps**:
```bash
# 1. Check API key is loaded
uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"

# 2. Check pydantic-ai is importable
uv run python -c "import pydantic_ai; print('Version:', pydantic_ai.__version__)"

# 3. Test minimal example
uv run python /tmp/test_pydantic_ai.py
```

### Rate Limits
If you hit OpenAI rate limits, use fallback agents:
```bash
# Temporarily disable LLM
gg run --rounds 20 --no-llm
```

Or reduce rounds:
```bash
gg run --rounds 2 --llm  # Only 2 rounds = 4 API calls
```

## Cost Estimation

**gpt-4o-mini pricing** (as of 2024):
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

**Per-round costs** (approximate):
- Sender: ~200 input tokens, ~20 output tokens
- Receiver: ~300 input tokens, ~10 output tokens
- **Total per round**: ~$0.0001 (0.01 cents)

**Example**:
- 20 rounds = ~$0.002 (0.2 cents)
- 1000 rounds = ~$0.10 (10 cents)

## Production Deployment

### Environment Variables
```bash
# Production .env
OPENAI_API_KEY=sk-prod-key
OPENAI_ORG_ID=org-xxx  # Optional
```

### CI/CD Testing
```bash
# Skip LLM tests in CI
pytest -v -k "not llm"

# Or mock API calls
export OPENAI_API_KEY="sk-mock-key"
pytest -v --mock-llm
```

### Monitoring
Use the `duration` field in results to monitor API latency:
```bash
gg list-runs --sort duration
```

Typical latencies:
- Heuristic agents: 0.1-0.5s per round
- LLM agents (gpt-4o-mini): 1-3s per round
