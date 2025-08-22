========================
CODE SNIPPETS
========================
TITLE: Run Pydantic AI Example with Zero Setup using uv
DESCRIPTION: This advanced one-liner command uses `uv` to run a Pydantic AI example (`pydantic_model`) by setting the OpenAI API key and installing dependencies on the fly. It's ideal for quick testing without prior setup.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/index.md#_snippet_5

LANGUAGE: bash
CODE:
```
OPENAI_API_KEY='your-api-key' \
  uv run --with "pydantic-ai[examples]" \
  -m pydantic_ai_examples.pydantic_model
```

----------------------------------------

TITLE: Install Pydantic AI with examples
DESCRIPTION: Installs the `pydantic-ai-examples` package via the `examples` optional group. This makes it easy to customize and run the provided Pydantic AI examples.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/install.md#_snippet_2

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai[examples]"
```

----------------------------------------

TITLE: Install Pydantic AI Example Dependencies
DESCRIPTION: Commands to install the necessary extra dependencies for running Pydantic AI examples. This includes the `examples` optional dependency group, which can be installed via `pip` or `uv` for PyPI installations, or `uv sync` if cloning the repository.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/index.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai[examples]"
```

LANGUAGE: bash
CODE:
```
uv sync --extra examples
```

----------------------------------------

TITLE: Copy Pydantic AI Examples to Local Directory
DESCRIPTION: This command copies the Pydantic AI example files to a specified local directory (e.g., `examples/`). This allows users to modify, experiment with, and develop upon the examples without affecting the installed package.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/index.md#_snippet_6

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples --copy-to examples/
```

----------------------------------------

TITLE: Perform slim installation for Pydantic AI with multiple optional groups
DESCRIPTION: Demonstrates how to install `pydantic-ai-slim` with multiple optional groups simultaneously. This allows for including dependencies for several specific models and features in a single installation command.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/install.md#_snippet_5

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[openai,vertexai,logfire]"
```

----------------------------------------

TITLE: Run Pydantic AI Flight Booking Example (Bash)
DESCRIPTION: Command to execute the Pydantic AI flight booking example. This command assumes that all necessary dependencies are installed and environment variables are properly configured as per the project's setup instructions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/flight-booking.md#_snippet_1

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.flight_booking
```

----------------------------------------

TITLE: Run Pydantic AI Bank Support Agent Example
DESCRIPTION: Command to execute the Pydantic AI bank support agent example. This requires prior installation of dependencies and setting up environment variables as per the project's usage instructions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/bank-support.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.bank_support
```

----------------------------------------

TITLE: Run Pydantic AI Data Analyst Example Script
DESCRIPTION: Execute the Pydantic AI data analyst example script from the command line. This command uses `python/uv-run` to launch the `pydantic_ai_examples.data_analyst` module, assuming necessary dependencies are installed and environment variables are configured as per the project's setup instructions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/data-analyst.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.data_analyst
```

----------------------------------------

TITLE: Run Specific Pydantic AI Example (pydantic_model)
DESCRIPTION: This command demonstrates how to run the `pydantic_model` example specifically. It uses `python` or `uv run` to execute the module, showcasing a common use case for running individual examples.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/index.md#_snippet_4

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.pydantic_model
```

----------------------------------------

TITLE: Run Pydantic AI Examples
DESCRIPTION: Commands to execute Pydantic AI examples using either `python` or `uv run`. This includes a general command for running any example module, a specific command for the `pydantic_model` example, and a convenient one-liner for `uv` that handles dependency installation and API key setting.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/index.md#_snippet_2

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.<example_module_name>
```

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.pydantic_model
```

LANGUAGE: bash
CODE:
```
OPENAI_API_KEY='your-api-key' \
  uv run --with "pydantic-ai[examples]" \
  -m pydantic_ai_examples.pydantic_model
```

----------------------------------------

TITLE: Install and Run clai with uv
DESCRIPTION: These commands demonstrate how to install `clai` globally using `uv tool install` and then run it. After installation, `clai` can be invoked directly from the command line to start an interactive AI chat session.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/clai/README.md#_snippet_2

LANGUAGE: bash
CODE:
```
uv tool install clai
...
clai
```

----------------------------------------

TITLE: Serve Pydantic AI Documentation Locally
DESCRIPTION: Runs the `mkdocs serve` command via `uv` to start a local web server, allowing contributors to preview and test changes to the project's documentation pages.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/contributing.md#_snippet_6

LANGUAGE: bash
CODE:
```
uv run mkdocs serve
```

----------------------------------------

TITLE: Install Pydantic AI Dependencies and Pre-commit Hooks
DESCRIPTION: Command to install all project dependencies and set up pre-commit hooks using the `make` utility. This command streamlines the setup process for development.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/contributing.md#_snippet_3

LANGUAGE: bash
CODE:
```
make install
```

----------------------------------------

TITLE: Perform slim installation for Pydantic AI with OpenAI model
DESCRIPTION: Installs the `pydantic-ai-slim` package with only the `openai` optional group. This is recommended when you intend to use only the `OpenAIModel` and wish to avoid installing superfluous packages.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/install.md#_snippet_3

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[openai]"
```

----------------------------------------

TITLE: Start Pydantic AI AG-UI Backend
DESCRIPTION: Initiates the Pydantic AI AG-UI example backend application. This command uses `uv-run` to execute the specified Python module, making the backend services available for the frontend to communicate with.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/ag-ui.md#_snippet_1

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.ag_ui
```

----------------------------------------

TITLE: Install Pydantic AI core package
DESCRIPTION: Installs the main `pydantic-ai` package and its core dependencies, including libraries required to use all models. This installation requires Python 3.9 or newer.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/install.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add pydantic-ai
```

----------------------------------------

TITLE: Install Gradio and Run Weather Agent UI
DESCRIPTION: This sequence of commands first installs the required Gradio library, then launches the web-based user interface for the Pydantic AI weather agent. The UI provides a chat-based interaction for the agent, requiring Python 3.10+.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/weather-agent.md#_snippet_1

LANGUAGE: bash
CODE:
```
pip install gradio>=5.9.0
python/uv-run -m pydantic_ai_examples.weather_agent_gradio
```

----------------------------------------

TITLE: Pydantic AI Slim Install Optional Groups
DESCRIPTION: This section details the available optional groups for `pydantic-ai-slim`, allowing users to install only the necessary dependencies for specific models or features, thereby avoiding superfluous packages. Each group corresponds to a set of external libraries.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/install.md#_snippet_4

LANGUAGE: APIDOC
CODE:
```
pydantic-ai-slim Optional Groups:
  logfire: Installs `logfire` for Logfire integration.
  evals: Installs `pydantic-evals`.
  openai: Installs `openai`.
  vertexai: Installs `google-auth` and `requests`.
  google: Installs `google-genai`.
  anthropic: Installs `anthropic`.
  groq: Installs `groq`.
  mistral: Installs `mistralai`.
  cohere: Installs `cohere`.
  bedrock: Installs `boto3`.
  huggingface: Installs `huggingface-hub[inference]`.
  duckduckgo: Installs `ddgs`.
  tavily: Installs `tavily-python`.
  cli: Installs `rich`, `prompt-toolkit`, and `argcomplete`.
  mcp: Installs `mcp`.
  a2a: Installs `fasta2a`.
  ag-ui: Installs `ag-ui-protocol` and `starlette`.
```

----------------------------------------

TITLE: Copy Pydantic AI Examples to Local Directory
DESCRIPTION: Command to copy the Pydantic AI example files to a specified local directory. This allows users to easily modify and experiment with the examples without affecting the original installed package files.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/index.md#_snippet_3

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples --copy-to examples/
```

----------------------------------------

TITLE: Install Pydantic AI with Logfire integration
DESCRIPTION: Installs Pydantic AI with the optional `logfire` group, enabling integration with Pydantic Logfire. This allows for enhanced viewing and understanding of agent runs.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/install.md#_snippet_1

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai[logfire]"
```

----------------------------------------

TITLE: Pydantic AI Hello World Example
DESCRIPTION: This minimal example demonstrates how to initialize and run a basic Pydantic AI agent. It configures the agent to use a specific LLM model (Gemini 1.5 Flash) and registers a static system prompt. The agent then synchronously runs a query, and its output is printed. This showcases the fundamental steps for setting up and interacting with a Pydantic AI agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/index.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent(  # (1)!
    'google-gla:gemini-1.5-flash',
    system_prompt='Be concise, reply with one sentence.',  # (2)!
)

result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

----------------------------------------

TITLE: Run Question Graph Example
DESCRIPTION: Executes the `pydantic_ai_examples.question_graph` module using `python/uv-run`. This command initiates the question graph application, which is designed for asking and evaluating questions. Users must ensure that all necessary dependencies are installed and environment variables are correctly configured as per the project's usage instructions before running.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/question-graph.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.question_graph
```

----------------------------------------

TITLE: Install and Run Pydantic AI CLI Globally with uv
DESCRIPTION: Install the `clai` CLI globally using `uv`'s tool installation feature. After installation, run `clai` to start an interactive chat session with the AI model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_2

LANGUAGE: bash
CODE:
```
uv tool install clai
...
clai
```

----------------------------------------

TITLE: Install Pydantic-AI with Groq Support
DESCRIPTION: This command installs the `pydantic-ai-slim` package along with the necessary `groq` optional dependencies, enabling Groq model integration. It uses either `pip` or `uv` for package management.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/groq.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[groq]"
```

----------------------------------------

TITLE: Run Pydantic AI Stream Whales Example Script
DESCRIPTION: This command executes the `stream_whales.py` example script, which demonstrates streaming structured responses from GPT-4 and displaying them dynamically. It requires dependencies to be installed and environment variables to be set up beforehand, as detailed in the project's usage instructions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/stream-whales.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.stream_whales
```

----------------------------------------

TITLE: Install and Run clai with pip
DESCRIPTION: These commands show how to install `clai` using `pip`, the Python package installer, and then run it. Once installed, `clai` can be executed to initiate an interactive chat session with an AI model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/clai/README.md#_snippet_3

LANGUAGE: bash
CODE:
```
pip install clai
...
clai
```

----------------------------------------

TITLE: Install Pydantic AI with Google Dependencies
DESCRIPTION: This command installs `pydantic-ai-slim` along with its `google` optional dependencies, which are required to use `GoogleModel` and access Google's Gemini models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[google]"
```

----------------------------------------

TITLE: Run Pydantic AI Weather Agent
DESCRIPTION: This command executes the main Pydantic AI weather agent script. It initializes the agent, allowing it to process user queries by leveraging configured tools and API keys. Ensure dependencies are installed and environment variables are set before running.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/weather-agent.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.weather_agent
```

----------------------------------------

TITLE: Install Pydantic AI with AG-UI Dependencies
DESCRIPTION: Instructions for installing Pydantic AI with AG-UI extra and Uvicorn for running ASGI applications, using pip or uv.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add 'pydantic-ai-slim[ag-ui]'
pip/uv-add uvicorn
```

----------------------------------------

TITLE: Execute Pydantic AI SQL Generation Example
DESCRIPTION: Commands to execute the Pydantic AI SQL generation script. The first command runs the example with default settings, while the second demonstrates passing a custom prompt string.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/sql-gen.md#_snippet_1

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.sql_gen
```

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.sql_gen "find me errors"
```

----------------------------------------

TITLE: Example Prompt for Haiku Generation
DESCRIPTION: A simple text prompt demonstrating how to instruct a generative AI model to create a haiku on a specific subject, such as Formula 1.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/ag-ui.md#_snippet_4

LANGUAGE: text
CODE:
```
Generate a haiku about formula 1
```

----------------------------------------

TITLE: Install Pre-commit with uv
DESCRIPTION: Command to install the `pre-commit` tool using `uv`, a fast Python package installer and resolver. This tool helps manage and run pre-commit hooks for code quality.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/contributing.md#_snippet_1

LANGUAGE: bash
CODE:
```
uv tool install pre-commit
```

----------------------------------------

TITLE: Run Pydantic AI with OpenTelemetry Example
DESCRIPTION: Terminal command to execute a Python script (`raw_otel.py`) that demonstrates Pydantic AI's integration with OpenTelemetry. It ensures necessary dependencies like `pydantic-ai-slim[openai]`, `opentelemetry-sdk`, and `opentelemetry-exporter-otlp` are included for a complete setup.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_8

LANGUAGE: txt
CODE:
```
uv run \
  --with 'pydantic-ai-slim[openai]' \
  --with opentelemetry-sdk \
  --with opentelemetry-exporter-otlp \
  raw_otel.py
```

----------------------------------------

TITLE: Run pydantic-ai stream_markdown example
DESCRIPTION: Execute the pydantic-ai example script to stream markdown output. This command uses `python` or `uv-run` to launch the module.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/stream-markdown.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.stream_markdown
```

----------------------------------------

TITLE: Install Pydantic-AI with Bedrock Support
DESCRIPTION: Instructions for installing the `pydantic-ai-slim` package with the `bedrock` optional group, which provides necessary dependencies for integrating with AWS Bedrock.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[bedrock]"
```

----------------------------------------

TITLE: Comprehensive Pydantic-AI Agent with Asynchronous Dependencies, Tools, and Output Validators
DESCRIPTION: Illustrates a complete Pydantic-AI agent setup utilizing asynchronous dependencies. This example integrates an `httpx.AsyncClient` with an `async` system prompt, an `async` tool, and an `async` output validator, showcasing how `RunContext` can be passed to all these components.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/dependencies.md#_snippet_3

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)


@agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    response = await ctx.deps.http_client.get('https://example.com')
    response.raise_for_status()
    return f'Prompt: {response.text}'


@agent.tool  # (1)!
async def get_joke_material(ctx: RunContext[MyDeps], subject: str) -> str:
    response = await ctx.deps.http_client.get(
        'https://example.com#jokes',
        params={'subject': subject},
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    response.raise_for_status()
    return response.text


@agent.output_validator  # (2)!
async def validate_output(ctx: RunContext[MyDeps], output: str) -> str:
    response = await ctx.deps.http_client.post(
        'https://example.com#validate',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
        params={'query': output},
    )
    if response.status_code == 400:
        raise ModelRetry(f'invalid response: {response.text}')
    response.raise_for_status()
    return output


async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('foobar', client)
        result = await agent.run('Tell me a joke.', deps=deps)
        print(result.output)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
```

----------------------------------------

TITLE: Install pydantic-ai-slim with Hugging Face support
DESCRIPTION: This command installs the `pydantic-ai-slim` package along with the `huggingface` optional group, providing necessary dependencies for integrating with Hugging Face models and inference providers.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/huggingface.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[huggingface]"
```

----------------------------------------

TITLE: Running Pydantic-AI Agents Example
DESCRIPTION: This Python example demonstrates how to use `pydantic_ai.Agent` to perform various types of runs: `agent.run_sync()` for a synchronous call, `agent.run()` for an asynchronous call (awaitable), and `agent.run_stream()` for streaming text output asynchronously. It showcases how to get a completed response or stream parts of it.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)

    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            print(text)
```

----------------------------------------

TITLE: Navigate to AG-UI TypeScript SDK Directory
DESCRIPTION: Changes the current working directory to `ag-ui/typescript-sdk`. This directory contains the TypeScript-based AG-UI Dojo example application, which serves as the frontend component for demonstrating Pydantic AI agent interactions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/ag-ui.md#_snippet_3

LANGUAGE: shell
CODE:
```
cd ag-ui/typescript-sdk
```

----------------------------------------

TITLE: Install pydantic-ai-slim with Cohere Support
DESCRIPTION: This command installs the `pydantic-ai-slim` package along with its `cohere` optional dependencies. This ensures that all necessary components for interacting with Cohere models via `pydantic-ai` are available in your environment.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/cohere.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[cohere]"
```

----------------------------------------

TITLE: Integrate single ACI.dev tool with Pydantic AI using tool_from_aci
DESCRIPTION: This Python example demonstrates how to use the `tool_from_aci` convenience method to integrate a specific ACI.dev tool, like `TAVILY__SEARCH`, into a Pydantic AI `Agent`. It shows the setup for initializing the tool with a linked account owner ID and then using the agent to run a query. Users need to install `aci-sdk` and set the `ACI_API_KEY` environment variable.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_19

LANGUAGE: python
CODE:
```
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import tool_from_aci


tavily_search = tool_from_aci(
    'TAVILY__SEARCH',
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent(
    'google-gla:gemini-2.0-flash',
    tools=[tavily_search],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')
print(result.output)
#> Elden Ring Nightreign is planned to be released on May 30, 2025.
```

----------------------------------------

TITLE: Install pydantic-ai with OpenAI support
DESCRIPTION: Instructions to install the `pydantic-ai-slim` package with the `openai` optional group using pip or uv-add. This enables the necessary dependencies for OpenAI model integration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[openai]"
```

----------------------------------------

TITLE: Install pydantic-ai-slim with Mistral support
DESCRIPTION: Instructions to install the `pydantic-ai-slim` package with the `mistral` optional group using `pip` or `uv-add`, which provides the necessary dependencies for Mistral integration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/mistral.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[mistral]"
```

----------------------------------------

TITLE: Clone AG-UI Protocol Repository
DESCRIPTION: Clones the official AG-UI protocol repository from GitHub. This step is necessary to obtain the source code for the AG-UI Dojo example frontend application, which complements the Pydantic AI backend.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/ag-ui.md#_snippet_2

LANGUAGE: shell
CODE:
```
git clone https://github.com/ag-ui-protocol/ag-ui.git
```

----------------------------------------

TITLE: Install Pydantic AI with MCP Support
DESCRIPTION: Provides the `pip` or `uv-add` command to install `pydantic-ai-slim` along with its `mcp` optional dependencies, which are necessary for MCP client functionality. This installation requires Python 3.10 or higher.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[mcp]"
```

----------------------------------------

TITLE: Run Pydantic AI Example
DESCRIPTION: Execute the Pydantic AI example using `python/uv-run`. This command runs the `pydantic_model` module from `pydantic_ai_examples` with default settings or allows specifying an alternative model like Gemini via the `PYDANTIC_AI_MODEL` environment variable.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/pydantic-model.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.pydantic_model
```

LANGUAGE: bash
CODE:
```
PYDANTIC_AI_MODEL=gemini-1.5-pro python/uv-run -m pydantic_ai_examples.pydantic_model
```

----------------------------------------

TITLE: Run PostgreSQL with pgvector using Docker
DESCRIPTION: This command starts a PostgreSQL container with the pgvector extension, mapping port 54320 and mounting a local volume for data persistence. It's used as the search database for the RAG example, avoiding port conflicts with other PostgreSQL instances.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/rag.md#_snippet_0

LANGUAGE: bash
CODE:
```
mkdir postgres-data
docker run --rm \
  -e POSTGRES_PASSWORD=postgres \
  -p 54320:5432 \
  -v `pwd`/postgres-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

----------------------------------------

TITLE: Install Pydantic Evals Package
DESCRIPTION: This snippet demonstrates how to install the Pydantic Evals library using `pip` or `uv`. The first command installs the base package, while the second command includes an optional `logfire` dependency for OpenTelemetry tracing and sending evaluation results to Logfire.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add pydantic-evals
```

LANGUAGE: bash
CODE:
```
pip/uv-add 'pydantic-evals[logfire]'
```

----------------------------------------

TITLE: Install Deno Runtime via Curl
DESCRIPTION: Command to install the Deno runtime using a `curl` script. Deno is a secure runtime for JavaScript and TypeScript, often used for web development and scripting.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/contributing.md#_snippet_2

LANGUAGE: bash
CODE:
```
curl -fsSL https://deno.land/install.sh | sh
```

----------------------------------------

TITLE: Install FastA2A Library
DESCRIPTION: This command installs the `fasta2a` library from PyPI, which provides a framework-agnostic implementation of the A2A protocol in Python. It's the foundational package for building A2A-compliant services.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/a2a.md#_snippet_1

LANGUAGE: bash
CODE:
```
pip/uv-add fasta2a
```

----------------------------------------

TITLE: Install Pydantic AI with A2A Extra
DESCRIPTION: This command installs the `pydantic-ai-slim` package along with its `a2a` extra, which automatically includes the `FastA2A` library as a dependency. This is the recommended installation method for users who want to leverage both Pydantic AI and FastA2A together.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/a2a.md#_snippet_2

LANGUAGE: bash
CODE:
```
pip/uv-add 'pydantic-ai-slim[a2a]'
```

----------------------------------------

TITLE: Run clai with uvx
DESCRIPTION: This command executes the `clai` command-line interface using `uvx`, a tool for running Python applications without global installation. It starts an interactive session where you can chat with an AI model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/clai/README.md#_snippet_1

LANGUAGE: bash
CODE:
```
uvx clai
```

----------------------------------------

TITLE: Pydantic AI Agent with Tools, Dependency Injection, and Structured Output
DESCRIPTION: This comprehensive example illustrates building a sophisticated Pydantic AI agent for a bank support system. It showcases key features such as defining agent dependencies (`SupportDependencies`), enforcing structured output with Pydantic models (`SupportOutput`), creating dynamic system prompts, and registering custom Python functions as LLM-callable tools (`customer_balance`). The example demonstrates how to run the agent asynchronously with injected dependencies and process its validated, structured output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/README.md#_snippet_2

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


# SupportDependencies is used to pass data, connections, and logic into the model that will be needed when running
# system prompt and tool functions. Dependency injection provides a type-safe way to customise the behavior of your agents.
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


# This pydantic model defines the structure of the output returned by the agent.
class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


# This agent will act as first-tier support in a bank.
# Agents are generic in the type of dependencies they accept and the type of output they return.
# In this case, the support agent has type `Agent[SupportDependencies, SupportOutput]`.
support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    # The response from the agent will, be guaranteed to be a SupportOutput,
    # if validation fails the agent is prompted to try again.
    output_type=SupportOutput,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


# Dynamic system prompts can make use of dependency injection.
# Dependencies are carried via the `RunContext` argument, which is parameterized with the `deps_type` from above.
# If the type annotation here is wrong, static type checkers will catch it.
@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


# `tool` let you register functions which the LLM may call while responding to a user.
# Again, dependencies are carried via `RunContext`, any other arguments become the tool schema passed to the LLM.
# Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
@support_agent.tool
async def customer_balance(
        ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    # The docstring of a tool is also passed to the LLM as the description of the tool.
    # Parameter descriptions are extracted from the docstring and added to the parameter schema sent to the LLM.
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return balance


# ...  # In a real use case, you'd add more tools and a longer system prompt


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    # Run the agent asynchronously, conducting a conversation with the LLM until a final response is reached.
    # Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.
    result = await support_agent.run('What is my balance?', deps=deps)
    # The `result.output` will be validated with Pydantic to guarantee it is a `SupportOutput`. Since the agent is generic,
    # it'll also be typed as a `SupportOutput` to aid with static type checking.
    print(result.output)
    # """
    # support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    # """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    # """
    # support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    # """
```

----------------------------------------

TITLE: Integrate MCP Run Python with Pydantic AI Agent
DESCRIPTION: This Python example demonstrates how to set up and use the MCP Run Python server as a toolset for a Pydantic AI Agent. It shows the initialization of the MCPServerStdio, configuring logging, creating an Agent instance, and executing an asynchronous task that leverages the Python sandbox.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/mcp-run-python/README.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

import logfire

logfire.configure()
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

server = MCPServerStdio('deno',
    args=[
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ])
agent = Agent('claude-3-5-haiku-latest', toolsets=[server])


async def main():
    async with agent:
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    # There are 9,208 days between January 1, 2000, and March 18, 2025.w

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

----------------------------------------

TITLE: Install and Run Pydantic AI CLI with pip
DESCRIPTION: Install the `clai` CLI using pip, Python's package installer. Once installed, execute `clai` to initiate an interactive chat session with the AI model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_3

LANGUAGE: bash
CODE:
```
pip install clai
...
clai
```

----------------------------------------

TITLE: Set LLM API Key Environment Variables
DESCRIPTION: Commands to set environment variables for authenticating with Large Language Models (LLMs) such as OpenAI or Google Gemini. These API keys are crucial for the Pydantic AI examples to interact with the respective model providers.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/index.md#_snippet_1

LANGUAGE: bash
CODE:
```
export OPENAI_API_KEY=your-api-key
```

LANGUAGE: bash
CODE:
```
export GEMINI_API_KEY=your-api-key
```

----------------------------------------

TITLE: Create a Streamable HTTP MCP Server in Python
DESCRIPTION: This Python example demonstrates how to set up a basic Model Context Protocol (MCP) server using `FastMCP`. It defines an `add` tool and configures the server to run using the `streamable-http` transport, which is a prerequisite for the client example.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_1

LANGUAGE: python
CODE:
```
from mcp.server.fastmcp import FastMCP

app = FastMCP()

@app.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == '__main__':
    app.run(transport='streamable-http')
```

----------------------------------------

TITLE: Install Pydantic-Graph Library
DESCRIPTION: This snippet provides the command to install the `pydantic-graph` library using `pip` or `uv-add`. It is a required dependency for `pydantic-ai` and an optional one for `pydantic-ai-slim`, enabling the use of graph-based state machines.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_0

LANGUAGE: Bash
CODE:
```
pip/uv-add pydantic-graph
```

----------------------------------------

TITLE: Install and Run Ollama Locally
DESCRIPTION: Instructions to download and run the Ollama server with a specific model, preparing it for local `pydantic-ai` integration. This command will pull the specified model if it's not already downloaded.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_14

LANGUAGE: bash
CODE:
```
ollama run llama3.2
```

----------------------------------------

TITLE: Run Pydantic AI Example with Gemini Model
DESCRIPTION: Command to execute the Pydantic AI example using the Gemini 1.5 Pro model by setting the PYDANTIC_AI_MODEL environment variable. This allows overriding the default model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/pydantic-model.md#_snippet_1

LANGUAGE: bash
CODE:
```
PYDANTIC_AI_MODEL=gemini-1.5-pro python/uv-run -m pydantic_ai_examples.pydantic_model
```

----------------------------------------

TITLE: Query Pydantic AI Agent with RAG Search
DESCRIPTION: This Python command allows users to ask questions to the Pydantic AI agent, leveraging the previously built RAG search database. The example demonstrates how to query the agent with a specific question about Logfire configuration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/rag.md#_snippet_2

LANGUAGE: python
CODE:
```
python/uv-run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
```

----------------------------------------

TITLE: Configure Pydantic-AI Models with Fallback
DESCRIPTION: This example shows how to initialize `OpenAIModel` and `AnthropicModel` with specific `ModelSettings` (e.g., temperature, max_tokens) and then combine them into a `FallbackModel`. An `Agent` is then created with the `FallbackModel` to execute a prompt, demonstrating automatic model failover.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/index.md#_snippet_2

LANGUAGE: python
CODE:
```
openai_model = OpenAIModel(
    'gpt-4o',
    settings=ModelSettings(temperature=0.7, max_tokens=1000)  # Higher creativity for OpenAI
)
anthropic_model = AnthropicModel(
    'claude-3-5-sonnet-latest',
    settings=ModelSettings(temperature=0.2, max_tokens=1000)  # Lower temperature for consistency
)

fallback_model = FallbackModel(openai_model, anthropic_model)
agent = Agent(fallback_model)

result = agent.run_sync('Write a creative story about space exploration')
print(result.output)
```

----------------------------------------

TITLE: Install and Run MCP Run Python Server with Deno
DESCRIPTION: This command installs and runs the MCP Run Python server using Deno. It specifies necessary permissions for network access and node_modules, and allows choosing a transport method (stdio, streamable_http, sse, or warmup) for server operation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/run-python.md#_snippet_0

LANGUAGE: bash
CODE:
```
deno run \
  -N -R=node_modules -W=node_modules --node-modules-dir=auto \
  jsr:@pydantic/mcp-run-python [stdio|streamable_http|sse|warmup]
```

----------------------------------------

TITLE: Install Anthropic dependency for pydantic-ai-slim
DESCRIPTION: This command installs the `anthropic` optional group for `pydantic-ai-slim`, enabling the use of Anthropic models. It ensures necessary dependencies are available for integration with Anthropic's API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/anthropic.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[anthropic]"
```

----------------------------------------

TITLE: Run MCP Python Server with Deno
DESCRIPTION: This snippet provides the Deno command to start the MCP Run Python server. It includes necessary flags for network access, read/write permissions to node_modules (required for Pyodide), and specifies different transport options like stdio, sse, or warmup for various deployment scenarios.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/mcp-run-python/README.md#_snippet_0

LANGUAGE: bash
CODE:
```
deno run \
  -N -R=node_modules -W=node_modules --node-modules-dir=auto \
  jsr:@pydantic/mcp-run-python [stdio|sse|warmup]
```

----------------------------------------

TITLE: Install Tavily Search Tool for Pydantic AI
DESCRIPTION: Provides the `pip` or `uv` command to install the `tavily` optional group for `pydantic-ai-slim`, which is required to use the Tavily search tool with Pydantic AI agents. Users need to sign up for a Tavily account and obtain an API key.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/common-tools.md#_snippet_2

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[tavily]"
```

----------------------------------------

TITLE: Serve Pydantic AI Documentation Locally
DESCRIPTION: Command to run the Pydantic AI documentation site locally using `uv` and `mkdocs serve`. This allows contributors to preview documentation changes before committing.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/contributing.md#_snippet_5

LANGUAGE: bash
CODE:
```
uv run mkdocs serve
```

----------------------------------------

TITLE: Authenticate Local Environment with Logfire
DESCRIPTION: Authenticates your local development environment with Pydantic Logfire. This command typically guides you through a process to link your local setup to your Logfire account, ensuring that data can be sent securely.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_1

LANGUAGE: bash
CODE:
```
py-cli logfire auth
```

----------------------------------------

TITLE: Pydantic AI Native Output Mode Example
DESCRIPTION: Demonstrates how to use Pydantic AI's `NativeOutput` mode to force a language model to return structured data matching a specified JSON schema. This mode leverages the model's native structured output capabilities, which are not supported by all models. The example shows an `Agent` configured to output either a `Fruit` or `Vehicle` object, and then runs a query to get a `Vehicle`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_7

LANGUAGE: python
CODE:
```
from tool_output import Fruit, Vehicle

from pydantic_ai import Agent, NativeOutput

agent = Agent(
    'openai:gpt-4o',
    output_type=NativeOutput(
        [Fruit, Vehicle], # (1)!
        name='Fruit_or_vehicle',
        description='Return a fruit or vehicle.'
    ),
)
result = agent.run_sync('What is a Ford Explorer?')
print(repr(result.output))
#> Vehicle(name='Ford Explorer', wheels=4)
```

----------------------------------------

TITLE: Configure Pydantic AI with Together AI
DESCRIPTION: Outlines the setup for using Together AI with Pydantic AI via the `TogetherProvider`. This configuration requires an API key from Together.ai and allows access to their model library, exemplified by 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_29

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.together import TogetherProvider

model = OpenAIModel(
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',  # model library available at https://www.together.ai/models
    provider=TogetherProvider(api_key='your-together-api-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Mermaid Diagram Definition for Fives Graph
DESCRIPTION: This Mermaid syntax defines the visual flow of the `fives_graph` example. It shows the state transitions between `DivisibleBy5` and `Increment` nodes, including the start and end points of the graph execution.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_6

LANGUAGE: Mermaid
CODE:
```
--- 
title: fives_graph
--- 
stateDiagram-v2
  [*] --> DivisibleBy5
  DivisibleBy5 --> Increment
  DivisibleBy5 --> [*]
  Increment --> DivisibleBy5
```

----------------------------------------

TITLE: Implementing Static and Dynamic System Prompts in Pydantic AI
DESCRIPTION: This example demonstrates how to define both static and dynamic system prompts for a Pydantic AI agent. It shows how static prompts are set during agent initialization and dynamic prompts are created using decorated functions, optionally leveraging `RunContext` for runtime information to tailor responses.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_18

LANGUAGE: Python
CODE:
```
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,  # (1)!
    system_prompt="Use the customer's name while replying to them.",  # (2)!
)


@agent.system_prompt  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.system_prompt
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.
```

----------------------------------------

TITLE: Dynamically Customize Pydantic-AI Tool Parameter Description
DESCRIPTION: This Python example demonstrates using the `prepare` method to dynamically modify a tool's definition before it's passed to the model. The `prepare_greet` function updates the `description` of the `name` parameter for the `greet` tool based on the `deps` value from the `RunContext`, showcasing how tool metadata can be adapted at runtime.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_10

LANGUAGE: python
CODE:
```
from __future__ import annotations

from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool, ToolDefinition


def greet(name: str) -> str:
    return f'hello {name}'


async def prepare_greet(
    ctx: RunContext[Literal['human', 'machine']], tool_def: ToolDefinition
) -> ToolDefinition | None:
    d = f'Name of the {ctx.deps} to greet.'
    tool_def.parameters_json_schema['properties']['name']['description'] = d
    return tool_def


greet_tool = Tool(greet, prepare=prepare_greet)
test_model = TestModel()
agent = Agent(test_model, tools=[greet_tool], deps_type=Literal['human', 'machine'])

result = agent.run_sync('testing...', deps='human')
print(result.output)
# {"greet":"hello a"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='greet',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {
                'name': {'type': 'string', 'description': 'Name of the human to greet.'}
            },
            'required': ['name'],
            'type': 'object',
        },
    )
]
"""
```

----------------------------------------

TITLE: Python Example: Agent Delegation with Shared Dependencies
DESCRIPTION: This Python example demonstrates how to set up agent delegation where a 'joke selection' agent delegates joke generation to a 'joke generation' agent. It highlights the use of `deps_type` to define shared dependencies (an HTTP client and API key) and how these dependencies are passed and utilized across agents to make external API calls efficiently. The example also shows how to track combined usage across delegated agents.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#_snippet_2

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext


@dataclass
class ClientAndKey:
    http_client: httpx.AsyncClient
    api_key: str


joke_selection_agent = Agent(
    'openai:gpt-4o',
    deps_type=ClientAndKey,
    system_prompt=(
        'Use the `joke_factory` tool to generate some jokes on the given subject, '
        'then choose the best. You must return just a single joke.'
    ),
)
joke_generation_agent = Agent(
    'gemini-1.5-flash',
    deps_type=ClientAndKey,
    output_type=list[str],
    system_prompt=(
        'Use the "get_jokes" tool to get some jokes on the given subject, '
        'then extract each joke into a list.'
    ),
)


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[ClientAndKey], count: int) -> list[str]:
    r = await joke_generation_agent.run(
        f'Please generate {count} jokes.',
        deps=ctx.deps,
        usage=ctx.usage,
    )
    return r.output


@joke_generation_agent.tool
async def get_jokes(ctx: RunContext[ClientAndKey], count: int) -> str:
    response = await ctx.deps.http_client.get(
        'https://example.com',
        params={'count': count},
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    response.raise_for_status()
    return response.text


async def main():
    async with httpx.AsyncClient() as client:
        deps = ClientAndKey(client, 'foobar')
        result = await joke_selection_agent.run('Tell me a joke.', deps=deps)
        print(result.output)
        # > Did you hear about the toothpaste scandal? They called it Colgate.
        print(result.usage())
        # > Usage(requests=4, request_tokens=309, response_tokens=32, total_tokens=341)
```

----------------------------------------

TITLE: Build RAG Search Database with OpenAI Embeddings
DESCRIPTION: This Python command initiates the process of building the search database for the RAG example. It requires the `OPENAI_API_KEY` environment variable and will make approximately 300 calls to the OpenAI embedding API to generate embeddings for documentation sections.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/rag.md#_snippet_1

LANGUAGE: python
CODE:
```
python/uv-run -m pydantic_ai_examples.rag build
```

----------------------------------------

TITLE: Run Gradio UI for Weather Agent
DESCRIPTION: Instructions to install Gradio and launch the web-based user interface for the Pydantic AI weather agent. This UI provides a multi-turn chat application built entirely in Python.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/weather-agent.md#_snippet_2

LANGUAGE: bash
CODE:
```
pip install gradio>=5.9.0
python/uv-run -m pydantic_ai_examples.weather_agent_gradio
```

----------------------------------------

TITLE: Initialize GoogleModel with Vertex AI using Service Account
DESCRIPTION: This Python example demonstrates how to authenticate `GoogleModel` with Vertex AI using a service account JSON file. It loads credentials from the specified file and passes them to the `GoogleProvider` for secure access.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_4

LANGUAGE: python
CODE:
```
from google.oauth2 import service_account

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

credentials = service_account.Credentials.from_service_account_file(
    'path/to/service-account.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform'],
)
provider = GoogleProvider(credentials=credentials)
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Use DuckDuckGo Search Tool with Pydantic AI Agent
DESCRIPTION: Demonstrates how to initialize a Pydantic AI `Agent` with the `duckduckgo_search_tool` and execute a synchronous web search query. The example shows importing the necessary components, configuring the agent with a system prompt, and processing the search results.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/common-tools.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:o3-mini',
    tools=[duckduckgo_search_tool()],
    system_prompt='Search DuckDuckGo for the given query and return the results.',
)

result = agent.run_sync(
    'Can you list the top five highest-grossing animated films of 2025?'
)
print(result.output)
```

----------------------------------------

TITLE: Initialize GoogleModel with Custom GoogleProvider and Base URL
DESCRIPTION: This Python example shows how to provide a custom `google.genai.Client` instance to `GoogleProvider` to configure advanced options like a custom `base_url`. This is useful for connecting to custom-compatible endpoints with the Google Generative Language API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_6

LANGUAGE: python
CODE:
```
from google.genai import Client
from google.genai.types import HttpOptions

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

client = Client(
    api_key='gemini-custom-api-key',
    http_options=HttpOptions(base_url='gemini-custom-base-url'),
)
provider = GoogleProvider(client=client)
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Install DuckDuckGo Search Tool for Pydantic AI
DESCRIPTION: Provides the `pip` or `uv` command to install the `duckduckgo` optional group for `pydantic-ai-slim`, which is required to use the DuckDuckGo search tool with Pydantic AI agents.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/common-tools.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai-slim[duckduckgo]"
```

----------------------------------------

TITLE: Initialize Pydantic-AI Agent with GroqModel Object
DESCRIPTION: This Python example shows how to explicitly create a `GroqModel` instance with a specific model name and then pass this model object to the `pydantic-ai` Agent. This approach offers more control over the model configuration compared to using a simple string name.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/groq.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

model = GroqModel('llama-3.3-70b-versatile')
agent = Agent(model)
...
```

----------------------------------------

TITLE: Logging Tool Execution with WrapperToolset
DESCRIPTION: This Python code defines a `LoggingToolset` that inherits from `WrapperToolset`. It overrides the `call_tool` method to log the start and end of each tool call, including its name, arguments, and result. An `asyncio.sleep` is included to simulate asynchronous operations and ensure consistent logging order for testing. The example then demonstrates how to use this custom toolset with an `Agent` to observe tool execution.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_7

LANGUAGE: python
CODE:
```
import asyncio
from typing_extensions import Any

from prepared_toolset import prepared_toolset

from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import WrapperToolset, ToolsetTool

LOG = []

class LoggingToolset(WrapperToolset):
    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        LOG.append(f'Calling tool {name!r} with args: {tool_args!r}')
        try:
            await asyncio.sleep(0.1 * len(LOG)) # (1)!

            result = await super().call_tool(name, tool_args, ctx, tool)
            LOG.append(f'Finished calling tool {name!r} with result: {result!r}')
        except Exception as e:
            LOG.append(f'Error calling tool {name!r}: {e}')
            raise e
        else:
            return result


logging_toolset = LoggingToolset(prepared_toolset)

agent = Agent(TestModel(), toolsets=[logging_toolset]) # (2)!
result = agent.run_sync('Call all the tools')
print(LOG)
"""
[
    "Calling tool 'temperature_celsius' with args: {'city': 'a'}",
    "Calling tool 'temperature_fahrenheit' with args: {'city': 'a'}",
    "Calling tool 'weather_conditions' with args: {'city': 'a'}",
    "Calling tool 'current_time' with args: {}",
    "Finished calling tool 'temperature_celsius' with result: 21.0",
    "Finished calling tool 'temperature_fahrenheit' with result: 69.8",
    'Finished calling tool \'weather_conditions\' with result: "It\'s raining"',
    "Finished calling tool 'current_time' with result: datetime.datetime(...)",
]
"""
```

----------------------------------------

TITLE: Run Pydantic AI Project Commands with Make
DESCRIPTION: Commands to interact with the Pydantic AI project using `make`. `make help` displays available commands, while `make` (without arguments) runs formatting, linting, static type checks, and tests with coverage.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/contributing.md#_snippet_4

LANGUAGE: bash
CODE:
```
make help
```

LANGUAGE: bash
CODE:
```
make
```

----------------------------------------

TITLE: Run FastAPI Chat Application
DESCRIPTION: Command to start the FastAPI chat application using `python/uv-run`. This command launches the application, making it accessible via a web browser, typically at `localhost:8000`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/chat-app.md#_snippet_0

LANGUAGE: bash
CODE:
```
python/uv-run -m pydantic_ai_examples.chat_app
```

----------------------------------------

TITLE: FastAPI Server Example for AG-UI Agent
DESCRIPTION: Demonstrates how to implement a FastAPI endpoint that accepts AG-UI run input, validates it, and streams AG-UI events back to the client using `run_ag_ui`, setting up a complete AG-UI server.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_2

LANGUAGE: python
CODE:
```
from ag_ui.core import RunAgentInput
from fastapi import FastAPI
from http import HTTPStatus
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
import json

from pydantic_ai import Agent
from pydantic_ai.ag_ui import run_ag_ui, SSE_CONTENT_TYPE


agent = Agent('openai:gpt-4.1', instructions='Be fun!')

app = FastAPI()


@app.post("/")
async def run_agent(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = RunAgentInput.model_validate(await request.json())
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    event_stream = run_ag_ui(agent, run_input, accept=accept)

    return StreamingResponse(event_stream, media_type=accept)
```

----------------------------------------

TITLE: Set OpenAI API Key
DESCRIPTION: Sets the `OPENAI_API_KEY` environment variable, which is essential for authenticating and interacting with the OpenAI API services used by the Pydantic AI agents. This key allows the application to make requests to OpenAI's models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/ag-ui.md#_snippet_0

LANGUAGE: bash
CODE:
```
export OPENAI_API_KEY=<your api key>
```

----------------------------------------

TITLE: OpenTelemetry Integration Example in Python
DESCRIPTION: This Python example demonstrates how to integrate OpenTelemetry tracing with Pydantic Evals. It defines a `SpanTracingEvaluator` that analyzes the `SpanTree` from the `EvaluatorContext` to extract information like total processing time and error occurrences. The example also includes a traced asynchronous function (`process_text`) and sets up a `Dataset` to evaluate its performance and tracing behavior using Logfire for OpenTelemetry configuration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_8

LANGUAGE: Python
CODE:
```
import asyncio
from typing import Any

import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.otel.span_tree import SpanQuery

logfire.configure(  # ensure that an OpenTelemetry tracer is configured
    send_to_logfire='if-token-present'
)


class SpanTracingEvaluator(Evaluator[str, str]):
    """Evaluator that analyzes the span tree generated during function execution."""

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, Any]:
        # Get the span tree from the context
        span_tree = ctx.span_tree
        if span_tree is None:
            return {'has_spans': False, 'performance_score': 0.0}

        # Find all spans with "processing" in the name
        processing_spans = span_tree.find(lambda node: 'processing' in node.name)

        # Calculate total processing time
        total_processing_time = sum(
            (span.duration.total_seconds() for span in processing_spans), 0.0
        )

        # Check for error spans
        error_query: SpanQuery = {'name_contains': 'error'}
        has_errors = span_tree.any(error_query)

        # Calculate a performance score (lower is better)
        performance_score = 1.0 if total_processing_time < 1.0 else 0.5

        return {
            'has_spans': True,
            'has_errors': has_errors,
            'performance_score': 0 if has_errors else performance_score,
        }


async def process_text(text: str) -> str:
    """Function that processes text with OpenTelemetry instrumentation."""
    with logfire.span('process_text'):
        # Simulate initial processing
        with logfire.span('text_processing'):
            await asyncio.sleep(0.1)
            processed = text.strip().lower()

        # Simulate additional processing
        with logfire.span('additional_processing'):
            if 'error' in processed:
                with logfire.span('error_handling'):
                    logfire.error(f'Error detected in text: {text}')
                    return f'Error processing: {text}'
            await asyncio.sleep(0.2)
            processed = processed.replace(' ', '_')

        return f'Processed: {processed}'


# Create test cases
dataset = Dataset(
    cases=[
        Case(
            name='normal_text',
            inputs='Hello World',
            expected_output='Processed: hello_world',
        ),
        Case(
            name='text_with_error',
            inputs='Contains error marker',
            expected_output='Error processing: Contains error marker',
        ),
    ],
    evaluators=[SpanTracingEvaluator()],
)

# Run evaluation - spans are automatically captured since logfire is configured
report = dataset.evaluate_sync(process_text)
```

----------------------------------------

TITLE: Connect to MCP Run Python Server using Python Client
DESCRIPTION: This Python example demonstrates how to connect to the MCP Run Python server using the `mcp` client library. It initializes a `StdioServerParameters` object to run the Deno server as a subprocess, then uses an asynchronous client session to initialize, list available tools (e.g., `run_python_code`), and execute Python code, capturing the results including output and return values.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/run-python.md#_snippet_1

LANGUAGE: python
CODE:
```
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

code = """
import numpy
a = numpy.array([1, 2, 3])
print(a)
a
"""
server_params = StdioServerParameters(
    command='deno',
    args=[
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(len(tools.tools))
            # > 1
            print(repr(tools.tools[0].name))
            # > 'run_python_code'
            print(repr(tools.tools[0].inputSchema))
            """
            {'type': 'object', 'properties': {'python_code': {'type': 'string', 'description': 'Python code to run'}}, 'required': ['python_code'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}
            """
            result = await session.call_tool('run_python_code', {'python_code': code})
            print(result.content[0].text)
            """
            <status>success</status>
            <dependencies>["numpy"]</dependencies>
            <output>
            [1 2 3]
            </output>
            <return_value>
            [
              1,
              2,
              3
            ]
            </return_value>
            """
```

----------------------------------------

TITLE: Install Pydantic AI with Retries Dependency
DESCRIPTION: Instructions to install `pydantic-ai-slim` with the `retries` dependency group, which includes `tenacity`, using `pip` or `uv-add`. This step is necessary to enable the retry functionality.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add 'pydantic-ai-slim[retries]'
```

----------------------------------------

TITLE: Run PostgreSQL for SQL Validation
DESCRIPTION: Launches a PostgreSQL Docker container on port 54320, configured for use with the Pydantic AI SQL generation example. This instance is used to validate generated SQL queries via `EXPLAIN`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/sql-gen.md#_snippet_0

LANGUAGE: bash
CODE:
```
docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 postgres
```

----------------------------------------

TITLE: Registering Pydantic-AI Agent Tools via Constructor
DESCRIPTION: This example demonstrates two methods for registering tools with a Pydantic-AI `Agent` during its construction. It shows how to pass a list of functions directly, allowing the agent to infer `RunContext` usage, or use `Tool` objects for explicit control over tool definitions, names, and descriptions. The snippet includes a simple dice game scenario to illustrate tool interaction and agent execution.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_3

LANGUAGE: python
CODE:
```
import random

from pydantic_ai import Agent, RunContext, Tool

system_prompt = """
You're a dice game, you should roll the die and see if the number
you get back matches the user's guess. If so, tell them they're a winner.
Use the player's name in the response.
"""

def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 1, 6))


def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


agent_a = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[roll_dice, get_player_name],
    system_prompt=system_prompt,
)
agent_b = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[
        Tool(roll_dice, takes_ctx=False),
        Tool(get_player_name, takes_ctx=True),
    ],
    system_prompt=system_prompt,
)

dice_result = {}
dice_result['a'] = agent_a.run_sync('My guess is 6', deps='Yashar')
dice_result['b'] = agent_b.run_sync('My guess is 4', deps='Anne')
print(dice_result['a'].output)
print(dice_result['b'].output)
```

----------------------------------------

TITLE: Install Pydantic Logfire SDK
DESCRIPTION: Installs the Pydantic Logfire Python SDK, including necessary dependencies for Pydantic AI integration, using either pip or uv package managers. This step is required to enable Logfire's observability features.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip/uv-add "pydantic-ai[logfire]"
```

----------------------------------------

TITLE: Create Stand-alone ASGI Application from Pydantic AI Agent
DESCRIPTION: This example illustrates how to convert a Pydantic AI `Agent` directly into a stand-alone ASGI application using the `Agent.to_ag_ui()` method. This simplifies deployment by allowing the agent to be served directly by any ASGI server. The shell command shows how to run this application with Uvicorn.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='Be fun!')
app = agent.to_ag_ui()
```

LANGUAGE: shell
CODE:
```
uvicorn agent_to_ag_ui:app
```

----------------------------------------

TITLE: Define Pydantic-AI Agent Instructions
DESCRIPTION: This example demonstrates how to configure a Pydantic-AI agent with both static and dynamic instructions. Static instructions are set during agent initialization, while dynamic instructions are defined via decorated functions that can access runtime context, allowing for flexible prompt generation based on current dependencies or other run-time information.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_19

LANGUAGE: python
CODE:
```
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,  # (1)!
    instructions="Use the customer's name while replying to them.",  # (2)!
)


@agent.instructions  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.
```

----------------------------------------

TITLE: Perform asynchronous model request with tool calling using pydantic-ai direct API
DESCRIPTION: This advanced example illustrates how to integrate tool calling with the `pydantic_ai.direct` API. It defines a Pydantic model (`Divide`) to represent a tool, generates its JSON schema, and passes it to `model_request` via `ModelRequestParameters`. The example shows how the model can then suggest a tool call based on the prompt.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/direct.md#_snippet_1

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel
from typing_extensions import Literal

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition


class Divide(BaseModel):
    """Divide two numbers."""

    numerator: float
    denominator: float
    on_inf: Literal['error', 'infinity'] = 'infinity'


async def main():
    # Make a request to the model with tool access
    model_response = await model_request(
        'openai:gpt-4.1-nano',
        [ModelRequest.user_text_prompt('What is 123 / 456?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name=Divide.__name__.lower(),
                    description=Divide.__doc__,
                    parameters_json_schema=Divide.model_json_schema(),
                )
            ],
            allow_text_output=True,  # Allow model to either use tools or respond directly
        ),
    )
    print(model_response)
    """
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='divide',
                args={'numerator': '123', 'denominator': '456'},
                tool_call_id='pyd_ai_2e0e396768a14fe482df90a29a78dc7b',
            )
        ],
        usage=Usage(requests=1, request_tokens=55, response_tokens=7, total_tokens=62),
        model_name='gpt-4.1-nano',
        timestamp=datetime.datetime(...),
    )
    """
```

----------------------------------------

TITLE: Pydantic AI Prompted Output Mode Example
DESCRIPTION: Illustrates the use of Pydantic AI's `PromptedOutput` mode, where the language model is prompted to generate structured data based on a provided JSON schema. This mode is usable with all models but is generally less reliable as it depends on the model's interpretation of instructions. The example includes configuring an `Agent` to return a `Vehicle` or `Device` object, and also demonstrates using a custom prompt template.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from tool_output import Vehicle

from pydantic_ai import Agent, PromptedOutput


class Device(BaseModel):
    name: str
    kind: str


agent = Agent(
    'openai:gpt-4o',
    output_type=PromptedOutput(
        [Vehicle, Device], # (1)!
        name='Vehicle or device',
        description='Return a vehicle or device.'
    ),
)
result = agent.run_sync('What is a MacBook?')
print(repr(result.output))
#> Device(name='MacBook', kind='laptop')

agent = Agent(
    'openai:gpt-4o',
    output_type=PromptedOutput(
        [Vehicle, Device],
        template='Gimme some JSON: {schema}'
    ),
)
result = agent.run_sync('What is a Ford Explorer?')
print(repr(result.output))
#> Vehicle(name='Ford Explorer', wheels=4)
```

----------------------------------------

TITLE: Initialize Pydantic AI Agent with System Prompt
DESCRIPTION: This snippet demonstrates how to create a basic Agent instance using the pydantic_ai library. It shows how to specify the LLM model to use and how to set a static system prompt, which defines the agent's initial behavior or instructions. This is a foundational step for building conversational AI applications with Pydantic AI.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/README.md#_snippet_0

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent

# Define a very simple agent including the model to use, you can also set the model when running the agent.
agent = Agent(
    'google-gla:gemini-1.5-flash',
    # Register a static system prompt using a keyword argument to the agent.
    # For more complex dynamically-generated system prompts, see the example below.
    system_prompt='Be concise, reply with one sentence.',
)

# Run the agent synchronously, conducting a conversation with the LLM.
```

----------------------------------------

TITLE: Interact with MCP Server using Python Client
DESCRIPTION: This Python client example demonstrates how to connect to an MCP server (like the one defined in `mcp_server.py`) using the `mcp.client.stdio` SDK. It initializes a client session, calls the 'poet' tool with a specific theme ('socks'), and then prints the generated poem, showcasing basic client-server communication.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/server.md#_snippet_1

LANGUAGE: python
CODE:
```
import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def client():
    server_params = StdioServerParameters(
        command='python', args=['mcp_server.py'], env=os.environ
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('poet', {'theme': 'socks'})
            print(result.content[0].text)
            """
            Oh, socks, those garments soft and sweet,
            That nestle softly 'round our feet,
            From cotton, wool, or blended thread,
            They keep our toes from feeling dread.
            """


if __name__ == '__main__':
    asyncio.run(client())
```

----------------------------------------

TITLE: Initialize GroqModel with Custom Provider and Async HTTP Client
DESCRIPTION: This advanced Python example demonstrates how to configure a `GroqModel` with a custom `GroqProvider` that uses a specific `httpx.AsyncClient`. This is useful for fine-tuning HTTP request behavior, such as setting custom timeouts or proxies, for interactions with the Groq API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/groq.md#_snippet_5

LANGUAGE: python
CODE:
```
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

custom_http_client = AsyncClient(timeout=30)
model = GroqModel(
    'llama-3.3-70b-versatile',
    provider=GroqProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Deploy FastAPI Web Endpoint on Modal with ASGI
DESCRIPTION: This snippet demonstrates deploying an ASGI application, such as FastAPI, as a web endpoint on Modal. It includes configuring `min_containers=1` to meet Slack's 3-second response time requirement and integrates Logfire setup and application imports within the Modal function.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_9

LANGUAGE: python
CODE:
```
# This function defines an ASGI web application for deployment on Modal.
# It ensures the web app is always running with min_containers=1 for fast responses.
import modal
# Assuming app.py contains the Pydantic AI app
# from .app import app as pydantic_ai_app

app = modal.App()

@app.function(min_containers=1)
@modal.asgi_app()
def web_app(): # type: ignore
    # Call setup_logfire here as logfire package is available in Modal context
    # setup_logfire() # Assuming setup_logfire is defined elsewhere
    # return pydantic_ai_app
    pass
```

----------------------------------------

TITLE: Configure Pydantic AI with Fireworks AI
DESCRIPTION: Provides an example of integrating Pydantic AI with Fireworks AI using the `FireworksProvider`. An API key obtained from Fireworks.AI account settings is required, and the snippet demonstrates the typical model naming convention for Fireworks models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_28

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.fireworks import FireworksProvider

model = OpenAIModel(
    'accounts/fireworks/models/qwq-32b',  # model library available at https://fireworks.ai/models
    provider=FireworksProvider(api_key='your-fireworks-api-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Define and Run a Basic Pydantic Graph
DESCRIPTION: This example demonstrates how to define a simple graph using `pydantic-graph`. It showcases the creation of nodes by subclassing `BaseNode`, defining their `run` methods for state transitions, and executing the graph synchronously. The graph increments a number until it becomes divisible by 5, illustrating a basic state machine flow.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/pydantic_graph/README.md#_snippet_0

LANGUAGE: python
CODE:
```
from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class DivisibleBy5(BaseNode[None, None, int]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode):
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)


fives_graph = Graph(nodes=[DivisibleBy5, Increment])
result = fives_graph.run_sync(DivisibleBy5(4))
print(result.output)
#> 5
```

----------------------------------------

TITLE: Configure Pydantic AI with OpenRouter
DESCRIPTION: Details the process of integrating `pydantic-ai` with OpenRouter. This setup requires an API key, which can be obtained from the OpenRouter platform, to authenticate and access various models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_18

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Pydantic-AI Tool with Advanced ToolReturn for Multi-modal Output
DESCRIPTION: This Python example demonstrates how to define a `pydantic-ai` tool using `@agent.tool_plain` that returns a `ToolReturn` object. It showcases sending multi-modal content (text and binary images via `BinaryContent`) to the LLM as context, while providing a separate `return_value` for programmatic use and including `metadata` not visible to the LLM. The `click_and_capture` function simulates a UI interaction, capturing before/after screenshots and sending them to the model for analysis.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_5

LANGUAGE: python
CODE:
```
import time
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturn, BinaryContent

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def click_and_capture(x: int, y: int) -> ToolReturn:
    """Click at coordinates and show before/after screenshots."""
    # Take screenshot before action
    before_screenshot = capture_screen()

    # Perform click operation
    perform_click(x, y)
    time.sleep(0.5)  # Wait for UI to update

    # Take screenshot after action
    after_screenshot = capture_screen()

    return ToolReturn(
        return_value=f"Successfully clicked at ({x}, {y})",
        content=[
            f"Clicked at coordinates ({x}, {y}). Here's the comparison:",
            "Before:",
            BinaryContent(data=before_screenshot, media_type="image/png"),
            "After:",
            BinaryContent(data=after_screenshot, media_type="image/png"),
            "Please analyze the changes and suggest next steps."
        ],
        metadata={
            "coordinates": {"x": x, "y": y},
            "action_type": "click_and_capture",
            "timestamp": time.time()
        }
    )

# The model receives the rich visual content for analysis
# while your application can access the structured return_value and metadata
result = agent.run_sync("Click on the submit button and tell me what happened")
print(result.output)
# The model can analyze the screenshots and provide detailed feedback
```

----------------------------------------

TITLE: Filter Out Tools Conditionally with prepare_tools
DESCRIPTION: This example illustrates how to use `prepare_tools` to conditionally filter out specific tools based on the agent's context or dependencies. It demonstrates disabling a tool ('launch_potato') if a boolean dependency (`ctx.deps`) is true, showing how to control tool availability dynamically.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_12

LANGUAGE: python
CODE:
```
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool, ToolDefinition


def launch_potato(target: str) -> str:
    return f'Potato launched at {target}!'


async def filter_out_tools_by_name(
    ctx: RunContext[bool], tool_defs: list[ToolDefinition]
) -> Union[list[ToolDefinition], None]:
    if ctx.deps:
        return [tool_def for tool_def in tool_defs if tool_def.name != 'launch_potato']
    return tool_defs


agent = Agent(
    'test',
    tools=[Tool(launch_potato)],
    prepare_tools=filter_out_tools_by_name,
    deps_type=bool,
)

result = agent.run_sync('testing...', deps=False)
print(result.output)
# > {"launch_potato":"Potato launched at a!"}
result = agent.run_sync('testing...', deps=True)
print(result.output)
# > success (no tool calls)
```

----------------------------------------

TITLE: Initialize AnthropicModel with custom AnthropicProvider
DESCRIPTION: This Python example illustrates how to provide a custom `AnthropicProvider` instance when initializing `AnthropicModel`. This allows for explicit configuration of the provider, such as passing the API key directly instead of relying on environment variables, or customizing other provider-specific settings for enhanced control.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/anthropic.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    'claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Define Pydantic AI Agent with Function Toolset
DESCRIPTION: This Python code demonstrates how to initialize a `pydantic-ai` agent using `FunctionToolset`. It defines two simple tools, `get_default_language` and `get_user_name`, and configures the agent to produce a `PersonalizedGreeting` BaseModel as output. The example shows a synchronous run of the agent and prints the structured result.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_10

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.toolsets.function import FunctionToolset

toolset = FunctionToolset()


@toolset.tool
def get_default_language():
    return 'en-US'


@toolset.tool
def get_user_name():
    return 'David'


class PersonalizedGreeting(BaseModel):
    greeting: str
    language_code: str


agent = Agent('openai:gpt-4o', toolsets=[toolset], output_type=PersonalizedGreeting)

result = agent.run_sync('Greet the user in a personalized way')
print(repr(result.output))
#> PersonalizedGreeting(greeting='Hello, David!', language_code='en-US')
```

----------------------------------------

TITLE: Integrate multiple ACI.dev tools with Pydantic AI Agent using Toolset
DESCRIPTION: This Python example illustrates how to integrate multiple ACI.dev tools, such as `OPEN_WEATHER_MAP__CURRENT_WEATHER` and `OPEN_WEATHER_MAP__FORECAST`, into a Pydantic AI `Agent` using the `ACIToolset`. The `ACIToolset` takes a list of ACI tool names and the `linked_account_owner_id` to enable the agent to access a collection of tools.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_17

LANGUAGE: python
CODE:
```
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import ACIToolset


toolset = ACIToolset(
    [
        'OPEN_WEATHER_MAP__CURRENT_WEATHER',
        'OPEN_WEATHER_MAP__FORECAST',
    ],
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

----------------------------------------

TITLE: Agent Delegation Control Flow Diagram
DESCRIPTION: This Mermaid diagram visualizes the execution flow of the agent delegation example. It illustrates how the 'joke_selection_agent' initiates a call to its 'joke_factory' tool, which then delegates to the 'joke_generation_agent', with control returning through the tool back to the parent agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#_snippet_1

LANGUAGE: mermaid
CODE:
```
graph TD
  START --> joke_selection_agent
  joke_selection_agent --> joke_factory["joke_factory (tool)"]
  joke_factory --> joke_generation_agent
  joke_generation_agent --> joke_factory
  joke_factory --> joke_selection_agent
  joke_selection_agent --> END
```

----------------------------------------

TITLE: Configure Pydantic AI with Grok (xAI)
DESCRIPTION: Explains how to connect `pydantic-ai` to Grok (xAI) models. This setup requires an API key obtained from the xAI API Console to authenticate and access Grok's language models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_21

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.grok import GrokProvider

model = OpenAIModel(
    'grok-2-1212',
    provider=GrokProvider(api_key='your-xai-api-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Configure Pydantic AI with Heroku AI
DESCRIPTION: Demonstrates how to initialize an `OpenAIModel` with `HerokuProvider` for Heroku AI. This setup requires a Heroku inference key to authenticate with the Heroku AI service.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_30

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.heroku import HerokuProvider

model = OpenAIModel(
    'claude-3-7-sonnet',
    provider=HerokuProvider(api_key='your-heroku-inference-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Run MCP SSE Server
DESCRIPTION: Command to start the Model Context Protocol (MCP) server using Deno, configured for Server-Sent Events (SSE) transport. This server must be running and accepting HTTP connections before the Pydantic AI agent can successfully connect to it.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_4

LANGUAGE: bash
CODE:
```
deno run \
  -N -R=node_modules -W=node_modules --node-modules-dir=auto \
  jsr:@pydantic/mcp-run-python sse
```

----------------------------------------

TITLE: Configure OpenAI Responses API with built-in tools
DESCRIPTION: Provides an example of configuring `OpenAIResponsesModel` with `OpenAIResponsesModelSettings` to enable built-in tools like web search. It demonstrates how to pass tool parameters and execute a query, showcasing the model's ability to perform external actions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_8

LANGUAGE: python
CODE:
```
from openai.types.responses import WebSearchToolParam  # (1)!

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
)
model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model=model, model_settings=model_settings)

result = agent.run_sync('What is the weather in Tokyo?')
print(result.output)
"""
As of 7:48 AM on Wednesday, April 2, 2025, in Tokyo, Japan, the weather is cloudy with a temperature of 53°F (12°C).
"""
```

----------------------------------------

TITLE: Pydantic-AI Programmatic Agent Hand-off Example
DESCRIPTION: This Python code demonstrates a programmatic agent hand-off using Pydantic-AI. It defines a `flight_search_agent` that uses an `openai:gpt-4o` model to find flight details via a `flight_search` tool. The example includes setting up usage limits and an asynchronous function `find_flight` that iteratively prompts the user and runs the agent until a flight is found or attempts are exhausted. It also defines a `SeatPreference` model, implying a subsequent agent for seat preference extraction.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#_snippet_3

LANGUAGE: python
CODE:
```
from typing import Literal, Union

from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits


class FlightDetails(BaseModel):
    flight_number: str


class Failed(BaseModel):
    """Unable to find a satisfactory choice."""


flight_search_agent = Agent[None, Union[FlightDetails, Failed]](  # (1)!
    'openai:gpt-4o',
    output_type=Union[FlightDetails, Failed],  # type: ignore
    system_prompt=(
        'Use the "flight_search" tool to find a flight '
        'from the given origin to the given destination.'
    ),
)


@flight_search_agent.tool  # (2)!
async def flight_search(
    ctx: RunContext[None], origin: str, destination: str
) -> Union[FlightDetails, None]:
    # in reality, this would call a flight search API or
    # use a browser to scrape a flight search website
    return FlightDetails(flight_number='AK456')


usage_limits = UsageLimits(request_limit=15)  # (3)!


async def find_flight(usage: Usage) -> Union[FlightDetails, None]:  # (4)!
    message_history: Union[list[ModelMessage], None] = None
    for _ in range(3):
        prompt = Prompt.ask(
            'Where would you like to fly from and to?',
        )
        result = await flight_search_agent.run(
            prompt,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, FlightDetails):
            return result.output
        else:
            message_history = result.all_messages(
                output_tool_return_content='Please try again.'
            )


class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']
```

----------------------------------------

TITLE: Conditionally Register Pydantic-AI Tool with `prepare` Method
DESCRIPTION: This example illustrates how to use the `prepare` method with a Pydantic-AI tool to control its registration dynamically. The `only_if_42` function, registered as the `prepare` method, checks the `RunContext`'s `deps` value and returns `None` if it's not 42, effectively preventing the `hitchhiker` tool from being available to the agent in that step.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_9

LANGUAGE: python
CODE:
```
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition

agent = Agent('test')


async def only_if_42(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> Union[ToolDefinition, None]:
    if ctx.deps == 42:
        return tool_def


@agent.tool(prepare=only_if_42)
def hitchhiker(ctx: RunContext[int], answer: str) -> str:
    return f'{ctx.deps} {answer}'


result = agent.run_sync('testing...', deps=41)
print(result.output)
# success (no tool calls)
result = agent.run_sync('testing...', deps=42)
print(result.output)
# {"hitchhiker":"42 a"}
```

----------------------------------------

TITLE: Demonstrating Toolset Registration and Override in Pydantic AI Agent
DESCRIPTION: This Python example illustrates the various ways to register and manage toolsets with a Pydantic AI `Agent`. It shows how toolsets can be provided at agent construction, during a specific `run_sync` call, and how they can be overridden using an `agent.override()` context manager. The `TestModel` is used to inspect the tools available to the agent at different stages, demonstrating the additive and overriding behaviors of toolset registration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset


def agent_tool():
    return "I'm registered directly on the agent"


def extra_tool():
    return "I'm passed as an extra tool for a specific run"


def override_tool():
    return "I override all other tools"


agent_toolset = FunctionToolset(tools=[agent_tool])
extra_toolset = FunctionToolset(tools=[extra_tool])
override_toolset = FunctionToolset(tools=[override_tool])

test_model = TestModel()
agent = Agent(test_model, toolsets=[agent_toolset])

result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool']

result = agent.run_sync('What tools are available?', toolsets=[extra_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool', 'extra_tool']

with agent.override(toolsets=[override_toolset]):
    result = agent.run_sync('What tools are available?', toolsets=[extra_toolset])
    print([t.name for t in test_model.last_model_request_parameters.function_tools])
    #> ['override_tool']
```

----------------------------------------

TITLE: Generate Pydantic Evals Dataset to YAML
DESCRIPTION: This Python example demonstrates how to generate a test dataset using `pydantic_evals.generation.generate_dataset`. It defines Pydantic `BaseModel` classes for `QuestionInputs`, `AnswerOutput`, and `MetadataType` to structure the data. The `generate_dataset` function is called with these schemas, a specified number of examples, and extra instructions for the LLM. The resulting dataset is then saved to a YAML file (`questions_cases.yaml`), automatically generating a corresponding JSON schema file for type checking and auto-completion. An example of the generated YAML output is provided in the original documentation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_10

LANGUAGE: python
CODE:
```
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from pydantic_evals import Dataset
from pydantic_evals.generation import generate_dataset


class QuestionInputs(BaseModel, use_attribute_docstrings=True):
    """Model for question inputs."""

    question: str
    """A question to answer"""
    context: str | None = None
    """Optional context for the question"""


class AnswerOutput(BaseModel, use_attribute_docstrings=True):
    """Model for expected answer outputs."""

    answer: str
    """The answer to the question"""
    confidence: float = Field(ge=0, le=1)
    """Confidence level (0-1)"""


class MetadataType(BaseModel, use_attribute_docstrings=True):
    """Metadata model for test cases."""

    difficulty: str
    """Difficulty level (easy, medium, hard)"""
    category: str
    """Question category"""


async def main():
    dataset = await generate_dataset(
        dataset_type=Dataset[QuestionInputs, AnswerOutput, MetadataType],
        n_examples=2,
        extra_instructions="""
        Generate question-answer pairs about world capitals and landmarks.
        Make sure to include both easy and challenging questions.
        """,
    )
    output_file = Path('questions_cases.yaml')
    dataset.to_file(output_file)
    print(output_file.read_text())
```

----------------------------------------

TITLE: Prepare Tool Definitions with PreparedToolset in Pydantic-AI
DESCRIPTION: This Python code demonstrates how to use `PreparedToolset` in `pydantic-ai` to dynamically modify `ToolDefinition`s. It defines an asynchronous function `add_descriptions` that takes `RunContext` and a list of `ToolDefinition`s, then uses `dataclasses.replace` to add descriptions to existing tools based on a predefined dictionary. The example shows how to apply this prepared toolset to an `Agent` and inspect the resulting tool definitions available to the model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_6

LANGUAGE: Python
CODE:
```
from dataclasses import replace
from typing import Union

from renamed_toolset import renamed_toolset

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

descriptions = {
    'temperature_celsius': 'Get the temperature in degrees Celsius',
    'temperature_fahrenheit': 'Get the temperature in degrees Fahrenheit',
    'weather_conditions': 'Get the current weather conditions',
    'current_time': 'Get the current time',
}

async def add_descriptions(ctx: RunContext, tool_defs: list[ToolDefinition]) -> Union[list[ToolDefinition], None]:
    return [
        replace(tool_def, description=description)
        if (description := descriptions.get(tool_def.name, None))
        else tool_def
        for tool_def
        in tool_defs
    ]

prepared_toolset = renamed_toolset.prepared(add_descriptions)

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[prepared_toolset])
result = agent.run_sync('What tools are available?')
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='temperature_celsius',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the temperature in degrees Celsius',
    ),
    ToolDefinition(
        name='temperature_fahrenheit',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the temperature in degrees Fahrenheit',
    ),
    ToolDefinition(
        name='weather_conditions',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the current weather conditions',
    ),
    ToolDefinition(
        name='current_time',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {},
            'type': 'object',
        },
        description='Get the current time',
    ),
]
"""
```

----------------------------------------

TITLE: Pydantic-AI Agent with Synchronous Dependencies
DESCRIPTION: Demonstrates how to configure a Pydantic-AI agent to use synchronous dependencies. This example shows a `httpx.Client` and a regular Python function for the system prompt, which `pydantic-ai` automatically runs in a thread pool using `run_in_executor`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/dependencies.md#_snippet_2

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext


@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.Client  # (1)!


agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)


@agent.system_prompt
def get_system_prompt(ctx: RunContext[MyDeps]) -> str:  # (2)!
    response = ctx.deps.http_client.get(
        'https://example.com', headers={'Authorization': f'Bearer {ctx.deps.api_key}'}
    )
    response.raise_for_status()
    return f'Prompt: {response.text}'


async def main():
    deps = MyDeps('foobar', httpx.Client())
    result = await agent.run(
        'Tell me a joke.',
        deps=deps,
    )
    print(result.output)
    #> Did you hear about the toothpaste scandal? They called it Colgate.
```

----------------------------------------

TITLE: Forcing Structured Output with Pydantic Model
DESCRIPTION: This example demonstrates how to configure a `pydantic-ai` agent to return structured data by specifying a Pydantic `BaseModel` as the `output_type`. It shows how to define the expected output structure, initialize the agent, run a query, and then access both the structured output and the usage statistics from the `AgentRunResult`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_0

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google-gla:gemini-1.5-flash', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
```

----------------------------------------

TITLE: Make a basic synchronous model request with pydantic-ai direct API
DESCRIPTION: This example demonstrates how to use `model_request_sync` from the `pydantic_ai.direct` module to send a simple text prompt to an LLM and retrieve its response. It shows how to construct a `ModelRequest` and access the content and usage statistics from the `ModelResponse`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/direct.md#_snippet_0

LANGUAGE: Python
CODE:
```
from pydantic_ai.direct import model_request_sync
from pydantic_ai.messages import ModelRequest

# Make a synchronous request to the model
model_response = model_request_sync(
    'anthropic:claude-3-5-haiku-latest',
    [ModelRequest.user_text_prompt('What is the capital of France?')]
)

print(model_response.parts[0].content)
# The capital of France is Paris.
print(model_response.usage)
# Usage(requests=1, request_tokens=56, response_tokens=7, total_tokens=63)
```

----------------------------------------

TITLE: Configure Hugging Face model with custom provider parameters
DESCRIPTION: This Python example illustrates how to programmatically instantiate `HuggingFaceProvider` with custom parameters like `api_key` and `provider_name`, and then pass this configured provider to the `HuggingFaceModel`. This allows fine-grained control over the inference provider settings, such as selecting a specific backend.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/huggingface.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider

model = HuggingFaceModel('Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(api_key='hf_token', provider_name='nebius'))
agent = Agent(model)
...
```

----------------------------------------

TITLE: Execute Python Code with Inline Script Dependencies via MCP Client
DESCRIPTION: This Python example demonstrates how to use the `mcp` client to execute a Python script that includes PEP 723 inline metadata for defining its dependencies (`pydantic`, `email-validator`). It shows connecting to a server via `stdio_client`, initializing a session, and calling the `run_python_code` tool. The output illustrates the successful execution and dependency recognition.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/run-python.md#_snippet_2

LANGUAGE: python
CODE:
```
from mcp import ClientSession
from mcp.client.stdio import stdio_client

# using `server_params` from the above example.
from mcp_run_python import server_params

code = """\
# /// script
# dependencies = ["pydantic", "email-validator"]
# ///
import pydantic

class Model(pydantic.BaseModel):
    email: pydantic.EmailStr

print(Model(email='hello@pydantic.dev'))
"""


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('run_python_code', {'python_code': code})
            print(result.content[0].text)
            """
            <status>success</status>
            <dependencies>["pydantic","email-validator"]</dependencies>
            <output>
            email='hello@pydantic.dev'
            </output>
            """
```

----------------------------------------

TITLE: Implement AG-UI State Management with Pydantic AI Agent
DESCRIPTION: This snippet demonstrates how to leverage AG-UI's state management capabilities with a Pydantic AI agent. It uses `StateDeps` with a Pydantic `BaseModel` (`DocumentState`) to define and automatically validate shared state between the UI and the server. The example shows how to initialize the agent with a `deps_type` and how to run the application using Uvicorn.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-4.1',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = agent.to_ag_ui(deps=StateDeps(DocumentState()))
```

LANGUAGE: shell
CODE:
```
uvicorn ag_ui_state:app --host 0.0.0.0 --port 9000
```

----------------------------------------

TITLE: Configure custom HTTP client for DeepSeekProvider
DESCRIPTION: This example illustrates how to provide a custom httpx.AsyncClient instance to a provider, such as DeepSeekProvider. This allows for advanced control over HTTP request behavior, including setting custom timeouts or other client-specific configurations.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_12

LANGUAGE: python
CODE:
```
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenAIModel(
    'deepseek-chat',
    provider=DeepSeekProvider(
        api_key='your-deepseek-api-key', http_client=custom_http_client
    ),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: pydantic-ai Anthropic Integration Classes and Configuration
DESCRIPTION: This section details the core classes for integrating Anthropic models with `pydantic-ai`, including `AnthropicModel` for model instantiation and `AnthropicProvider` for configuring API access and HTTP client settings. It outlines their constructors and key parameters for flexible setup and customization.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/anthropic.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
AnthropicModel:
  __init__(model_name: str, provider: AnthropicProvider | None = None)
    model_name: The specific Anthropic model identifier (e.g., 'claude-3-5-sonnet-latest').
    provider: An optional custom AnthropicProvider instance for advanced configuration.

AnthropicProvider:
  __init__(api_key: str | None = None, http_client: httpx.AsyncClient | None = None)
    api_key: Your Anthropic API key. If not provided, it defaults to the ANTHROPIC_API_KEY environment variable.
    http_client: An optional custom httpx.AsyncClient instance to control HTTP request behavior (e.g., timeouts).
```

----------------------------------------

TITLE: Generate Pydantic Evals Dataset to JSON
DESCRIPTION: This Python example illustrates how to save a generated `pydantic_evals` dataset to a JSON file. It reuses the Pydantic models defined previously and calls `generate_dataset` with similar parameters. The key difference is saving the output to `questions_cases.json`, which also generates a corresponding JSON schema file, enabling structured editing and validation. An example of the generated JSON output is provided in the original documentation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_11

LANGUAGE: python
CODE:
```
from pathlib import Path

from generate_dataset_example import AnswerOutput, MetadataType, QuestionInputs

from pydantic_evals import Dataset
from pydantic_evals.generation import generate_dataset


async def main():
    dataset = await generate_dataset(
        dataset_type=Dataset[QuestionInputs, AnswerOutput, MetadataType],
        n_examples=2,
        extra_instructions="""
        Generate question-answer pairs about world capitals and landmarks.
        Make sure to include both easy and challenging questions.
        """,
    )
    output_file = Path('questions_cases.json')
    dataset.to_file(output_file)
    print(output_file.read_text())
```

----------------------------------------

TITLE: Initialize Agent with Explicit CohereModel Object
DESCRIPTION: This Python example illustrates how to explicitly create a `CohereModel` object and then pass it to the `Agent` constructor. This method provides more granular control over the model instance before it's used by the agent, allowing for potential pre-configuration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/cohere.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command')
agent = Agent(model)
...
```

----------------------------------------

TITLE: Customize OpenAIModel behavior with ModelProfile
DESCRIPTION: This example illustrates how to fine-tune the behavior of OpenAIModel requests using ModelProfile or OpenAIModelProfile. This allows for handling provider-specific requirements, such as different JSON schema transformations for tool definitions or support for strict tool definitions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_10

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>.com', api_key='your-api-key'
    ),
    profile=OpenAIModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,  # Supported by any model class on a plain ModelProfile
        openai_supports_strict_tool_definition=False  # Supported by OpenAIModel only, requires OpenAIModelProfile
    )
)
agent = Agent(model)
```

----------------------------------------

TITLE: Pydantic-AI Development and Testing Commands
DESCRIPTION: Common `make` commands for setting up the development environment, running checks (format, lint, typecheck, test), building documentation, and executing specific tests within the Pydantic-AI project.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/CLAUDE.md#_snippet_0

LANGUAGE: Shell
CODE:
```
make install
make
make format
make lint
make typecheck
make typecheck-both
make test
make docs
make docs-serve
uv run pytest tests/test_agent.py::test_function_name -v
uv run pytest tests/test_agent.py -v
uv run pytest tests/test_agent.py -v -s
make install-all-python
make test-all-python
```

----------------------------------------

TITLE: Define Pydantic Evals Dataset and Case
DESCRIPTION: This Python example illustrates the creation of a `Case` and a `Dataset` in Pydantic Evals. A `Case` represents a single test scenario with inputs, expected outputs, and metadata. A `Dataset` is a collection of such cases, forming the basis for an evaluation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_evals import Case, Dataset

case1 = Case(
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)

dataset = Dataset(cases=[case1])
```

----------------------------------------

TITLE: Pydantic AI OpenTelemetry SDK Configuration Example
DESCRIPTION: Python code demonstrating how to manually configure OpenTelemetry for Pydantic AI without Logfire. It sets up an `OTLPSpanExporter` to send traces via HTTP to a local endpoint, initializes a `TracerProvider`, and instruments the `Agent` class for automatic tracing of AI operations.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_9

LANGUAGE: python
CODE:
```
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

from pydantic_ai.agent import Agent

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'
exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(exporter)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(span_processor)

set_tracer_provider(tracer_provider)

Agent.instrument_all()
agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> Paris
```

----------------------------------------

TITLE: Pydantic-AI Agent with Output Functions and Chained Agents Example
DESCRIPTION: This comprehensive Python example demonstrates the use of Pydantic-AI's output functions to define the expected output of an agent. It showcases a multi-agent architecture where a `router_agent` delegates natural language queries to a `sql_agent`. The `sql_agent` uses a `run_sql_query` output function to simulate database interaction, handling valid queries and raising `ModelRetry` for unsupported or invalid ones. Custom Pydantic models (`SQLFailure`, `RouterFailure`) are used for structured error reporting, illustrating robust error handling and inter-agent communication within the Pydantic-AI framework.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_3

LANGUAGE: Python
CODE:
```
import re
from typing import Union

from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior


class Row(BaseModel):
    name: str
    country: str


tables = {
    'capital_cities': [
        Row(name='Amsterdam', country='Netherlands'),
        Row(name='Mexico City', country='Mexico'),
    ]
}


class SQLFailure(BaseModel):
    """An unrecoverable failure. Only use this when you can't change the query to make it work."""

    explanation: str


def run_sql_query(query: str) -> list[Row]:
    """Run a SQL query on the database."""

    select_table = re.match(r'SELECT (.+) FROM (\w+)', query)
    if select_table:
        column_names = select_table.group(1)
        if column_names != '*':
            raise ModelRetry("Only 'SELECT *' is supported, you'll have to do column filtering manually.")

        table_name = select_table.group(2)
        if table_name not in tables:
            raise ModelRetry(
                f"Unknown table '{table_name}' in query '{query}'. Available tables: {', '.join(tables.keys())}."
            )

        return tables[table_name]

    raise ModelRetry(f"Unsupported query: '{query}'.")


sql_agent = Agent[None, Union[list[Row], SQLFailure]](
    'openai:gpt-4o',
    output_type=[run_sql_query, SQLFailure],
    instructions='You are a SQL agent that can run SQL queries on a database.',
)


async def hand_off_to_sql_agent(ctx: RunContext, query: str) -> list[Row]:
    """I take natural language queries, turn them into SQL, and run them on a database."""

    # Drop the final message with the output tool call, as it shouldn't be passed on to the SQL agent
    messages = ctx.messages[:-1]
    try:
        result = await sql_agent.run(query, message_history=messages)
        output = result.output
        if isinstance(output, SQLFailure):
            raise ModelRetry(f'SQL agent failed: {output.explanation}')
        return output
    except UnexpectedModelBehavior as e:
        # Bubble up potentially retryable errors to the router agent
        if (cause := e.__cause__) and isinstance(cause, ModelRetry):
            raise ModelRetry(f'SQL agent failed: {cause.message}') from e
        else:
            raise


class RouterFailure(BaseModel):
    """Use me when no appropriate agent is found or the used agent failed."""

    explanation: str


router_agent = Agent[None, Union[list[Row], RouterFailure]](
    'openai:gpt-4o',
    output_type=[hand_off_to_sql_agent, RouterFailure],
    instructions='You are a router to other agents. Never try to solve a problem yourself, just pass it on.',
)

result = router_agent.run_sync('Select the names and countries of all capitals')
print(result.output)
"""
[
    Row(name='Amsterdam', country='Netherlands'),
    Row(name='Mexico City', country='Mexico'),
]
"""

result = router_agent.run_sync('Select all pets')
print(repr(result.output))
"""
RouterFailure(explanation="The requested table 'pets' does not exist in the database. The only available table is 'capital_cities', which does not contain data about pets.")
"""

result = router_agent.run_sync('How do I fly from Amsterdam to Mexico City?')
print(repr(result.output))
"""
RouterFailure(explanation='I am not equipped to provide travel information, such as flights from Amsterdam to Mexico City.')
"""
```

----------------------------------------

TITLE: Customize Tools with prepare_tools based on Model Type
DESCRIPTION: This example demonstrates how to use the `prepare_tools` function to dynamically modify tool definitions. Specifically, it shows how to set all tools to 'strict' mode if the agent's model is identified as an OpenAI model, showcasing global modification of tool properties.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_11

LANGUAGE: python
CODE:
```
from dataclasses import replace
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models.test import TestModel


async def turn_on_strict_if_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> Union[list[ToolDefinition], None]:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs


test_model = TestModel()
agent = Agent(test_model, prepare_tools=turn_on_strict_if_openai)


@agent.tool_plain
def echo(message: str) -> str:
    return message


agent.run_sync('testing...')
assert test_model.last_model_request_parameters.function_tools[0].strict is None

# Set the system attribute of the test_model to 'openai'
test_model._system = 'openai'

agent.run_sync('testing with openai...')
assert test_model.last_model_request_parameters.function_tools[0].strict
```

----------------------------------------

TITLE: Run Pydantic AI Slack Qualifier as Ephemeral Modal App
DESCRIPTION: Executes the Slack lead qualifier application as a temporary Modal app. This command starts a web function that processes incoming events and runs until manually terminated, useful for local development and testing.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_3

LANGUAGE: bash
CODE:
```
python/uv-run -m modal serve -m pydantic_ai_examples.slack_lead_qualifier.modal
```

----------------------------------------

TITLE: Integrate multiple ACI.dev tools with Pydantic AI using ACIToolset
DESCRIPTION: This Python example illustrates how to integrate multiple ACI.dev tools into a Pydantic AI `Agent` using the `ACIToolset`. It demonstrates initializing the `ACIToolset` with a list of tool names and a linked account owner ID, then passing the toolset to the `Agent` constructor. This approach is suitable for applications requiring access to several ACI functionalities.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_20

LANGUAGE: python
CODE:
```
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import ACIToolset


toolset = ACIToolset(
    [
        'OPEN_WEATHER_MAP__CURRENT_WEATHER',
        'OPEN_WEATHER_MAP__FORECAST',
    ],
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

----------------------------------------

TITLE: Generate Tool Schema with Docstring Descriptions in Pydantic AI
DESCRIPTION: This Python example demonstrates how Pydantic AI extracts detailed JSON schemas for tools directly from function signatures and docstrings. It showcases the use of `docstring_format='google'` and `require_parameter_descriptions=True` to ensure parameter descriptions are included in the generated schema, which is then printed using `FunctionModel`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

agent = Agent()


@agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def foobar(a: int, b: str, c: dict[str, list[float]]) -> str:
    """Get me foobar.

    Args:
        a: apple pie
        b: banana cake
        c: carrot smoothie
    """
    return f'{a} {b} {c}'


def print_schema(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool = info.function_tools[0]
    print(tool.description)
    #> Get me foobar.
    print(tool.parameters_json_schema)
    """
    {
        'additionalProperties': False,
        'properties': {
            'a': {'description': 'apple pie', 'type': 'integer'},
            'b': {'description': 'banana cake', 'type': 'string'},
            'c': {
                'additionalProperties': {'items': {'type': 'number'}, 'type': 'array'},
                'description': 'carrot smoothie',
                'type': 'object',
            },
        },
        'required': ['a', 'b', 'c'],
        'type': 'object',
    }
    """
    return ModelResponse(parts=[TextPart('foobar')])


agent.run_sync('hello', model=FunctionModel(print_schema))
```

----------------------------------------

TITLE: Pydantic-AI Type-Safe Agent Output Definition
DESCRIPTION: Demonstrates how to define a type-safe agent in Pydantic-AI by specifying an `output_type` using a Pydantic `BaseModel`. This ensures that the agent's output conforms to a predefined schema, enabling robust data validation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/CLAUDE.md#_snippet_3

LANGUAGE: Python
CODE:
```
class OutputModel(BaseModel):
    result: str
    confidence: float

agent: Agent[MyDeps, OutputModel] = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
    output_type=OutputModel
)
```

----------------------------------------

TITLE: Creating a custom TypeAdapter for Pydantic AI ModelMessage list
DESCRIPTION: This example shows how to manually create a `TypeAdapter` for a list of `ModelMessage` objects from `pydantic_ai.messages`. This is an alternative to using the pre-exported `ModelMessagesTypeAdapter` and can be useful for custom serialization/deserialization needs.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage
ModelMessagesTypeAdapter = TypeAdapter(list[ModelMessage])
```

----------------------------------------

TITLE: Integrate Single LangChain Tool with Pydantic AI Agent in Python
DESCRIPTION: This example shows how to integrate a single LangChain tool, such as `DuckDuckGoSearchRun`, into a Pydantic AI agent using the `tool_from_langchain` convenience method. It highlights the necessary `langchain-community` dependency and notes that Pydantic AI delegates argument validation to the LangChain tool in this integration scenario.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_14

LANGUAGE: python
CODE:
```
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import tool_from_langchain


search = DuckDuckGoSearchRun()
search_tool = tool_from_langchain(search)

agent = Agent(
    'google-gla:gemini-2.0-flash',
    tools=[search_tool],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')
print(result.output)
```

----------------------------------------

TITLE: Define and Register Agent Tools (Python)
DESCRIPTION: This Python code demonstrates how to initialize a `pydantic-ai` agent and register custom functions as tools using `@agent.tool_plain` (for context-independent tools) and `@agent.tool` (for tools requiring `RunContext`). The example implements a dice game where the agent uses `roll_dice` and `get_player_name` tools to determine a winner based on a user's guess and name.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_0

LANGUAGE: python
CODE:
```
import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    'google-gla:gemini-1.5-flash',  # (1)!
    deps_type=str,  # (2)!
    system_prompt=(
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    ),
)


@agent.tool_plain  # (3)!
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool  # (4)!
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


dice_result = agent.run_sync('My guess is 4', deps='Anne')  # (5)!
print(dice_result.output)
```

----------------------------------------

TITLE: Evaluate Python Function with Pydantic Evals
DESCRIPTION: This example demonstrates the core workflow of Pydantic Evals. It shows how to define a test case with inputs and expected output, create a custom evaluator (`MatchAnswer`) to assess the function's output, and combine these with a built-in evaluator (`IsInstance`) into a `Dataset`. Finally, it illustrates running an asynchronous Python function (`answer_question`) against the defined dataset and printing the evaluation report, which summarizes scores and assertions, similar to the provided output:

```
                                    Evaluation Summary: answer_question
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID          ┃ Inputs                         ┃ Outputs ┃ Scores            ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ capital_question │ What is the capital of France? │ Paris   │ MatchAnswer: 1.00 │ ✔          │     10ms │
├──────────────────┼────────────────────────────────┼─────────┼───────────────────┼────────────┼──────────┤
│ Averages         │                                │         │ MatchAnswer: 1.00 │ 100.0% ✔   │     10ms │
└──────────────────┴────────────────────────────────┴─────────┴───────────────────┴────────────┴──────────┘
```

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/pydantic_evals/README.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

# Define a test case with inputs and expected output
case = Case(
    name='capital_question',
    inputs='What is the capital of France?',
    expected_output='Paris',
)

# Define a custom evaluator
class MatchAnswer(Evaluator[str, str]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        elif isinstance(ctx.output, str) and ctx.expected_output.lower() in ctx.output.lower():
            return 0.8
        return 0.0

# Create a dataset with the test case and evaluators
dataset = Dataset(
    cases=[case],
    evaluators=[IsInstance(type_name='str'), MatchAnswer()],
)

# Define the function to evaluate
async def answer_question(question: str) -> str:
    return 'Paris'

# Run the evaluation
report = dataset.evaluate_sync(answer_question)
report.print(include_input=True, include_output=True)
```

----------------------------------------

TITLE: Stream User Profile with Fine-Grained Validation (pydantic-ai)
DESCRIPTION: This example illustrates advanced streaming with `pydantic-ai`, providing fine-grained control over validation using `stream_structured` and `validate_structured_output`. It demonstrates how to handle `ValidationError` during streaming and leverage `allow_partial` for robust partial data validation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_16

LANGUAGE: python
CODE:
```
from datetime import date

from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_ai import Agent


class UserProfile(TypedDict, total=False):
    name: str
    dob: date
    bio: str


agent = Agent('openai:gpt-4o', output_type=UserProfile)


async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for message, last in result.stream_structured(debounce_by=0.01):
            try:
                profile = await result.validate_structured_output(
                    message,
                    allow_partial=not last,
                )
            except ValidationError:
                continue
            print(profile)
```

----------------------------------------

TITLE: Demonstrate Pydantic AI FallbackModel usage with OpenAI and Anthropic
DESCRIPTION: This Python example demonstrates how to use `FallbackModel` from `pydantic_ai` to chain multiple language models. It shows how the agent automatically falls back from an initial failing OpenAI model to a working Anthropic model, and how to inspect the `model_name` in the `ModelResponse` to identify the successful model. Dependencies include `pydantic_ai`, `OpenAIModel`, `AnthropicModel`, and `FallbackModel`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/index.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
response = agent.run_sync('What is the capital of France?')
print(response.data)
#> Paris

print(response.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='What is the capital of France?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[TextPart(content='Paris', part_kind='text')],
        model_name='claude-3-5-sonnet-latest',
        timestamp=datetime.datetime(...),
        kind='response',
        vendor_id=None,
    ),
]
"""
```

----------------------------------------

TITLE: Provide local document input to LLM using Pydantic-AI BinaryContent
DESCRIPTION: This example shows how to send the binary content of a local document to an LLM using `pydantic-ai`'s `BinaryContent` class. It reads the bytes of a local PDF file and passes them to the `Agent` along with the appropriate `media_type`. This method is useful for processing documents stored locally or generated dynamically.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/input.md#_snippet_3

LANGUAGE: python
CODE:
```
from pathlib import Path
from pydantic_ai import Agent, BinaryContent

pdf_path = Path('document.pdf')
agent = Agent(model='anthropic:claude-3-sonnet')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf'),
    ]
)
print(result.output)
```

----------------------------------------

TITLE: Accessing Agent Run Output in Pydantic AI
DESCRIPTION: This Python example demonstrates how to initialize a Pydantic AI agent, run a synchronous command, and then access the agent's output from the `result` object. It showcases a basic interaction where the agent responds to a prompt and its output is printed.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result = agent.run_sync('Tell me a joke.')
print(result.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.
```

----------------------------------------

TITLE: Limit Pydantic AI Agent Response Tokens
DESCRIPTION: This example demonstrates how to apply `UsageLimits` to a Pydantic AI agent to restrict the number of response tokens. It shows a successful run within the limit and a `UsageLimitExceeded` exception when the response exceeds the configured `response_tokens_limit`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

agent = Agent('anthropic:claude-3-5-sonnet-latest')

result_sync = agent.run_sync(
    'What is the capital of Italy? Answer with just the city.',
    usage_limits=UsageLimits(response_tokens_limit=10),
)
print(result_sync.output)
# Rome
print(result_sync.usage())
# Usage(requests=1, request_tokens=62, response_tokens=1, total_tokens=63)

try:
    result_sync = agent.run_sync(
        'What is the capital of Italy? Answer with a paragraph.',
        usage_limits=UsageLimits(response_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    # Exceeded the response_tokens_limit of 10 (response_tokens=32)
```

----------------------------------------

TITLE: Integrate LangChain Tools with Pydantic AI Agent
DESCRIPTION: This snippet shows how to integrate tools from LangChain's community library into a Pydantic AI agent using `LangChainToolset`. It highlights the necessary `langchain-community` package installation and notes that Pydantic AI does not validate arguments for LangChain tools, leaving it to the model and the LangChain tool itself.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_13

LANGUAGE: python
CODE:
```
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset


toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-4o', toolsets=[toolset])
# ...
```

----------------------------------------

TITLE: Define Custom Tool from Schema for Pydantic-AI Agent
DESCRIPTION: This Python example demonstrates how to create a `Tool` for a Pydantic-AI agent from a function (`foobar`) that lacks proper documentation. It uses `Tool.from_schema` to explicitly define the tool's name, description, and a JSON schema for its arguments, allowing the agent to correctly interpret and use the function. Note that argument validation is not performed, and all arguments are passed as keyword arguments.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, Tool
from pydantic_ai.models.test import TestModel


def foobar(**kwargs) -> str:
    return kwargs['a'] + kwargs['b']

tool = Tool.from_schema(
    function=foobar,
    name='sum',
    description='Sum two numbers.',
    json_schema={
        'additionalProperties': False,
        'properties': {
            'a': {'description': 'the first number', 'type': 'integer'},
            'b': {'description': 'the second number', 'type': 'integer'},
        },
        'required': ['a', 'b'],
        'type': 'object',
    }
)

test_model = TestModel()
agent = Agent(test_model, tools=[tool])

result = agent.run_sync('testing...')
print(result.output)
# {"sum":0}
```

----------------------------------------

TITLE: Pydantic AI Text Output with TextOutput
DESCRIPTION: Demonstrates how to configure a Pydantic AI agent to produce plain text output using the `TextOutput` marker class. The example shows an agent processing a query and returning a list of words from the model's text response, illustrating how to transform raw model output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, TextOutput

def split_into_words(text: str) -> list[str]:
    return text.split()


agent = Agent(
    'openai:gpt-4o',
    output_type=TextOutput(split_into_words),
)
result = agent.run_sync('Who was Albert Einstein?')
print(result.output)
#> ['Albert', 'Einstein', 'was', 'a', 'German-born', 'theoretical', 'physicist.']
```

----------------------------------------

TITLE: Agent Delegation Example with Pydantic AI
DESCRIPTION: This Python code demonstrates agent delegation in Pydantic AI. A 'joke_selection_agent' utilizes a 'joke_factory' tool, which in turn calls a 'joke_generation_agent' to produce jokes. It illustrates how to pass 'ctx.usage' to the delegate agent to ensure combined usage tracking and defines two agents with distinct models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

joke_selection_agent = Agent(  # (1)!
    'openai:gpt-4o',
    system_prompt=(
        'Use the `joke_factory` to generate some jokes, then choose the best. '
        'You must return just a single joke.'
    ),
)
joke_generation_agent = Agent(  # (2)!
    'google-gla:gemini-1.5-flash', output_type=list[str]
)


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    r = await joke_generation_agent.run(  # (3)!
        f'Please generate {count} jokes.',
        usage=ctx.usage,  # (4)!
    )
    return r.output  # (5)!


result = joke_selection_agent.run_sync(
    'Tell me a joke.',
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=300),
)
print(result.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.
print(result.usage())
#> Usage(requests=3, request_tokens=204, response_tokens=24, total_tokens=228)
```

----------------------------------------

TITLE: Provide local image input to LLM using Pydantic-AI BinaryContent
DESCRIPTION: This example illustrates how to send local image data to an LLM using `pydantic-ai`'s `BinaryContent` class. It first fetches image bytes (simulating a local file read) and then wraps them in `BinaryContent`, specifying the correct `media_type`. This allows the LLM to process image content that is not accessible via a public URL.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/input.md#_snippet_1

LANGUAGE: python
CODE:
```
import httpx

from pydantic_ai import Agent, BinaryContent

image_response = httpx.get('https://iili.io/3Hs4FMg.png')  # Pydantic logo

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(
    [
        'What company is this logo from?',
        BinaryContent(data=image_response.content, media_type='image/png'),
    ]
)
print(result.output)
```

----------------------------------------

TITLE: Implement MCP Server with Pydantic AI Sampling
DESCRIPTION: This Python code extends the MCP server example to incorporate sampling, allowing the Pydantic AI agent to make LLM calls back through the MCP client. By using `MCPSamplingModel`, the agent's `run` method directs LLM requests to the client session, enabling more flexible and controlled LLM interactions within the MCP ecosystem.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/server.md#_snippet_2

LANGUAGE: python
CODE:
```
from mcp.server.fastmcp import Context, FastMCP

from pydantic_ai import Agent
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

server = FastMCP('Pydantic AI Server with sampling')
server_agent = Agent(system_prompt='always reply in rhyme')


@server.tool()
async def poet(ctx: Context, theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}', model=MCPSamplingModel(session=ctx.session))
    return r.output


if __name__ == '__main__':
    server.run()
```

----------------------------------------

TITLE: Make Synchronous AI Model Request (Python)
DESCRIPTION: This snippet demonstrates how to make a synchronous request to an AI model, such as 'anthropic:claude-3-5-haiku-latest', using the `model_request_sync` function. It shows how to pass a user text prompt and enable instrumentation for observability. The example then accesses and prints the content of the model's response.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/direct.md#_snippet_4

LANGUAGE: Python
CODE:
```
model_response = model_request_sync(
    'anthropic:claude-3-5-haiku-latest',
    [ModelRequest.user_text_prompt('What is the capital of France?')],
    instrument=True
)

print(model_response.parts[0].content)
```

----------------------------------------

TITLE: Initialize AnthropicModel directly and pass to Agent
DESCRIPTION: This Python code shows how to explicitly create an `AnthropicModel` instance with a specific model name and then pass this model object to the `Agent` constructor. This approach provides more direct control over the model configuration before it's used by the agent, allowing for more complex setups.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/anthropic.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('claude-3-5-sonnet-latest')
agent = Agent(model)
...
```

----------------------------------------

TITLE: Support MCP Sampling in a Python Client
DESCRIPTION: This Python client example demonstrates how to support MCP sampling by providing a `sampling_callback` function to the `ClientSession`. This callback intercepts LLM calls originating from the server, allowing the client to handle them (e.g., by routing them to an actual LLM or providing a mock response), thus enabling a full sampling workflow.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/server.md#_snippet_3

LANGUAGE: python
CODE:
```
import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import CreateMessageRequestParams, CreateMessageResult, ErrorData, TextContent


async def sampling_callback(
    context: RequestContext[ClientSession, Any], params: CreateMessageRequestParams
) -> CreateMessageResult | ErrorData:
    print('sampling system prompt:', params.systemPrompt)
    #> sampling system prompt: always reply in rhyme
    print('sampling messages:', params.messages)
    """
    sampling messages:
    [
        SamplingMessage(
            role='user',
            content=TextContent(
                type='text',
                text='write a poem about socks',
                annotations=None,
                meta=None,
            ),
        )
    ]
    """

    # TODO get the response content by calling an LLM...
    response_content = 'Socks for a fox.'

    return CreateMessageResult(
        role='assistant',
        content=TextContent(type='text', text=response_content),
        model='fictional-llm',
    )


async def client():
    server_params = StdioServerParameters(command='python', args=['mcp_server_sampling.py'])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=sampling_callback) as session:
            await session.initialize()
            result = await session.call_tool('poet', {'theme': 'socks'})
            print(result.content[0].text)
            #> Socks for a fox.


if __name__ == '__main__':
    asyncio.run(client())
```

----------------------------------------

TITLE: Customize CohereProvider with Custom HTTP Client
DESCRIPTION: This Python example shows how to configure the `CohereProvider` with a custom `httpx.AsyncClient`, enabling advanced HTTP client settings such as custom timeouts. This provides fine-grained control over network requests made to the Cohere API, which can be crucial for performance or specific network environments.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/cohere.md#_snippet_5

LANGUAGE: python
CODE:
```
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

custom_http_client = AsyncClient(timeout=30)
model = CohereModel(
    'command',
    provider=CohereProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Run Pydantic AI Agent with Event Streaming
DESCRIPTION: This asynchronous Python example demonstrates how to run a Pydantic AI agent using `agent.run()` with an `event_stream_handler`. It captures and prints all events generated during the agent's execution, including tool calls and final text output, showcasing the detailed flow of an agent's response.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_3

LANGUAGE: python
CODE:
```
from run_stream_events import weather_agent, event_stream_handler, output_messages

import asyncio


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    run = await weather_agent.run(user_prompt, event_stream_handler=event_stream_handler)

    output_messages.append(f'[Final Output] {run.output}')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        "[Request] Part 0 text delta: 'warm and sunny '",
        "[Request] Part 0 text delta: 'in Paris on '",
        "[Request] Part 0 text delta: 'Tuesday.'",
        '[Final Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """
```

----------------------------------------

TITLE: Define and Run a Simple Pydantic Graph
DESCRIPTION: This Python example demonstrates how to create a basic computational graph using `pydantic-graph`. It defines two nodes, `DivisibleBy5` and `Increment`, which interact to find the next multiple of 5. The snippet shows node parameterization, asynchronous `run` methods, graph instantiation, and synchronous execution using `run_sync`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_4

LANGUAGE: Python
CODE:
```
from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class DivisibleBy5(BaseNode[None, None, int]):  # (1)!
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode):  # (2)!
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)


fives_graph = Graph(nodes=[DivisibleBy5, Increment])  # (3)!
result = fives_graph.run_sync(DivisibleBy5(4))  # (4)!
print(result.output)
```

----------------------------------------

TITLE: Integrate Tavily Search with Pydantic-AI Agent
DESCRIPTION: This Python code snippet demonstrates how to set up and use a Pydantic-AI agent with the Tavily search tool. It requires a `TAVILY_API_KEY` environment variable for authentication. The agent is configured to search Tavily for a given query, and the example shows it retrieving top news in the GenAI world, including links and summaries, then printing the output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/common-tools.md#_snippet_3

LANGUAGE: Python
CODE:
```
import os

from pydantic_ai.agent import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

api_key = os.getenv('TAVILY_API_KEY')
assert api_key is not None

agent = Agent(
    'openai:o3-mini',
    tools=[tavily_search_tool(api_key)],
    system_prompt='Search Tavily for the given query and return the results.',
)

result = agent.run_sync('Tell me the top news in the GenAI world, give me links.')
print(result.output)
"""
Here are some of the top recent news articles related to GenAI:

1. How CLEAR users can improve risk analysis with GenAI – Thomson Reuters
   Read more: https://legal.thomsonreuters.com/blog/how-clear-users-can-improve-risk-analysis-with-genai/
   (This article discusses how CLEAR's new GenAI-powered tool streamlines risk analysis by quickly summarizing key information from various public data sources.)

2. TELUS Digital Survey Reveals Enterprise Employees Are Entering Sensitive Data Into AI Assistants More Than You Think – FT.com
   Read more: https://markets.ft.com/data/announce/detail?dockey=600-202502260645BIZWIRE_USPRX____20250226_BW490609-1
   (This news piece highlights findings from a TELUS Digital survey showing that many enterprise employees use public GenAI tools and sometimes even enter sensitive data.)

3. The Essential Guide to Generative AI – Virtualization Review
   Read more: https://virtualizationreview.com/Whitepapers/2025/02/SNOWFLAKE-The-Essential-Guide-to-Generative-AI.aspx
   (This guide provides insights into how GenAI is revolutionizing enterprise strategies and productivity, with input from industry leaders.)

Feel free to click on the links to dive deeper into each story!
"""
```

----------------------------------------

TITLE: Multi-Agent Flight Booking Control Flow (Mermaid)
DESCRIPTION: Illustrates the delegation and hand-off process between agents in the flight booking system, showing the interaction sequence from search to seat selection and purchase. It visualizes the flow from initial search to final purchase, including human confirmation and seat choice steps.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/flight-booking.md#_snippet_0

LANGUAGE: mermaid
CODE:
```
graph TD
  START --> search_agent("search agent")
  search_agent --> extraction_agent("extraction agent")
  extraction_agent --> search_agent
  search_agent --> human_confirm("human confirm")
  human_confirm --> search_agent
  search_agent --> FAILED
  human_confirm --> find_seat_function("find seat function")
  find_seat_function --> human_seat_choice("human seat choice")
  human_seat_choice --> find_seat_agent("find seat agent")
  find_seat_agent --> find_seat_function
  find_seat_function --> buy_flights("buy flights")
  buy_flights --> SUCCESS
```

----------------------------------------

TITLE: Pydantic-AI error handling with retry mechanisms
DESCRIPTION: This Python example illustrates the context for handling errors when using `pydantic-ai` with retry transports. It highlights that if all retry attempts fail, the last exception will be re-raised, necessitating appropriate error handling in the application code.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_12

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

----------------------------------------

TITLE: Set Custom OpenTelemetry SDK Providers in Pydantic AI
DESCRIPTION: Python example demonstrating how to override the default global OpenTelemetry `TracerProvider` and `EventLoggerProvider` in Pydantic AI. This is achieved by passing custom instances of these providers via `InstrumentationSettings` when initializing an `Agent` or instrumenting all agents globally.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_11

LANGUAGE: python
CODE:
```
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk.trace import TracerProvider

from pydantic_ai.agent import Agent, InstrumentationSettings

instrumentation_settings = InstrumentationSettings(
    tracer_provider=TracerProvider(),
    event_logger_provider=EventLoggerProvider(),
)

agent = Agent('gpt-4o', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)
```

----------------------------------------

TITLE: Simplify Tool Schema for Single Pydantic Model Parameters
DESCRIPTION: This Python example illustrates how Pydantic AI simplifies the JSON schema for a tool when its function accepts a single parameter that is a Pydantic `BaseModel`. Instead of nesting, the tool's schema directly reflects the schema of the Pydantic model, as demonstrated by inspecting `test_model.last_model_request_parameters.function_tools`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent()


class Foobar(BaseModel):
    """This is a Foobar"""

    x: int
    y: str
    z: float = 3.14


@agent.tool_plain
def foobar(f: Foobar) -> str:
    return str(f)


test_model = TestModel()
result = agent.run_sync('hello', model=test_model)
print(result.output)
#> {"foobar":"x=0 y='a' z=3.14"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='foobar',
        parameters_json_schema={
            'properties': {
                'x': {'type': 'integer'},
                'y': {'type': 'string'},
                'z': {'default': 3.14, 'type': 'number'},
            },
            'required': ['x', 'y'],
            'title': 'Foobar',
            'type': 'object',
        },
        description='This is a Foobar',
    )
]
"""
```

----------------------------------------

TITLE: Initialize GoogleModel with Generative Language API Key
DESCRIPTION: This Python code demonstrates how to explicitly create a `GoogleProvider` instance using an API key and then use it to initialize a `GoogleModel` for interacting with Google's Gemini models via the Generative Language API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your-api-key')
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Stream AI Agent Response (Deltas)
DESCRIPTION: This example shows how to stream text responses as deltas (incremental changes) rather than the entire text. By passing `delta=True` to `stream_text()`, each yielded item represents a new piece of the response, suitable for real-time UI updates.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_13

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text(delta=True):
            print(message)
            # The first known
            # use of "hello,"
            # world" was in
            # a 1974 textbook
            # about the C
            # programming language.
```

----------------------------------------

TITLE: Iterate Pydantic-AI Agent Nodes with async for
DESCRIPTION: This example demonstrates using `async for` with `agent.iter()` to automatically record each node executed by a Pydantic-AI agent. The `AgentRun` object acts as an async-iterable, yielding `BaseNode` or `End` objects representing steps in the agent's execution graph. This method is suitable for observing the complete flow.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')


async def main():
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)
    """
    [
        UserPromptNode(
            user_prompt='What is the capital of France?',
            instructions=None,
            instructions_functions=[],
            system_prompts=(),
            system_prompt_functions=[],
            system_prompt_dynamic_functions={},
        ),
        ModelRequestNode(
            request=ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=datetime.datetime(...),
                    )
                ]
            )
        ),
        CallToolsNode(
            model_response=ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=
Usage(
                    requests=1, request_tokens=56, response_tokens=7, total_tokens=63
                ),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    """
    print(agent_run.result.output)
    # > The capital of France is Paris.
```

----------------------------------------

TITLE: Clone and Navigate Pydantic AI Repository
DESCRIPTION: Instructions to clone the Pydantic AI repository from GitHub and change the current directory into the newly cloned repository. This is the first step for local development.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/contributing.md#_snippet_0

LANGUAGE: bash
CODE:
```
git clone git@github.com:<your username>/pydantic-ai.git
cd pydantic-ai
```

----------------------------------------

TITLE: Handle Deferred Tool Calls in Pydantic AI Frontend
DESCRIPTION: This Python example demonstrates how a frontend application can manage deferred tool calls from a Pydantic AI agent. It defines local tool functions, processes agent outputs, and handles tool returns to continue the conversation loop. The `run_agent` function is assumed to be an API call to the backend.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_12

LANGUAGE: python
CODE:
```
from deferred_toolset_api import run_agent

from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.output import DeferredToolCalls

frontend_tool_definitions = [
    ToolDefinition(
        name='get_preferred_language',
        parameters_json_schema={'type': 'object', 'properties': {'default_language': {'type': 'string'}}},
        description="Get the user's preferred language from their browser",
    )
]

def get_preferred_language(default_language: str) -> str:
    return 'es-MX' # (1)!

frontend_tool_functions = {'get_preferred_language': get_preferred_language}

messages: list[ModelMessage] = [
    ModelRequest(
        parts=[
            UserPromptPart(content='Greet the user in a personalized way')
        ]
    )
]

final_output = None
while True:
    output, new_messages = run_agent(messages, frontend_tool_definitions)
    messages += new_messages

    if not isinstance(output, DeferredToolCalls):
        final_output = output
        break

    print(output.tool_calls)
    """
    [
        ToolCallPart(
            tool_name='get_preferred_language',
            args={'default_language': 'en-US'},
            tool_call_id='pyd_ai_tool_call_id',
        )
    ]
    """
    for tool_call in output.tool_calls:
        if function := frontend_tool_functions.get(tool_call.tool_name):
            part = ToolReturnPart(
                tool_name=tool_call.tool_name,
                content=function(**tool_call.args_as_dict()),
                tool_call_id=tool_call.tool_call_id,
            )
        else:
            part = RetryPromptPart(
                tool_name=tool_call.tool_name,
                content=f'Unknown tool {tool_call.tool_name!r}',
                tool_call_id=tool_call.tool_call_id,
            )
        messages.append(ModelRequest(parts=[part]))

print(repr(final_output))
"""
PersonalizedGreeting(greeting='Hola, David! Espero que tengas un gran día!', language_code='es-MX')
"""
```

----------------------------------------

TITLE: Apply Multiple History Processors in Python
DESCRIPTION: This example demonstrates how to chain multiple history processors together. When multiple processors are provided to an `Agent`, they are applied sequentially in the order they are listed. This allows for complex message manipulation pipelines, combining different filtering and summarization strategies.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_13

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    return [msg for msg in messages if isinstance(msg, ModelRequest)]


def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-5:]


agent = Agent('openai:gpt-4o', history_processors=[filter_responses, summarize_old_messages])
```

----------------------------------------

TITLE: MCP Server Example with Sampling for Image Generation
DESCRIPTION: This Python script defines an MCP server that exposes an `image_generator` tool. It demonstrates how an MCP server can use `ctx.session.create_message` to perform an LLM call via the connected MCP client (sampling) to generate an SVG image based on user input. The script handles potential markdown wrapping of the SVG output and writes the result to a file.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_11

LANGUAGE: python
CODE:
```
import re
from pathlib import Path

from mcp import SamplingMessage
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent

app = FastMCP()


@app.tool()
async def image_generator(ctx: Context, subject: str, style: str) -> str:
    prompt = f'{subject=} {style=}'
    # `ctx.session.create_message` is the sampling call
    result = await ctx.session.create_message(
        [SamplingMessage(role='user', content=TextContent(type='text', text=prompt))],
        max_tokens=1_024,
        system_prompt='Generate an SVG image as per the user input',
    )
    assert isinstance(result.content, TextContent)

    path = Path(f'{subject}_{style}.svg')
    # remove triple backticks if the svg was returned within markdown
    if m := re.search(r'^```\w*$(.+?)```$', result.content.text, re.S | re.M):
        path.write_text(m.group(1))
    else:
        path.write_text(result.content.text)
    return f'See {path}'


if __name__ == '__main__':
    # run the server via stdio
    app.run()
```

----------------------------------------

TITLE: Integrate Retrying HTTP Client with Anthropic API in Python
DESCRIPTION: Demonstrates passing a custom `httpx` client with retry capabilities (e.g., from `smart_retry_example.py`) to the `AnthropicProvider` in `pydantic-ai`. This setup ensures robust communication with the Anthropic API, handling transient errors and rate limits automatically.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_10

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(http_client=client))
agent = Agent(model)
```

----------------------------------------

TITLE: Integrate single ACI.dev tool with Pydantic AI Agent
DESCRIPTION: This Python example demonstrates how to integrate a single ACI.dev tool, such as `TAVILY__SEARCH`, into a Pydantic AI `Agent` using the `tool_from_aci` convenience method. It requires the `aci-sdk` package and setting the `ACI_API_KEY` environment variable, along with providing the `linked_account_owner_id`. Note that Pydantic AI does not validate arguments for ACI tools.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_16

LANGUAGE: python
CODE:
```
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import tool_from_aci


tavily_search = tool_from_aci(
    'TAVILY__SEARCH',
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent(
    'google-gla:gemini-2.0-flash',
    tools=[tavily_search],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')
print(result.output)
```

----------------------------------------

TITLE: Run Pydantic AI agent without explicit HTTPX instrumentation
DESCRIPTION: This Python example shows the default execution of a `pydantic-ai` agent without explicit HTTPX instrumentation. While `pydantic-ai` is instrumented, this snippet highlights the absence of detailed HTTP request/response capture, serving as a comparison to the fully instrumented version.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_5

LANGUAGE: python
CODE:
```
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.output)

```

----------------------------------------

TITLE: Access Dependencies in Pydantic AI System Prompt
DESCRIPTION: This example illustrates how to access defined dependencies within a Pydantic AI agent's system prompt function. It shows that the `RunContext` object, parameterized with the dependency type, is passed as the first argument to the system prompt function. Dependencies are then accessed via the `.deps` attribute of the `RunContext`, enabling the prompt to use external services like an HTTP client with an API key.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/dependencies.md#_snippet_1

LANGUAGE: Python
CODE:
```
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext


@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)


@agent.system_prompt  # (1)!
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:  # (2)!
    response = await ctx.deps.http_client.get(  # (3)!
        'https://example.com',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},  # (4)!
    )
    response.raise_for_status()
    return f'Prompt: {response.text}'


async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('foobar', client)
        result = await agent.run('Tell me a joke.', deps=deps)
        print(result.output)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
```

----------------------------------------

TITLE: Integrate ACI.dev Tools with Pydantic AI Agent
DESCRIPTION: This example illustrates how to incorporate tools from the ACI.dev library into a Pydantic AI agent using `ACIToolset`. It specifies the requirement for the `aci-sdk` package, setting the `ACI_API_KEY` environment variable, and providing a `linked_account_owner_id`. Similar to LangChain, Pydantic AI does not validate arguments for ACI tools.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_14

LANGUAGE: python
CODE:
```
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import ACIToolset


toolset = ACIToolset(
    [
        'OPEN_WEATHER_MAP__CURRENT_WEATHER',
        'OPEN_WEATHER_MAP__FORECAST',
    ],
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

----------------------------------------

TITLE: Reusing messages in a conversation with Pydantic AI agents
DESCRIPTION: This example demonstrates how to continue a conversation with a Pydantic AI agent by passing the message history from a previous run. By providing `message_history` to `Agent.run_sync`, the agent maintains context without generating a new system prompt, allowing for multi-turn interactions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.output)
# Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.output)
# This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=Usage(requests=1, request_tokens=60, response_tokens=12, total_tokens=72),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.'
            )
        ],
        usage=Usage(requests=1, request_tokens=61, response_tokens=26, total_tokens=87),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
]
"""
```

----------------------------------------

TITLE: Handling Diverse Tool Output Types in Pydantic-AI
DESCRIPTION: This example illustrates the flexibility of Pydantic-AI tools in returning various data types, including standard Python objects like `datetime`, Pydantic `BaseModel` instances, and specialized multi-modal types like `ImageUrl` and `DocumentUrl`. It demonstrates how the agent automatically handles serialization of these outputs to JSON or processes them according to the model's capabilities, enabling rich interactions and responses.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_4

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic import BaseModel

from pydantic_ai import Agent, DocumentUrl, ImageUrl
from pydantic_ai.models.openai import OpenAIResponsesModel


class User(BaseModel):
    name: str
    age: int


agent = Agent(model=OpenAIResponsesModel('gpt-4o'))


@agent.tool_plain
def get_current_time() -> datetime:
    return datetime.now()


@agent.tool_plain
def get_user() -> User:
    return User(name='John', age=30)


@agent.tool_plain
def get_company_logo() -> ImageUrl:
    return ImageUrl(url='https://iili.io/3Hs4FMg.png')


@agent.tool_plain
def get_document() -> DocumentUrl:
    return DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')


result = agent.run_sync('What time is it?')
print(result.output)

result = agent.run_sync('What is the user name?')
print(result.output)

result = agent.run_sync('What is the company name in the logo?')
print(result.output)

result = agent.run_sync('What is the main content of the document?')
print(result.output)
```

----------------------------------------

TITLE: Control Concurrency in Pydantic Evals Dataset Evaluation
DESCRIPTION: This Python example demonstrates how to evaluate a dataset with and without concurrency limits using `pydantic-evals`. It showcases the use of `dataset.evaluate_sync` to run a function against multiple test cases, first with unlimited concurrency and then with `max_concurrency=1` to illustrate the performance difference when operations are serialized.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_7

LANGUAGE: python
CODE:
```
import asyncio
import time

from pydantic_evals import Case, Dataset

# Create a dataset with multiple test cases
dataset = Dataset(
    cases=[
        Case(
            name=f'case_{i}',
            inputs=i,
            expected_output=i * 2,
        )
        for i in range(5)
    ]
)


async def double_number(input_value: int) -> int:
    """Function that simulates work by sleeping for a tenth of a second before returning double the input."""
    await asyncio.sleep(0.1)  # Simulate work
    return input_value * 2


# Run evaluation with unlimited concurrency
t0 = time.time()
report_default = dataset.evaluate_sync(double_number)
print(f'Evaluation took less than 0.5s: {time.time() - t0 < 0.5}')
# Evaluation took less than 0.5s: True

report_default.print(include_input=True, include_output=True, include_durations=False)
#       Evaluation Summary:
#          double_number
# ┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
# ┃ Case ID  ┃ Inputs ┃ Outputs ┃
# ┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
# │ case_0   │ 0      │ 0       │
# ├──────────┼────────┼─────────┤
# │ case_1   │ 1      │ 2       │
# ├──────────┼────────┼─────────┤
# │ case_2   │ 2      │ 4       │
# ├──────────┼────────┼─────────┤
# │ case_3   │ 3      │ 6       │
# ├──────────┼────────┼─────────┤
# │ case_4   │ 4      │ 8       │
# ├──────────┼────────┼─────────┤
# │ Averages │        │         │
# └──────────┴────────┴─────────┘

# Run evaluation with limited concurrency
t0 = time.time()
report_limited = dataset.evaluate_sync(double_number, max_concurrency=1)
print(f'Evaluation took more than 0.5s: {time.time() - t0 > 0.5}')
# Evaluation took more than 0.5s: True

report_limited.print(include_input=True, include_output=True, include_durations=False)
#       Evaluation Summary:
#          double_number
# ┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
# ┃ Case ID  ┃ Inputs ┃ Outputs ┃
# ┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
# │ case_0   │ 0      │ 0       │
# ├──────────┼────────┼─────────┤
# │ case_1   │ 1      │ 2       │
# ├──────────┼────────┼─────────┤
# │ case_2   │ 2      │ 4       │
# ├──────────┼────────┼─────────┤
# │ case_3   │ 3      │ 6       │
# ├──────────┼────────┼─────────┤
# │ case_4   │ 4      │ 8       │
# ├──────────┼────────┼─────────┤
# │ Averages │        │         │
# └──────────┴────────┴─────────┘
```

----------------------------------------

TITLE: Pydantic-Graph: Dependency Injection with ProcessPoolExecutor
DESCRIPTION: This Python example demonstrates how to implement dependency injection within `pydantic-graph` nodes. It shows how to define a `GraphDeps` class to hold dependencies (e.g., a `ProcessPoolExecutor`), pass these dependencies to the graph run, and access them within `BaseNode` subclasses via `GraphRunContext.deps` to offload computation to a separate process.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_18

LANGUAGE: Python
CODE:
```
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, FullStatePersistence, Graph, GraphRunContext


@dataclass
class GraphDeps:
    executor: ProcessPoolExecutor


@dataclass
class DivisibleBy5(BaseNode[None, GraphDeps, int]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext[None, GraphDeps],
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode[None, GraphDeps]):
    foo: int

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> DivisibleBy5:
        loop = asyncio.get_running_loop()
        compute_result = await loop.run_in_executor(
            ctx.deps.executor,
            self.compute,
        )
        return DivisibleBy5(compute_result)

    def compute(self) -> int:
        return self.foo + 1


fives_graph = Graph(nodes=[DivisibleBy5, Increment])


async def main():
    with ProcessPoolExecutor() as executor:
        deps = GraphDeps(executor)
        result = await fives_graph.run(DivisibleBy5(3), deps=deps, persistence=FullStatePersistence())
    print(result.output)
    #> 5
    # the full history is quite verbose (see below), so we'll just print the summary
    print([item.node for item in result.persistence.history])
    """
    [
        DivisibleBy5(foo=3),
        Increment(foo=3),
        DivisibleBy5(foo=4),
        Increment(foo=4),
        DivisibleBy5(foo=5),
        End(data=5),
    ]
    """
```

----------------------------------------

TITLE: Rename tools in Pydantic-AI toolset
DESCRIPTION: Shows how to rename tools in a Pydantic-AI toolset using `RenamedToolset` or the `renamed()` method. This is useful for clarifying ambiguous names or resolving conflicts without lengthy prefixes. The example renames specific weather and datetime tools, and `TestModel` is used to confirm the new tool names.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_5

LANGUAGE: Python
CODE:
```
from combined_toolset import combined_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


renamed_toolset = combined_toolset.renamed(
    {
        'current_time': 'datetime_now',
        'temperature_celsius': 'weather_temperature_celsius',
        'temperature_fahrenheit': 'weather_temperature_fahrenheit'
    }
)

test_model = TestModel()
agent = Agent(test_model, toolsets=[renamed_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
"""
['temperature_celsius', 'temperature_fahrenheit', 'weather_conditions', 'current_time']
"""
```

----------------------------------------

TITLE: Main Application Flow for Flight and Seat Booking
DESCRIPTION: This `main` asynchronous function orchestrates the application's flow. It initializes `Usage` tracking, then calls `find_flight` (assumed to be defined elsewhere) to get flight details. If a flight is found, it proceeds to call `find_seat` to determine the user's seat preference, demonstrating the sequential interaction with the defined AI agents.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#_snippet_5

LANGUAGE: python
CODE:
```
async def main():  # (7)!
    usage: Usage = Usage()

    opt_flight_details = await find_flight(usage)
    if opt_flight_details is not None:
        print(f'Flight found: {opt_flight_details.flight_number}')
        #> Flight found: AK456
        seat_preference = await find_seat(usage)
        print(f'Seat preference: {seat_preference}')
        #> Seat preference: row=1 seat='A'
```

----------------------------------------

TITLE: Prefix tool names in Pydantic-AI toolset
DESCRIPTION: Illustrates how to add prefixes to tool names within a Pydantic-AI toolset using `PrefixedToolset` or the `prefixed()` method. This technique helps prevent naming conflicts when combining multiple toolsets. The example demonstrates prefixing 'weather' and 'datetime' tools, then verifies the prefixed names via `TestModel`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_4

LANGUAGE: Python
CODE:
```
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import CombinedToolset


combined_toolset = CombinedToolset(
    [
        weather_toolset.prefixed('weather'),
        datetime_toolset.prefixed('datetime')
    ]
)

test_model = TestModel()
agent = Agent(test_model, toolsets=[combined_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
"""
[
    'weather_temperature_celsius',
    'weather_temperature_fahrenheit',
    'weather_conditions',
    'datetime_now',
]
"""
```

----------------------------------------

TITLE: Save and Load Pydantic Evals Datasets
DESCRIPTION: This example demonstrates how to persist and retrieve `pydantic-evals` datasets to and from file systems. It uses the `to_file()` method to save a `Dataset` object to a YAML file, showcasing the structured output. Subsequently, it uses the `from_file()` class method to load the dataset back into memory, verifying that the cases are correctly loaded. This functionality is crucial for managing and reusing evaluation configurations.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_6

LANGUAGE: python
CODE:
```
from pathlib import Path

from judge_recipes import CustomerOrder, Recipe, recipe_dataset

from pydantic_evals import Dataset

recipe_transforms_file = Path('recipe_transform_tests.yaml')
recipe_dataset.to_file(recipe_transforms_file)
print(recipe_transforms_file.read_text())

# Load dataset from file
loaded_dataset = Dataset[CustomerOrder, Recipe, dict].from_file(recipe_transforms_file)

print(f'Loaded dataset with {len(loaded_dataset.cases)} cases')
```

----------------------------------------

TITLE: Configure Logfire for Pydantic Evals Tracing
DESCRIPTION: This Python snippet demonstrates the basic setup for integrating Pydantic Evals with Logfire. It configures the Logfire SDK to send traces, specifying the environment as 'development' and the service name as 'evals'. The `send_to_logfire` parameter ensures traces are sent only if a token is present, enabling detailed monitoring of evaluation runs.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/pydantic_evals/README.md#_snippet_1

LANGUAGE: python
CODE:
```
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='evals',
)

...

my_dataset.evaluate_sync(my_task)
```

----------------------------------------

TITLE: Configure Logfire for Pydantic Evals Tracing
DESCRIPTION: This Python code demonstrates how to configure Pydantic Logfire to send OpenTelemetry traces from Pydantic Evals. It shows setting `send_to_logfire` based on token presence, defining the `environment` for filtering, and specifying a `service_name` for identification in the Logfire UI. It also includes an example of evaluating a dataset synchronously.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_13

LANGUAGE: Python
CODE:
```
import logfire
from judge_recipes import recipe_dataset, transform_recipe

logfire.configure(
    send_to_logfire='if-token-present',  # (1)!
    environment='development',  # (2)!
    service_name='evals',  # (3)!
)

recipe_dataset.evaluate_sync(transform_recipe)
```

----------------------------------------

TITLE: FunctionModel Usage Example for Agent Testing
DESCRIPTION: This Python code demonstrates how to use `FunctionModel` to override an `Agent`'s behavior for unit testing. It defines an asynchronous `model_function` that simulates a model's response, then uses `FunctionModel` within an `Agent`'s `override` context to test the agent's output, asserting the expected 'hello world' result.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/api/models/function.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

my_agent = Agent('openai:gpt-4o')


async def model_function(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    print(messages)
    """
    [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='Testing my agent...',
                    timestamp=datetime.datetime(...),
                )
            ]
        )
    ]
    """
    print(info)
    """
    AgentInfo(
        function_tools=[], allow_text_output=True, output_tools=[], model_settings=None
    )
    """
    return ModelResponse(parts=[TextPart('hello world')])


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    with my_agent.override(model=FunctionModel(model_function)):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'hello world'
```

----------------------------------------

TITLE: Implement Network Error Retries for HTTP Clients in Python
DESCRIPTION: Defines a client setup that automatically retries requests upon common network issues such as timeouts, connection errors, and read errors. It leverages `tenacity` to catch specific `httpx` exceptions and apply an exponential backoff strategy.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_7

LANGUAGE: Python
CODE:
```
import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, wait_exponential, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport

def create_network_resilient_client():
    """Create a client that handles network errors with retries."""
    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            retry=retry_if_exception_type((
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError
            )),
            wait=wait_exponential(multiplier=1, max=10),
            stop=stop_after_attempt(3),
            reraise=True
        )
    )
    return httpx.AsyncClient(transport=transport)

# Example usage
client = create_network_resilient_client()
# Client will now retry on timeout, connection, and read errors
```

----------------------------------------

TITLE: Extracting Box Dimensions or Text with Pydantic AI
DESCRIPTION: This example demonstrates configuring a `pydantic-ai.Agent` to extract structured data (a `Box` object) or return a plain string if complete information is not available. It showcases how the agent prompts for missing data and then successfully parses the structured output once all details are provided, highlighting the flexibility of `output_type` with multiple choices.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str


agent = Agent(
    'openai:gpt-4o-mini',
    output_type=[Box, str], # (1)!
    system_prompt=(
        "Extract me the dimensions of a box, "
        "if you can't extract all data, ask the user to try again."
    ),
)

result = agent.run_sync('The box is 10x20x30')
print(result.output)
#> Please provide the units for the dimensions (e.g., cm, in, m).

result = agent.run_sync('The box is 10x20x30 cm')
print(result.output)
#> width=10 height=20 depth=30 units='cm'
```

----------------------------------------

TITLE: Filter tools in Pydantic-AI toolset
DESCRIPTION: Demonstrates how to filter available tools in a Pydantic-AI toolset using `FilteredToolset` or the `filtered()` method. This example shows how to exclude tools whose names contain 'fahrenheit' by providing a lambda function that evaluates the tool's context and definition. It uses `TestModel` to inspect the tools available to the agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_3

LANGUAGE: Python
CODE:
```
from combined_toolset import combined_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

filtered_toolset = combined_toolset.filtered(lambda ctx, tool_def: 'fahrenheit' not in tool_def.name)

test_model = TestModel()
agent = Agent(test_model, toolsets=[filtered_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['weather_temperature_celsius', 'weather_conditions', 'datetime_now']
```

----------------------------------------

TITLE: Iterating Pydantic Graph nodes with `Graph.iter` and `async for`
DESCRIPTION: This example demonstrates how to use `Graph.iter` to gain direct control over graph execution. It returns a context manager yielding a `GraphRun` object, which is an async-iterable over the graph's nodes. This allows recording or modifying nodes as they execute, with the final result available via `run.result`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_13

LANGUAGE: python
CODE:
```
from __future__ import annotations as _annotations

from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End, GraphRunContext


@dataclass
class CountDownState:
    counter: int


@dataclass
class CountDown(BaseNode[CountDownState, None, int]):
    async def run(self, ctx: GraphRunContext[CountDownState]) -> CountDown | End[int]:
        if ctx.state.counter <= 0:
            return End(ctx.state.counter)
        ctx.state.counter -= 1
        return CountDown()


count_down_graph = Graph(nodes=[CountDown])


async def main():
    state = CountDownState(counter=3)
    async with count_down_graph.iter(CountDown(), state=state) as run:
        async for node in run:
            print('Node:', node)
    print('Final output:', run.result.output)
```

----------------------------------------

TITLE: Configure WebSearchTool with Advanced Options
DESCRIPTION: Illustrates how to configure the `WebSearchTool` with various parameters such as `search_context_size`, `user_location`, `blocked_domains`, `allowed_domains`, and `max_uses` to customize web search behavior. Shows how to pass these configurations during agent initialization for more controlled search operations.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/builtin-tools.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

agent = Agent(
    'anthropic:claude-sonnet-4-0',
    builtin_tools=[
        WebSearchTool(
            search_context_size='high',
            user_location=WebSearchUserLocation(
                city='San Francisco',
                country='US',
                region='CA',
                timezone='America/Los_Angeles',
            ),
            blocked_domains=['example.com', 'spam-site.net'],
            allowed_domains=None,  # Cannot use both blocked_domains and allowed_domains with Anthropic
            max_uses=5,  # Anthropic only: limit tool usage
        )
    ],
)

result = agent.run_sync('Use the web to get the current time.')
# > In San Francisco, it's 8:21:41 pm PDT on Wednesday, August 6, 2025.
```

----------------------------------------

TITLE: Pydantic AI Agent Run Event Processing Example
DESCRIPTION: This snippet illustrates how Pydantic AI agents process different event types during a run, such as `FunctionToolResultEvent` and reaching an `End` node. It shows how to capture and display output messages, including the final agent output, and demonstrates the execution flow using `asyncio.run(main())`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_10

LANGUAGE: python
CODE:
```
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            output_messages.append(
                                f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}'
                            )
            elif Agent.is_end_node(node):
                # Once an End node is reached, the agent run is complete
                assert run.result is not None
                assert run.result.output == node.data.output
                output_messages.append(f'=== Final Agent Output: {run.result.output} ===')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
```

----------------------------------------

TITLE: Define and Register Functions as AI Agent Tools
DESCRIPTION: This snippet demonstrates how to create a `FunctionToolset` in pydantic-ai to expose Python functions as tools for an AI agent. It illustrates three primary methods for registering functions: initializing the toolset with a list of functions, using the `@toolset.tool` decorator for class methods or standalone functions, and dynamically adding functions using `add_function()`. The example also shows how to verify which tools are made available to the agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_1

LANGUAGE: python
CODE:
```
from datetime import datetime

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset


def temperature_celsius(city: str) -> float:
    return 21.0


def temperature_fahrenheit(city: str) -> float:
    return 69.8


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"
    else:
        return "It's raining"


datetime_toolset = FunctionToolset()
datetime_toolset.add_function(lambda: datetime.now(), name='now')

test_model = TestModel()
agent = Agent(test_model)

result = agent.run_sync('What tools are available?', toolsets=[weather_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions']

result = agent.run_sync('What tools are available?', toolsets=[datetime_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['now']
```

----------------------------------------

TITLE: Example Model Exchange Messages for Pydantic AI Agent Testing
DESCRIPTION: This snippet illustrates a typical message exchange between a Pydantic AI agent and its model during a test run. It shows the structure of `ModelResponse` (containing `ToolCallPart`), `ModelRequest` (containing `ToolReturnPart`), and a final `ModelResponse` with `TextPart`, demonstrating how tool calls and their results are represented. The `IsStr()` and `IsNow()` helpers are used for flexible assertion of dynamic values.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#_snippet_2

LANGUAGE: Python
CODE:
```
[
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='weather_forecast',
                args={'location': 'London'},
                tool_call_id=IsStr(),
            )
        ],
        usage=Usage(
            requests=1,
            request_tokens=71,
            response_tokens=7,
            total_tokens=78,
            details=None,
        ),
        model_name='test',
        timestamp=IsNow(tz=timezone.utc),
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='weather_forecast',
                content='Sunny with a chance of rain',
                tool_call_id=IsStr(),
                timestamp=IsNow(tz=timezone.utc),
            ),
        ],
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='{"weather_forecast":"Sunny with a chance of rain"}',
            )
        ],
        usage=Usage(
            requests=1,
            request_tokens=77,
            response_tokens=16,
            total_tokens=93,
            details=None,
        ),
        model_name='test',
        timestamp=IsNow(tz=timezone.utc),
    ),
]
```

----------------------------------------

TITLE: Define Dependencies for Pydantic AI Agent
DESCRIPTION: This code demonstrates how to define dependencies for a Pydantic AI agent. It shows creating a `dataclass` to hold dependency objects like an API key and an HTTP client, passing the `dataclass` type to the `Agent` constructor for type checking, and then providing an instance of the `dataclass` when running the agent. Although the dependencies are defined, they are not actively used within the agent's logic in this specific example.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/dependencies.md#_snippet_0

LANGUAGE: Python
CODE:
```
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent


@dataclass
class MyDeps:  # (1)!
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,  # (2)!
)


async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('foobar', client)
        result = await agent.run(
            'Tell me a joke.',
            deps=deps,  # (3)!
        )
        print(result.output)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
```

----------------------------------------

TITLE: Display Pydantic AI CLI Help
DESCRIPTION: Use the `--help` flag with `uvx clai` to display the command-line interface's help message. This provides information on available commands, options, and usage.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_4

LANGUAGE: bash
CODE:
```
uvx clai --help
```

----------------------------------------

TITLE: Manually Drive Pydantic-AI Agent Iteration with next()
DESCRIPTION: This example illustrates how to manually control the agent's execution flow using the `agent_run.next(...)` method. By passing the current node to `next()`, you can inspect or modify it before execution, or even skip nodes based on custom logic. The iteration continues until an `End` node is returned, signifying the completion of the agent run.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o')


async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
        node = agent_run.next_node  # (1)!

        all_nodes = [node]

        # Drive the iteration manually:
        while not isinstance(node, End):  # (2)!
            node = await agent_run.next(node)  # (3)!
            all_nodes.append(node)  # (4)!

        print(all_nodes)
        """
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions=None,
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ]
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=(
                        Usage(
                            requests=1,
                            request_tokens=56,
                            response_tokens=7,
                            total_tokens=63,
                        )
                    ),
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        """
```

----------------------------------------

TITLE: Implement API Endpoint for Pydantic AI Agent with Deferred Tools
DESCRIPTION: This Python function, `run_agent`, simulates an API endpoint designed to interact with a `pydantic-ai` agent, incorporating deferred tools. It demonstrates how to dynamically add a `DeferredToolset` at runtime, override the agent's `output_type` to include `DeferredToolCalls`, and manage message history for continued agent runs. This setup is crucial for handling scenarios where tool results are produced externally.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_11

LANGUAGE: Python
CODE:
```
from deferred_toolset_agent import agent, PersonalizedGreeting

from typing import Union

from pydantic_ai.output import DeferredToolCalls
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import DeferredToolset
from pydantic_ai.messages import ModelMessage

def run_agent(
    messages: list[ModelMessage] = [], frontend_tools: list[ToolDefinition] = {}
) -> tuple[Union[PersonalizedGreeting, DeferredToolCalls], list[ModelMessage]]:
    deferred_toolset = DeferredToolset(frontend_tools)
    result = agent.run_sync(
        toolsets=[deferred_toolset], # (1)!
        output_type=[agent.output_type, DeferredToolCalls], # (2)!
        message_history=messages, # (3)!
    )
    return result.output, result.new_messages()
```

----------------------------------------

TITLE: Generate Mermaid Diagram for Pydantic-AI Graph
DESCRIPTION: This Python snippet demonstrates how to generate a Mermaid diagram representation of a `pydantic-ai` graph. It imports the graph definition and a starting node from `vending_machine.py`, then calls the `mermaid_code` method on the graph object, passing the initial node. The output is a Mermaid syntax string that can be rendered into a visual state diagram.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_9

LANGUAGE: Python
CODE:
```
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin)
```

----------------------------------------

TITLE: Pydantic AI Bank Support Agent with Tools and Dependency Injection
DESCRIPTION: This Python code demonstrates the implementation of a bank support agent using Pydantic AI. It defines `SupportDependencies` for injecting contextual data like `customer_id` and a `DatabaseConn`, and `SupportOutput` as a Pydantic `BaseModel` to enforce a structured response from the AI, including advice, card blocking status, and risk level. The `Agent` is configured with an OpenAI model, a static system prompt, and dynamic system prompts and tools (e.g., `customer_balance`) that leverage dependency injection via `RunContext`. The `main` function illustrates how to run the agent with specific dependencies and process its validated output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/index.md#_snippet_1

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # (3)!
    customer_id: int
    db: DatabaseConn  # (12)!


class SupportOutput(BaseModel):  # (13)!
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(  # (1)!
    'openai:gpt-4o',  # (2)!
    deps_type=SupportDependencies,
    output_type=SupportOutput,  # (9)!
    system_prompt=(  # (4)!
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


@support_agent.system_prompt  # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool  # (6)!
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""  # (7)!
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


...  # (11)!


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)  # (8)!
    print(result.output)  # (10)!
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
```

----------------------------------------

TITLE: Define and Run a Pydantic AI Agent with a Tool
DESCRIPTION: This example demonstrates how to create a Pydantic AI Agent that simulates a roulette wheel. It defines an Agent with specific dependency and output types, registers a tool function using the `@roulette_agent.tool` decorator, and then runs the agent synchronously with different inputs to show its behavior. The agent expects an integer dependency and produces a boolean output, while the tool checks if a given square matches the dependency.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, RunContext

roulette_agent = Agent(  # (1)!
    'openai:gpt-4o',
    deps_type=int,
    output_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  # (2)!
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


# Run the agent
success_number = 18  # (3)!
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.output)  # (4)!
# > True

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.output)
# > False
```

----------------------------------------

TITLE: Pydantic AI Core Classes: Agent and OpenAIModel
DESCRIPTION: Documentation for the core `Agent` and `OpenAIModel` classes in `pydantic-ai`, including their constructors and key methods for model interaction and structured output. It also details the `AgentResult` and `Usage` objects returned by the agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_23

LANGUAGE: APIDOC
CODE:
```
Agent:
  __init__(model: OpenAIModel, output_type: BaseModel = None)
    model: An instance of OpenAIModel configured for a specific LLM.
    output_type: (Optional) A Pydantic BaseModel subclass to enforce structured output.
  run_sync(prompt: str) -> AgentResult
    prompt: The input prompt string for the LLM.
    Returns: An AgentResult object containing the LLM's output and usage statistics.
  AgentResult:
    output: The parsed output, typically an instance of the specified output_type or a string.
    usage() -> Usage
      Returns: A Usage object detailing request, response, and total token counts.
    Usage:
      requests: Number of requests made.
      request_tokens: Tokens in the request.
      response_tokens: Tokens in the response.
      total_tokens: Total tokens used.

OpenAIModel:
  __init__(model_name: str, provider: Provider = OpenAIProvider())
    model_name: The name of the LLM model (e.g., 'llama3.2', 'gpt-4o').
    provider: An instance of a Provider class (e.g., OpenAIProvider, AzureProvider) to handle API communication.
```

----------------------------------------

TITLE: Limit Pydantic AI Agent Requests and Handle Tool Retries
DESCRIPTION: This example illustrates how to prevent infinite loops or excessive tool calling in a Pydantic AI agent by setting a `request_limit` using `UsageLimits`. It also demonstrates how a tool can raise a `ModelRetry` exception to trigger retries, and how the `request_limit` can ultimately prevent an infinite retry loop.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_12

LANGUAGE: python
CODE:
```
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits


class NeverOutputType(TypedDict):
    """
    Never ever coerce data to this type.
    """

    never_use_this: str


agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    retries=3,
    output_type=NeverOutputType,
    system_prompt='Any time you get a response, call the `infinite_retry_tool` to produce another response.',
)


@agent.tool_plain(retries=5)  # (1)!
def infinite_retry_tool() -> int:
    raise ModelRetry('Please try again.')


try:
    result_sync = agent.run_sync(
        'Begin infinite retry loop!', usage_limits=UsageLimits(request_limit=3)  # (2)!
    )
except UsageLimitExceeded as e:
    print(e)
    # The next request would exceed the request_limit of 3
```

----------------------------------------

TITLE: Implement Asynchronous Output Validation with Pydantic AI Agent Decorator
DESCRIPTION: Illustrates how to add custom asynchronous validation logic to an agent's output using the `@agent.output_validator` decorator. This is useful for validations requiring I/O or complex checks. The example shows validating a generated SQL query against a database and raising `ModelRetry` on failure.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_10

LANGUAGE: python
CODE:
```
from typing import Union

from fake_database import DatabaseConn, QueryError
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry


class Success(BaseModel):
    sql_query: str


class InvalidRequest(BaseModel):
    error_message: str


Output = Union[Success, InvalidRequest]
agent = Agent[DatabaseConn, Output](
    'google-gla:gemini-1.5-flash',
    output_type=Output,  # type: ignore
    deps_type=DatabaseConn,
    system_prompt='Generate PostgreSQL flavored SQL queries based on user input.',
)


@agent.output_validator
async def validate_sql(ctx: RunContext[DatabaseConn], output: Output) -> Output:
    if isinstance(output, InvalidRequest):
        return output
    try:
        await ctx.deps.execute(f'EXPLAIN {output.sql_query}')
    except QueryError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return output


result = agent.run_sync(
    'get me users who were last active yesterday.', deps=DatabaseConn()
)
# sql_query='SELECT * FROM users WHERE last_active::date = today() - interval 1 day'
```

----------------------------------------

TITLE: Summarize Old Messages with LLM in Python
DESCRIPTION: This example illustrates how to use a separate, potentially cheaper, language model to summarize older parts of the conversation history. This technique helps in reducing the total token count sent to the main agent while preserving essential context, making long conversations more efficient and cost-effective.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

# Use a cheaper model to summarize old messages.
summarize_agent = Agent(
    'openai:gpt-4o-mini',
    instructions="""
Summarize this conversation, omitting small talk and unrelated topics.
Focus on the technical discussion and next steps.
""",
)


async def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Summarize the oldest 10 messages
    if len(messages) > 10:
        oldest_messages = messages[:10]
        summary = await summarize_agent.run(message_history=oldest_messages)
        # Return the last message and the summary
        return summary.new_messages() + messages[-1:]

    return messages


agent = Agent('openai:gpt-4o', history_processors=[summarize_old_messages])
```

----------------------------------------

TITLE: Pydantic-AI Agent with Union Return Type
DESCRIPTION: This Python example demonstrates configuring a `pydantic-ai` Agent to handle multiple possible output types using `typing.Union`. The agent is initialized with `Union[list[str], list[int]]` as its `output_type`, allowing it to extract either colors (strings) or sizes (integers) from text. It showcases how the agent dynamically adapts its output based on the input provided.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_2

LANGUAGE: Python
CODE:
```
from typing import Union

from pydantic_ai import Agent

agent = Agent[None, Union[list[str], list[int]]](
    'openai:gpt-4o-mini',
    output_type=Union[list[str], list[int]],  # type: ignore # (1)!
    system_prompt='Extract either colors or sizes from the shapes provided.',
)

result = agent.run_sync('red square, blue circle, green triangle')
print(result.output)
#> ['red', 'blue', 'green']

result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.output)
#> [10, 20, 30]
```

----------------------------------------

TITLE: Vending Machine State Diagram (Mermaid)
DESCRIPTION: This Mermaid code block defines a state diagram visualizing the `vending_machine_graph`'s flow. It illustrates the transitions between different states (nodes) like `InsertCoin`, `CoinsInserted`, `SelectProduct`, and `Purchase`, including the start `[*]` and end `[*]` states. This diagram helps in understanding the graph's execution path and node interactions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_10

LANGUAGE: Mermaid
CODE:
```
--- 
title: vending_machine_graph
---
stateDiagram-v2
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```

----------------------------------------

TITLE: Run Pydantic AI AG-UI Application with Uvicorn
DESCRIPTION: This Bash command illustrates how to serve the Pydantic AI application, which is an ASGI application, using the Uvicorn server. It specifies the module and application object (`ag_ui_tool_events:app`) along with host and port settings, making the AG-UI integrated application accessible.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_8

LANGUAGE: bash
CODE:
```
uvicorn ag_ui_tool_events:app --host 0.0.0.0 --port 9000
```

----------------------------------------

TITLE: Configure Google Model Settings
DESCRIPTION: Demonstrates how to initialize `GoogleModelSettings` with various parameters like `temperature`, `max_tokens`, `google_thinking_config`, and `google_safety_settings` to customize the behavior of a `GoogleModel` when used with an `Agent`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_7

LANGUAGE: python
CODE:
```
from google.genai.types import HarmBlockThreshold, HarmCategory

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

settings = GoogleModelSettings(
    temperature=0.2,
    max_tokens=1024,
    google_thinking_config={'thinking_budget': 2048},
    google_safety_settings=[
        {
            'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)
model = GoogleModel('gemini-1.5-flash')
agent = Agent(model, model_settings=settings)
...
```

----------------------------------------

TITLE: Mermaid Diagram for Question Graph
DESCRIPTION: This Mermaid `stateDiagram-v2` code visualizes the `question_graph` defined in the Python example. It illustrates the flow between the 'Ask', 'Answer', 'Evaluate', and 'Reprimand' nodes, including an explicit edge label 'Ask the question', a multi-line note attached to the 'Ask' node, and highlights the 'Answer' node using a custom CSS class definition.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_20

LANGUAGE: mermaid
CODE:
```
--- 
title: question_graph 
--- 
stateDiagram-v2 
  Ask --> Answer: Ask the question 
  note right of Ask 
    Judge the answer. 
    Decide on next step. 
  end note 
  Answer --> Evaluate 
  Evaluate --> Reprimand 
  Evaluate --> [*]: success 
  Reprimand --> Ask 

classDef highlighted fill:#fdff32 
class Answer highlighted
```

----------------------------------------

TITLE: Implement Context-Aware History Processor in Python
DESCRIPTION: This snippet demonstrates how to create a history processor that leverages the `RunContext` parameter to access real-time information about the current agent run, such as token usage. It shows how to dynamically filter messages, for example, by keeping only recent messages when token usage exceeds a threshold, to optimize costs or performance.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_10

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext


def context_aware_processor(
    ctx: RunContext[None],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    # Access current usage
    current_tokens = ctx.usage.total_tokens

    # Filter messages based on context
    if current_tokens > 1000:
        return messages[-3:]  # Keep only recent messages when token usage is high
    return messages

agent = Agent('openai:gpt-4o', history_processors=[context_aware_processor])
```

----------------------------------------

TITLE: Implement Pydantic-AI Agent Tool Retries with ModelRetry
DESCRIPTION: This example illustrates how to implement self-correction in a Pydantic-AI agent by retrying tool calls. It demonstrates raising a `ModelRetry` exception within a tool function when a condition, such as a user not being found, is met. The agent is configured to retry the tool call a specified number of times, allowing the model to attempt a corrected response.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_20

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry

from fake_database import DatabaseConn


class ChatResult(BaseModel):
    user_id: int
    message: str


agent = Agent(
    'openai:gpt-4o',
    deps_type=DatabaseConn,
    output_type=ChatResult,
)


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    #> John
    #> John Doe
    user_id = ctx.deps.users.get(name=name)
    if user_id is None:
        raise ModelRetry(
            f'No user found with name {name!r}, remember to provide their full name'
        )
    return user_id


result = agent.run_sync(
    'Send a message to John Doe asking for coffee next week', deps=DatabaseConn()
)
print(result.output)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""
```

----------------------------------------

TITLE: Python Vending Machine Stateful Graph Example
DESCRIPTION: This Python code demonstrates a stateful graph implementation using `pydantic-graph` to simulate a vending machine. It defines a `MachineState` dataclass to track the user's balance and selected product. Various `BaseNode` classes (`InsertCoin`, `CoinsInserted`, `SelectProduct`, `Purchase`) represent the vending machine's operations, updating the shared `MachineState` as the graph progresses through user interactions like inserting coins and selecting products.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_8

LANGUAGE: python
CODE:
```
from __future__ import annotations

from dataclasses import dataclass

from rich.prompt import Prompt

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class MachineState:
    user_balance: float = 0.0
    product: str | None = None


@dataclass
class InsertCoin(BaseNode[MachineState]):
    async def run(self, ctx: GraphRunContext[MachineState]) -> CoinsInserted:
        return CoinsInserted(float(Prompt.ask('Insert coins')))


@dataclass
class CoinsInserted(BaseNode[MachineState]):
    amount: float

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> SelectProduct | Purchase:
        ctx.state.user_balance += self.amount
        if ctx.state.product is not None:
            return Purchase(ctx.state.product)
        else:
            return SelectProduct()


@dataclass
class SelectProduct(BaseNode[MachineState]):
    async def run(self, ctx: GraphRunContext[MachineState]) -> Purchase:
        return Purchase(Prompt.ask('Select product'))


PRODUCT_PRICES = {
    'water': 1.25,
    'soda': 1.50,
    'crisps': 1.75,
    'chocolate': 2.00,
}


@dataclass
class Purchase(BaseNode[MachineState, None, None]):
    product: str

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> End | InsertCoin | SelectProduct:
        if price := PRODUCT_PRICES.get(self.product):
            ctx.state.product = self.product
            if ctx.state.user_balance >= price:
                ctx.state.user_balance -= price
                return End(None)
            else:
                diff = price - ctx.state.user_balance
                print(f'Not enough money for {self.product}, need {diff:0.2f} more')
                return InsertCoin()
        else:
            print(f'No such product: {self.product}, try again')
            return SelectProduct()


vending_machine_graph = Graph(
    nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase]
)


async def main():
    state = MachineState()
    await vending_machine_graph.run(InsertCoin(), state=state)
    print(f'purchase successful item={state.product} change={state.user_balance:0.2f}')
```

----------------------------------------

TITLE: Initialize Bedrock Model with Agent by Name
DESCRIPTION: Shows a concise way to initialize an `Agent` instance in `pydantic-ai` by directly providing a Bedrock model name string, leveraging the library's internal model resolution.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('bedrock:anthropic.claude-3-sonnet-20240229-v1:0')
...
```

----------------------------------------

TITLE: Unit Testing Pydantic AI Agent with TestModel
DESCRIPTION: This Python example demonstrates how to use `TestModel` from `pydantic_ai.models.test` to facilitate unit testing of a `pydantic_ai.Agent`. By overriding the agent's model with `TestModel` using a context manager, developers can simulate responses and assert expected behavior without making actual external API calls. The snippet specifically checks for no tool calls and a 'success' output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/api/models/test.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

my_agent = Agent('openai:gpt-4o', system_prompt='...')


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    m = TestModel()
    with my_agent.override(model=m):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'success (no tool calls)'
    assert m.last_model_request_parameters.function_tools == []
```

----------------------------------------

TITLE: Run AG-UI FastAPI Server with Uvicorn
DESCRIPTION: Command to launch the FastAPI application as an ASGI server using Uvicorn, making the Pydantic AI agent accessible via the AG-UI protocol.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_3

LANGUAGE: bash
CODE:
```
uvicorn run_ag_ui:app
```

----------------------------------------

TITLE: Streaming results and messages with Pydantic-AI Agent
DESCRIPTION: Illustrates how to use `StreamedRunResult` to process agent responses asynchronously. This example sets up an agent, initiates a streaming run with `agent.run_stream`, and then demonstrates accessing incomplete messages before the stream finishes, iterating over streamed text parts using `result.stream_text()`, and finally retrieving complete messages once the stream has concluded. It highlights the dynamic nature of message objects during streaming.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')


async def main():
    async with agent.run_stream('Tell me a joke.') as result:
        # incomplete messages before the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='Be a helpful assistant.',
                        timestamp=datetime.datetime(...),
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    ),
                ]
            )
        ]
        """

        async for text in result.stream_text():
            print(text)
            # > Did you hear
            # > Did you hear about the toothpaste
            # > Did you hear about the toothpaste scandal? They called
            # > Did you hear about the toothpaste scandal? They called it Colgate.

        # complete messages once the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='Be a helpful assistant.',
                        timestamp=datetime.datetime(...),
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Did you hear about the toothpaste scandal? They called it Colgate.'
                    )
                ],
                usage=Usage(request_tokens=50, response_tokens=12, total_tokens=62),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
            ),
        ]
        """
```

----------------------------------------

TITLE: Evaluate AI Model for Recipe Generation with LLMJudge
DESCRIPTION: This snippet demonstrates how to set up and run an evaluation for an AI recipe generation agent using `pydantic-evals`. It defines `CustomerOrder` and `Recipe` models, initializes an `Agent` for recipe generation, and creates a `Dataset` with multiple `Case` entries. The evaluation utilizes `LLMJudge` for both case-specific and dataset-level rubrics, allowing for flexible assessment of the generated recipes based on dietary restrictions and general quality. The `evaluate_sync` method runs the evaluation and prints a summary report.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_5

LANGUAGE: python
CODE:
```
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pydantic_ai import Agent, format_as_xml
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge


class CustomerOrder(BaseModel):
    dish_name: str
    dietary_restriction: str | None = None


class Recipe(BaseModel):
    ingredients: list[str]
    steps: list[str]


recipe_agent = Agent(
    'groq:llama-3.3-70b-versatile',
    output_type=Recipe,
    system_prompt=(
        'Generate a recipe to cook the dish that meets the dietary restrictions.'
    ),
)


async def transform_recipe(customer_order: CustomerOrder) -> Recipe:
    r = await recipe_agent.run(format_as_xml(customer_order))
    return r.output


recipe_dataset = Dataset[CustomerOrder, Recipe, Any](
    cases=[
        Case(
            name='vegetarian_recipe',
            inputs=CustomerOrder(
                dish_name='Spaghetti Bolognese', dietary_restriction='vegetarian'
            ),
            expected_output=None,
            metadata={'focus': 'vegetarian'},
            evaluators=(
                LLMJudge(
                    rubric='Recipe should not contain meat or animal products',
                ),
            ),
        ),
        Case(
            name='gluten_free_recipe',
            inputs=CustomerOrder(
                dish_name='Chocolate Cake', dietary_restriction='gluten-free'
            ),
            expected_output=None,
            metadata={'focus': 'gluten-free'},
            # Case-specific evaluator with a focused rubric
            evaluators=(
                LLMJudge(
                    rubric='Recipe should not contain gluten or wheat products',
                ),
            ),
        ),
    ],
    evaluators=[
        IsInstance(type_name='Recipe'),
        LLMJudge(
            rubric='Recipe should have clear steps and relevant ingredients',
            include_input=True,
            model='anthropic:claude-3-7-sonnet-latest',
        ),
    ],
)


report = recipe_dataset.evaluate_sync(transform_recipe)
print(report)
```

----------------------------------------

TITLE: Fixing Jupyter Notebook RuntimeError with nest-asyncio
DESCRIPTION: This snippet demonstrates how to resolve the 'RuntimeError: This event loop is already running' in Jupyter Notebooks, Google Colab, and Marimo. The error arises from conflicts between Jupyter's event loop and Pydantic AI's. Applying `nest_asyncio` before executing any agent runs helps manage these conflicts, allowing for smooth operation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/troubleshooting.md#_snippet_0

LANGUAGE: Python
CODE:
```
import nest_asyncio

nest_asyncio.apply()
```

----------------------------------------

TITLE: Initialize GroqModel with Custom GroqProvider
DESCRIPTION: This Python snippet illustrates how to provide a custom `GroqProvider` instance to the `GroqModel` constructor. This allows for direct configuration of the API key or other provider-specific settings, overriding environment variables if needed.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/groq.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

model = GroqModel(
    'llama-3.3-70b-versatile', provider=GroqProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Stream Agent Output and Events with run_stream() in Python
DESCRIPTION: This Python snippet demonstrates how to use `pydantic_ai.Agent.run_stream()` to stream both the final text output and intermediate events during an agent's execution. It shows how to define an `event_stream_handler` to process various `AgentStreamEvent` types, providing insights into tool calls, part deltas, and final result production. The example includes a `weather_forecast` tool and illustrates the sequence of events during a tool-augmented agent run.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_2

LANGUAGE: python
CODE:
```
import asyncio
from collections.abc import AsyncIterable
from datetime import date
from typing import Union

from pydantic_ai import Agent
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
from pydantic_ai.tools import RunContext

weather_agent = Agent(
    'openai:gpt-4o',
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext,
    location: str,
    forecast_date: date,
) -> str:
    return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'


output_messages: list[str] = []


async def event_stream_handler(
    ctx: RunContext,
    event_stream: AsyncIterable[Union[AgentStreamEvent, HandleResponseEvent]],
):
    async for event in event_stream:
        if isinstance(event, PartStartEvent):
            output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                output_messages.append(f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
            elif isinstance(event.delta, ThinkingPartDelta):
                output_messages.append(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
            elif isinstance(event.delta, ToolCallPartDelta):
                output_messages.append(f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
        elif isinstance(event, FunctionToolCallEvent):
            output_messages.append(
                f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
            )
        elif isinstance(event, FunctionToolResultEvent):
            output_messages.append(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
        elif isinstance(event, FinalResultEvent):
            output_messages.append(f'[Result] The model starting producing a final result (tool_name={event.tool_name})')


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async with weather_agent.run_stream(user_prompt, event_stream_handler=event_stream_handler) as run:
        async for output in run.stream_text():
            output_messages.append(f'[Output] {output}')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        '[Output] It will be ',
        '[Output] It will be warm and sunny ',
        '[Output] It will be warm and sunny in Paris on ',
        '[Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """
```

----------------------------------------

TITLE: Limiting Pydantic AI Message History to Recent Messages
DESCRIPTION: This example illustrates how to use an asynchronous `history_processor` to manage token usage by keeping only the most recent messages in the conversation history. The `keep_recent_messages` function ensures that only the last 5 messages are sent to the model, regardless of the total conversation length, which is crucial for cost optimization and maintaining context relevance.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_9

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage


async def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only the last 5 messages to manage token usage."""
    return messages[-5:] if len(messages) > 5 else messages

agent = Agent('openai:gpt-4o', history_processors=[keep_recent_messages])

# Example: Even with a long conversation history, only the last 5 messages are sent to the model
long_conversation_history: list[ModelMessage] = []  # Your long conversation history here
# result = agent.run_sync('What did we discuss?', message_history=long_conversation_history)
```

----------------------------------------

TITLE: Define a Node with Conditional Graph Termination
DESCRIPTION: This example extends the `MyNode` to allow for conditional graph termination. The `run` method's return type is a union of `AnotherNode` and `End[int]`, enabling the node to either continue the graph execution or terminate it by returning an `End` object with a specified value. The `BaseNode` is parameterized with `None` for dependencies and `int` for the graph's return type.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_3

LANGUAGE: Python
CODE:
```
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class MyNode(BaseNode[MyState, None, int]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext[MyState],
    ) -> AnotherNode | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return AnotherNode()
```

----------------------------------------

TITLE: Run Graph from File Persistence (Python)
DESCRIPTION: This Python example demonstrates how to use `FileStatePersistence` with `pydantic-graph` to manage and resume graph execution. It initializes a `count_down_graph` with a `FileStatePersistence` object, then repeatedly calls `run_node` which loads the graph state from the file and executes the next available node. This showcases how `pydantic-graph` allows for distributed or interrupted execution by persisting and resuming graph state without relying on external application state.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_15

LANGUAGE: Python
CODE:
```
from pathlib import Path

from pydantic_graph import End
from pydantic_graph.persistence.file import FileStatePersistence

from count_down import CountDown, CountDownState, count_down_graph


async def main():
    run_id = 'run_abc123'
    persistence = FileStatePersistence(Path(f'count_down_{run_id}.json'))  # (1)!
    state = CountDownState(counter=5)
    await count_down_graph.initialize(  # (2)!
        CountDown(), state=state, persistence=persistence
    )

    done = False
    while not done:
        done = await run_node(run_id)


async def run_node(run_id: str) -> bool:  # (3)!
    persistence = FileStatePersistence(Path(f'count_down_{run_id}.json'))
    async with count_down_graph.iter_from_persistence(persistence) as run:  # (4)!
        node_or_end = await run.next()  # (5)!

    print('Node:', node_or_end)
    # Node: CountDown()
    # Node: CountDown()
    # Node: CountDown()
    # Node: CountDown()
    # Node: CountDown()
    # Node: End(data=0)
    return isinstance(node_or_end, End)  # (6)!
```

----------------------------------------

TITLE: Initialize Pydantic-AI Agent with Groq Model by String Name
DESCRIPTION: This Python snippet demonstrates how to initialize a `pydantic-ai` Agent by directly providing a Groq model name as a string. The Agent automatically resolves the Groq model using the `groq:` prefix, assuming the `GROQ_API_KEY` environment variable is set.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/groq.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('groq:llama-3.3-70b-versatile')
...
```

----------------------------------------

TITLE: Overriding Agent Model via Pytest Fixture
DESCRIPTION: This Python code provides an example of using a Pytest fixture to override an AI agent's model. The `override_weather_agent` fixture uses `weather_agent.override(model=TestModel())` to replace the agent's model, ensuring that subsequent tests within its scope use the `TestModel` for predictable behavior. This approach promotes reusability and cleaner test code for scenarios requiring model mocking.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#_snippet_5

LANGUAGE: python
CODE:
```
import pytest
from weather_app import weather_agent

from pydantic_ai.models.test import TestModel


@pytest.fixture
def override_weather_agent():
    with weather_agent.override(model=TestModel()):
        yield


async def test_forecast(override_weather_agent: None):
    ...
    # test code here
```

----------------------------------------

TITLE: Configure Pydantic AI Agent to Exclude Sensitive Content
DESCRIPTION: This example illustrates how to prevent Pydantic AI agents from sending sensitive data like user prompts, model completions, and tool call arguments to observability platforms. By setting `include_content=False`, only structural information is sent, which is crucial for privacy and compliance in production environments. It covers both individual agent and global instrumentation settings.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_14

LANGUAGE: python
CODE:
```
from pydantic_ai.agent import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

instrumentation_settings = InstrumentationSettings(include_content=False)

agent = Agent('gpt-4o', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)
```

----------------------------------------

TITLE: Configure Custom TLS/SSL with httpx.AsyncClient in Pydantic AI
DESCRIPTION: This Python example demonstrates how to set up a custom `httpx.AsyncClient` with specific TLS/SSL configurations and pass it to a Pydantic AI `MCPServerSSE` instance. It shows how to create an `ssl.SSLContext` to trust a custom CA file and optionally load a client certificate for mutual TLS. The configured `httpx.AsyncClient` is then used by the Pydantic AI agent for all MCP traffic, allowing advanced control over network requests.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_9

LANGUAGE: python
CODE:
```
import httpx
import ssl

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE


# Trust an internal / self-signed CA
ssl_ctx = ssl.create_default_context(cafile="/etc/ssl/private/my_company_ca.pem")

# OPTIONAL: if the server requires **mutual TLS** load your client certificate
ssl_ctx.load_cert_chain(certfile="/etc/ssl/certs/client.crt", keyfile="/etc/ssl/private/client.key",)

http_client = httpx.AsyncClient(
    verify=ssl_ctx,
    timeout=httpx.Timeout(10.0),
)

server = MCPServerSSE(
    url="http://localhost:3001/sse",
    http_client=http_client,  # (1)!
)
agent = Agent("openai:gpt-4o", toolsets=[server])

async def main():
    async with agent:
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.
```

----------------------------------------

TITLE: Reusing Pydantic AI Messages Across Different Models
DESCRIPTION: This example demonstrates how to reuse message history generated by one Pydantic AI agent (using `openai:gpt-4o`) in a subsequent agent run with a different model (`google-gla:gemini-1.5-pro`). It highlights the model-agnostic nature of Pydantic AI's message format, allowing seamless transfer of conversation context using `result.new_messages()`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync(
    'Explain?',
    model='google-gla:gemini-1.5-pro',
    message_history=result1.new_messages(),
)
print(result2.output)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=Usage(requests=1, request_tokens=60, response_tokens=12, total_tokens=72),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.'
            )
        ],
        usage=Usage(requests=1, request_tokens=61, response_tokens=26, total_tokens=87),
        model_name='gemini-1.5-pro',
        timestamp=datetime.datetime(...),
    ),
]
"""
```

----------------------------------------

TITLE: Integrate Pydantic AI with Local Ollama
DESCRIPTION: Demonstrates how to configure `pydantic-ai` to use a locally running Ollama instance, defining a Pydantic model for structured output and executing a query. It shows how to parse the output into a `BaseModel` and retrieve usage statistics.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_15

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIModel(
    model_name='llama3.2', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
agent = Agent(ollama_model, output_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
# city='London' country='United Kingdom'
print(result.usage())
# Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
```

----------------------------------------

TITLE: Pydantic AI Agent for AG-UI Event and State Management
DESCRIPTION: This Python code demonstrates how to define a Pydantic AI `Agent` that interacts with AG-UI for event handling and state management. It showcases the use of `StateDeps` to manage application state, and defines `tool` and `tool_plain` functions that return `StateSnapshotEvent` for state updates and `CustomEvent` for custom event emission, respectively. This setup allows Pydantic AI tools to send structured events to the AG-UI frontend.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_7

LANGUAGE: python
CODE:
```
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-4.1',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = agent.to_ag_ui(deps=StateDeps(DocumentState()))


@agent.tool
async def update_state(ctx: RunContext[StateDeps[DocumentState]]) -> StateSnapshotEvent:
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool_plain
async def custom_events() -> list[CustomEvent]:
    return [
        CustomEvent(
            type=EventType.CUSTOM,
            name='count',
            value=1,
        ),
        CustomEvent(
            type=EventType.CUSTOM,
            name='count',
            value=2,
        ),
    ]
```

----------------------------------------

TITLE: Handling and Diagnosing Agent Model Errors with capture_run_messages in Python
DESCRIPTION: This Python example illustrates how to gracefully handle `UnexpectedModelBehavior` when an agent's tool, `calc_volume`, intentionally raises `ModelRetry` multiple times, simulating a persistent error. It showcases the critical role of `capture_run_messages` in capturing the entire communication flow (requests and responses) between the agent and the model. This captured message history is invaluable for debugging and understanding the root cause of model failures or unexpected retries, providing detailed insights into the interaction leading to the exception.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_21

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages

agent = Agent('openai:gpt-4o')


@agent.tool_plain
def calc_volume(size: int) -> int:  # (1)!
    if size == 42:
        return size**3
    else:
        raise ModelRetry('Please try again.')


with capture_run_messages() as messages:  # (2)!
    try:
        result = agent.run_sync('Please get me the volume of a box with size 6.')
    except UnexpectedModelBehavior as e:
        print('An error occurred:', e)
        # > An error occurred: Tool 'calc_volume' exceeded max retries count of 1
        print('cause:', repr(e.__cause__))
        # > cause: ModelRetry('Please try again.')
        print('messages:', messages)
        """
        messages:
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Please get me the volume of a box with size 6.',
                        timestamp=datetime.datetime(...),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=(
                    Usage(
                        requests=1, request_tokens=62, response_tokens=4, total_tokens=66
                    )
                ),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please try again.',
                        tool_name='calc_volume',
                        tool_call_id='pyd_ai_tool_call_id',
                        timestamp=datetime.datetime(...),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=(
                    Usage(
                        requests=1, request_tokens=72, response_tokens=8, total_tokens=80
                    )
                ),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
            ),
        ]
        """
    else:
        print(result.output)
```

----------------------------------------

TITLE: Initialize pydantic-ai Agent with Hugging Face model by name
DESCRIPTION: This Python code demonstrates how to create an `Agent` instance by directly passing a Hugging Face model name string (prefixed with 'huggingface:') to its constructor. This is a convenient way to quickly use a Hugging Face model with default provider settings.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/huggingface.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('huggingface:Qwen/Qwen3-235B-A22B')
...
```

----------------------------------------

TITLE: Provide Custom Bedrock Provider with Boto3 Client
DESCRIPTION: Illustrates how to use a pre-configured `boto3` client to initialize a `BedrockProvider`, which is then passed to `BedrockConverseModel`. This is useful for scenarios where a `boto3` client is already set up with specific configurations or session management.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_6

LANGUAGE: python
CODE:
```
import boto3

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Using a pre-configured boto3 client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
model = BedrockConverseModel(
    'anthropic.claude-3-sonnet-20240229-v1:0',
    provider=BedrockProvider(bedrock_client=bedrock_client),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Set Vercel AI Gateway Environment Variables
DESCRIPTION: Instructions for setting environment variables (`VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN`) to authenticate with Vercel AI Gateway. These variables allow `pydantic-ai` to automatically pick up credentials.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_19

LANGUAGE: bash
CODE:
```
export VERCEL_AI_GATEWAY_API_KEY='your-ai-gateway-api-key'
# OR
export VERCEL_OIDC_TOKEN='your-oidc-token'
```

----------------------------------------

TITLE: Define Dataset, Custom Evaluator, and Run Evaluation
DESCRIPTION: This Python snippet demonstrates the complete workflow for setting up an evaluation with `pydantic-evals`. It shows how to define a `Case`, implement a custom `Evaluator`, create a `Dataset` with both built-in and custom evaluators, run a synchronous evaluation against a target function, and print the resulting `EvaluationReport`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

case1 = Case(
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)


class MyEvaluator(Evaluator[str, str]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


dataset = Dataset(
    cases=[case1],
    evaluators=[IsInstance(type_name='str'), MyEvaluator()],
)


async def guess_city(question: str) -> str:
    return 'Paris'


report = dataset.evaluate_sync(guess_city)
report.print(include_input=True, include_output=True, include_durations=False)
"""
                              Evaluation Summary: guess_city
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Case ID     ┃ Inputs                         ┃ Outputs ┃ Scores            ┃ Assertions ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ simple_case │ What is the capital of France? │ Paris   │ MyEvaluator: 1.00 │ ✔          │
├─────────────┼────────────────────────────────┼─────────┼───────────────────┼────────────┤
│ Averages    │                                │         │ MyEvaluator: 1.00 │ 100.0% ✔   │
└─────────────┴────────────────────────────────┴─────────┴───────────────────┴────────────┘
"""
```

----------------------------------------

TITLE: Configure Pydantic AI with GitHub Models
DESCRIPTION: Demonstrates how to initialize an `OpenAIModel` with `GitHubProvider` using a personal access token. This requires a GitHub token with `models: read` permission. The model name uses a prefixed format, e.g., 'xai/grok-3-mini'.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_25

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.github import GitHubProvider

model = OpenAIModel(
    'xai/grok-3-mini',  # GitHub Models uses prefixed model names
    provider=GitHubProvider(api_key='your-github-token'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Initialize MistralModel with custom provider and API key
DESCRIPTION: Demonstrates how to configure a `MistralModel` with a custom `MistralProvider` instance. This allows for direct specification of the API key and a custom base URL, overriding any environment variables.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/mistral.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

model = MistralModel(
    'mistral-large-latest', provider=MistralProvider(api_key='your-api-key', base_url='https://<mistral-provider-endpoint>')
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Run Pydantic AI A2A Server with Uvicorn
DESCRIPTION: This bash command illustrates how to launch the ASGI application generated by `agent.to_a2a()` using the Uvicorn server. It specifies the module (`agent_to_a2a`) and the application object (`app`), along with the host and port for network accessibility. This command is essential for making the Pydantic AI agent available as a live A2A service.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/a2a.md#_snippet_4

LANGUAGE: bash
CODE:
```
uvicorn agent_to_a2a:app --host 0.0.0.0 --port 8000
```

----------------------------------------

TITLE: Provide image input to LLM using Pydantic-AI ImageUrl
DESCRIPTION: This snippet demonstrates how to send an image to an LLM by providing its direct URL using the `ImageUrl` class from `pydantic-ai`. It initializes an `Agent` with a specified model (e.g., 'openai:gpt-4o') and passes a list containing the text prompt and the `ImageUrl` object to the `run_sync` method. The LLM then processes the image and provides a textual response.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/input.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, ImageUrl

agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(
    [
        'What company is this logo from?',
        ImageUrl(url='https://iili.io/3Hs4FMg.png'),
    ]
)
print(result.output)
```

----------------------------------------

TITLE: Run Pydantic AI Agent Synchronously for Basic Text Response
DESCRIPTION: This snippet demonstrates the most basic usage of a Pydantic AI agent, executing a synchronous run to query an LLM and retrieve a simple text-based response. It shows how to initiate a conversation and print the agent's output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/README.md#_snippet_1

LANGUAGE: python
CODE:
```
result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
```

----------------------------------------

TITLE: clai Command Line Interface Reference
DESCRIPTION: Comprehensive reference for the `clai` command-line tool, detailing its usage, available options, and special interactive mode commands. It allows users to interact with AI models, specify models, agents, and control streaming behavior.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/clai/README.md#_snippet_4

LANGUAGE: APIDOC
CODE:
```
usage: clai [-h] [-m [MODEL]] [-a AGENT] [-l] [-t [CODE_THEME]] [--no-stream] [--version] [prompt]

Pydantic AI CLI v...

Special prompts:
* `/exit` - exit the interactive mode (ctrl-c and ctrl-d also work)
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode

positional arguments:
  prompt                AI Prompt, if omitted fall into interactive mode

options:
  -h, --help            show this help message and exit
  -m [MODEL], --model [MODEL]
                        Model to use, in format "<provider>:<model>" e.g. "openai:gpt-4.1" or "anthropic:claude-sonnet-4-0". Defaults to "openai:gpt-4.1".
  -a AGENT, --agent AGENT
                        Custom Agent to use, in format "module:variable", e.g. "mymodule.submodule:my_agent"
  -l, --list-models     List all available models and exit
  -t [CODE_THEME], --code-theme [CODE_THEME]
                        Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "dark" which works well on dark terminals.
  --no-stream           Disable streaming from the model
  --version             Show version and exit
```

----------------------------------------

TITLE: Create a Pydantic AI MCP Server
DESCRIPTION: This Python code sets up a basic MCP server using `FastMCP` and integrates a Pydantic AI `Agent`. It defines an asynchronous `poet` tool that uses the agent to generate rhyming poems based on a provided theme, demonstrating a simple AI-powered tool within the MCP framework.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/server.md#_snippet_0

LANGUAGE: python
CODE:
```
from mcp.server.fastmcp import FastMCP

from pydantic_ai import Agent

server = FastMCP('Pydantic AI Server')
server_agent = Agent(
    'anthropic:claude-3-5-haiku-latest', system_prompt='always reply in rhyme'
)


@server.tool()
async def poet(theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}')
    return r.output


if __name__ == '__main__':
    server.run()
```

----------------------------------------

TITLE: Configure Pydantic AI Agent with Logfire Instrumentation
DESCRIPTION: This Python snippet demonstrates how to integrate Pydantic Logfire for instrumentation with a Pydantic AI `Agent`. It shows configuring Logfire and instrumenting `asyncpg` for database query logging, along with the `Agent` initialization including `instrument=True` to enable tracing.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/index.md#_snippet_2

LANGUAGE: python
CODE:
```
...
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn

import logfire

logfire.configure()  # (1)!
logfire.instrument_asyncpg()  # (2)!

...

support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
    instrument=True,
)
```

----------------------------------------

TITLE: Deploy Pydantic AI Slack Qualifier to Modal Production
DESCRIPTION: Use this command to deploy the Pydantic AI Slack lead qualifier application persistently to your Modal workspace. This makes the application available for continuous operation in a production environment, accessible via a stable URL.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_7

LANGUAGE: bash
CODE:
```
python/uv-run -m modal deploy -m pydantic_ai_examples.slack_lead_qualifier.modal
```

----------------------------------------

TITLE: Initialize OpenAIModel directly in pydantic-ai
DESCRIPTION: Illustrates how to explicitly instantiate an `OpenAIModel` object with a model name and then pass this model instance to the `Agent` constructor. This provides more granular control over the model configuration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o')
agent = Agent(model)
...
```

----------------------------------------

TITLE: Configure Pydantic AI Model Settings Precedence
DESCRIPTION: Demonstrates how `ModelSettings` are applied and merged in Pydantic AI, showing the precedence order from model-level defaults, to agent-level defaults, and finally run-time overrides. It illustrates how specific parameters like `temperature` and `max_tokens` are resolved based on this hierarchy.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_14

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

# 1. Model-level defaults
model = OpenAIModel(
    'gpt-4o',
    settings=ModelSettings(temperature=0.8, max_tokens=500)  # Base defaults
)

# 2. Agent-level defaults (overrides model defaults by merging)
agent = Agent(model, model_settings=ModelSettings(temperature=0.5))

# 3. Run-time overrides (highest priority)
result_sync = agent.run_sync(
    'What is the capital of Italy?',
    model_settings=ModelSettings(temperature=0.0)  # Final temperature: 0.0
)
print(result_sync.output)
```

----------------------------------------

TITLE: Initialize pydantic-ai Agent with OpenAI model by name
DESCRIPTION: Shows how to create an `Agent` instance in `pydantic-ai` by directly specifying an OpenAI model name (e.g., 'openai:gpt-4o'). This method assumes the API key is configured via an environment variable.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
...
```

----------------------------------------

TITLE: Provide document input to LLM using Pydantic-AI DocumentUrl
DESCRIPTION: This snippet demonstrates how to provide a document to an LLM using its direct URL via the `DocumentUrl` class from `pydantic-ai`. It initializes an `Agent` with a model capable of document understanding (e.g., 'anthropic:claude-3-sonnet') and passes the document URL along with a text prompt. The LLM then processes the document content to answer questions or summarize its main points.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/input.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, DocumentUrl

agent = Agent(model='anthropic:claude-3-sonnet')
result = agent.run_sync(
    [
        'What is the main content of this document?',
        DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
    ]
)
print(result.output)
```

----------------------------------------

TITLE: Connect Pydantic AI Agent to MCP SSE Server
DESCRIPTION: Demonstrates how to initialize an `MCPServerSSE` client with a specified URL and integrate it with a Pydantic AI `Agent`. The agent then uses the server to run a natural language query, showcasing communication over Server-Sent Events (SSE) for tool execution.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE(url='http://localhost:3001/sse')  # (1)!
agent = Agent('openai:gpt-4o', toolsets=[server])  # (2)!


async def main():
    async with agent:  # (3)!
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    # There are 9,208 days between January 1, 2000, and March 18, 2025.
```

----------------------------------------

TITLE: Pydantic AI Components for OpenAI-compatible Model Configuration
DESCRIPTION: Comprehensive API documentation for configuring OpenAI-compatible models within Pydantic AI. This includes details on the OpenAIModel, various provider classes (OpenAIProvider, DeepSeekProvider), model profiles for behavior customization, and the Agent class for model interaction.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_13

LANGUAGE: APIDOC
CODE:
```
Pydantic AI Components for OpenAI-compatible Models:

OpenAIModel:
  __init__(model_name: str, provider: Union[OpenAIProvider, str], profile: Optional[ModelProfile] = None)
    model_name: The name of the OpenAI-compatible model to use (e.g., 'model_name', 'deepseek-chat').
    provider: The provider to use. Can be an instance of OpenAIProvider (or a specific subclass like DeepSeekProvider) or a string shorthand (e.g., 'deepseek', 'openrouter').
    profile: An optional ModelProfile or OpenAIModelProfile instance to customize model behavior.

OpenAIProvider:
  __init__(base_url: str, api_key: str, http_client: Optional[AsyncClient] = None)
    base_url: The base URL of the OpenAI-compatible API endpoint.
    api_key: The API key for authentication with the service.
    http_client: An optional custom httpx.AsyncClient instance for advanced HTTP request configuration (e.g., timeouts).

DeepSeekProvider (subclass of OpenAIProvider):
  __init__(api_key: str, http_client: Optional[AsyncClient] = None)
    api_key: The DeepSeek API key.
    http_client: An optional custom httpx.AsyncClient instance.

ModelProfile:
  Used to tweak various aspects of how model requests are constructed, shared among all model classes.
  Example attribute:
    json_schema_transformer: A transformer class (e.g., InlineDefsJsonSchemaTransformer) to modify JSON schemas for tool definitions.

OpenAIModelProfile (subclass of ModelProfile):
  Used for behaviors specific to OpenAIModel.
  Example attribute:
    openai_supports_strict_tool_definition: A boolean indicating if the OpenAI-compatible API supports strict tool definitions.

Agent:
  __init__(model: OpenAIModel)
    model: An initialized OpenAIModel instance.
  Shorthand initialization:
    Agent("<provider>:<model>"): A convenient shorthand to initialize an Agent with a specific provider and model (e.g., Agent("deepseek:deepseek-chat")).
```

----------------------------------------

TITLE: Verify Model Understanding of DuckDB SQL with clai CLI
DESCRIPTION: Demonstrates how to use the `clai` command-line interface to query a specified large language model (e.g., a Bedrock Claude model) about its understanding of DuckDB SQL. The interactive output shows the model's affirmative response and a detailed explanation of DuckDB's features and capabilities, confirming its suitability for SQL-based data analysis tasks.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/data-analyst.md#_snippet_1

LANGUAGE: sh
CODE:
```
clai -m bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
clai - Pydantic AI CLI v0.0.1.dev920+41dd069 with bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
clai ➤ do you understand duckdb sql?
# DuckDB SQL

Yes, I understand DuckDB SQL. DuckDB is an in-process analytical SQL database
that uses syntax similar to PostgreSQL. It specializes in analytical queries
and is designed for high-performance analysis of structured data.

Some key features of DuckDB SQL include:

 • OLAP (Online Analytical Processing) optimized
 • Columnar-vectorized query execution
 • Standard SQL support with PostgreSQL compatibility
 • Support for complex analytical queries
 • Efficient handling of CSV/Parquet/JSON files

I can help you with DuckDB SQL queries, schema design, optimization, or other
DuckDB-related questions.
```

----------------------------------------

TITLE: Configure Pydantic AI with MoonshotAI
DESCRIPTION: Shows how to integrate `pydantic-ai` with MoonshotAI. This configuration requires an API key, which can be created in the Moonshot Console, to access their language models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_22

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.moonshotai import MoonshotAIProvider

model = OpenAIModel(
    'kimi-k2-0711-preview',
    provider=MoonshotAIProvider(api_key='your-moonshot-api-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Pydantic AI Output Marker Classes and Tool Preparation API
DESCRIPTION: Documents key classes and types in Pydantic AI for managing model output, including plain text and structured tool-based output, and functions for dynamic tool preparation. This includes `TextOutput`, `ToolOutput`, `ToolsPrepareFunc`, `RunContext`, and `ToolDefinition`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
pydantic_ai.output.TextOutput:
  Description: Marker class to configure an agent for plain text output.
  Usage: TextOutput(output_function)
  Parameters:
    output_function: Callable[[str], Any] - A function that takes the model's raw string output and processes it into the desired type.

pydantic_ai.output.ToolOutput:
  Description: Marker class for structured output via tool calls. Allows customization of tool name, description, and strictness.
  Usage: ToolOutput(type_or_function, name: str = None, description: str = None, strict: bool = None)
  Parameters:
    type_or_function: Type | Callable - A Pydantic BaseModel or a function whose signature defines the output schema.
    name: str (optional) - Custom name for the output tool.
    description: str (optional) - Custom description for the output tool. Defaults to the docstring of the type/function.
    strict: bool (optional) - If True, enforces strict schema validation.

pydantic_ai.tools.ToolsPrepareFunc:
  Description: Type alias for a function used to dynamically modify or filter available output tools before an agent step.
  Signature: Callable[[RunContext, list[ToolDefinition]], list[ToolDefinition] | None]
  Parameters:
    context: pydantic_ai.tools.RunContext - The current run context.
    tool_definitions: list[pydantic_ai.tools.ToolDefinition] - A list of tool definitions available for the current step.
  Returns: list[pydantic_ai.tools.ToolDefinition] | None - A new list of tool definitions to use, or None to disable all tools for the step.

pydantic_ai.tools.RunContext:
  Description: An object providing context for the current agent run, used in tool preparation functions. (Details not provided in source text)

pydantic_ai.tools.ToolDefinition:
  Description: Represents the definition of a tool, including its schema and metadata. (Details not provided in source text)
```

----------------------------------------

TITLE: Initialize Agent with CodeExecutionTool
DESCRIPTION: Shows how to set up a PydanticAI agent with the `CodeExecutionTool` to enable code execution in a secure environment. This allows the agent to perform computational tasks, data analysis, and mathematical operations, such as calculating factorials.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/builtin-tools.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, CodeExecutionTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[CodeExecutionTool()])

result = agent.run_sync('Calculate the factorial of 15 and show your work')
# > The factorial of 15 is **1,307,674,368,000**.
```

----------------------------------------

TITLE: Pydantic-AI Bedrock Model and Provider Classes
DESCRIPTION: Comprehensive documentation for key classes used to interact with AWS Bedrock within `pydantic-ai`. This includes `BedrockConverseModel` for instantiating Bedrock models, `BedrockProvider` for managing AWS authentication and client configurations, and `BedrockModelSettings` for customizing runtime API parameters like guardrails and performance.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_7

LANGUAGE: APIDOC
CODE:
```
BedrockConverseModel:
  __init__(model_name: str, provider: BedrockProvider = None)
    model_name: The name of the Bedrock model to use (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0').
    provider: An optional BedrockProvider instance for custom AWS credentials or boto3 client.

BedrockProvider:
  __init__(region_name: str = None, aws_access_key_id: str = None, aws_secret_access_key: str = None, bedrock_client: boto3.client = None)
    region_name: The AWS region for the Bedrock service (e.g., 'us-east-1').
    aws_access_key_id: Your AWS access key ID.
    aws_secret_access_key: Your AWS secret access key.
    bedrock_client: An optional pre-configured boto3 Bedrock runtime client.
    Note: Either explicit credentials (region_name, aws_access_key_id, aws_secret_access_key) or a bedrock_client must be provided.

BedrockModelSettings:
  __init__(bedrock_guardrail_config: dict = None, bedrock_performance_configuration: dict = None)
    bedrock_guardrail_config: A dictionary for Bedrock guardrail configurations, as per AWS Bedrock documentation (e.g., {'guardrailIdentifier': 'v1', 'guardrailVersion': 'v1', 'trace': 'enabled'}).
    bedrock_performance_configuration: A dictionary for Bedrock performance settings, as per AWS Bedrock documentation (e.g., {'latency': 'optimized'}).
```

----------------------------------------

TITLE: Pydantic AI Agent Client Using MCP Sampling
DESCRIPTION: This Python script demonstrates how a Pydantic AI `Agent` can act as a client to an MCP server that supports sampling. It initializes an agent with an `MCPServerStdio` instance, sets the agent's sampling model using `agent.set_mcp_sampling_model()`, and then runs a prompt that triggers the server's sampling capability to generate an image.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_12

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(command='python', args=['generate_svg.py'])
agent = Agent('openai:gpt-4o', toolsets=[server])


async def main():
    async with agent:
        agent.set_mcp_sampling_model()
        result = await agent.run('Create an image of a robot in a punk style.')
    print(result.output)
```

----------------------------------------

TITLE: Configure Logfire Instrumentation for Pydantic AI on Modal
DESCRIPTION: This snippet defines a Python function to set up Logfire instrumentation for Pydantic AI and HTTPX. It must be defined within a Modal function context as `logfire` and other requested packages are only available in that environment, not locally.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_8

LANGUAGE: python
CODE:
```
# This function sets up Logfire instrumentation for Pydantic AI and HTTPX.
# It is designed to run within a Modal function context.
def setup_logfire():
    # Example: logfire.instrument_httpx()
    # Example: logfire.instrument_pydantic_ai()
    pass
```

----------------------------------------

TITLE: Initialize pydantic-ai with OpenAI Responses API Model
DESCRIPTION: Explains how to use the `OpenAIResponsesModel` class to leverage OpenAI's Responses API, which provides access to built-in tools for enhanced model capabilities.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_7

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model)
...
```

----------------------------------------

TITLE: Initialize MistralModel with custom httpx.AsyncClient
DESCRIPTION: Shows how to provide a custom `httpx.AsyncClient` to the `MistralProvider`. This enables fine-grained control over HTTP requests, such as setting custom timeouts or configuring proxies for communication with the Mistral API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/mistral.md#_snippet_5

LANGUAGE: python
CODE:
```
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

custom_http_client = AsyncClient(timeout=30)
model = MistralModel(
    'mistral-large-latest',
    provider=MistralProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: PydanticAI Builtin Tools API Reference
DESCRIPTION: Comprehensive API documentation for PydanticAI's builtin tools, `WebSearchTool` and `CodeExecutionTool`. This section details their purpose, provider compatibility, and configurable parameters for `WebSearchTool`, including notes on provider-specific limitations.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/builtin-tools.md#_snippet_3

LANGUAGE: APIDOC
CODE:
```
WebSearchTool:
  Purpose: Allows agents to search the web for up-to-date data.
  Provider Support:
    - OpenAI: Full feature support.
    - Anthropic: Full feature support.
    - Groq: Limited parameter support (requires compound models).
    - Google, Bedrock, Mistral, Cohere, HuggingFace: Not supported.
  Configuration Parameters:
    - search_context_size: 'high' (string) - Context size for search results. Supported by OpenAI only.
    - user_location: WebSearchUserLocation (object) - Specifies user's location for localized search results.
      - city: string
      - country: string
      - region: string
      - timezone: string
      Supported by OpenAI, Anthropic.
    - blocked_domains: list[str] - Domains to exclude from search results. Supported by Anthropic, Groq.
      Note for Anthropic: Cannot be used simultaneously with 'allowed_domains'.
    - allowed_domains: list[str] - Domains to restrict search results to. Supported by Anthropic, Groq.
      Note for Anthropic: Cannot be used simultaneously with 'blocked_domains'.
    - max_uses: int - Maximum number of times the tool can be used. Supported by Anthropic only.

CodeExecutionTool:
  Purpose: Enables agents to execute code in a secure environment for computational tasks, data analysis, and mathematical operations.
  Provider Support:
    - OpenAI: Supported.
    - Anthropic: Supported.
    - Google: Supported.
    - Groq, Bedrock, Mistral, Cohere, HuggingFace: Not supported.
```

----------------------------------------

TITLE: Connect Pydantic AI Agent to Streamable HTTP MCP Server
DESCRIPTION: This Python code illustrates how to establish a connection from a Pydantic AI `Agent` to an MCP server using the `MCPServerStreamableHTTP` client. It shows how to define the server URL, register the server as a toolset with the agent, and use an `async with` context manager to manage the connection for agent runs.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
agent = Agent('openai:gpt-4o', toolsets=[server])

async def main():
    async with agent:
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
```

----------------------------------------

TITLE: Handle Starlette/FastAPI Request with Pydantic AI Agent
DESCRIPTION: This snippet demonstrates how to integrate a Pydantic AI `Agent` with a Starlette-based web framework like FastAPI. It shows how to use `handle_ag_ui_request()` to process incoming HTTP requests and return responses, effectively exposing the agent as an AG-UI server. The accompanying shell command illustrates how to run the FastAPI application using Uvicorn.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_4

LANGUAGE: python
CODE:
```
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ag_ui import handle_ag_ui_request


agent = Agent('openai:gpt-4.1', instructions='Be fun!')

app = FastAPI()

@app.post("/")
async def run_agent(request: Request) -> Response:
    return await handle_ag_ui_request(agent, request)
```

LANGUAGE: shell
CODE:
```
uvicorn handle_ag_ui_request:app
```

----------------------------------------

TITLE: Pydantic AI AG-UI Integration Methods Overview
DESCRIPTION: Comprehensive overview of the three primary methods provided by Pydantic AI for integrating agents with the AG-UI protocol, detailing their purpose, parameters, return values, and suitable use cases for various web frameworks.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/ag-ui.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
run_ag_ui(agent: Agent, run_input: RunAgentInput, accept: str = SSE_CONTENT_TYPE, **kwargs)
  - Description: Processes an AG-UI RunAgentInput object directly, returning a stream of AG-UI events.
  - Parameters:
    - agent (Agent): The Pydantic AI agent instance to run.
    - run_input (RunAgentInput): The input object containing the agent's run details.
    - accept (str, optional): The desired content type for the event stream (default: SSE_CONTENT_TYPE).
    - **kwargs: Optional arguments passed to Agent.iter(), including 'deps' for dependency injection.
  - Returns: An iterable stream of AG-UI events, encoded as strings.
  - Use Case: Ideal for integrating with web frameworks other than Starlette (e.g., Django, Flask) or when fine-grained control over input/output processing is required.

handle_ag_ui_request(agent: Agent, request: Request, **kwargs)
  - Description: Handles an incoming Starlette Request from an AG-UI frontend, processing it with the agent and returning a streaming Starlette Response.
  - Parameters:
    - agent (Agent): The Pydantic AI agent instance.
    - request (Request): The Starlette Request object.
    - **kwargs: Optional arguments passed to Agent.iter(), including 'deps'. These can be dynamically varied per request, for example, based on the authenticated user.
  - Returns: A Starlette StreamingResponse object containing AG-UI events.
  - Use Case: The typical method for integration within Starlette-based web frameworks like FastAPI.

Agent.to_ag_ui(**kwargs) -> ASGI Application
  - Description: Converts the Pydantic AI agent into a standalone ASGI application that can handle AG-UI requests.
  - Parameters:
    - **kwargs: Optional arguments passed to Agent.iter(), including 'deps'. These arguments are static for the ASGI app, except for AG-UI state which is injected.
  - Returns: An ASGI application instance.
  - Use Case: Can be mounted as a sub-application within an existing FastAPI application, providing a dedicated endpoint for the agent.
```

----------------------------------------

TITLE: Configure OpenAIProvider with API key programmatically
DESCRIPTION: Demonstrates how to programmatically configure the `OpenAIProvider` by passing the API key directly during its instantiation. This method is useful for managing API keys within the application code rather than relying on environment variables.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

----------------------------------------

TITLE: Launch Pydantic AI CLI from Agent Instance (Synchronous)
DESCRIPTION: Launch an interactive `clai` CLI session directly from an `Agent` instance using the `to_cli_sync()` method. This allows immediate interaction with the AI model configured by the agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='You always respond in Italian.')
agent.to_cli_sync()
```

----------------------------------------

TITLE: Running Pydantic AI A2A App with Uvicorn
DESCRIPTION: This Bash command illustrates how to serve the Pydantic AI A2A application, created using the `to_a2a` method, with the Uvicorn ASGI server. It specifies the Python module and the ASGI application instance to run, binding the server to all network interfaces on port 8000, making the agent accessible as an A2A endpoint.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/a2a.md#_snippet_6

LANGUAGE: bash
CODE:
```
uvicorn agent_to_a2a:app --host 0.0.0.0 --port 8000
```

----------------------------------------

TITLE: Configure per-model settings for Pydantic AI FallbackModel
DESCRIPTION: This Python snippet illustrates how to apply individual `ModelSettings` to each model within a `FallbackModel` chain. It highlights the importance of configuring settings like `base_url` or `api_key` directly on each model instance rather than on the `FallbackModel` itself, allowing for distinct configurations for different providers. Dependencies include `pydantic_ai`, `OpenAIModel`, `AnthropicModel`, `FallbackModel`, and `ModelSettings`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/index.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings
```

----------------------------------------

TITLE: Pydantic-AI Agent Dependency Injection Pattern
DESCRIPTION: Demonstrates how to use dependency injection with the `Agent` class in Pydantic-AI, allowing tools to access shared resources like a database connection via the `RunContext`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/CLAUDE.md#_snippet_1

LANGUAGE: Python
CODE:
```
@dataclass
class MyDeps:
    database: DatabaseConn

agent = Agent('openai:gpt-4o', deps_type=MyDeps)

@agent.tool
async def get_data(ctx: RunContext[MyDeps]) -> str:
    return await ctx.deps.database.fetch_data()
```

----------------------------------------

TITLE: Run Pydantic AI CLI with uvx
DESCRIPTION: Execute the `clai` command-line interface using `uvx`, a tool runner from `uv`. This command launches an interactive session for chatting with an AI model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_1

LANGUAGE: bash
CODE:
```
uvx clai
```

----------------------------------------

TITLE: Configure New Logfire Project
DESCRIPTION: Creates and configures a new project within Logfire for sending application data. Alternatively, you can use an existing project. This command sets up the necessary configuration files (e.g., in a .logfire directory) that the Logfire SDK will use at runtime.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_2

LANGUAGE: bash
CODE:
```
py-cli logfire projects new
```

----------------------------------------

TITLE: Initialize pydantic-ai Agent with MistralModel object
DESCRIPTION: Illustrates how to explicitly instantiate a `MistralModel` object and then pass it to the `Agent` constructor. This approach provides more explicit control over the model configuration.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/mistral.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

model = MistralModel('mistral-small-latest')
agent = Agent(model)
...
```

----------------------------------------

TITLE: Initialize Agent with WebSearchTool
DESCRIPTION: Demonstrates how to initialize a PydanticAI agent with the `WebSearchTool` to enable web search capabilities. The agent can then be used to run queries requiring up-to-date information, such as current news.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/builtin-tools.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, WebSearchTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
# > Scientists have developed a universal AI detector that can identify deepfake videos.
```

----------------------------------------

TITLE: Streaming Agent Events and Output in Pydantic-AI
DESCRIPTION: This snippet demonstrates how to set up an AI agent using `pydantic-ai` with custom tools and stream its execution. It shows how to define a `WeatherService` with `get_forecast` and `get_historic_weather` methods, register `weather_forecast` as an agent tool, and then iterate asynchronously over agent run events to capture and process various stages like user prompts, model requests, and tool calls. It highlights the use of `async for` with `Agent.iter()` and `node.stream()` for real-time event handling.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_9

LANGUAGE: python
CODE:
```
import asyncio
from dataclasses import dataclass
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
from pydantic_ai.tools import RunContext


@dataclass
class WeatherService:
    async def get_forecast(self, location: str, forecast_date: date) -> str:
        # In real code: call weather API, DB queries, etc.
        return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'

    async def get_historic_weather(self, location: str, forecast_date: date) -> str:
        # In real code: call a historical weather API or DB
        return f'The weather in {location} on {forecast_date} was 18°C and partly cloudy.'


weather_agent = Agent[WeatherService, str](
    'openai:gpt-4o',
    deps_type=WeatherService,
    output_type=str,  # We'll produce a final answer as plain text
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext[WeatherService],
    location: str,
    forecast_date: date,
) -> str:
    if forecast_date >= date.today():
        return await ctx.deps.get_forecast(location, forecast_date)
    else:
        return await ctx.deps.get_historic_weather(location, forecast_date)


output_messages: list[str] = []


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    # Begin a node-by-node, streaming iteration
    async with weather_agent.iter(user_prompt, deps=WeatherService()) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                # A user prompt node => The user has provided input
                output_messages.append(f'=== UserPromptNode: {node.user_prompt} ===')
            elif Agent.is_model_request_node(node):
                # A model request node => We can stream tokens from the model's request
                output_messages.append('=== ModelRequestNode: streaming partial request tokens ===')
                async with node.stream(run.ctx) as request_stream:
                    final_result_found = False
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}'
                                )
                            elif isinstance(event.delta, ThinkingPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}'
                                )
                            elif isinstance(event.delta, ToolCallPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} args delta: {event.delta.args_delta}'
                                )
                        elif isinstance(event, FinalResultEvent):
                            output_messages.append(
                                f'[Result] The model started producing a final result (tool_name={event.tool_name})'
                            )
                            final_result_found = True
                            break

                    if final_result_found:
                        # Once the final result is found, we can call `AgentStream.stream_text()` to stream the text.
                        # A similar `AgentStream.stream_output()` method is available to stream structured output.
                        async for output in request_stream.stream_text():
                            output_messages.append(f'[Output] {output}')
            elif Agent.is_call_tools_node(node):
                # A handle-response node => The model returned some data, potentially calls a tool
                output_messages.append('=== CallToolsNode: streaming partial response & tool usage ===')
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            output_messages.append(
                                f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'

```

----------------------------------------

TITLE: Authenticate Modal CLI
DESCRIPTION: This command authenticates the Modal CLI, which is a necessary prerequisite for deploying or running any applications on the Modal platform.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_2

LANGUAGE: bash
CODE:
```
python/uv-run -m modal setup
```

----------------------------------------

TITLE: Set Groq API Key Environment Variable
DESCRIPTION: This command sets the `GROQ_API_KEY` environment variable, which is required for authenticating with the Groq API. Replace 'your-api-key' with your actual API key obtained from the Groq console.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/groq.md#_snippet_1

LANGUAGE: bash
CODE:
```
export GROQ_API_KEY='your-api-key'
```

----------------------------------------

TITLE: Configure Pydantic AI with Perplexity API
DESCRIPTION: Shows how to configure an `OpenAIModel` to interact with the Perplexity API. It utilizes `OpenAIProvider` by specifying a custom `base_url` and the Perplexity API key, enabling access to models like 'sonar-pro'.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_27

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'sonar-pro',
    provider=OpenAIProvider(
        base_url='https://api.perplexity.ai',
        api_key='your-perplexity-api-key',
    ),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Integrate Multiple LangChain Tools with Pydantic AI using LangChainToolset in Python
DESCRIPTION: This snippet demonstrates how to integrate multiple LangChain tools or an entire LangChain toolkit, like `SlackToolkit`, into a Pydantic AI agent using the `LangChainToolset`. This approach allows for grouping related LangChain functionalities and providing them as a single toolset to the Pydantic AI agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_15

LANGUAGE: python
CODE:
```
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset


toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-4o', toolsets=[toolset])
# ...
```

----------------------------------------

TITLE: Initialize GoogleModel with Vertex AI using Application Default Credentials
DESCRIPTION: This Python code shows how to configure `GoogleModel` to use Vertex AI by setting `vertexai=True` in the `GoogleProvider`. This method leverages Application Default Credentials for authentication, suitable when running within a GCP environment or with `gcloud` CLI configured.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(vertexai=True)
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Use custom AsyncOpenAI client with pydantic-ai
DESCRIPTION: Shows how to pass a pre-configured `AsyncOpenAI` client instance to `OpenAIProvider`. This allows for custom settings such as `max_retries`, `organization`, or `project` as defined in the OpenAI API documentation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_5

LANGUAGE: python
CODE:
```
from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncOpenAI(max_retries=3)
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
agent = Agent(model)
...
```

----------------------------------------

TITLE: Deploy Pydantic AI Slack Qualifier to Modal
DESCRIPTION: Deploys the Slack lead qualifier application to a Modal workspace for persistent operation. This command makes the application continuously available and accessible without requiring manual execution, suitable for production environments.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_5

LANGUAGE: bash
CODE:
```
python/uv-run -m modal deploy -m pydantic_ai_examples.slack_lead_qualifier.modal
```

----------------------------------------

TITLE: Manage Tool Naming Conflicts with Prefixes in Pydantic AI
DESCRIPTION: Demonstrates how to use the `tool_prefix` parameter with `MCPServerSSE` to avoid naming conflicts when integrating multiple MCP servers that might expose tools with identical names. Each server's tools are prefixed, ensuring unique identification and preventing clashes.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Create two servers with different prefixes
weather_server = MCPServerSSE(
    url='http://localhost:3001/sse',
    tool_prefix='weather'  # Tools will be prefixed with 'weather_'
)

calculator_server = MCPServerSSE(
    url='http://localhost:3002/sse',
    tool_prefix='calc'  # Tools will be prefixed with 'calc_'
)

# Both servers might have a tool named 'get_data', but they'll be exposed as:
# - 'weather_get_data'
# - 'calc_get_data'
agent = Agent('openai:gpt-4o', toolsets=[weather_server, calculator_server])
```

----------------------------------------

TITLE: Stream User Profile with Basic Streaming (pydantic-ai)
DESCRIPTION: This snippet demonstrates how to stream structured output, specifically a `UserProfile` using `pydantic-ai`'s `Agent.run_stream`. It shows defining a `TypedDict` for the output schema and asynchronously iterating over the streamed partial results, printing each update as the profile is built.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_15

LANGUAGE: python
CODE:
```
from datetime import date

from typing_extensions import TypedDict, NotRequired

from pydantic_ai import Agent


class UserProfile(TypedDict):
    name: str
    dob: NotRequired[date]
    bio: NotRequired[str]


agent = Agent(
    'openai:gpt-4o',
    output_type=UserProfile,
    system_prompt='Extract a user profile from the input',
)


async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for profile in result.stream():
            print(profile)
```

----------------------------------------

TITLE: Pydantic AI Structured Output with ToolOutput
DESCRIPTION: Illustrates how to define multiple structured output types using `ToolOutput` marker classes for a Pydantic AI agent. The agent can then return instances of `BaseModel` subclasses (e.g., `Fruit`, `Vehicle`) based on the query, demonstrating how to map model responses to specific data structures and customize tool names.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent, ToolOutput


class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int


agent = Agent(
    'openai:gpt-4o',
    output_type=[ # (1)!
        ToolOutput(Fruit, name='return_fruit'),
        ToolOutput(Vehicle, name='return_vehicle'),
    ],
)
result = agent.run_sync('What is a banana?')
print(repr(result.output))
#> Fruit(name='banana', color='yellow')
```

----------------------------------------

TITLE: Gradio UI Implementation for Weather Agent
DESCRIPTION: The Python code responsible for building the Gradio web interface for the Pydantic AI weather agent. This file includes the UI components and integrates with the agent's backend logic to provide an interactive chat experience.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/weather-agent.md#_snippet_3

LANGUAGE: python
CODE:
```
// Code from /examples/pydantic_ai_examples/weather_agent_gradio.py
```

----------------------------------------

TITLE: Initialize pydantic-ai Agent with HuggingFaceModel object
DESCRIPTION: This Python code shows how to explicitly instantiate a `HuggingFaceModel` object with a specific model name and then pass this model object to the `Agent` constructor. This approach provides more control and is a prerequisite for further model or provider customization.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/huggingface.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel

model = HuggingFaceModel('Qwen/Qwen3-235B-A22B')
agent = Agent(model)
...
```

----------------------------------------

TITLE: pydantic-ai Mistral Model and Provider Configuration
DESCRIPTION: Details the configuration options for `MistralModel` and `MistralProvider` within `pydantic-ai`, including how to specify model names, API keys, base URLs, and custom HTTP clients for interacting with the Mistral API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/mistral.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
MistralModel(model_name: str, provider: Optional[MistralProvider] = None)
  - model_name: The name of the Mistral model to use (e.g., 'mistral-large-latest', 'mistral-small-latest').
  - provider: An optional instance of MistralProvider for custom configuration.

MistralProvider(api_key: str, base_url: Optional[str] = None, http_client: Optional[httpx.AsyncClient] = None)
  - api_key: Your Mistral API key. Can also be set via MISTRAL_API_KEY environment variable.
  - base_url: Optional custom base URL for the Mistral API endpoint.
  - http_client: Optional custom httpx.AsyncClient instance for advanced HTTP request control (e.g., timeouts, proxies).

Agent(model: Union[str, MistralModel])
  - model: The model to use, either as a string identifier (e.g., 'mistral:mistral-large-latest') or a configured MistralModel object.
```

----------------------------------------

TITLE: Define Pydantic AI Agent with Custom Dependencies
DESCRIPTION: This Python code defines a `Pydantic AI Agent` (`joke_agent`) that utilizes custom dependencies encapsulated in the `MyDeps` dataclass. `MyDeps` includes an `httpx.AsyncClient` and a `system_prompt_factory` method for dynamically generating system prompts. The `application_code` function demonstrates how to instantiate `MyDeps` and pass it to `joke_agent.run` for execution, showcasing dependency injection.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/dependencies.md#_snippet_4

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext


@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

    async def system_prompt_factory(self) -> str:  # (1)!
        response = await self.http_client.get('https://example.com')
        response.raise_for_status()
        return f'Prompt: {response.text}'


joke_agent = Agent('openai:gpt-4o', deps_type=MyDeps)


@joke_agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    return await ctx.deps.system_prompt_factory()  # (2)!


async def application_code(prompt: str) -> str:  # (3)!
    ...
    ...
    # now deep within application code we call our agent
    async with httpx.AsyncClient() as client:
        app_deps = MyDeps('foobar', client)
        result = await joke_agent.run(prompt, deps=app_deps)  # (4)!
    return result.output
```

----------------------------------------

TITLE: In-Browser TypeScript Transpilation and Loading Utility
DESCRIPTION: This JavaScript function demonstrates a client-side approach to load and transpile TypeScript code from a specified URL (`/chat_app.ts`) using a global `ts` object (e.g., from a CDN). The transpiled JavaScript is then dynamically appended to the document as a module script. This method is explicitly stated as a non-production-ready hack for development or demonstration purposes.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/chat_app.html#_snippet_1

LANGUAGE: JavaScript
CODE:
```
async function loadTs() {
  const response = await fetch('/chat_app.ts');
  const tsCode = await response.text();
  const jsCode = window.ts.transpile(tsCode, { target: "es2015" });
  let script = document.createElement('script');
  script.type = 'module';
  script.text = jsCode;
  document.body.appendChild(script);
}
loadTs().catch((e) => {
  console.error(e);
  document.getElementById('error').classList.remove('d-none');
  document.getElementById('spinner').classList.remove('active');
});
```

----------------------------------------

TITLE: Pydantic-AI Agent Streaming API Reference
DESCRIPTION: Comprehensive documentation for the core API components involved in streaming text responses from pydantic-ai agents, including the Agent class, `run_stream` method, and `stream_text` method.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_14

LANGUAGE: APIDOC
CODE:
```
Pydantic-AI Agent Streaming API:

Agent:
  - Class: pydantic_ai.Agent
  - Description: The main class for interacting with AI models.
  - Instantiation: Agent(model_name: str)

Agent.run_stream:
  - Method of: pydantic_ai.agent.AbstractAgent
  - Signature: async with agent.run_stream(prompt: str) -> StreamedRunResult
  - Parameters:
    - prompt (str): The input prompt for the AI model.
  - Returns: An asynchronous context manager that yields a StreamedRunResult object.
  - Purpose: Initiates a streaming interaction with the AI model, allowing for real-time text output. The context manager ensures proper connection closure.

StreamedRunResult.stream_text:
  - Method of: pydantic_ai.result.StreamedRunResult
  - Signature: async for message in result.stream_text(delta: bool = False) -> AsyncIterator[str]
  - Parameters:
    - delta (bool, optional): If True, streams incremental text changes (deltas). If False (default), streams the complete text response with each update.
  - Returns: An asynchronous iterator yielding string messages.
  - Purpose: Provides access to the streamed text output from the AI model.
  - Note: If `delta=True`, the final output message will NOT be added to the result messages.
```

----------------------------------------

TITLE: Pydantic AI Agent Testing Utilities Reference
DESCRIPTION: This section provides a reference for key utilities used in testing Pydantic AI agents. It covers `Agent.override` for mocking models, `TestModel` for simulating LLM responses, `capture_run_messages` for inspecting agent-model interactions, and `IsNow` for handling dynamic timestamps in assertions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#_snippet_3

LANGUAGE: APIDOC
CODE:
```
Agent.override
  Description: A context manager used to temporarily replace an agent's underlying model, typically with a mock model like `TestModel`, without needing to modify the agent's `run*` method call sites.
  Usage: Used in testing to control the agent's model behavior.

TestModel
  Description: A mock model used in testing Pydantic AI agents. By default, it returns a JSON string summarizing tool calls and their returns.
  Parameters:
    custom_output_text (str, optional): Allows customizing the model's response to a specific string instead of the default JSON summary.
  Usage: Replaces the real LLM during tests to prevent actual API calls and control responses.

capture_run_messages
  Description: A utility function to inspect the message exchange between an agent and its model during a run.
  Returns: A list of messages (e.g., `ModelRequest`, `ModelResponse`, `ToolReturnPart`, `TextPart`) representing the interaction.
  Usage: Used for asserting the exact sequence and content of messages exchanged during an agent's operation.

IsNow (from dirty_equals)
  Description: A helper for declarative assertions, particularly useful for comparing data that includes timestamps, allowing for flexible matching of current or recent timestamps.
  Usage: Enables robust assertions in tests where exact timestamp matching is problematic.
```

----------------------------------------

TITLE: Initialize Bedrock Model Directly
DESCRIPTION: Illustrates how to explicitly initialize a `BedrockConverseModel` instance with a specific model name and then pass this model object to an `Agent`, providing more control over model instantiation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

model = BedrockConverseModel('anthropic.claude-3-sonnet-20240229-v1:0')
agent = Agent(model)
...
```

----------------------------------------

TITLE: pydantic_ai.toolsets Module Members
DESCRIPTION: This entry documents the key classes and functions available within the `pydantic_ai.toolsets` module. These components provide flexible ways to define, combine, filter, and manage tools for AI applications, supporting patterns like deferred execution, prefixing, renaming, and wrapping.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/api/toolsets.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
pydantic_ai.toolsets Module Members:

AbstractToolset
  - Description: Base abstract class for defining custom toolsets. Provides a common interface for managing and interacting with collections of tools.

CombinedToolset
  - Description: A concrete toolset implementation that aggregates tools from multiple other toolsets. Useful for creating a unified interface from disparate tool sources.

DeferredToolset
  - Description: A toolset that defers the loading or initialization of its tools until they are actually needed. Improves performance by avoiding unnecessary resource allocation.

FilteredToolset
  - Description: Creates a new toolset by applying a filter to an existing toolset, including only tools that match specified criteria.

FunctionToolset
  - Description: A toolset constructed directly from a collection of Python functions, treating each function as a callable tool.

PrefixedToolset
  - Description: Modifies an existing toolset by adding a specified prefix to the names of all its contained tools. Useful for avoiding naming conflicts.

RenamedToolset
  - Description: Allows for the renaming of specific tools within an existing toolset, providing more descriptive or consistent naming conventions.

PreparedToolset
  - Description: A toolset where tools undergo a preparation or transformation step before being made available.

WrapperToolset
  - Description: A toolset that wraps another toolset, allowing for the interception or modification of tool access and behavior without altering the original toolset.

ToolsetFunc
  - Description: Likely a type alias or a utility function signature related to the creation or manipulation of toolsets.
```

----------------------------------------

TITLE: Visualize Agent-LLM Tool Interaction Flow (Mermaid)
DESCRIPTION: This Mermaid sequence diagram visually represents the interaction flow between a `pydantic-ai` Agent and the Language Model (LLM) during a tool-enabled conversation. It highlights the steps where the LLM calls agent tools (`roll_dice`, `get_player_name`), the agent executes them, and returns results, ultimately leading to the LLM's final response.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_2

LANGUAGE: mermaid
CODE:
```
sequenceDiagram
    participant Agent
    participant LLM

    Note over Agent: Send prompts
    Agent ->> LLM: System: "You're a dice game..."<br>User: "My guess is 4"
    activate LLM
    Note over LLM: LLM decides to use<br>a tool

    LLM ->> Agent: Call tool<br>roll_dice()
    deactivate LLM
    activate Agent
    Note over Agent: Rolls a six-sided die

    Agent -->> LLM: ToolReturn<br>"4"
    deactivate Agent
    activate LLM
    Note over LLM: LLM decides to use<br>another tool

    LLM ->> Agent: Call tool<br>get_player_name()
    deactivate LLM
    activate Agent
    Note over Agent: Retrieves player name
    Agent -->> LLM: ToolReturn<br>"Anne"
    deactivate Agent
    activate LLM
    Note over LLM: LLM constructs final response

    LLM ->> Agent: ModelResponse<br>"Congratulations Anne, ..."
    deactivate LLM
    Note over Agent: Game session complete
```

----------------------------------------

TITLE: Configure Hugging Face provider with custom AsyncInferenceClient
DESCRIPTION: This Python code demonstrates how to create a custom `AsyncInferenceClient` from `huggingface_hub` with specific configurations (e.g., `bill_to`, `api_key`, `provider`) and then pass this client to the `HuggingFaceProvider`. This enables advanced customization of the underlying Hugging Face client behavior, such as billing to a specific organization or using a custom base URL.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/huggingface.md#_snippet_5

LANGUAGE: python
CODE:
```
from huggingface_hub import AsyncInferenceClient

from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider

client = AsyncInferenceClient(
    bill_to='openai',
    api_key='hf_token',
    provider='fireworks-ai',
)

model = HuggingFaceModel(
    'Qwen/Qwen3-235B-A22B',
    provider=HuggingFaceProvider(hf_client=client),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Pydantic AI Provider Classes
DESCRIPTION: Documentation for various provider classes in `pydantic-ai`, enabling integration with different LLM services. Each provider class handles specific authentication and endpoint configurations for its respective service.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_24

LANGUAGE: APIDOC
CODE:
```
OpenAIProvider:
  __init__(base_url: str = 'https://api.openai.com/v1', api_key: str = None)
    base_url: The base URL for the OpenAI-compatible API endpoint. Defaults to OpenAI's official API.
    api_key: (Optional) Your OpenAI API key. Can be omitted if set via environment variables.

AzureProvider:
  __init__(azure_endpoint: str, api_version: str, api_key: str)
    azure_endpoint: The base URL for your Azure AI service endpoint.
    api_version: The API version to use (e.g., '2024-02-01').
    api_key: Your Azure API key.

OpenRouterProvider:
  __init__(api_key: str)
    api_key: Your OpenRouter API key.

VercelProvider:
  __init__(api_key: str = None)
    api_key: (Optional) Your Vercel AI Gateway API key. If not provided, attempts to read from VERCEL_AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN environment variables.

GrokProvider:
  __init__(api_key: str)
    api_key: Your xAI API key for Grok models.

MoonshotAIProvider:
  __init__(api_key: str)
    api_key: Your MoonshotAI API key.
```

----------------------------------------

TITLE: Integrate Pydantic AI with Vercel AI Gateway
DESCRIPTION: Demonstrates configuring `pydantic-ai` to use Vercel AI Gateway. It shows two methods: automatic credential detection via environment variables and direct API key passing during provider instantiation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_20

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.vercel import VercelProvider

# Uses environment variable automatically
model = OpenAIModel(
    'anthropic/claude-4-sonnet',
    provider=VercelProvider(),
)
agent = Agent(model)

# Or pass the API key directly
model = OpenAIModel(
    'anthropic/claude-4-sonnet',
    provider=VercelProvider(api_key='your-vercel-ai-gateway-api-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Set OpenAI API key environment variable
DESCRIPTION: Demonstrates how to set the `OPENAI_API_KEY` environment variable. This is the default method `pydantic-ai` uses to authenticate with the OpenAI API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_1

LANGUAGE: bash
CODE:
```
export OPENAI_API_KEY='your-api-key'
```

----------------------------------------

TITLE: pydantic-ai Direct API Request Methods Reference
DESCRIPTION: Comprehensive reference for the low-level request methods available in `pydantic_ai.direct`. These functions provide direct control over LLM interactions, supporting various request patterns (sync/async, streamed/non-streamed) while handling input/output schema translation. They are ideal for custom abstractions or fine-grained control.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/direct.md#_snippet_3

LANGUAGE: APIDOC
CODE:
```
pydantic_ai.direct.model_request(
    model_name: str,
    messages: list[ModelRequest],
    model_request_parameters: ModelRequestParameters = None,
    **kwargs
) -> ModelResponse
  - Purpose: Make a non-streamed asynchronous request to a model.
  - Parameters:
    - model_name (str): The identifier for the LLM (e.g., 'anthropic:claude-3-5-haiku-latest', 'openai:gpt-4.1-nano').
    - messages (list[ModelRequest]): A list of message objects representing the conversation history or prompt.
    - model_request_parameters (ModelRequestParameters, optional): Additional parameters for the model request, such as function tools or output allowances.
  - Returns: ModelResponse object containing the model's reply, usage information, etc.

pydantic_ai.direct.model_request_sync(
    model_name: str,
    messages: list[ModelRequest],
    model_request_parameters: ModelRequestParameters = None,
    **kwargs
) -> ModelResponse
  - Purpose: Make a non-streamed synchronous request to a model.
  - Parameters: Same as model_request.
  - Returns: Same as model_request.

pydantic_ai.direct.model_request_stream(
    model_name: str,
    messages: list[ModelRequest],
    model_request_parameters: ModelRequestParameters = None,
    **kwargs
) -> AsyncIterator[ModelResponse]
  - Purpose: Make a streamed asynchronous request to a model.
  - Parameters: Same as model_request.
  - Returns: An asynchronous iterator yielding ModelResponse objects as the stream progresses.

pydantic_ai.direct.model_request_stream_sync(
    model_name: str,
    messages: list[ModelRequest],
    model_request_parameters: ModelRequestParameters = None,
    **kwargs
) -> Iterator[ModelResponse]
  - Purpose: Make a streamed synchronous request to a model.
  - Parameters: Same as model_request.
  - Returns: A synchronous iterator yielding ModelResponse objects as the stream progresses.
```

----------------------------------------

TITLE: Integrate DeepSeek provider with OpenAIModel
DESCRIPTION: This snippet demonstrates how to use Pydantic AI's OpenAIModel with the DeepSeek API. It shows the instantiation of DeepSeekProvider with an API key, simplifying the connection to DeepSeek's services without needing to manually specify the base URL.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIModel(
    'deepseek-chat',
    provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Convert Pydantic AI Agent to ASGI Application
DESCRIPTION: This Python snippet demonstrates how to instantiate a Pydantic AI `Agent` and convert it into a standard ASGI application using the `agent.to_a2a()` method. The resulting `app` object can then be served by any ASGI-compatible web server, enabling the agent to function as an A2A server. This method simplifies the exposure of AI agents as network services.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/a2a.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='Be fun!')
app = agent.to_a2a()
```

----------------------------------------

TITLE: Connect Pydantic AI Agent to MCP Stdio Server
DESCRIPTION: Illustrates connecting a Pydantic AI `Agent` to an MCP server using the standard I/O (stdio) transport via `MCPServerStdio`. The server is launched as a subprocess with specified arguments, and the agent communicates with it over stdin/stdout for tool execution.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(  # (1)!
    'deno',
    args=[
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ]
)
agent = Agent('openai:gpt-4o', toolsets=[server])


async def main():
    async with agent:
        result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    # There are 9,208 days between January 1, 2000, and March 18, 2025.
```

----------------------------------------

TITLE: Set GitHub API Key Environment Variable
DESCRIPTION: Illustrates how to set the `GITHUB_API_KEY` environment variable. This provides an alternative method for authenticating with GitHub Models, allowing the API key to be managed outside the code.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_26

LANGUAGE: Bash
CODE:
```
export GITHUB_API_KEY='your-github-token'
```

----------------------------------------

TITLE: Initialize CohereModel with Custom Provider and API Key
DESCRIPTION: This Python code demonstrates how to provide a custom `CohereProvider` instance to the `CohereModel` constructor, allowing direct specification of the API key within the code. This is useful when the API key is not set as an environment variable or needs to be dynamically managed for different model instances.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/cohere.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

model = CohereModel('command', provider=CohereProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

----------------------------------------

TITLE: pydantic-ai GoogleModelSettings API Reference
DESCRIPTION: Defines the configurable parameters for `GoogleModel` instances within the `pydantic-ai` framework, allowing fine-tuning of model behavior, thinking processes, and safety thresholds.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_10

LANGUAGE: APIDOC
CODE:
```
Class: pydantic_ai.models.google.GoogleModelSettings

Parameters:
  temperature (float, optional): Controls the randomness of the output. Higher values mean more random.
  max_tokens (int, optional): The maximum number of tokens to generate in the response.
  google_thinking_config (dict, optional): Configuration for the model's 'thinking' process.
    - thinking_budget (int): Budget for thinking, setting to 0 disables thinking.
  google_safety_settings (list[dict], optional): A list of safety settings to apply.
    Each dictionary should contain:
    - category (HarmCategory): The safety category (e.g., HARM_CATEGORY_HATE_SPEECH).
    - threshold (HarmBlockThreshold): The blocking threshold for the category (e.g., BLOCK_LOW_AND_ABOVE).

Usage Context:
  Used to initialize pydantic_ai.Agent with a GoogleModel, e.g., Agent(model, model_settings=settings).
```

----------------------------------------

TITLE: Integrate Multiple LangChain Tools with Pydantic AI using LangChainToolset
DESCRIPTION: This Python code demonstrates how to integrate multiple LangChain tools or an entire LangChain toolkit into a Pydantic AI agent using `LangChainToolset`. This is useful for grouping related functionalities, such as all tools from the `SlackToolkit`, and providing them to the agent as a single toolset.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_18

LANGUAGE: python
CODE:
```
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset


toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-4o', toolsets=[toolset])
# ...
```

----------------------------------------

TITLE: Dynamically Registering Agent Toolsets with Pydantic-AI
DESCRIPTION: This Python snippet demonstrates how to dynamically register toolsets with an `Agent` in Pydantic-AI using a function decorated with `@agent.toolset`. It shows how to switch between different toolsets (weather or datetime) based on agent dependencies (`RunContext`) and how to inspect available tools using `TestModel`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_8

LANGUAGE: python
CODE:
```
from dataclasses import dataclass
from typing import Literal

from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


@dataclass
class ToggleableDeps:
    active: Literal['weather', 'datetime']

    def toggle(self):
        if self.active == 'weather':
            self.active = 'datetime'
        else:
            self.active = 'weather'

test_model = TestModel()  # (1)!
agent = Agent(
    test_model,
    deps_type=ToggleableDeps  # (2)!
)

@agent.toolset
def toggleable_toolset(ctx: RunContext[ToggleableDeps]):
    if ctx.deps.active == 'weather':
        return weather_toolset
    else:
        return datetime_toolset

@agent.tool
def toggle(ctx: RunContext[ToggleableDeps]):
    ctx.deps.toggle()

deps = ToggleableDeps('weather')

result = agent.run_sync('Toggle the toolset', deps=deps)
print([t.name for t in test_model.last_model_request_parameters.function_tools])  # (3)!
# > ['toggle', 'now']

result = agent.run_sync('Toggle the toolset', deps=deps)
print([t.name for t in test_model.last_model_request_parameters.function_tools])
# > ['toggle', 'temperature_celsius', 'temperature_fahrenheit', 'conditions']
```

----------------------------------------

TITLE: Initialize pydantic-ai Agent with Mistral model name string
DESCRIPTION: Shows how to create an `Agent` instance in `pydantic-ai` by directly passing a string identifier for a Mistral model. This method assumes the Mistral API key is configured via an environment variable.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/mistral.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('mistral:mistral-large-latest')
...
```

----------------------------------------

TITLE: Initialize Agent with Cohere Model by Name String
DESCRIPTION: This Python code demonstrates how to initialize an `Agent` instance from `pydantic_ai` by directly specifying a Cohere model using its 'cohere:model_name' string identifier. This approach simplifies model instantiation when the API key is managed via environment variables.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/cohere.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('cohere:command')
...
```

----------------------------------------

TITLE: Python: GenAI Email Feedback System Implementation
DESCRIPTION: This Python code implements a generative AI system for writing and refining welcome emails using Pydantic AI and Pydantic Graph. It defines `User`, `Email`, and `State` dataclasses, along with `Agent` instances for email generation and feedback. The `WriteEmail` and `Feedback` nodes orchestrate the graph's execution, demonstrating how to integrate AI agents into a structured workflow.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_12

LANGUAGE: Python
CODE:
```
from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, EmailStr

from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


email_writer_agent = Agent(
    'google-vertex:gemini-1.5-pro',
    output_type=Email,
    system_prompt='Write a welcome email to our tech blog.',
)


@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f'Rewrite the email for the user:\n'
                f'{format_as_xml(ctx.state.user)}\n'
                f'Feedback: {self.email_feedback}'
            )
        else:
            prompt = (
                f'Write a welcome email for the user:\n'
                f'{format_as_xml(ctx.state.user)}'
            )

        result = await email_writer_agent.run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        ctx.state.write_agent_messages += result.new_messages()
        return Feedback(result.output)


class EmailRequiresWrite(BaseModel):
    feedback: str


class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    'openai:gpt-4o',
    output_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=(
        'Review the email and provide feedback, email must reference the users specific interests.'
    ),
)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> WriteEmail | End[Email]:
        prompt = format_as_xml({'user': ctx.state.user, 'email': self.email})
        result = await feedback_agent.run(prompt)
        if isinstance(result.output, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.output.feedback)
        else:
            return End(self.email)


async def main():
    user = User(
        name='John Doe',
        email='john.joe@example.com',
        interests=['Haskel', 'Lisp', 'Fortran'],
    )
    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))
    result = await feedback_graph.run(WriteEmail(), state=state)
    print(result.output)
    """
    Email(
        subject='Welcome to our tech blog!',
        body='Hello John, Welcome to our tech blog! ...',
    )
    """
```

----------------------------------------

TITLE: Initialize GoogleModel with Vertex AI and Custom Location
DESCRIPTION: This Python code illustrates how to specify a custom location (region) when using `GoogleModel` with Vertex AI. By passing the `location` argument to the `GoogleProvider`, you can control data residency and potentially improve latency.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(vertexai=True, location='asia-east1')
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Chat Application UI Styling
DESCRIPTION: This CSS defines the visual appearance and layout for a chat application's main container, conversation elements (user and AI messages), and an animated loading spinner. It includes basic styling for text, display, and a keyframe animation for the spinner.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/chat_app.html#_snippet_0

LANGUAGE: CSS
CODE:
```
main { max-width: 700px; }
#conversation .user::before { content: 'You asked: '; font-weight: bold; display: block; }
#conversation .model::before { content: 'AI Response: '; font-weight: bold; display: block; }
#spinner { opacity: 0; transition: opacity 500ms ease-in; width: 30px; height: 30px; border: 3px solid #222; border-bottom-color: transparent; border-radius: 50%; animation: rotation 1s linear infinite; }
@keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
#spinner.active { opacity: 1; }
```

----------------------------------------

TITLE: Execute and Persist Human-in-the-Loop AI Q&A Graph
DESCRIPTION: This Python script demonstrates how to execute the previously defined `question_graph` using `pydantic-graph`'s state persistence. It shows how to load the graph's state from a file, provide user input (an answer via command line), and continue the graph's execution from the last saved point. This enables a multi-invocation, interactive human-in-the-loop workflow.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_17

LANGUAGE: python
CODE:
```
import sys
from pathlib import Path

from pydantic_graph import End
from pydantic_graph.persistence.file import FileStatePersistence
from pydantic_ai.messages import ModelMessage  # noqa: F401

from ai_q_and_a_graph import Ask, question_graph, Evaluate, QuestionState, Answer


async def main():
    answer: str | None = sys.argv[1] if len(sys.argv) > 1 else None  # (1)!
    persistence = FileStatePersistence(Path('question_graph.json'))  # (2)!
    persistence.set_graph_types(question_graph)  # (3)!

    if snapshot := await persistence.load_next():  # (4)!
        state = snapshot.state
        assert answer is not None
        node = Evaluate(answer)
    else:
        state = QuestionState()
        node = Ask()  # (5)!

    async with question_graph.iter(node, state=state, persistence=persistence) as run:
        while True:
            node = await run.next()  # (6)!
            if isinstance(node, End):  # (7)!
                print('END:', node.data)
                history = await persistence.load_all()  # (8)!
                print([e.node for e in history])
                break
            elif isinstance(node, Answer):  # (9)!
                print(node.question)
                #> What is the capital of France?
                break
            # otherwise just continue
```

----------------------------------------

TITLE: Initialize Pydantic-AI with OpenAI-compatible provider and custom retry client
DESCRIPTION: This Python code demonstrates how to configure `pydantic-ai` to work with any OpenAI-compatible API endpoint. It shows the initialization of an `OpenAIModel` using a custom `OpenAIProvider` that integrates a retrying HTTP client, enabling robust handling of transient network issues and rate limits.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_11

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIModel(
    'your-model-name',  # Replace with actual model name
    provider=OpenAIProvider(
        base_url='https://api.example.com/v1',  # Replace with actual API URL
        api_key='your-api-key',  # Replace with actual API key
        http_client=client
    )
)
agent = Agent(model)
```

----------------------------------------

TITLE: Configure AWS Credentials via Environment Variables
DESCRIPTION: Demonstrates how to set AWS credentials and default region as environment variables, which `boto3` and `pydantic-ai` can automatically pick up for authenticating with AWS Bedrock.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_1

LANGUAGE: bash
CODE:
```
export AWS_BEARER_TOKEN_BEDROCK='your-api-key'
# or:
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='us-east-1'  # or your preferred region
```

----------------------------------------

TITLE: Customize Tool Calls in Pydantic AI MCP Server
DESCRIPTION: Shows how to implement a `process_tool_call` function to inject custom metadata (like dependencies) into tool call requests before they are sent to the MCP server. This allows for dynamic modification of tool arguments or context, enabling more flexible tool interactions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_7

LANGUAGE: python
CODE:
```
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.mcp import CallToolFunc, MCPServerStdio, ToolResult
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext


async def process_tool_call(
    ctx: RunContext[int],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """A tool call processor that passes along the deps."""
    return await call_tool(name, tool_args, {'deps': ctx.deps})


server = MCPServerStdio('python', ['mcp_server.py'], process_tool_call=process_tool_call)
agent = Agent(
    model=TestModel(call_tools=['echo_deps']),
    deps_type=int,
    toolsets=[server]
)


async def main():
    async with agent:
        result = await agent.run('Echo with deps set to 42', deps=42)
    print(result.output)
    # {"echo_deps":{"echo":"This is an echo message","deps":42}}
```

----------------------------------------

TITLE: Simulate Slack team_join Event for Testing
DESCRIPTION: This API documentation describes how to simulate a Slack 'team_join' event by sending a POST request to a specified webhook endpoint. It includes the required JSON payload structure for the event, useful for testing the lead qualification agent's response to new user signups.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
POST <webhook endpoint URL>
Content-Type: application/json

Body (JSON):
{
    "type": "event_callback",
    "event": {
        "type": "team_join",
        "user": {
            "profile": {
                "email": "samuel@pydantic.dev",
                "first_name": "Samuel",
                "last_name": "Colvin",
                "display_name": "Samuel Colvin"
            }
        }
    }
}

Example Usage (cURL):
curl -X POST <webhook endpoint URL> \
-H "Content-Type: application/json" \
-d '{
    "type": "event_callback",
    "event": {
        "type": "team_join",
        "user": {
            "profile": {
                "email": "samuel@pydantic.dev",
                "first_name": "Samuel",
                "last_name": "Colvin",
                "display_name": "Samuel Colvin"
            }
        }
    }
}'
```

----------------------------------------

TITLE: Configure Pydantic AI with Azure AI Foundry
DESCRIPTION: Illustrates how to set up `pydantic-ai` to leverage Azure AI Foundry. This configuration requires specifying the Azure endpoint, API version, and an API key for authentication with the Azure service.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_17

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider

model = OpenAIModel(
    'gpt-4o',
    provider=AzureProvider(
        azure_endpoint='your-azure-endpoint',
        api_version='your-api-version',
        api_key='your-api-key',
    ),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Customize Bedrock Runtime API Settings
DESCRIPTION: Demonstrates how to apply custom settings to Bedrock Runtime API calls using `BedrockModelSettings`. This includes configuring guardrails and performance optimizations for the model, which are then passed to the `Agent`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_4

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

# Define Bedrock model settings with guardrail and performance configurations
bedrock_model_settings = BedrockModelSettings(
    bedrock_guardrail_config={
        'guardrailIdentifier': 'v1',
        'guardrailVersion': 'v1',
        'trace': 'enabled'
    },
    bedrock_performance_configuration={
        'latency': 'optimized'
    }
)


model = BedrockConverseModel(model_name='us.amazon.nova-pro-v1:0')

agent = Agent(model=model, model_settings=bedrock_model_settings)
```

----------------------------------------

TITLE: Pydantic AI Agent.to_a2a Method Reference
DESCRIPTION: This API documentation describes the `Agent.to_a2a` method, a convenience function for transforming a Pydantic AI agent into an ASGI application suitable for A2A server deployment. It highlights that the method accepts arguments mirroring the `FastA2A` constructor and details its built-in capabilities, including automatic conversation history storage, context management for `context_id`-based messages, and the structured persistence of agent results as `TextPart` or `DataPart` artifacts with rich metadata.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/a2a.md#_snippet_5

LANGUAGE: APIDOC
CODE:
```
Agent.to_a2a(*args, **kwargs)
  - Converts the Pydantic AI Agent into an ASGI application.
  - Parameters:
    - *args, **kwargs: Arguments are passed directly to the `FastA2A` constructor.
  - Returns: An ASGI application object.
  - Automatic functionalities provided:
    - Stores complete conversation history (including tool calls and responses) in the context storage.
    - Ensures that subsequent messages with the same `context_id` have access to the full conversation history.
    - Persists agent results as A2A artifacts:
      - String results become `TextPart` artifacts and also appear in the message history.
      - Structured data (Pydantic models, dataclasses, tuples, etc.) become `DataPart` artifacts with the data wrapped as `{"result": <your_data>}`.
      - Artifacts include metadata with type information and JSON schema when available.
```

----------------------------------------

TITLE: pydantic_ai.models Module API Reference
DESCRIPTION: Detailed API documentation for the `pydantic_ai.models` module, outlining its public members. This module is central to defining and managing AI model interactions, including data models for request parameters, handling streamed responses, and controlling access permissions for model requests.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/api/models/base.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Module: pydantic_ai.models

This module provides core components for defining and managing AI model interactions within the pydantic_ai library.

Members:
  - KnownModelName:
      Type: Class/Enum
      Description: Represents a predefined set of known AI model names.
  - ModelRequestParameters:
      Type: Class
      Description: Pydantic model defining the parameters for making a request to an AI model.
  - Model:
      Type: Class
      Description: Abstract base class or interface for AI models, likely defining common methods or properties.
  - AbstractToolDefinition:
      Type: Class
      Description: Abstract base class for defining tools that AI models can use or interact with.
  - StreamedResponse:
      Type: Class
      Description: Handles and processes streamed responses from AI models, often used for real-time output.
  - ALLOW_MODEL_REQUESTS:
      Type: Constant (bool)
      Description: A global flag or configuration variable indicating whether AI model requests are generally permitted.
  - check_allow_model_requests():
      Type: Function
      Description: Checks the current status of the ALLOW_MODEL_REQUESTS flag to determine if model requests are allowed.
      Signature: check_allow_model_requests() -> bool
  - override_allow_model_requests(allow: bool):
      Type: Function
      Description: Temporarily overrides the ALLOW_MODEL_REQUESTS flag, typically used as a context manager or decorator.
      Signature: override_allow_model_requests(allow: bool) -> None
```

----------------------------------------

TITLE: Initialize AnthropicModel using Agent with model name string
DESCRIPTION: This Python snippet demonstrates how to initialize an `Agent` instance from `pydantic_ai` by directly passing a string identifier for an Anthropic model. The `Agent` internally handles the creation of the `AnthropicModel` based on the provided name, simplifying model instantiation.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/anthropic.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('anthropic:claude-3-5-sonnet-latest')
...
```

----------------------------------------

TITLE: Pydantic AI File URL Processing Behavior
DESCRIPTION: Describes the default and model-specific behaviors for handling file URLs (e.g., ImageUrl, DocumentUrl) within Pydantic AI, including when files are downloaded by the user's application versus when URLs are passed directly to the AI model. It also covers the 'force_download' option for Google models.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/input.md#_snippet_4

LANGUAGE: APIDOC
CODE:
```
Pydantic AI File URL Handling:

General Rule:
  - When providing a URL using `ImageUrl`, `AudioUrl`, `VideoUrl`, or `DocumentUrl`, Pydantic AI typically downloads the file content and sends it as part of the API request.

Model-Specific Behaviors:

AnthropicModel:
  - `DocumentUrl` (specifically for PDF documents): The URL is sent directly in the API request; no user-side download occurs.

GoogleModel (on Vertex AI):
  - All `FileUrl` types (`ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`): URLs are sent as-is in the API request; no data is downloaded beforehand.
  - Supported URLs (as per Gemini API docs for Vertex AI):
    - Cloud Storage bucket URIs (e.g., `gs://bucket/path/to/file`)
    - Public HTTP(S) URLs
    - Public YouTube video URL (maximum one URL per request)
  - Crawling Restrictions & `force_download`:
    - If Gemini cannot access certain URLs due to crawling restrictions, you can instruct Pydantic AI to download the file content and send that instead of the URL.
    - To do this, set the boolean flag `force_download` to `True` on any object inheriting from `FileUrl`.
    - `force_download`: boolean, defaults to `False`. When `True`, forces Pydantic AI to download the content locally and send the data, rather than the URL.

GoogleModel (on GLA):
  - YouTube video URLs: Sent directly in the request to the model.
```

----------------------------------------

TITLE: Run OpenTelemetry TUI backend via Docker
DESCRIPTION: This shell command provides instructions to launch the `otel-tui` OpenTelemetry terminal user interface backend using Docker. It exposes port 4318, allowing other applications to send OTLP (OpenTelemetry Protocol) traces to this local viewer for real-time debugging and visualization.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_6

LANGUAGE: shell
CODE:
```
docker run --rm -it -p 4318:4318 --name otel-tui ymtdzzz/otel-tui:latest

```

----------------------------------------

TITLE: AgentRun Object API Reference
DESCRIPTION: Documentation for accessing usage statistics and final results from the `AgentRun` object in `pydantic-ai`. This includes the `usage()` method for runtime metrics and the `result` attribute for the final output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_8

LANGUAGE: APIDOC
CODE:
```
AgentRun Object API:
  Method: usage()
    Description: Retrieves usage statistics (tokens, requests, etc.) at any time from the AgentRun object.
    Returns: A `pydantic_ai.usage.Usage` object containing the usage data.

  Attribute: result
    Description: Becomes available once the run finishes. It holds the final output and related metadata.
    Type: `pydantic_ai.agent.AgentRunResult` object.
```

----------------------------------------

TITLE: Set OpenAI API Key Environment Variable
DESCRIPTION: This command sets the `OPENAI_API_KEY` environment variable, which is required for `clai` to authenticate with the OpenAI API. Replace 'your-api-key-here' with your actual API key before running `clai`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/clai/README.md#_snippet_0

LANGUAGE: bash
CODE:
```
export OPENAI_API_KEY='your-api-key-here'
```

----------------------------------------

TITLE: Integrate Pydantic AI with Remote Ollama
DESCRIPTION: Shows how to connect `pydantic-ai` to a remote Ollama server by specifying its IP address and port in the `base_url` parameter. This enables distributed model inference using a model hosted elsewhere.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_16

LANGUAGE: python
CODE:
```
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

ollama_model = OpenAIModel(
    model_name='qwen2.5-coder:7b',  # (1)!
    provider=OpenAIProvider(base_url='http://192.168.1.74:11434/v1'),  # (2)!
)


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent(model=ollama_model, output_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
# city='London' country='United Kingdom'
print(result.usage())
# Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
```

----------------------------------------

TITLE: Simulate Slack Team Join Event via cURL
DESCRIPTION: This cURL command sends a POST request to the application's webhook endpoint, simulating a 'team_join' event. It includes a JSON payload with a mock user profile, which is highly useful for testing the lead qualification logic without requiring an actual Slack signup.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_4

LANGUAGE: bash
CODE:
```
curl -X POST <webhook endpoint URL> \
-H "Content-Type: application/json" \
-d '{
    "type": "event_callback",
    "event": {
        "type": "team_join",
        "user": {
            "profile": {
                "email": "samuel@pydantic.dev",
                "first_name": "Samuel",
                "last_name": "Colvin",
                "display_name": "Samuel Colvin"
            }
        }
    }
}'
```

LANGUAGE: APIDOC
CODE:
```
{
    "type": "event_callback",
    "event": {
        "type": "team_join",
        "user": {
            "profile": {
                "email": "samuel@pydantic.dev",
                "first_name": "Samuel",
                "last_name": "Colvin",
                "display_name": "Samuel Colvin"
            }
        }
    }
}
```

----------------------------------------

TITLE: AI Question and Answer Graph with Customizations
DESCRIPTION: This Python code defines a question-answering graph using Pydantic Graph, demonstrating how to integrate AI agents (e.g., GPT-4o) into a state-based flow. It showcases advanced customization for Mermaid diagram generation by adding labels to edges using `Edge(label='...')` and enabling docstring-based notes for nodes via `docstring_notes = True` to enrich the visual representation of the graph.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_19

LANGUAGE: python
CODE:
```
from typing import Annotated

from pydantic_graph import BaseNode, End, Graph, GraphRunContext, Edge

ask_agent = Agent('openai:gpt-4o', output_type=str, instrument=True)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-4o."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'openai:gpt-4o',
    output_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
        if result.output.correct:
            return End(result.output.comment)
        else:
            return Reprimand(result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f'Comment: {self.comment}')
        ctx.state.question = None
        return Ask()


question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)
```

----------------------------------------

TITLE: Configure Hugging Face API token environment variable
DESCRIPTION: This command sets the `HF_TOKEN` environment variable, which is required for authenticating with Hugging Face Inference Providers. Replace 'hf_token' with your actual Hugging Face access token obtained from your Hugging Face settings.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/huggingface.md#_snippet_1

LANGUAGE: bash
CODE:
```
export HF_TOKEN='hf_token'
```

----------------------------------------

TITLE: Use custom AsyncAzureOpenAI client with pydantic-ai
DESCRIPTION: Demonstrates configuring `pydantic-ai` to use Azure OpenAI by providing an `AsyncAzureOpenAI` client instance. This client requires specific parameters like `azure_endpoint`, `api_version`, and `api_key` for connecting to Azure's service.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_6

LANGUAGE: python
CODE:
```
from openai import AsyncAzureOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncAzureOpenAI(
    azure_endpoint='...',
    api_version='2024-07-01-preview',
    api_key='your-api-key',
)

model = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=client),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Set Google API Key Environment Variable
DESCRIPTION: This command sets the `GOOGLE_API_KEY` environment variable, which is used by `GoogleModel` to authenticate with the Generative Language API. Replace 'your-api-key' with your actual Google API key.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_1

LANGUAGE: bash
CODE:
```
export GOOGLE_API_KEY=your-api-key
```

----------------------------------------

TITLE: Pydantic AI Output Definition and Agent Execution API
DESCRIPTION: Comprehensive documentation for Pydantic AI's core functionalities related to defining structured outputs and executing agents, including methods for streaming results and handling validation. This covers `StructuredDict` for custom schemas, the `@agent.output_validator` decorator for custom validation, and various agent execution methods like `run_stream()`, `run()`, and `iter()` for different streaming and completion behaviors.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_11

LANGUAGE: APIDOC
CODE:
```
pydantic_ai.output.StructuredDict(schema: dict, name: str = None, description: str = None) -> Type[dict[str, Any]]
  - Purpose: Generates a `dict[str, Any]` subclass with an attached JSON schema that Pydantic AI will pass to the model.
  - Parameters:
    - `schema` (dict): The JSON schema definition (e.g., `{"type": "object", "properties": {...}}`).
    - `name` (str, optional): A descriptive name for the structured output, providing additional context to the model.
    - `description` (str, optional): A detailed description for the structured output, aiding the model's understanding.
  - Returns: A `dict[str, Any]` subclass that can be used as an `output_type` for an agent.
  - Notes: Pydantic AI does not perform validation of the received JSON object against this schema; it's up to the model to correctly interpret it. Your code should defensively read from the resulting dictionary.

pydantic_ai.Agent.output_validator(func: Callable[[RunContext, OutputType], Awaitable[OutputType]]) -> Callable
  - Purpose: A decorator to register an asynchronous validation function for an agent's output.
  - Parameters:
    - `func` (Callable): An async function that takes `RunContext` and the agent's `output` as arguments. It should return the validated output or raise `ModelRetry` to prompt the model to try again.
  - Usage: Ideal for validation logic that requires I/O (e.g., database checks) or is asynchronous, complementing Pydantic's built-in validators.

pydantic_ai.Agent.run_stream(...) -> pydantic_ai.result.StreamedRunResult
  - Purpose: Executes the agent and streams the results, performing 'partial validation' of structured responses as they arrive.
  - Behavior: Streams just enough of the response to determine if it's a tool call or an output. Once the first output matching the `output_type` is identified, it is considered the final output, and the agent graph stops running, meaning no further tool calls will be executed.
  - Returns: A `StreamedRunResult` object, allowing access to the streamed output.

pydantic_ai.agent.AbstractAgent.run(...) -> Any
  - Purpose: Runs the agent to completion, processing all model responses and executing any tool calls.
  - Usage: Can be used with an `event_stream_handler` to stream all events (model responses, tool calls, etc.) from the agent's execution, providing full visibility into the agent's workflow.

pydantic_ai.agent.AbstractAgent.iter(...) -> AsyncIterator[Any]
  - Purpose: Provides an asynchronous iterator over all events and outputs generated during the agent's execution.
  - Usage: Similar to `run()` with an `event_stream_handler`, it ensures the agent graph runs to completion and allows for processing all intermediate events and the final output in a streaming fashion.
```

----------------------------------------

TITLE: Configure OpenAI model for thinking parts
DESCRIPTION: This snippet demonstrates how to enable thinking parts for OpenAI models using `OpenAIResponsesModel`. It requires setting `openai_reasoning_effort` and `openai_reasoning_summary` fields within the `OpenAIResponsesModelSettings` to control the level of detail for the reasoning output.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/thinking.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('o3-mini')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
    openai_reasoning_summary='detailed',
)
agent = Agent(model, model_settings=settings)
...
```

----------------------------------------

TITLE: Customize AnthropicProvider with httpx.AsyncClient
DESCRIPTION: This Python snippet demonstrates how to inject a custom `httpx.AsyncClient` into the `AnthropicProvider` when initializing `AnthropicModel`. This allows for fine-grained control over HTTP request parameters like timeouts, proxies, or other client-specific configurations, enhancing flexibility for network interactions with the Anthropic API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/anthropic.md#_snippet_5

LANGUAGE: python
CODE:
```
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

custom_http_client = AsyncClient(timeout=30)
model = AnthropicModel(
    'claude-3-5-sonnet-latest',
    provider=AnthropicProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Configure OpenAIModel with custom base URL and API key
DESCRIPTION: This snippet demonstrates how to connect Pydantic AI's OpenAIModel to any OpenAI-compatible API endpoint by explicitly specifying the base_url and api_key via an OpenAIProvider instance. This is useful for self-hosted or alternative OpenAI-compatible services.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_9

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>.com', api_key='your-api-key'
    ),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Pydantic-AI Custom Toolset API
DESCRIPTION: This API documentation describes how to define a fully custom toolset in Pydantic-AI by subclassing `AbstractToolset`. It outlines the required methods for listing and calling tools, as well as optional asynchronous context manager methods for resource management.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_9

LANGUAGE: APIDOC
CODE:
```
AbstractToolset:
  - get_tools() -> List[Tool]:
      Abstract method. Subclasses must implement this to return a list of available tools.
  - call_tool(tool_name: str, **kwargs) -> Any:
      Abstract method. Subclasses must implement this to handle the invocation of a specific tool by its name and arguments.
  - __aenter__() -> Self:
      Optional asynchronous context manager entry point. If implemented, this method will be called when the agent using the toolset is entered via an `async with` statement. Useful for setting up network connections or sessions.
  - __aexit__(exc_type, exc_val, exc_tb) -> None:
      Optional asynchronous context manager exit point. If implemented, this method will be called when the agent using the toolset exits an `async with` block. Useful for cleaning up resources.
```

----------------------------------------

TITLE: Launch Pydantic AI CLI from Agent Instance (Asynchronous)
DESCRIPTION: Launch an interactive `clai` CLI session asynchronously from an `Agent` instance using the `to_cli()` method. This method is suitable for integration into asynchronous Python applications.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_9

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='You always respond in Italian.')

async def main():
    await agent.to_cli()
```

----------------------------------------

TITLE: Pydantic-AI Type-Safe Agent Output Validation
DESCRIPTION: Illustrates how to define a type-safe `Agent` with a Pydantic `BaseModel` for output validation, ensuring structured and predictable responses from the agent.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/CLAUDE.md#_snippet_2

LANGUAGE: Python
CODE:
```
class OutputModel(BaseModel):
    result: str
    confidence: float

agent: Agent[MyDeps, OutputModel] = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
    output_type=OutputModel
)
```

----------------------------------------

TITLE: Run Pydantic AI CLI with Custom Agent
DESCRIPTION: Execute the `clai` CLI using a custom agent defined in a Python module. The `--agent` flag takes a `module:variable` path, allowing the CLI to load and use the specified `Agent` instance for interactions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_7

LANGUAGE: bash
CODE:
```
uvx clai --agent custom_agent:agent "What's the weather today?"
```

----------------------------------------

TITLE: Pydantic Evals Core API Reference
DESCRIPTION: This section provides an overview of key classes and methods within the `pydantic-evals` library, essential for defining evaluation workflows. It covers how to structure test cases, implement custom evaluation logic, and manage the evaluation process.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_4

LANGUAGE: APIDOC
CODE:
```
pydantic_evals.Case:
  Represents a single test case for evaluation.
  __init__(name: str, inputs: Any, expected_output: Any, metadata: dict)
    - name: A unique identifier for the case.
    - inputs: The input(s) to be passed to the function under evaluation.
    - expected_output: The expected output for the given inputs.
    - metadata: Optional dictionary for additional case-specific data.

pydantic_evals.evaluators.Evaluator:
  Abstract base class for custom evaluation logic.
  evaluate(self, ctx: EvaluatorContext[InputType, OutputType]) -> float
    - ctx: An EvaluatorContext object containing the function's output, expected output, and other context.
    - Returns: A float score between 0.0 and 1.0, indicating the evaluation result.

pydantic_evals.evaluators.EvaluatorContext:
  Context object passed to an Evaluator's `evaluate` method.
  Attributes:
    - output: The actual output from the function being evaluated.
    - expected_output: The expected output defined in the Case.
    - inputs: The inputs defined in the Case.
    - case: The full Case object being evaluated.

pydantic_evals.evaluators.IsInstance:
  A built-in evaluator that checks if the output is an instance of a specified type.
  __init__(type_name: str)
    - type_name: The name of the type to check against (e.g., 'str', 'int').

pydantic_evals.Dataset:
  Manages a collection of test cases and evaluators for a comprehensive evaluation.
  __init__(cases: list[Case], evaluators: list[Evaluator])
    - cases: A list of Case objects to be evaluated.
    - evaluators: A list of Evaluator instances (custom or built-in) to apply to each case.
  evaluate_sync(self, func: Callable) -> EvaluationReport
    - func: The synchronous function to be evaluated against all cases in the dataset.
    - Returns: An EvaluationReport object containing the results.

pydantic_evals.reporting.EvaluationReport:
  Stores and presents the results of an evaluation run.
  print(self, include_input: bool = False, include_output: bool = False, include_durations: bool = False)
    - Prints a formatted summary of the evaluation results to the console.
    - include_input: If True, includes the input for each case in the report.
    - include_output: If True, includes the actual output for each case in the report.
    - include_durations: If True, includes the execution duration for each case.
```

----------------------------------------

TITLE: Pydantic-Graph Core Components API Reference
DESCRIPTION: This section details the core components of the `pydantic-graph` library, including `GraphRunContext` for managing graph state and `End` for signaling graph termination. Both are generic types, allowing for flexible state and return value definitions within graph runs.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
GraphRunContext:
  Description: The context for the graph run, similar to Pydantic AI's RunContext. This holds the state of the graph and dependencies and is passed to nodes when they're run.
  Generics:
    StateT: The state type of the graph it's used in.

End:
  Description: Return value to indicate the graph run should end.
  Generics:
    RunEndT: The graph return type of the graph it's used in.
```

----------------------------------------

TITLE: Modal Application Secrets Configuration
DESCRIPTION: Explains how to configure essential API keys and tokens (Slack, Logfire, OpenAI) as custom secrets within the Modal platform. These secrets are crucial for the application's secure operation and access to external services.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
Modal Secrets Configuration:
  - Name: slack
    Key: SLACK_API_KEY
    Value: Your Slack Access Token (obtained during Slack app installation)
  - Name: logfire
    Key: LOGFIRE_TOKEN
    Value: Your Logfire Write Token (obtained from Logfire project settings)
  - Name: openai
    Key: OPENAI_API_KEY
    Value: Your OpenAI API Key (obtained from OpenAI platform settings)
```

----------------------------------------

TITLE: Question Graph State Diagram
DESCRIPTION: A Mermaid state diagram illustrating the workflow of the question graph. It visualizes the progression through different states: 'Ask', 'Answer', 'Evaluate', 'Congratulate', and 'Castigate'. The diagram clearly shows the transitions between these states, including the conditions for moving from 'Evaluate' to either 'Congratulate' (success) or 'Castigate' (try again), and the loop back to 'Ask' if 'Castigate' occurs.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/question-graph.md#_snippet_1

LANGUAGE: mermaid
CODE:
```
--- 
title: question_graph
---
stateDiagram-v2
  [*] --> Ask
  Ask --> Answer: ask the question
  Answer --> Evaluate: answer the question
  Evaluate --> Congratulate
  Evaluate --> Castigate
  Congratulate --> [*]: success
  Castigate --> Ask: try again
```

----------------------------------------

TITLE: Demonstrating Type Mismatches in Pydantic AI Agents
DESCRIPTION: This snippet illustrates how Pydantic AI agents leverage type hints for static analysis. It shows common type errors when `deps_type` and `output_type` are mismatched with function signatures or usage, and how `mypy` identifies these issues, ensuring robust application development.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_17

LANGUAGE: Python
CODE:
```
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class User:
    name: str


agent = Agent(
    'test',
    deps_type=User,  # (1)!
    output_type=bool,
)


@agent.system_prompt
def add_user_name(ctx: RunContext[str]) -> str:  # (2)!
    return f"The user's name is {ctx.deps}."


def foobar(x: bytes) -> None:
    pass


result = agent.run_sync('Does their name start with "A"?', deps=User('Anne'))
foobar(result.output)  # (3)!
```

LANGUAGE: Bash
CODE:
```
➤ uv run mypy type_mistakes.py
type_mistakes.py:18: error: Argument 1 to "system_prompt" of "Agent" has incompatible type "Callable[[RunContext[str]], str]"; expected "Callable[[RunContext[User]], str]"  [arg-type]
type_mistakes.py:28: error: Argument 1 to "foobar" has incompatible type "bool"; expected "bytes"  [arg-type]
Found 2 errors in 1 file (checked 1 source file)
```

----------------------------------------

TITLE: Instrument Pydantic AI Agent with Logfire
DESCRIPTION: This Python snippet demonstrates how to integrate Logfire for enhanced observability with Pydantic AI. By configuring Logfire and instrumenting `pydantic_ai`, developers can gain detailed insights into agent operations, tool calls, and the overall flow of AI interactions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_3

LANGUAGE: python
CODE:
```
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

----------------------------------------

TITLE: Pydantic AI Agent Execution and Event Streaming
DESCRIPTION: This section details the `agent.run()` and `agent.run_stream()` methods for executing Pydantic AI agents, emphasizing their event streaming capabilities. It describes how to use an `event_stream_handler` to capture `PartStartEvent` and `PartDeltaEvent` for real-time insights into the agent's processing.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_4

LANGUAGE: APIDOC
CODE:
```
AbstractAgent.run(user_prompt: str, event_stream_handler: Optional[Callable] = None)
  - Purpose: Runs the agent graph to completion, optionally streaming all events.
  - Parameters:
    - user_prompt (str): The input prompt for the agent.
    - event_stream_handler (Optional[Callable]): A handler function to process streaming events.
  - Behavior with event_stream_handler: Requires manual reassembly of streamed text from PartStartEvent and PartDeltaEvent.
  - Returns: The final output of the agent run.

AbstractAgent.run_stream()
  - Purpose: Streams the agent's response, allowing for partial results before completion.

Events for Streaming:
  - PartStartEvent: Indicates the start of a new part in the streamed response.
  - PartDeltaEvent: Provides incremental updates for a part in the streamed response.
```

----------------------------------------

TITLE: Unit Testing Tool Calls with FunctionModel
DESCRIPTION: This Python code demonstrates how to unit test an AI agent's tool calls using Pydantic-AI's `FunctionModel`. It defines a custom `call_weather_forecast` function that intercepts and simulates the LLM's behavior, extracting a date from the prompt and returning a `ToolCallPart` or `TextPart`. The `FunctionModel` is used to override the agent's default model, allowing for controlled testing of tool interactions and ensuring the `WeatherService.get_forecast` is properly exercised.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#_snippet_4

LANGUAGE: python
CODE:
```
import re

import pytest

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


def call_weather_forecast(  # (1)!
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # first call, call the weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}  # (2)!
        return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
    else:
        # second call, return the forecast
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])


async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):  # (3)!
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == 'The forecast is: Rainy with a chance of sun'
```

----------------------------------------

TITLE: Instrument pydantic-ai direct API calls with Logfire
DESCRIPTION: This snippet demonstrates how to enable OpenTelemetry/Logfire instrumentation for `pydantic-ai` direct API calls. By configuring Logfire and instrumenting `pydantic-ai`, all subsequent model requests will automatically generate traces and logs, aiding in observability and debugging.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/direct.md#_snippet_2

LANGUAGE: Python
CODE:
```
import logfire

from pydantic_ai.direct import model_request_sync
from pydantic_ai.messages import ModelRequest

logfire.configure()
logfire.instrument_pydantic_ai()

# Make a synchronous request to the model
model_response = model_request_sync(
    'anthropic:claude-3-5-haiku-latest',
    [ModelRequest.user_text_prompt('What is the capital of France?')],
)

print(model_response.parts[0].content)
# The capital of France is Paris.
```

----------------------------------------

TITLE: Control Flow Diagram for Flight and Seat Booking Agents
DESCRIPTION: This Mermaid graph visualizes the control flow of the application. It illustrates the interaction between the user, the `flight_search_agent`, and the `seat_preference_agent`, showing how the application progresses from asking for flight details to asking for seat preferences, and the iterative nature of agent interactions within their respective subgraphs.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#_snippet_6

LANGUAGE: mermaid
CODE:
```
graph TB
  START --> ask_user_flight["ask user for flight"]

  subgraph find_flight
    flight_search_agent --> ask_user_flight
    ask_user_flight --> flight_search_agent
  end

  flight_search_agent --> ask_user_seat["ask user for seat"]
  flight_search_agent --> END

  subgraph find_seat
    seat_preference_agent --> ask_user_seat
    ask_user_seat --> seat_preference_agent
  end

  seat_preference_agent --> END
```

----------------------------------------

TITLE: Expose Pydantic AI Agent as A2A Server
DESCRIPTION: This Python code snippet demonstrates how to instantiate a Pydantic AI Agent and convert it into an A2A (Agent2Agent) server using the `to_a2a()` convenience method. This allows the agent to communicate and receive requests following the A2A protocol, making it interoperable with other A2A-compliant agents.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/a2a.md#_snippet_0

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='Be fun!')
app = agent.to_a2a()
```

----------------------------------------

TITLE: Instrument a Specific Pydantic AI Model Instance
DESCRIPTION: Python code showing how to apply `InstrumentationSettings` to a particular `Pydantic AI Model` instance. This allows for granular control over observability settings for individual models, enabling different tracing or logging configurations for distinct AI components within an application.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_12

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel

settings = InstrumentationSettings()
model = InstrumentedModel('gpt-4o', settings)
agent = Agent(model)
```

----------------------------------------

TITLE: Pydantic AI Agent Configuration and Usage Limits API
DESCRIPTION: This section details key API components for configuring Pydantic AI agents, managing usage limits, and handling model retries. It covers the `UsageLimits` class for controlling token and request consumption, the `Agent` class methods for running agents and defining tools, and relevant exceptions like `ModelRetry` and `UsageLimitExceeded`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_13

LANGUAGE: APIDOC
CODE:
```
pydantic_ai.usage.UsageLimits:
  A structure to help limit usage (tokens and/or requests) on model runs.
  Parameters:
    response_tokens_limit: int, optional
      Maximum number of response tokens allowed.
    request_limit: int, optional
      Maximum number of requests allowed.

pydantic_ai.Agent:
  __init__(model_name: str, retries: int = 0, output_type: Type = None, system_prompt: str = None, ...)
    Initializes an Agent instance.
    Parameters:
      model_name: str
        The name of the model to use (e.g., 'anthropic:claude-3-5-sonnet-latest').
      retries: int, optional
        Number of retries for model calls (default: 0).
      output_type: Type, optional
        The expected output type for the agent.
      system_prompt: str, optional
        A system prompt to guide the agent's behavior.

  run_sync(prompt: str, usage_limits: UsageLimits = None, ...) -> AgentRunResult
    Synchronously runs the agent with a given prompt.
    Parameters:
      prompt: str
        The input prompt for the agent.
      usage_limits: UsageLimits, optional
        An instance of UsageLimits to apply restrictions on tokens and requests.
    Returns:
      AgentRunResult: The result of the agent run, including output and usage.

  tool_plain(retries: int = 0) -> Callable
    Decorator for defining plain Python functions as agent tools.
    Parameters:
      retries: int, optional
        Number of retries for this specific tool call (default: 0).

pydantic_ai.exceptions.UsageLimitExceeded:
  Exception raised when a configured usage limit (e.g., response_tokens_limit, request_limit) is exceeded during an agent run.

pydantic_ai.ModelRetry:
  Exception that can be raised within a tool to signal the model to retry the current step.
```

----------------------------------------

TITLE: Customize Google Model Safety Settings
DESCRIPTION: Illustrates how to apply custom safety settings to a Google model by defining `google_safety_settings` with a list of dictionaries, each specifying a `HarmCategory` and its corresponding `HarmBlockThreshold`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_9

LANGUAGE: python
CODE:
```
from google.genai.types import HarmBlockThreshold, HarmCategory

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(
    google_safety_settings=[
        {
            'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)
model = GoogleModel('gemini-2.0-flash')
agent = Agent(model, model_settings=model_settings)
...
```

----------------------------------------

TITLE: Pydantic Evals Dataset Generation and Management API
DESCRIPTION: This section details the core API for generating and managing test datasets within Pydantic Evals. It covers the `generate_dataset` asynchronous function for creating datasets using LLMs based on defined Pydantic schemas, and the `Dataset` class methods for saving these datasets to various file formats.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_12

LANGUAGE: APIDOC
CODE:
```
pydantic_evals.generation.generate_dataset:
  async def generate_dataset(
      dataset_type: Dataset[InputSchema, OutputSchema, MetadataSchema],
      n_examples: int,
      extra_instructions: str
  ) -> Dataset:
    - description: Generates a test dataset using an LLM based on provided Pydantic schemas.
    - parameters:
      - name: dataset_type
        type: Dataset[InputSchema, OutputSchema, MetadataSchema]
        description: A Pydantic Evals Dataset type hint specifying the Pydantic models for inputs, expected outputs, and optional metadata.
      - name: n_examples
        type: int
        description: The number of examples (test cases) to generate for the dataset.
      - name: extra_instructions
        type: str
        description: Additional instructions or context to guide the LLM during dataset generation.
    - returns:
      - type: Dataset
        description: An instance of the generated Dataset containing the test cases.

pydantic_evals.Dataset:
  class Dataset[InputSchema, OutputSchema, MetadataSchema]:
    - description: Represents a collection of test cases, each with inputs, expected outputs, and optional metadata.
    - methods:
      - name: to_file
        signature: to_file(output_path: Path) -> None
        description: Saves the dataset to the specified file path. Automatically generates a corresponding JSON schema file alongside the dataset for validation and auto-completion.
        parameters:
          - name: output_path
            type: Path
            description: The file path where the dataset should be saved. Supports .yaml and .json extensions.
```

----------------------------------------

TITLE: Run Background Slack Member Processing on Modal with Tracing
DESCRIPTION: This snippet defines a Modal function to wrap `process_slack_member`, enabling it to run as a background task. It ensures that the Logfire context is attached to maintain distributed tracing across the application flow.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_11

LANGUAGE: python
CODE:
```
# This function wraps process_slack_member to run it in the background on Modal.
# It ensures Logfire context is propagated for distributed tracing.
import modal
# Assuming process_slack_member is in app.py
# from .app import process_slack_member

app = modal.App()

@app.function()
def background_process_slack_member(member_data, logfire_context=None):
    # Attach Logfire context if provided for distributed tracing
    if logfire_context:
        # logfire.set_context(logfire_context) # Example of how context might be used
        pass
    # process_slack_member(member_data)
    pass
```

----------------------------------------

TITLE: Print Pydantic AI Evaluation Report
DESCRIPTION: This Python snippet demonstrates how to print a comprehensive evaluation report using the `report.print` method. It allows for customization of the output by including or excluding specific details such as input values, processed outputs, and performance durations. Setting `include_durations` to `False` ensures consistent output across multiple runs by omitting time-based metrics.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_9

LANGUAGE: Python
CODE:
```
report.print(include_input=True, include_output=True, include_durations=False)
```

----------------------------------------

TITLE: Set Cohere API Key Environment Variable
DESCRIPTION: This bash command sets the `CO_API_KEY` environment variable, which is used by `pydantic-ai` to authenticate with the Cohere API. It is a common and secure way to manage API keys without hardcoding them directly into your application code.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/cohere.md#_snippet_1

LANGUAGE: bash
CODE:
```
export CO_API_KEY='your-api-key'
```

----------------------------------------

TITLE: Set Heroku AI Environment Variables
DESCRIPTION: Explains how to set the `HEROKU_INFERENCE_KEY` and `HEROKU_INFERENCE_URL` environment variables. These variables provide a flexible way to configure authentication and the base URL for Heroku AI outside of the application code.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/openai.md#_snippet_31

LANGUAGE: Bash
CODE:
```
export HEROKU_INFERENCE_KEY='your-heroku-inference-key'
export HEROKU_INFERENCE_URL='https://us.inference.heroku.com'
```

----------------------------------------

TITLE: Provide Custom Bedrock Provider with AWS Credentials
DESCRIPTION: Shows how to explicitly pass AWS credentials (access key, secret key, region) to a `BedrockProvider` instance, which is then used to initialize `BedrockConverseModel`. This method offers direct control over authentication.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/bedrock.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Using AWS credentials directly
model = BedrockConverseModel(
    'anthropic.claude-3-sonnet-20240229-v1:0',
    provider=BedrockProvider(
        region_name='us-east-1',
        aws_access_key_id='your-access-key',
        aws_secret_access_key='your-secret-key',
    ),
)
agent = Agent(model)
...
```

----------------------------------------

TITLE: Configure Groq model for thinking parts format
DESCRIPTION: This snippet illustrates how to enable and specify the format for thinking parts when using Groq models. The `groq_reasoning_format` field in `GroqModelSettings` can be set to 'raw' (included in text with tags), 'hidden' (not in text), or 'parsed' (as a separate `ThinkingPart` object).

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/thinking.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

model = GroqModel('qwen-qwq-32b')
settings = GroqModelSettings(groq_reasoning_format='parsed')
agent = Agent(model, model_settings=settings)
...
```

----------------------------------------

TITLE: Pydantic AI MCP Sampling Configuration API
DESCRIPTION: API details for configuring MCP sampling behavior in Pydantic AI, including setting the sampling model on the server or agent, and controlling whether sampling is allowed. These components enable fine-grained control over how LLM calls are proxied via the MCP client.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_14

LANGUAGE: APIDOC
CODE:
```
pydantic_ai.mcp.MCPServerStdio.sampling_model
  - Description: Property to set or get the sampling model for an MCP server instance. This model will be used for LLM calls proxied through the client.
  - Type: str (model identifier) or None
  - Usage: Can be set via the constructor keyword argument or directly on the property after instantiation.

pydantic_ai.Agent.set_mcp_sampling_model(model: Optional[str] = None)
  - Description: Sets the specified model (or the agent's default model if not provided) as the sampling model on all MCP servers registered with the agent. This simplifies configuring multiple servers.
  - Parameters:
    - model (Optional[str]): The model name to use for sampling. If None, the agent's primary model is used.
  - Returns: None

pydantic_ai.mcp.MCPServerStdio(..., allow_sampling: bool = True, ...)
  - Description: Constructor parameter for `MCPServerStdio` to control whether sampling is permitted for the server instance.
  - Parameters:
    - allow_sampling (bool): If set to `False`, the server will explicitly not be able to proxy LLM calls through the client. Defaults to `True`.
```

----------------------------------------

TITLE: Set Mistral API Key environment variable
DESCRIPTION: Demonstrates how to set the `MISTRAL_API_KEY` environment variable in a bash shell. This environment variable is automatically picked up by `pydantic-ai` for authenticating with the Mistral API.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/mistral.md#_snippet_1

LANGUAGE: bash
CODE:
```
export MISTRAL_API_KEY='your-api-key'
```

----------------------------------------

TITLE: Specify AI Model for Pydantic AI CLI
DESCRIPTION: Specify a particular AI model to use with the `clai` CLI by using the `--model` flag. The format is `provider:model_name`, allowing selection of models like Anthropic's Claude Sonnet.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_5

LANGUAGE: bash
CODE:
```
uvx clai --model anthropic:claude-sonnet-4-0
```

----------------------------------------

TITLE: Slack App Required API Scopes
DESCRIPTION: Details the necessary API scopes for the Slack application to function correctly, allowing it to read user information for lead qualification. These scopes are requested during the Slack app creation process.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Slack App Scopes:
  - users.read: Allows the app to read basic user information.
  - users.read.email: Allows the app to read user email addresses.
  - users.profile.read: Allows the app to read user profile information.
```

----------------------------------------

TITLE: Inspect Agent Conversation Messages (Python)
DESCRIPTION: This Python snippet shows how to retrieve and print the complete message history of an agent's run. It illustrates the sequence of `ModelRequest`, `ModelResponse`, `ToolCallPart`, and `ToolReturnPart` objects, providing a detailed trace of the agent's interactions with the language model and its tool executions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_1

LANGUAGE: python
CODE:
```
from dice_game import dice_result

print(dice_result.all_messages())
```

----------------------------------------

TITLE: Configure Logfire to send traces to an alternative OpenTelemetry backend
DESCRIPTION: This Python code illustrates how to redirect `pydantic-ai`'s OpenTelemetry traces from Logfire's default backend to an alternative OTLP endpoint, such as `otel-tui`. It involves setting the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable and configuring Logfire with `send_to_logfire=False` to prevent duplicate data transmission, enabling integration with any OTel-compatible system.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_7

LANGUAGE: python
CODE:
```
import os

import logfire

from pydantic_ai import Agent

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.output)

```

----------------------------------------

TITLE: Specify Dependency Version Constraints with Inline Metadata
DESCRIPTION: This snippet demonstrates how to use PEP 723 inline script metadata to specify version constraints for Python package dependencies, such as pinning `rich` to a version less than 13. This is useful for ensuring compatibility or specific behavior.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/run-python.md#_snippet_3

LANGUAGE: python
CODE:
```
# /// script
# dependencies = ["rich<13"]
# ///
```

----------------------------------------

TITLE: Apply Google-Specific Model Settings in Pydantic AI
DESCRIPTION: Shows how to use a subclass of `ModelSettings`, specifically `GoogleModelSettings`, to configure model-specific parameters like `gemini_safety_settings`. It also demonstrates handling `UnexpectedModelBehavior` when safety thresholds are exceeded by the model's response.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_15

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.google import GoogleModelSettings

agent = Agent('google-gla:gemini-1.5-flash')

try:
    result = agent.run_sync(
        'Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:',
        model_settings=GoogleModelSettings(
            temperature=0.0,  # general model settings can also be specified
            gemini_safety_settings=[
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
            ],
        ),
    )
except UnexpectedModelBehavior as e:
    print(e)
```

----------------------------------------

TITLE: Set OpenAI API Key Environment Variable
DESCRIPTION: Before using the `clai` CLI with OpenAI, set the `OPENAI_API_KEY` environment variable to your personal API key. This authenticates your requests to the OpenAI service.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_0

LANGUAGE: bash
CODE:
```
export OPENAI_API_KEY='your-api-key-here'
```

----------------------------------------

TITLE: Mermaid Diagram: Email Feedback Graph Structure
DESCRIPTION: This Mermaid diagram visualizes the state transitions within the email feedback graph. It shows the flow from an initial state to email writing, then to feedback, with a loop back to writing based on feedback, and a final end state.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_11

LANGUAGE: Mermaid
CODE:
```
---
title: feedback_graph
---
stateDiagram-v2
  [*] --> WriteEmail
  WriteEmail --> Feedback
  Feedback --> WriteEmail
  Feedback --> [*]
```

----------------------------------------

TITLE: Manually controlling Pydantic Graph iteration with `GraphRun.next()`
DESCRIPTION: This snippet shows how to manually drive graph iteration using `GraphRun.next(node)`, allowing selective execution or skipping of nodes. It illustrates breaking the loop early based on state, which results in `run.result` being `None` if the graph doesn't complete. The `FullStatePersistence` is used to track the history of executed steps.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_14

LANGUAGE: python
CODE:
```
from pydantic_graph import End, FullStatePersistence
from count_down import CountDown, CountDownState, count_down_graph


async def main():
    state = CountDownState(counter=5)
    persistence = FullStatePersistence()
    async with count_down_graph.iter(
        CountDown(), state=state, persistence=persistence
    ) as run:
        node = run.next_node
        while not isinstance(node, End):
            print('Node:', node)
            if state.counter == 2:
                break
            node = await run.next(node)

        print(run.result)

        for step in persistence.history:
            print('History Step:', step.state, step.state)
```

----------------------------------------

TITLE: Integrate Retrying HTTP Client with OpenAI API in Python
DESCRIPTION: Shows how to use a pre-configured `httpx` client, equipped with retry logic (e.g., from `smart_retry_example.py`), when initializing the `OpenAIProvider` for `pydantic-ai`. This ensures all API calls to OpenAI benefit from the defined retry mechanisms.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_9

LANGUAGE: Python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

----------------------------------------

TITLE: MCP Sampling Data Flow Diagram
DESCRIPTION: This Mermaid diagram illustrates the interaction and data flow between an LLM, MCP Client, and MCP Server during an MCP sampling operation. It visually represents how the client proxies LLM calls on behalf of the server, including tool calls and sampling responses.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_10

LANGUAGE: mermaid
CODE:
```
sequenceDiagram
    participant LLM
    participant MCP_Client as MCP client
    participant MCP_Server as MCP server

    MCP_Client->>LLM: LLM call
    LLM->>MCP_Client: LLM tool call response

    MCP_Client->>MCP_Server: tool call
    MCP_Server->>MCP_Client: sampling "create message"

    MCP_Client->>LLM: LLM call
    LLM->>MCP_Client: LLM text response

    MCP_Client->>MCP_Server: sampling response
    MCP_Server->>MCP_Client: tool call response
```

----------------------------------------

TITLE: Define Human-in-the-Loop AI Q&A Graph with Pydantic-Graph
DESCRIPTION: This Python code defines a `pydantic-graph` for a human-in-the-loop AI question-and-answer system. It includes nodes for an AI to ask questions, a user to provide answers, and an AI to evaluate those answers, demonstrating state persistence and agent interaction. The graph allows for interruption and resumption, enabling interactive user input.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_16

LANGUAGE: python
CODE:
```
from __future__ import annotations as _annotations

from typing import Annotated
from pydantic_graph import Edge
from dataclasses import dataclass, field
from pydantic import BaseModel
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage

ask_agent = Agent('openai:gpt-4o', output_type=str, instrument=True)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-4o."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'openai:gpt-4o',
    output_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
        if result.output.correct:
            return End(result.output.comment)
        else:
            return Reprimand(result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f'Comment: {self.comment}')
        ctx.state.question = None
        return Ask()


question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)
```

----------------------------------------

TITLE: Define Custom Pydantic AI Agent
DESCRIPTION: Define a custom `Agent` instance in Python, specifying the AI model and initial instructions. This agent can then be used with the `clai` CLI to customize AI behavior.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/cli.md#_snippet_6

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='You always respond in Italian.')
```

----------------------------------------

TITLE: Set Anthropic API Key environment variable
DESCRIPTION: This command sets the `ANTHROPIC_API_KEY` environment variable, which `pydantic-ai` uses to authenticate with the Anthropic API. It is the recommended way to manage your API key securely. Remember to replace 'your-api-key' with your actual API key obtained from the Anthropic console.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/anthropic.md#_snippet_1

LANGUAGE: bash
CODE:
```
export ANTHROPIC_API_KEY='your-api-key'
```

----------------------------------------

TITLE: Schedule Daily Summary Function on Modal via Cron
DESCRIPTION: This snippet illustrates how to define a scheduled function on Modal using the `@app.function()` decorator with a `schedule` argument. It configures the `send_daily_summary` function to run automatically every day at 8 AM UTC.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/examples/slack-lead-qualifier.md#_snippet_10

LANGUAGE: python
CODE:
```
# This function schedules the daily summary task to run every day at 8 AM UTC.
import modal
from modal import Daily
# Assuming send_daily_summary is in app.py
# from .app import send_daily_summary

app = modal.App()

@app.function(schedule=Daily(8, 0))
def scheduled_daily_summary_task():
    # send_daily_summary()
    pass
```

----------------------------------------

TITLE: Monitor HTTP requests with Logfire and HTTPX instrumentation
DESCRIPTION: This Python snippet demonstrates how to enable comprehensive monitoring of HTTP requests made by `pydantic-ai` agents using Logfire's HTTPX instrumentation. By calling `logfire.instrument_httpx(capture_all=True)`, it ensures that both request and response headers and bodies are captured, providing deep visibility into interactions with model providers like OpenAI.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_4

LANGUAGE: python
CODE:
```
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)
agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.output)

```

----------------------------------------

TITLE: Define Pydantic-AI Agent and Weather Tool
DESCRIPTION: This Python code defines a `pydantic-ai` agent (`weather_agent`) with a `weather_forecast` tool. The tool retrieves historic or future weather data based on the provided date and location, interacting with external services like `WeatherService` and `DatabaseConn`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#_snippet_0

LANGUAGE: python
CODE:
```
import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext

from fake_database import DatabaseConn  # (1)!
from weather_service import WeatherService  # (2)!

weather_agent = Agent(
    'openai:gpt-4o',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():  # (3)!
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)


async def run_weather_forecast(  # (4)!
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.output)

        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )
```

----------------------------------------

TITLE: Combine Multiple Function Toolsets for AI Agent
DESCRIPTION: This snippet demonstrates how to use `CombinedToolset` in pydantic-ai to merge multiple `FunctionToolset` instances into a single, unified toolset. It shows how to initialize `CombinedToolset` with a list of existing toolsets and then integrate this combined toolset with an `Agent`. This allows the agent to access all tools defined across the individual toolsets as if they were part of a single collection.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/toolsets.md#_snippet_2

LANGUAGE: python
CODE:
```
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import CombinedToolset


combined_toolset = CombinedToolset([weather_toolset, datetime_toolset])

test_model = TestModel()
agent = Agent(test_model, toolsets=[combined_toolset])
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions', 'now']
```

----------------------------------------

TITLE: Display Mermaid Graph Image in Jupyter Notebook
DESCRIPTION: This Python code demonstrates how to render and display a `pydantic-graph` as an image within a Jupyter Notebook environment. It utilizes `IPython.display` to show the generated Mermaid diagram image, making graph visualization interactive.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_7

LANGUAGE: Python
CODE:
```
from graph_example import DivisibleBy5, fives_graph
from IPython.display import Image, display

display(Image(fives_graph.mermaid_image(start_node=DivisibleBy5)))
```

----------------------------------------

TITLE: Mermaid Graph for ModelMessage Structure
DESCRIPTION: Visual representation of the `ModelMessage` components and their relationships within the `pydantic_ai.messages` module, showing how different prompt and response parts contribute to the overall message structure. This graph helps in understanding the data flow and composition of messages in the pydantic-ai library.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/api/messages.md#_snippet_0

LANGUAGE: mermaid
CODE:
```
graph RL
    SystemPromptPart(SystemPromptPart) --- ModelRequestPart
    UserPromptPart(UserPromptPart) --- ModelRequestPart
    ToolReturnPart(ToolReturnPart) --- ModelRequestPart
    RetryPromptPart(RetryPromptPart) --- ModelRequestPart
    TextPart(TextPart) --- ModelResponsePart
    ToolCallPart(ToolCallPart) --- ModelResponsePart
    ThinkingPart(ThinkingPart) --- ModelResponsePart
    ModelRequestPart("ModelRequestPart<br>(Union)") --- ModelRequest
    ModelRequest("ModelRequest(parts=list[...])") --- ModelMessage
    ModelResponsePart("ModelResponsePart<br>(Union)") --- ModelResponse
    ModelResponse("ModelResponse(parts=list[...])") --- ModelMessage("ModelMessage<br>(Union)")
```

----------------------------------------

TITLE: Handle Fallback Model Failures (Python 3.11+)
DESCRIPTION: Demonstrates how to catch `ModelHTTPError` exceptions using the `except*` syntax introduced in Python 3.11. This allows for handling `FallbackExceptionGroup` which contains all individual exceptions encountered when all models in the `FallbackModel` chain fail.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/index.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
try:
    response = agent.run_sync('What is the capital of France?')
except* ModelHTTPError as exc_group:
    for exc in exc_group.exceptions:
        print(exc)
```

----------------------------------------

TITLE: Generate Mermaid Diagram Code for a Pydantic Graph
DESCRIPTION: This Python snippet illustrates how to programmatically generate the Mermaid diagram code for a `pydantic-graph` instance. It imports the previously defined `fives_graph` and uses its `mermaid_code` method to output a string representation suitable for Mermaid rendering.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_5

LANGUAGE: Python
CODE:
```
from graph_example import DivisibleBy5, fives_graph

fives_graph.mermaid_code(start_node=DivisibleBy5)
```

----------------------------------------

TITLE: Configure Anthropic model for thinking
DESCRIPTION: This snippet shows how to enable thinking for Anthropic models. Unlike other providers, Anthropic includes a signature in the thinking part for tamper-proofing. Thinking is enabled by configuring the `anthropic_thinking` field in the `AnthropicModelSettings`, specifying its type and an optional token budget.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/thinking.md#_snippet_1

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-3-7-sonnet-latest')
settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
)
agent = Agent(model, model_settings=settings)
...
```

----------------------------------------

TITLE: Handle Fallback Model Failures (Python <3.11)
DESCRIPTION: Illustrates how to manage `ModelHTTPError` exceptions for Python versions older than 3.11. It uses the `exceptiongroup` backport package and its `catch` context manager to handle `FallbackExceptionGroup` when all models in the `FallbackModel` chain fail.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/index.md#_snippet_4

LANGUAGE: python
CODE:
```
from exceptiongroup import catch

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel


def model_status_error_handler(exc_group: BaseExceptionGroup) -> None:
    for exc in exc_group.exceptions:
        print(exc)


openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
with catch({ModelHTTPError: model_status_error_handler}):
    response = agent.run_sync('What is the capital of France?')
```

----------------------------------------

TITLE: Unit Test Pydantic-AI Agent with TestModel
DESCRIPTION: This Python unit test demonstrates how to use `pydantic-ai`'s `TestModel` to test an agent's functionality without external API calls. It captures messages and asserts the agent's behavior and output, showcasing `TestModel`'s ability to simulate tool calls and generate structured responses.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md#_snippet_1

LANGUAGE: python
CODE:
```
from datetime import timezone
import pytest

from dirty_equals import IsNow, IsStr

from pydantic_ai import models, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    ModelRequest,
)
from pydantic_ai.usage import Usage

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio  # (1)!
models.ALLOW_MODEL_REQUESTS = False  # (2)!


async def test_forecast():
    conn = DatabaseConn()
    user_id = 1
    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):  # (3)!
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)  # (4)!

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'  # (5)!

    assert messages == [  # (6)!
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='Providing a weather forecast at the locations the user provides.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),  # (7)!
                ),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args={
                        'location': 'a',
                        'forecast_date': '2024-01-01',  # (8)!
```

----------------------------------------

TITLE: Override Pydantic AI Agent Dependencies for Testing
DESCRIPTION: This Python test code demonstrates how to override the dependencies of a `Pydantic AI Agent` for testing purposes. It defines `TestMyDeps`, a subclass of `MyDeps`, to mock the `system_prompt_factory` method. The `joke_agent.override` context manager is used to temporarily replace the agent's dependencies with `TestMyDeps` instances, allowing for isolated testing of the `application_code` without external HTTP calls.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/dependencies.md#_snippet_5

LANGUAGE: python
CODE:
```
from joke_app import MyDeps, application_code, joke_agent


class TestMyDeps(MyDeps):  # (1)!
    async def system_prompt_factory(self) -> str:
        return 'test prompt'


async def test_application_code():
    test_deps = TestMyDeps('test_key', None)  # (2)!
    with joke_agent.override(deps=test_deps):  # (3)!
        joke = await application_code('Tell me a joke.')  # (4)!
    assert joke.startswith('Did you hear about the toothpaste scandal?')
```

----------------------------------------

TITLE: Configure Google model for thinking
DESCRIPTION: This snippet demonstrates how to enable thinking for Google models. The `google_thinking_config` field within `GoogleModelSettings` is used to include thoughts, typically by setting `include_thoughts` to `True`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/thinking.md#_snippet_3

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-2.5-pro-preview-03-25')
settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
agent = Agent(model, model_settings=settings)
...
```

----------------------------------------

TITLE: Pydantic AI Agent Graph Iteration
DESCRIPTION: This section describes how to gain deeper control and insight into a Pydantic AI agent's execution flow by iterating over its underlying `pydantic-graph`. It covers the `Agent.iter()` method, the `AgentRun` object returned, and how to manually advance the graph using `next()` until an `End` node is reached.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_5

LANGUAGE: APIDOC
CODE:
```
Agent.iter()
  - Purpose: Returns an AgentRun object to asynchronously iterate over the agent's execution graph or drive it node-by-node.
  - Returns: AgentRun

AgentRun
  - Purpose: An object representing an ongoing agent execution, allowing iteration over graph steps.
  - Methods:
    - next(): Manually advances the agent's execution to the next node in the graph.
  - Termination: Iteration concludes when the graph returns an End node.

pydantic_graph.nodes.End
  - Purpose: Represents the final state or completion of a pydantic-graph execution.
```

----------------------------------------

TITLE: Pydantic AI Agent.override Method
DESCRIPTION: The `override` method of the `pydantic_ai.Agent` class provides a mechanism to temporarily replace the agent's registered dependencies. It is designed to be used as a context manager, ensuring that the original dependencies are restored upon exiting the `with` block. This functionality is crucial for testing, allowing developers to inject mock or test-specific dependency implementations without altering the agent's global state.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/dependencies.md#_snippet_6

LANGUAGE: APIDOC
CODE:
```
pydantic_ai.Agent.override(deps: Any) -> ContextManager
  - Temporarily overrides the agent's dependencies within a context.
  - Parameters:
    - deps: An instance of the dependency type (or a subclass) to be used during the override.
  - Returns: A context manager that restores original dependencies upon exit.
  - Usage:
    with agent_instance.override(deps=test_dependencies):
        # Code that uses the agent with overridden dependencies
        pass
```

----------------------------------------

TITLE: Stream AI Agent Response (Full Text)
DESCRIPTION: This snippet demonstrates how to stream the complete text response from a pydantic-ai agent. It uses the `Agent.run_stream()` method as an asynchronous context manager and iterates over `result.stream_text()` to receive the full response progressively.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_12

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text():
            print(message)
            # The first known
            # The first known use of "hello,"
            # The first known use of "hello, world" was in
            # The first known use of "hello, world" was in a 1974 textbook
            # The first known use of "hello, world" was in a 1974 textbook about the C
            # The first known use of "hello, world" was in a 1974 textbook about the C programming language.
```

----------------------------------------

TITLE: FallbackModel Constructor and Customization
DESCRIPTION: This API documentation describes the `FallbackModel` constructor, focusing on its `fallback_on` parameter. This parameter allows developers to customize which exception types will trigger the `FallbackModel` to attempt the next model in its sequence. By default, the `FallbackModel` only falls back if a `ModelHTTPError` is raised by the current model, but this behavior can be extended to other exception types.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/index.md#_snippet_5

LANGUAGE: APIDOC
CODE:
```
FallbackModel(
    *models: BaseModel,
    fallback_on: Type[Exception] | tuple[Type[Exception], ...] = ModelHTTPError
)
  - models: One or more instances of `BaseModel` (e.g., `OpenAIModel`, `AnthropicModel`) that the `FallbackModel` will attempt to use in the specified order.
  - fallback_on: An exception type or a tuple of exception types. If any of these exceptions are raised by the current model during execution, the `FallbackModel` will automatically attempt to use the next model in the sequence. Defaults to `ModelHTTPError`.
```

----------------------------------------

TITLE: Define Custom JSON Schema with Pydantic AI StructuredDict
DESCRIPTION: Demonstrates how to use `StructuredDict` to define a custom JSON schema for structured output when Pydantic `BaseModel`, dataclass, or `TypedDict` are not suitable (e.g., for dynamically generated or external schemas). Pydantic AI will pass this schema to the model, but will not perform validation itself.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md#_snippet_9

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent, StructuredDict

HumanDict = StructuredDict(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    },
    name="Human",
    description="A human with a name and age",
)

agent = Agent('openai:gpt-4o', output_type=HumanDict)
result = agent.run_sync("Create a person")
# {'name': 'John Doe', 'age': 30}
```

----------------------------------------

TITLE: Instrument Pydantic AI Agent Runs with Logfire
DESCRIPTION: Demonstrates how to integrate Pydantic Logfire with Pydantic AI to automatically trace agent runs. It configures the Logfire SDK and enables instrumentation for Pydantic AI, ensuring that a trace is generated for each agent run, with spans emitted for model calls and tool function executions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_3

LANGUAGE: python
CODE:
```
import logfire

from pydantic_ai import Agent

logfire.configure()  # (1)! Configures the SDK, finding the write token from the .logfire directory or accepting it directly.
logfire.instrument_pydantic_ai()  # (2)! Enables instrumentation of Pydantic AI.

agent = Agent('openai:gpt-4o', instructions='Be concise, reply with one sentence.')
result = agent.run_sync('Where does "hello world" come from?')  # (3)! Generates a trace for each run with spans for model calls and tool execution.
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

----------------------------------------

TITLE: Accessing all messages from a Pydantic-AI run
DESCRIPTION: Demonstrates how to retrieve all messages (requests and responses) from a completed `pydantic-ai` agent run using `result.all_messages()`. The accompanying multi-line string shows the structured `ModelRequest` and `ModelResponse` objects that are returned, including system prompts, user prompts, and the model's text response along with usage statistics.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_2

LANGUAGE: python
CODE:
```
print(result.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=Usage(requests=1, request_tokens=60, response_tokens=12, total_tokens=72),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
]
"""
```

----------------------------------------

TITLE: Configure Pydantic AI Agent to Exclude Binary Content
DESCRIPTION: This snippet demonstrates how to configure a Pydantic AI agent to exclude binary content from its instrumentation events. This is useful for reducing data volume or avoiding sending large binary data to observability platforms. It shows both per-agent configuration and global configuration for all agents.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_13

LANGUAGE: python
CODE:
```
from pydantic_ai.agent import Agent, InstrumentationSettings

instrumentation_settings = InstrumentationSettings(include_binary_content=False)

agent = Agent('gpt-4o', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)
```

----------------------------------------

TITLE: Mermaid State Diagram for Vending Machine (LR Direction)
DESCRIPTION: This Mermaid code defines a state diagram for a vending machine, illustrating its various states and transitions. The diagram is configured to flow from left to right (LR), showing the sequence of operations from coin insertion to product purchase and reset.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_24

LANGUAGE: mermaid
CODE:
```
---
title: vending_machine_graph
---
stateDiagram-v2
  direction LR
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```

----------------------------------------

TITLE: Manage Conversations Across Multiple Pydantic AI Runs
DESCRIPTION: Illustrates how to maintain context in a conversation by chaining multiple `agent.run_sync` calls. It shows how to pass `message_history` from a previous run to ensure the model understands the context of subsequent queries, enabling a continuous dialogue.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/agents.md#_snippet_16

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.output)

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),  # (1)!
)
print(result2.output)
```

----------------------------------------

TITLE: Mermaid State Diagram with Left-to-Right Flow
DESCRIPTION: This Mermaid code defines a state diagram for a vending machine, explicitly setting its flow direction to 'LR' (Left to Right) using the `direction LR` directive. It illustrates the transitions between states such as `InsertCoin`, `CoinsInserted`, `SelectProduct`, and `Purchase`, providing a visual representation of the vending machine's operational flow.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_22

LANGUAGE: Mermaid
CODE:
```
--- 
title: vending_machine_graph
---
stateDiagram-v2
  direction LR
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```

----------------------------------------

TITLE: Configure AsyncClient with Smart HTTP Request Retries in Pydantic AI
DESCRIPTION: Demonstrates how to create an `httpx.AsyncClient` with advanced retry logic using `pydantic_ai.retries.AsyncTenacityTransport`. It configures retries for `HTTPStatusError` and `ConnectionError`, implements a smart `wait_retry_after` strategy respecting `Retry-After` headers, and integrates the client with `Pydantic AI` models for resilient API calls.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_1

LANGUAGE: python
CODE:
```
from httpx import AsyncClient, HTTPStatusError
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after
from pydantic_ai.providers.openai import OpenAIProvider

def create_retrying_client():
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""
        if response.status_code in (429, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300
            ),
            # Stop after 5 attempts
            stop=stop_after_attempt(5),
            # Re-raise the last exception if all retries fail
            reraise=True
        ),
        validate_response=should_retry_status
    )
    return AsyncClient(transport=transport)

# Use the retrying client with a model
client = create_retrying_client()
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

----------------------------------------

TITLE: Initialize HTTP Client with Custom Transport
DESCRIPTION: Demonstrates the fundamental step of creating an `httpx` client instance by passing a configured transport layer. This transport is where custom behaviors like retry logic are injected.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_5

LANGUAGE: Python
CODE:
```
client = Client(transport=transport)
```

----------------------------------------

TITLE: Define Pydantic AI Agent for Seat Preference and Extraction Logic
DESCRIPTION: This snippet defines an AI agent (`seat_preference_agent`) using `pydantic-ai` to extract seat preferences from user input. It specifies `openai:gpt-4o` as the model and `Union[SeatPreference, Failed]` as the output type for structured data or failure. The `find_seat` asynchronous function continuously prompts the user for their seat preference, runs the agent, and handles cases where the preference cannot be understood, maintaining message history for context.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#_snippet_4

LANGUAGE: python
CODE:
```
seat_preference_agent = Agent[None, Union[SeatPreference, Failed]](  # (5)!
    'openai:gpt-4o',
    output_type=Union[SeatPreference, Failed],  # type: ignore
    system_prompt=(
        "Extract the user's seat preference. "
        'Seats A and F are window seats. '
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room. '
    ),
)


async def find_seat(usage: Usage) -> SeatPreference:  # (6)!
    message_history: Union[list[ModelMessage], None] = None
    while True:
        answer = Prompt.ask('What seat would you like?')

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, SeatPreference):
            return result.output
        else:
            print('Could not understand seat preference. Please try again.')
            message_history = result.all_messages()
```

----------------------------------------

TITLE: Implement Custom and Built-in Pydantic Evals Evaluators
DESCRIPTION: This Python code demonstrates how to add both built-in and custom evaluators to a Pydantic Evals `Dataset`. It shows the use of `IsInstance` for basic type checking and defines a custom `MyEvaluator` class that inherits from `Evaluator` to provide a scoring logic based on output matching the expected output, including partial matches.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals.md#_snippet_2

LANGUAGE: python
CODE:
```
from dataclasses import dataclass

from simple_eval_dataset import dataset

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import IsInstance

dataset.add_evaluator(IsInstance(type_name='str'))  # (1)!


@dataclass
class MyEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:  # (2)!
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


dataset.add_evaluator(MyEvaluator())
```

----------------------------------------

TITLE: pydantic_graph.nodes Module Members
DESCRIPTION: This entry documents the public members exposed by the `pydantic_graph.nodes` module. It lists types and classes such as `StateT`, `GraphRunContext`, `BaseNode`, `End`, `Edge`, `DepsT`, `RunEndT`, and `NodeRunEndT`, which are integral to defining and managing graph structures within the pydantic_graph library.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/api/pydantic_graph/nodes.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Module: pydantic_graph.nodes

Members:
- StateT
- GraphRunContext
- BaseNode
- End
- Edge
- DepsT
- RunEndT
- NodeRunEndT
```

----------------------------------------

TITLE: Disabling MCP Sampling for a Pydantic AI Server
DESCRIPTION: This Python snippet shows how to explicitly disallow sampling when configuring an `MCPServerStdio` instance. By setting the `allow_sampling` parameter to `False` during server initialization, the server will not be able to proxy LLM calls through the client, overriding the default behavior.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md#_snippet_13

LANGUAGE: python
CODE:
```
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(
    command='python',
    args=['generate_svg.py'],
    allow_sampling=False,
)
```

----------------------------------------

TITLE: Implement Smart Wait Strategy with wait_retry_after for HTTP Retries
DESCRIPTION: Illustrates the usage of `pydantic_ai.retries.wait_retry_after` for intelligent backoff in retry mechanisms. It shows how to configure it to automatically respect HTTP `Retry-After` headers, fall back to exponential backoff when no header is present, and set a maximum wait time to prevent excessive delays during retries.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic_ai.retries import wait_retry_after
from tenacity import wait_exponential

# Basic usage - respects Retry-After headers, falls back to exponential backoff
wait_strategy_1 = wait_retry_after()

# Custom configuration
wait_strategy_2 = wait_retry_after(
    fallback_strategy=wait_exponential(multiplier=2, max=120),
    max_wait=600  # Never wait more than 10 minutes
)
```

----------------------------------------

TITLE: Configure Synchronous HTTP Client with TenacityTransport for Retries
DESCRIPTION: Illustrates how to set up an `httpx.Client` with `pydantic_ai.retries.TenacityTransport` for synchronous HTTP request retries. It demonstrates integrating a `Retrying` controller from `tenacity` and an optional `validate_response` function to treat specific HTTP status codes (e.g., 4xx/5xx) as retryable failures, ensuring robust synchronous API interactions.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_4

LANGUAGE: python
CODE:
```
from httpx import Client
from tenacity import Retrying, stop_after_attempt
from pydantic_ai.retries import TenacityTransport

# Create the basic components
retrying = Retrying(stop=stop_after_attempt(3), reraise=True)

def validator(response):
    """Treat responses with HTTP status 4xx/5xx as failures that need to be retried.
    Without a response validator, only network errors and timeouts will result in a retry.
    """
    response.raise_for_status()

# Create the transport
transport = TenacityTransport(
    controller=retrying,       # Retrying instance
    validate_response=validator # Optional response validator
)
```

----------------------------------------

TITLE: Configure Asynchronous HTTP Client with AsyncTenacityTransport for Retries
DESCRIPTION: Shows how to set up an `httpx.AsyncClient` with `pydantic_ai.retries.AsyncTenacityTransport` for asynchronous HTTP request retries. It demonstrates integrating an `AsyncRetrying` controller from `tenacity` and an optional `validate_response` function to treat specific HTTP status codes (e.g., 4xx/5xx) as retryable failures, beyond just network errors.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_3

LANGUAGE: python
CODE:
```
from httpx import AsyncClient
from tenacity import AsyncRetrying, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport

# Create the basic components
async_retrying = AsyncRetrying(stop=stop_after_attempt(3), reraise=True)

def validator(response):
    """Treat responses with HTTP status 4xx/5xx as failures that need to be retried.
    Without a response validator, only network errors and timeouts will result in a retry.
    """
    response.raise_for_status()

# Create the transport
transport = AsyncTenacityTransport(
    controller=async_retrying,   # AsyncRetrying instance
    validate_response=validator  # Optional response validator
)

# Create a client using the transport:
client = AsyncClient(transport=transport)
```

----------------------------------------

TITLE: Define an Intermediate Node in Pydantic-Graph
DESCRIPTION: This Python code demonstrates how to create a basic intermediate node (`MyNode`) in a pydantic-graph. It's defined as a dataclass with a field `foo` and an asynchronous `run` method that processes context and returns another node, indicating an outgoing edge. This node cannot terminate the graph run.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_2

LANGUAGE: Python
CODE:
```
from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class MyNode(BaseNode[MyState]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext[MyState],
    ) -> AnotherNode:
        ...
        return AnotherNode()
```

----------------------------------------

TITLE: Generate Vending Machine State Diagram with LR Direction in Python
DESCRIPTION: This Python snippet demonstrates how to generate a Mermaid state diagram for a vending machine, explicitly setting its direction to 'Left to Right' (LR) using the `mermaid_code` method. It requires the `vending_machine.py` module for its functionality.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_23

LANGUAGE: python
CODE:
```
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin, direction='LR')
```

----------------------------------------

TITLE: Handle Rate Limits with Retry-After Headers in Python
DESCRIPTION: This function creates an `httpx.AsyncClient` configured to automatically respect `Retry-After` headers from 429 (Too Many Requests) responses. It uses `AsyncTenacityTransport` with `wait_retry_after` for intelligent waiting, falling back to exponential backoff if no header is present.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_6

LANGUAGE: Python
CODE:
```
from httpx import AsyncClient, HTTPStatusError
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type, wait_exponential
from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after

def create_rate_limit_client():
    """Create a client that respects Retry-After headers from rate limiting responses."""
    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            retry=retry_if_exception_type(HTTPStatusError),
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300  # Don't wait more than 5 minutes
            ),
            stop=stop_after_attempt(10),
            reraise=True
        ),
        validate_response=lambda r: r.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
    )
    return AsyncClient(transport=transport)

# Example usage
client = create_rate_limit_client()
# Client is now ready to use with any HTTP requests and will respect Retry-After headers
```

----------------------------------------

TITLE: Handle Tool Execution Retries with ModelRetry in Python
DESCRIPTION: This snippet demonstrates how to explicitly request a tool execution retry using the `ModelRetry` exception in Pydantic AI. Raising `ModelRetry` allows a tool's internal logic to inform the LLM of an issue, prompting it to correct parameters and retry the call, similar to how `ValidationError` works for argument validation failures.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md#_snippet_13

LANGUAGE: python
CODE:
```
from pydantic_ai import ModelRetry

def my_flaky_tool(query: str) -> str:
    if query == 'bad':
        # Tell the LLM the query was bad and it should try again
        raise ModelRetry("The query 'bad' is not allowed. Please provide a different query.")
    # ... process query ...
    return 'Success!'
```

----------------------------------------

TITLE: Serializing and deserializing Pydantic AI messages to JSON
DESCRIPTION: This snippet demonstrates how to persist Pydantic AI message histories by serializing them to JSON-compatible Python objects or direct JSON strings, and then deserializing them back. It utilizes `ModelMessagesTypeAdapter` for validation and conversion, enabling storage or sharing of conversation states.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_5

LANGUAGE: python
CODE:
```
from pydantic_core import to_jsonable_python

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
history_step_1 = result1.all_messages()
as_python_objects = to_jsonable_python(history_step_1)
same_history_as_step_1 = ModelMessagesTypeAdapter.validate_python(as_python_objects)

result2 = agent.run_sync(
    'Tell me a different joke.', message_history=same_history_as_step_1
)
```

LANGUAGE: python
CODE:
```
from pydantic_core import to_json
...
as_json_objects = to_json(history_step_1)
same_history_as_step_1 = ModelMessagesTypeAdapter.validate_json(as_json_objects)
```

----------------------------------------

TITLE: Disable Google Model Thinking
DESCRIPTION: Shows how to disable the 'thinking' feature for a Google model by setting the `thinking_budget` to `0` within the `google_thinking_config` dictionary when initializing `GoogleModelSettings`.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/models/google.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(google_thinking_config={'thinking_budget': 0})
model = GoogleModel('gemini-2.0-flash')
agent = Agent(model, model_settings=model_settings)
...
```

----------------------------------------

TITLE: Create Custom Retry Logic for HTTP Status Codes and Network Errors in Python
DESCRIPTION: Illustrates how to define a custom retry condition to selectively retry requests based on HTTP status codes (e.g., 5xx server errors) and network exceptions. This allows fine-grained control over when retries are performed, combining smart `Retry-After` handling with custom backoff.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/retries.md#_snippet_8

LANGUAGE: Python
CODE:
```
import httpx
from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after

def create_custom_retry_client():
    """Create a client with custom retry logic."""
    def custom_retry_condition(exception):
        """Custom logic to determine if we should retry."""
        if isinstance(exception, httpx.HTTPStatusError):
            # Retry on server errors but not client errors
            return 500 <= exception.response.status_code < 600
        return isinstance(exception, (httpx.TimeoutException, httpx.ConnectError))

    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            retry=custom_retry_condition,
            # Use wait_retry_after for smart waiting on rate limits,
            # with custom exponential backoff as fallback
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=2, max=30),
                max_wait=120
            ),
            stop=stop_after_attempt(5),
            reraise=True
        ),
        validate_response=lambda r: r.raise_for_status()
    )
    return httpx.AsyncClient(transport=transport)

client = create_custom_retry_client()
# Client will retry server errors (5xx) and network errors, but not client errors (4xx)
```

----------------------------------------

TITLE: Configure Pydantic AI OpenTelemetry Event Mode
DESCRIPTION: Python code illustrating how to change the default OpenTelemetry event collection behavior in Pydantic AI. By setting `event_mode='logs'` during `logfire.instrument_pydantic_ai`, messages are captured as individual events (logs) rather than a single large JSON array attribute on the request span.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/logfire.md#_snippet_10

LANGUAGE: python
CODE:
```
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai(event_mode='logs')
agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

----------------------------------------

TITLE: Set State Diagram Direction in Python
DESCRIPTION: This Python snippet demonstrates how to specify the layout direction for a state diagram generated by `vending_machine_graph.mermaid_code`. It sets the direction to 'LR' (Left to Right) using the `direction` parameter. This functionality requires the `vending_machine` module and is compatible with Python 3.10 or newer.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md#_snippet_21

LANGUAGE: Python
CODE:
```
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin, direction='LR')
```

----------------------------------------

TITLE: Test History Processors with FunctionModel in Python
DESCRIPTION: This snippet demonstrates a robust method for testing history processors using `pytest` and `FunctionModel`. By injecting a `FunctionModel` that captures the messages sent to the underlying model provider, developers can verify that their history processors are correctly modifying the message history before it's processed by the AI model.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_12

LANGUAGE: python
CODE:
```
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.fixture
def received_messages() -> list[ModelMessage]:
    return []


@pytest.fixture
def function_model(received_messages: list[ModelMessage]) -> FunctionModel:
    def capture_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Capture the messages that the provider actually receives
        received_messages.clear()
        received_messages.extend(messages)
        return ModelResponse(parts=[TextPart(content='Provider response')])

    return FunctionModel(capture_model_function)


def test_history_processor(function_model: FunctionModel, received_messages: list[ModelMessage]):
    def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, history_processors=[filter_responses])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
    ]

    agent.run_sync('Question 2', message_history=message_history)
    assert received_messages == [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelRequest(parts=[UserPromptPart(content='Question 2')]),
    ]
```

----------------------------------------

TITLE: Pydantic AI Agent Run Result Message Access
DESCRIPTION: This section details the methods available on `RunResult` and `StreamedRunResult` objects for accessing messages exchanged during an agent run. These methods allow retrieval of all messages (including prior runs) or only new messages from the current run, with options for JSON byte output. It also clarifies how message retrieval behaves with `StreamedRunResult` after streaming operations.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
RunResult / StreamedRunResult:
  all_messages(): list[Message]
    - Returns all messages from the current and prior runs.
  all_messages_json(): bytes
    - Returns all messages as JSON bytes.
  new_messages(): list[Message]
    - Returns only the messages from the current run.
  new_messages_json(): bytes
    - Returns only the messages from the current run as JSON bytes.

StreamedRunResult specific methods (affecting message completeness):
  stream(): AsyncIterator[Any]
    - Awaits the stream to complete, making final result message available.
  stream_text(): AsyncIterator[str]
    - Awaits the stream to complete, making final result message available (unless delta=True).
  stream_structured(): AsyncIterator[Any]
    - Awaits the stream to complete, making final result message available.
  get_output(): Any
    - Awaits the output to be retrieved, making final result message available.
```

----------------------------------------

TITLE: Filtering Pydantic AI Message History with a Custom Processor
DESCRIPTION: This snippet demonstrates how to use the `history_processors` parameter on a Pydantic AI `Agent` to modify message history before it's sent to the model. The `filter_responses` function serves as a custom processor, removing all `ModelResponse` messages and retaining only `ModelRequest` messages, which can be useful for specific privacy or context management scenarios.

SOURCE: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#_snippet_8

LANGUAGE: python
CODE:
```
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove all ModelResponse messages, keeping only ModelRequest messages."""
    return [msg for msg in messages if isinstance(msg, ModelRequest)]

# Create agent with history processor
agent = Agent('openai:gpt-4o', history_processors=[filter_responses])

# Example: Create some conversation history
message_history = [
    ModelRequest(parts=[UserPromptPart(content='What is 2+2?')]),
    ModelResponse(parts=[TextPart(content='2+2 equals 4')]),  # This will be filtered out
]

# When you run the agent, the history processor will filter out ModelResponse messages
# result = agent.run_sync('What about 3+3?', message_history=message_history)
```