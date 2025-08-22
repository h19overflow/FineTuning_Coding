========================
CODE SNIPPETS
========================
TITLE: LangChain Introduction and Tutorials
DESCRIPTION: Guides for getting started with LangChain and a list of tutorials for building various LLM applications.

SOURCE: https://python.langchain.com/docs/how_to/tools_error/

LANGUAGE: APIDOC
CODE:
```
Introduction: /docs/introduction/
Tutorials: /docs/tutorials/
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides guides on implementing specific features and functionalities within LangChain. Topics include using tools in chains, vectorstores as retrievers, adding memory to chatbots, example selectors, semantic layers over graph databases, parallel runnable invocation, streaming chat model responses, default invocation arguments, retrieval for chatbots, few-shot examples, function calling, package installation, query analysis examples, routing, and structured output from models.

SOURCE: https://python.langchain.com/docs/how_to/sql_prompting/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: How to Use Example Selectors
DESCRIPTION: This guide covers the use of example selectors in LangChain, which dynamically select relevant examples to include in prompts based on the current input. This is useful for few-shot learning.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import LengthWithoutTokensExampleSelector
from langchain_text_splitters import TextSplitter

# Example using SemanticSimilarityExampleSelector
# examples = [
#     {"input": "apple", "output": "fruit"},
#     {"input": "carrot", "output": "vegetable"},
#     {"input": "banana", "output": "fruit"},
# ]
# 
# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     examples,
#     OpenAIEmbeddings(),
#     Chroma,
#     k=1,
# )
# 
# prompt = PromptTemplate(
#     input_variables=["input", "examples"],
#     template="Input: {input}\nOutput: {examples}",
# )
# 
# chain = prompt | OpenAI() # Assuming OpenAI is configured
# 
# print(chain.invoke({"input": "grape", "examples": example_selector}))

```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This how-to guide covers the usage of example selectors in LangChain. These selectors help in dynamically choosing few-shot examples for prompts.

SOURCE: https://python.langchain.com/docs/tutorials/sql_qa/

LANGUAGE: python
CODE:
```
print("How-to: How to use example selectors")
# Further implementation details would follow here.
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: Provides a list of how-to guides for implementing specific functionalities in LangChain, such as using tools, vectorstores, memory, example selectors, parallel execution, streaming, function calling, and more.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_mmr/

LANGUAGE: text
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Use Example Selectors
DESCRIPTION: This guide explains how to use LangChain's example selectors to dynamically select relevant few-shot examples for prompts. This helps in optimizing prompt length and improving model performance by providing the most pertinent examples.

SOURCE: https://python.langchain.com/docs/how_to/binding/

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Initialize the chat model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define few-shot examples
examples = [
    {"input": "Q: What is the capital of France?", "output": "A: Paris"},
    {"input": "Q: What is the capital of Japan?", "output": "A: Tokyo"},
    {"input": "Q: What is the capital of Canada?", "output": "A: Ottawa"},
    {"input": "Q: What is the capital of Germany?", "output": "A: Berlin"},
    {"input": "Q: What is the capital of Brazil?", "output": "A: Brasilia"}
]

# Convert examples to chat messages format
example_messages = []
for example in examples:
    example_messages.append(HumanMessage(content=example['input']))
    example_messages.append(AIMessage(content=example['output']))

# Create a semantic similarity example selector
# This requires embedding the example inputs
example_selector = SemanticSimilarityExampleSelector.from_examples(
    example_messages,
    OpenAIEmbeddings(),
    FAISS,
    k=2 # Select top 2 similar examples
)

# Create a FewShotChatMessagePromptTemplate
# This template will use the example selector to fetch examples
example_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_messages=example_messages, # Provide all examples here for the selector to use
    input_variables=["input"],
    # The prompt structure for each example pair (HumanMessage, AIMessage)
    # This is implicitly handled by FewShotChatMessagePromptTemplate when using example_messages
)

# Create the final prompt template including the few-shot examples
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about capitals."),
    example_prompt,
    ("human", "{input}")
])

# Create a chain
chain = final_prompt | llm

# Invoke the chain with a new question
# The example_selector will pick the 2 most relevant examples based on semantic similarity
# result = chain.invoke({"input": "What is the capital of Italy?"})
# print(result.content)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Lists how-to guides for LangChain, covering practical implementation details such as using tools, vectorstores, memory, example selectors, parallel execution, streaming, function calling, and more.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_constructor/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: LangGraph Quickstarts
DESCRIPTION: Provides quickstart guides for getting started with LangGraph, including building custom workflows and running local servers.

SOURCE: https://langchain-ai.github.io/langgraph/

LANGUAGE: python
CODE:
```
Start with a prebuilt agent
Build a custom workflow
Run a local server
```

----------------------------------------

TITLE: LangChain Setup and Usage Guide
DESCRIPTION: Provides a step-by-step guide on how to use LangChain, including environment setup, understanding core concepts like Agents, Chains, and Tools, choosing components based on use cases, integrating with language models (e.g., OpenAI, Anthropic), implementing application logic, and testing. It highlights LangChain's role as a flexible framework for building language-based applications.

SOURCE: https://python.langchain.com/docs/how_to/routing/

LANGUAGE: python
CODE:
```
import os

# Example of setting up environment variables (e.g., for API keys)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- Core LangChain Concepts ---
# Agents: Agents use a language model to decide which actions to take and in what order.
# Chains: Chains allow you to combine LLMs with other components or data sources.
# Tools: Tools are functions that agents can call to interact with the outside world.

# --- Example Usage (Conceptual) ---
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate

# llm = OpenAI(temperature=0.9)
# prompt = PromptTemplate(
#     input_variables=["product_name"],
#     template="Tell me a joke about {product_name}.",
# )
# chain = LLMChain(llm=llm, prompt=prompt)

# print(chain.run("LangChain"))

# --- RunnableBranch Example (Conceptual) ---
# from langchain.schema import StrOutputParser
# from langchain.schema.runnable import RunnableBranch

# def is_even(n):
#     return n % 2 == 0

# def is_positive(n):
#     return n > 0

# branch = RunnableBranch(
#     ("even", RunnableLambda(is_even) | StrOutputParser()),
#     ("positive", RunnableLambda(is_positive) | StrOutputParser()),
#     name="number_branch"
# )

# print(branch.invoke(4))
# print(branch.invoke(-2))
# print(branch.invoke(3))

```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This how-to guide provides instructions on how to install the necessary LangChain packages. It covers the prerequisites and commands for setting up LangChain.

SOURCE: https://python.langchain.com/docs/tutorials/sql_qa/

LANGUAGE: bash
CODE:
```
pip install langchain
```

----------------------------------------

TITLE: How to Use Few Shot Examples
DESCRIPTION: This guide covers the general practice of using few-shot examples in LangChain prompts. It explains how to format examples to guide language models in tasks like classification, summarization, or translation.

SOURCE: https://python.langchain.com/docs/how_to/tool_artifacts/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# example_prompt = PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}")
# examples = [
#     {"input": "apple", "output": "fruit"},
#     {"input": "carrot", "output": "vegetable"},
# ]
# prompt = FewShotPromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
#     suffix="Input: {input}\nOutput:",
#     input_variables=["input"],
# )
# formatted_prompt = prompt.format(input="banana")

```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This how-to guide explains how to leverage few-shot learning by providing examples directly in the prompt to guide the LLM's behavior.

SOURCE: https://python.langchain.com/docs/tutorials/rag/

LANGUAGE: python
CODE:
```
# This is a placeholder for the actual code in the how-to guide.
# The guide focuses on the concepts and steps involved in using few-shot examples.
# For specific code examples, please refer to the official LangChain documentation.

# Example conceptual steps:
# 1. Create a prompt template that includes placeholders for examples.
# 2. Manually provide a few input-output pairs as examples within the prompt.
# 3. Append the actual user query to the prompt.
# 4. Send the complete prompt to the LLM.
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain. These cover topics like using tools, vectorstores, memory, example selectors, parallel execution, streaming, and more.

SOURCE: https://python.langchain.com/docs/concepts/example_selectors/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
- How to use tools in a chain: /docs/how_to/tools_chain/
- How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
- How to add memory to chatbots: /docs/how_to/chatbots_memory/
- How to use example selectors: /docs/how_to/example_selectors/
- How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
- How to invoke runnables in parallel: /docs/how_to/parallel/
- How to stream chat model responses: /docs/how_to/chat_streaming/
- How to add default invocation args to a Runnable: /docs/how_to/binding/
- How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
- How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
- How to do tool/function calling: /docs/how_to/function_calling/
- How to install LangChain packages: /docs/how_to/installation/
- How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
- How to use few shot examples: /docs/how_to/few_shot_examples/
- How to run custom functions: /docs/how_to/functions/
- How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
- How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
- How to route between sub-chains: /docs/how_to/routing/
- How to return structured data from a model: /docs/how_to/structured_output/
- How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This how-to guide covers the usage of example selectors in LangChain, which help in dynamically selecting few-shot examples for prompts.

SOURCE: https://python.langchain.com/docs/tutorials/rag/

LANGUAGE: python
CODE:
```
# This is a placeholder for the actual code in the how-to guide.
# The guide focuses on the concepts and steps involved in using example selectors.
# For specific code examples, please refer to the official LangChain documentation.

# Example conceptual steps:
# 1. Prepare a list of example prompts and outputs
# 2. Choose an example selector strategy (e.g., LengthBasedExampleSelector, SemanticSimilarityExampleSelector)
# 3. Instantiate the selector with examples and a prompt template
# 4. Use the selector to format the prompt with selected examples
```

----------------------------------------

TITLE: How-to Guides Navigation
DESCRIPTION: Lists various how-to guides for LangChain, covering essential tasks like installation, using tools, adding memory, and handling structured output.

SOURCE: https://python.langchain.com/docs/how_to/filter_messages/

LANGUAGE: HTML
CODE:
```
*   [How-to guides](/docs/how_to/)
    
    *   [How-to guides](/docs/how_to/)
    *   [How to use tools in a chain](/docs/how_to/tools_chain/)
    *   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
    *   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
    *   [How to use example selectors](/docs/how_to/example_selectors/)
    *   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
    *   [How to invoke runnables in parallel](/docs/how_to/parallel/)
    *   [How to stream chat model responses](/docs/how_to/chat_streaming/)
    *   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
    *   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
    *   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
    *   [How to do tool/function calling](/docs/how_to/function_calling/)
    *   [How to install LangChain packages](/docs/how_to/installation/)
    *   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
    *   [How to use few shot examples](/docs/how_to/few_shot_examples/)
    *   [How to run custom functions](/docs/how_to/functions/)
    *   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
    *   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
    *   [How to route between sub-chains](/docs/how_to/routing/)
    *   [How to return structured data from a model](/docs/how_to/structured_output/)
    *   [How to summarize text through parallelization](/docs/how_to/summarize_map_reduce/)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides guides on how to perform specific tasks with LangChain. It includes instructions on using tools, vectorstores, memory, example selectors, parallel execution, streaming, function calling, and more.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_xml/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
*   [How to summarize text through parallelization](/docs/how_to/summarize_map_reduce/)
```

----------------------------------------

TITLE: How to Use Few Shot Examples
DESCRIPTION: This guide covers the general approach to using few-shot examples in LangChain, applicable to various model types and tasks. It emphasizes the importance of prompt engineering with examples.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
# This is a conceptual guide. Actual implementation depends on the specific model and prompt structure.
# Example structure for few-shot prompting:
# prompt = """
# Given the following examples:
# 
# Input: apple
# Output: fruit
# 
# Input: carrot
# Output: vegetable
# 
# Input: {user_input}
# Output:
# """
# 
# chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("banana"))
```

----------------------------------------

TITLE: Install Dependencies
DESCRIPTION: Installs necessary libraries including langchain-core, langchain-openai, and langgraph for the examples.

SOURCE: https://python.langchain.com/docs/how_to/convert_runnable_to_tool/

LANGUAGE: bash
CODE:
```
%%capture --no-stderr%pip install -U langchain-core langchain-openai langgraph
```

----------------------------------------

TITLE: LangGraph Examples
DESCRIPTION: Guided examples for getting started with LangGraph, demonstrating various use cases and functionalities.

SOURCE: https://langchain-ai.github.io/langgraph/

LANGUAGE: python
CODE:
```
# Example: Simple graph execution
# from langgraph.graph import StateGraph
# def add(state):
#     return state + 1
# workflow = StateGraph(int)
# workflow.add_node("add", add)
# workflow.set_entry_point("add")
# app = workflow.compile()
# result = app.invoke(5)
# print(result)
```

----------------------------------------

TITLE: Install LangChain Dependencies
DESCRIPTION: Installs the necessary LangChain core and OpenAI packages for the project. This is a prerequisite for running the example code.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
# %pip install -qU langchain-core langchain-openai
```

----------------------------------------

TITLE: How to Use Few Shot Examples in Chat Models
DESCRIPTION: This guide explains how to provide few-shot examples to chat models in LangChain to improve their performance on specific tasks. It demonstrates how to format prompts with examples.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French. "
        "Only return the translation and nothing else."
    ),
    HumanMessage(content="Translate this English text to French: 'I am a helpful assistant.'"),
    AIMessage(content="Je suis un assistant utile."),
    HumanMessage(content="Translate
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features within LangChain. It covers topics such as using tools in chains, vectorstore retrievers, adding memory to chatbots, example selectors, semantic layers over graph databases, parallel runnable invocation, streaming chat responses, default invocation arguments, retrieval for chatbots, few-shot examples, function calling, package installation, query analysis, routing, and structured output from models.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_directory/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: How to Add Examples to the Prompt for Query Analysis
DESCRIPTION: This guide explains how to include few-shot examples within prompts for query analysis tasks. It demonstrates how to structure prompts with examples to improve the accuracy and relevance of query analysis.

SOURCE: https://python.langchain.com/docs/how_to/tool_artifacts/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate

# template = """
# Analyze the following queries and categorize them.

# Query: "What is the capital of France?"
# Category: Geography

# Query: "{user_query}"
# Category:
# """
# prompt = PromptTemplate.from_template(template)
# formatted_prompt = prompt.format(user_query="Who won the world series in 2020?")

```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on using various features of LangChain, including tools in chains, vectorstore retrievers, memory for chatbots, example selectors, parallel runnable invocation, streaming chat responses, default invocation arguments, retrieval for chatbots, few-shot examples, function calling, installation, query analysis, routing, and structured output.

SOURCE: https://python.langchain.com/docs/how_to/self_query/

LANGUAGE: python
CODE:
```
# Example: How to use tools in a chain
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(['serpapi', 'llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What is the weather in San Francisco?")

# Example: How to use a vectorstore as a retriever
# This involves creating embeddings for documents and storing them in a vectorstore.

# Example: How to add memory to chatbots
# This allows the chatbot to remember previous interactions.

# Example: How to stream chat model responses
# This enables real-time responses from the chat model.

# Example: How to do tool/function calling
# This allows LLMs to call external tools or functions.

# Example: How to return structured data from a model
# This uses output parsers to get structured output from LLM responses.
```

----------------------------------------

TITLE: How to Use Few Shot Examples in Chat Models
DESCRIPTION: This guide explains how to provide few-shot examples to chat models in LangChain to improve their performance on specific tasks. It demonstrates how to format prompts with examples.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French. "
        "Only return the translation and nothing else."
    ),
    HumanMessage(content="Translate this English text to French: 'I am a helpful assistant.'"),
    # Example of a few-shot example:
    # HumanMessage(content="Translate this English text to French: 'Hello world.'"),
    # AIMessage(content="Bonjour le monde.")
]

# print(chat.invoke(messages))
```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for Python. It covers using pip to install the core library and common integrations, ensuring you have the right tools to start building LLM applications.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

LANGUAGE: bash
CODE:
```
# Install the core LangChain library
pip install langchain

# Install common integrations (e.g., OpenAI, Hugging Face)
pip install langchain-openai langchain-huggingface

# Install specific components like LangChain Community
pip install langchain-community

# Install LangChain partners (e.g., LangSmith)
pip install langsmith

# Install LangServe for deploying chains as APIs
pip install langchain-serve

# Install LangGraph for stateful, multi-agent applications
pip install langchain-graph

# To install all common integrations:
pip install "langchain[all]"
```

----------------------------------------

TITLE: Use Few Shot Examples
DESCRIPTION: This guide explains how to incorporate few-shot examples into LangChain prompts to enhance model performance on specific tasks. It covers structuring prompts with examples to guide the LLM's output.

SOURCE: https://python.langchain.com/docs/how_to/binding/

LANGUAGE: python
CODE:
```
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Define few-shot examples
examples = [
    {"input": "Q: What is the capital of France?", "output": "A: Paris"},
    {"input": "Q: What is the capital of Japan?", "output": "A: Tokyo"},
    {"input": "Q: What is the capital of Canada?", "output": "A: Ottawa"}
]

# Format the examples into a string suitable for the prompt
example_str = ""
for example in examples:
    example_str += f"{example['input']}\n{example['output']}\n\n"

# Create a prompt template that includes the formatted examples
prompt_template = PromptTemplate(
    input_variables=["input"],
    template=f"{{example_str}}Q: {{input}}\nA:",
    partial_variables={"example_str": example_str}
)

# Create a chain
chain = prompt_template | llm

# Invoke the chain with a new question
# The model will use the provided examples to answer the question
# result = chain.invoke("What is the capital of Germany?")
# print(result)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and patterns in LangChain, such as using tools, memory, vectorstores, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/installation/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Semantic Similarity Example Selector Setup
DESCRIPTION: Illustrates the setup for SemanticSimilarityExampleSelector, which selects examples based solely on their semantic similarity to the input. This is contrasted with the MMR approach to highlight the diversity aspect. It also requires examples, an embedding model, and a vector store.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_mmr/

LANGUAGE: python
CODE:
```
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # The number of examples to produce.
    k=2,
)

similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

print(similar_prompt.format(adjective="worried"))
```

----------------------------------------

TITLE: LangGraph Examples
DESCRIPTION: Features guided examples to help users get started with LangGraph, demonstrating practical applications and common use cases.

SOURCE: https://langchain-ai.github.io/langgraph/

LANGUAGE: python
CODE:
```
https://langchain-ai.github.io/langgraph/examples/
```

----------------------------------------

TITLE: PipelinePromptTemplate Example
DESCRIPTION: Demonstrates creating a pipeline of prompt templates for sequential composition. It defines an introduction, an example, and a start prompt, then combines them into a final prompt using PipelinePromptTemplate.

SOURCE: https://python.langchain.com/docs/how_to/prompts_composition/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

full_template = """{introduction}{example}{start}"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """Here's an example of an interaction:\nQ: {example_q}\nA: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

start_template = """Now, do this for real!\nQ: {input}\nA:"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

print(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on how to perform specific tasks within LangChain, such as creating and querying vector stores, and loading web pages. These guides offer step-by-step instructions and code examples.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_web/

LANGUAGE: APIDOC
CODE:
```
How to create and query vector stores:
  Description: Steps to initialize a vector store and perform similarity searches.
  Dependencies: Embedding model, vector store implementation (e.g., Chroma).
  Example:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        ["hello world", "bye world"],
        embeddings,
        metadatas=[{"source": "doc1"}, {"source": "doc2"}]
    )
    results = vectorstore.similarity_search("hello")

How to load web pages:
  Description: Methods for fetching and parsing content from web pages.
  Dependencies: BeautifulSoup, requests.
  Example:
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader("https://www.example.com")
    docs = loader.load()

```

----------------------------------------

TITLE: How to Use Few Shot Examples in Chat Models
DESCRIPTION: This guide details how to provide few-shot examples to chat models in LangChain. It covers formatting examples and including them in the prompt to guide the model's responses.

SOURCE: https://python.langchain.com/docs/how_to/tool_artifacts/

LANGUAGE: python
CODE:
```
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

# chat = ChatOpenAI()
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Translate 'hello' to French."),
#     AIMessage(content="Bonjour"),
#     HumanMessage(content="Translate 'goodbye' to French.")
# ]
# response = chat.invoke(messages)

```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for implementing specific functionalities within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/convert_runnable_to_tool/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide covers the usage of example selectors in LangChain, which help in dynamically selecting few-shot examples for prompts.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_tools/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "apple", "output": "fruit"},
    {"input": "banana", "output": "fruit"},
    {"input": "carrot", "output": "vegetable"}
]

example_prompt = PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}")

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    maxLength=5
)

selector_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the category of the input.",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide explains how to use example selectors in LangChain to dynamically select few-shot examples for prompts.

SOURCE: https://python.langchain.com/docs/tutorials/extraction/

LANGUAGE: python
CODE:
```
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAI

# Example data
# examples = [
#     {"input": "I love this product!", "output": "positive"},
#     {"input": "This is terrible.", "output": "negative"}
# ]

# Example selector
# example_selector = LengthBasedExampleSelector(
#     examples=examples,
#     example_prompt=PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}"),
#     maxLength=1
# )

# Create the few-shot prompt
# prefix = "Classify the sentiment of the following input."
# suffix = "Input: {input}\nOutput:"
# few_shot_prompt = FewShotPromptTemplate(
#     example_selector=example_selector,
#     example_prompt=PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}"),
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["input"]
# )

# Example usage:
# formatted_prompt = few_shot_prompt.format(input="This is okay.")
# print(formatted_prompt)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for LangChain, demonstrating practical implementation of features like tools, memory, output parsers, and parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/multimodal_prompts/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/how_to/contextual_compression/

LANGUAGE: markdown
CODE:
```
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: SambaStudio Chat Integration
DESCRIPTION: Guides users on getting started with SambaStudio chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatSambaStudio

# Example usage (requires SambaStudio API key)
# llm = ChatSambaStudio(api_key="YOUR_SAMBASTUDIO_API_KEY")
# response = llm.invoke("Write a Python script to read a CSV file.")
# print(response)
```

----------------------------------------

TITLE: Pipeshift Chat Integration
DESCRIPTION: Guides users on getting started with Pipeshift chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatPipeshift

# Example usage (requires Pipeshift API key)
# llm = ChatPipeshift(api_key="YOUR_PIPESHIFT_API_KEY")
# response = llm.invoke("Write a product description for a new gadget.")
# print(response)
```

----------------------------------------

TITLE: Use Few Shot Examples in Chat Models
DESCRIPTION: This guide explains how to provide few-shot examples to LangChain chat models to improve their performance on specific tasks. It covers formatting examples within the prompt to guide the model's responses.

SOURCE: https://python.langchain.com/docs/how_to/binding/

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Initialize the chat model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define few-shot examples
examples = [
    HumanMessage(content="Translate English to French: see => voir"),
    AIMessage(content="Translate English to French: run => courir"),
    HumanMessage(content="Translate English to French: eat => manger"),
    AIMessage(content="Translate English to French: sleep => dormir"),
]

# Create a prompt template that includes the examples
# The 'messages' format is used for chat models
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    *examples, # Unpack the examples into the prompt
    ("human", "Translate English to French: {input}")
])

# Create a chain
chain = chat_prompt | llm

# Invoke the chain with a new input
# The model will use the provided examples to guide its translation
# result = chain.invoke({"input": "walk"})
# print(result.content)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section offers practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, vectorstores, and handling different output formats.

SOURCE: https://python.langchain.com/docs/how_to/vectorstores/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing various features within LangChain, such as using tools, memory, output parsers, and managing installations.

SOURCE: https://python.langchain.com/docs/how_to/passthrough/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
    
    *   [How-to guides](/docs/how_to/)
    *   [How to use tools in a chain](/docs/how_to/tools_chain/)
    *   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
    *   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
    *   [How to use example selectors](/docs/how_to/example_selectors/)
    *   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
    *   [How to invoke runnables in parallel](/docs/how_to/parallel/)
    *   [How to stream chat model responses](/docs/how_to/chat_streaming/)
    *   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
    *   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
    *   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
    *   [How to do tool/function calling](/docs/how_to/function_calling/)
    *   [How to install LangChain packages](/docs/how_to/installation/)
    *   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
    *   [How to use few shot examples](/docs/how_to/few_shot_examples/)
    *   [How to run custom functions](/docs/how_to/functions/)
    *   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
    *   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
    *   [How to route between sub-chains](/docs/how_to/routing/)
    *   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on specific functionalities and implementation details within LangChain, such as using tools, vectorstores, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_similarity/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: Lists practical guides for implementing specific functionalities within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/summarize_refine/

LANGUAGE: python
CODE:
```
# How-to guides available:
# - How to use tools in a chain
# - How to use a vectorstore as a retriever
# - How to add memory to chatbots
# - How to use example selectors
# - How to add a semantic layer over graph database
# - How to invoke runnables in parallel
# - How to stream chat model responses
# - How to add default invocation args to a Runnable
# - How to add retrieval to chatbots
# - How to use few shot examples in chat models
# - How to do tool/function calling
# - How to install LangChain packages
# - How to add examples to the prompt for query analysis
# - How to use few shot examples
# - How to run custom functions
# - How to use output parsers to parse an LLM response into structured format
# - How to handle cases where no queries are generated
# - How to route between sub-chains
# - How to return structured data from a model
```

----------------------------------------

TITLE: Install LangChain Packages
DESCRIPTION: This guide provides instructions on how to install LangChain and its related packages. It covers using pip for installation and managing dependencies for different LangChain components.

SOURCE: https://python.langchain.com/docs/how_to/binding/

LANGUAGE: python
CODE:
```
# Install the core LangChain library
pip install langchain

# Install specific integrations, e.g., for OpenAI models
pip install langchain-openai

# Install for vectorstores, e.g., FAISS
pip install langchain-community

# Install for specific tools or frameworks
# pip install langchain-aws
# pip install langchain-google-genai
# pip install langchain-anthropic

# To install all community integrations (use with caution):
# pip install "langchain[all]"

# To install specific optional dependencies:
# pip install langchain-core
# pip install langchain-text-splitters

# Example: Installing LangChain with OpenAI and FAISS support
# pip install langchain langchain-openai langchain-community

# Verify installation
# python -c "import langchain; print(langchain.__version__)"
```

----------------------------------------

TITLE: Netmind Chat Integration
DESCRIPTION: Guides users on getting started with Netmind chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatNetmind

# Example usage (requires Netmind API key)
# llm = ChatNetmind(api_key="YOUR_NETMIND_API_KEY")
# response = llm.invoke("What are the benefits of mindfulness?")
# print(response)
```

----------------------------------------

TITLE: How to Install LangChain Packages
DESCRIPTION: This guide provides instructions on installing LangChain and its related packages. It covers using pip to install the core library and optional dependencies for specific integrations.

SOURCE: https://python.langchain.com/docs/how_to/tool_artifacts/

LANGUAGE: bash
CODE:
```
pip install langchain
# For specific integrations, e.g., OpenAI:
# pip install langchain-openai

```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide explains the general concept and implementation of few-shot learning in LangChain. It covers how to structure prompts with examples to guide the language model's responses, improving accuracy and relevance for specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate

# Define a prompt template with few-shot examples
prompt = PromptTemplate(
    input_variables=["query"],
    template=[
        "Q: What is the capital of France?",
        "A: Paris",
        "Q: What is the capital of Germany?",
        "A: Berlin",
        "Q: {query}",
        "A:"
    ].join("\n")
)

# Example usage
# formatted_prompt = prompt.format(query="What is the capital of Spain?")
# print(formatted_prompt)
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This how-to guide covers the general usage of few-shot examples in LangChain. Providing examples can significantly improve the accuracy and relevance of model outputs.

SOURCE: https://python.langchain.com/docs/tutorials/sql_qa/

LANGUAGE: python
CODE:
```
print("How-to: How to use few shot examples")
# Further implementation details would follow here.
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section lists various how-to guides for LangChain, covering practical implementation details for common tasks and features.

SOURCE: https://python.langchain.com/docs/how_to/qa_sources/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on how to perform specific tasks within LangChain. It covers using tools in chains, vectorstores as retrievers, adding memory to chatbots, using example selectors, adding semantic layers, invoking runnables in parallel, streaming chat responses, adding default invocation args, adding retrieval to chatbots, using few-shot examples, function calling, package installation, query analysis, routing, structured output, and summarization through parallelization.

SOURCE: https://python.langchain.com/docs/concepts/chat_models/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: Volc Engine Maas Chat Integration
DESCRIPTION: Provides a guide on getting started with Volc Engine Maas chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatVolcEngineMaas

# Example usage (requires Volc Engine credentials)
# llm = ChatVolcEngineMaas(secret_id="YOUR_SECRET_ID", secret_key="YOUR_SECRET_KEY")
# response = llm.invoke("Write a short story about a robot.")
# print(response)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain. Includes topics like using tools, vectorstores, memory, example selectors, parallel execution, streaming, and function calling.

SOURCE: https://python.langchain.com/docs/concepts/structured_outputs/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
  - How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: Few-Shot Prompting Setup
DESCRIPTION: Constructs a few-shot prompt using `ChatPromptTemplate` and a list of example messages (`AIMessage`, `HumanMessage`, `ToolMessage`). This prompt guides the language model on how to correctly use tools for mathematical operations, especially concerning the order of operations.

SOURCE: https://python.langchain.com/docs/how_to/tools_few_shot/

LANGUAGE: python
CODE:
```
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

examples = [
    HumanMessage(
        "What's the product of 317253 and 128472 plus four", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "Multiply", "args": {"x": 317253, "y": 128472}, "id": "1"}
        ],
    ),
    ToolMessage("16505054784", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "Add", "args": {"x": 16505054784, "y": 4}, "id": "2"}],
    ),
    ToolMessage("16505054788", tool_call_id="2"),
    AIMessage(
        "The product of 317253 and 128472 plus four is 16505054788",
        name="example_assistant",
    ),
]
system = """You are bad at math but are an expert at using a calculator. Use past tool usage as an example of how to correctly use the tools."""

few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: Offers practical guides on implementing specific features and functionalities within LangChain applications, including tool usage, memory, and streaming.

SOURCE: https://python.langchain.com/docs/how_to/tools_few_shot/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: Few-Shot Examples in LangChain
DESCRIPTION: Explains how to incorporate few-shot examples into prompts to guide language model outputs. This includes creating formatters, constructing example sets, and utilizing example selectors like `SemanticSimilarityExampleSelector` for dynamic example retrieval.

SOURCE: https://context7_llms

LANGUAGE: python
CODE:
```
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Example usage:
# examples = [
#     {"question": "Q: What is the capital of France?", "answer": "A: Paris"},
#     {"question": "Q: What is the capital of Germany?", "answer": "A: Berlin"}
# ]
# example_prompt = PromptTemplate(input_variables=["question", "answer"], template="{question}\n{answer}")
# selector = SemanticSimilarityExampleSelector.from_examples(
#     examples,
#     OpenAIEmbeddings(),
#     Chroma,
#     k=1
# )
# few_shot_prompt = FewShotPromptTemplate(
#     example_selector=selector,
#     example_prompt=example_prompt,
#     prefix="Here are some examples:",
#     suffix="Question: {input}\nAnswer: "
# )
```

----------------------------------------

TITLE: LangChain How-to Guide - Select Examples by Similarity
DESCRIPTION: This guide details how to select examples for few-shot prompting based on similarity. It involves embedding a set of examples and then finding the most similar ones to a given input query, which helps in providing relevant context to the LLM.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_similarity/

LANGUAGE: python
CODE:
```
# Conceptual example of selecting examples by similarity:
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# examples = [
#     {"input": "...', "output": "..."},
#     # ... more examples
# ]

# # Embed examples and create a vector store
# embeddings = OpenAIEmbeddings()
# example_selector = ... # Logic to select examples based on similarity

# # Create a FewShotPromptTemplate using the selector
# example_prompt = PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}")
# prompt = FewShotPromptTemplate(
#     example_selector=example_selector,
#     example_prompt=example_prompt,
#     prefix="Given the following examples:",
#     suffix="Input: {input}\nOutput:",
#     input_variables=["input"],
# )

# # Use the prompt with an LLM
# # formatted_prompt = prompt.format(input="New input query")

```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as adding memory, using tools, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_ngram/

LANGUAGE: APIDOC
CODE:
```
LangChain How-to Guides:
  - Installation: How to install LangChain packages.
  - Tools: Using tools in a LangChain chain.
  - Memory: Adding memory to chatbots.
  - Example Selectors: Utilizing example selectors for LLM prompts.
  - Parallel Execution: Invoking runnables in parallel.
  - Streaming: Streaming chat model responses.
  - Function Calling: Implementing tool/function calling.
  - Output Parsers: Parsing LLM responses into structured formats.
  - Routing: Routing between sub-chains.
  - Structured Output: Returning structured data from models.
```

----------------------------------------

TITLE: Getting Started with Yi Chat Models
DESCRIPTION: This guide helps users begin with Yi chat models. It offers detailed documentation for integrating and utilizing Yi models effectively within the LangChain ecosystem.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatYi

# Example usage (assuming Yi API credentials are set)
# chat = ChatYi()
# response = chat.invoke("What are the capabilities of Yi models?")
# print(response.content)
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide explains how to use example selectors in LangChain, which help in dynamically selecting few-shot examples for prompts.

SOURCE: https://python.langchain.com/docs/concepts/rag/

LANGUAGE: python
CODE:
```
print('How-to: How to use example selectors')
```

----------------------------------------

TITLE: Streaming Runnables in LangChain
DESCRIPTION: This guide explains how to stream runnables in LangChain, enabling real-time output for LLM applications. It includes examples and setup instructions for integrating streaming capabilities.

SOURCE: https://python.langchain.com/docs/how_to/streaming/

LANGUAGE: python
CODE:
```
from langchain_core.runnables import Runnable

def stream_runnable(runnable: Runnable, input_data: dict):
    """Streams output from a LangChain runnable."""
    for chunk in runnable.stream(input_data):
        print(chunk)

# Example usage (assuming 'my_runnable' is defined elsewhere)
# stream_runnable(my_runnable, {"input": "Hello, world!"})
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide covers the use of few-shot examples in LangChain to enhance model performance. By providing a few input-output pairs, you can guide the model towards generating more accurate and relevant responses.

SOURCE: https://python.langchain.com/docs/how_to/tool_choice/

LANGUAGE: python
CODE:
```
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define the prompt template with few-shot examples
template = """
Translate English to French:

English: Hello world
French: Bonjour le monde

English: How are you?
French: Comment allez-vous?

English: {english_text}
French: """

prompt = PromptTemplate(template=template, input_variables=["english_text"])

# Initialize the LLM
llm = OpenAI()

# Create an LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Run the chain with a new input
response = llm_chain.run("I love programming.")
print(response)
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide explains how to use example selectors in LangChain, which help in dynamically selecting few-shot examples for prompts based on input.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_memory/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(), Chroma, k=1, 
)

```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide covers the use of example selectors in LangChain, which help in dynamically selecting relevant examples to include in prompts for few-shot learning.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Define few-shot examplesexamples = [
    {"input": "I love dogs.", "output": "I love cats.", "input_language": "English", "output_language": "English"},
    {"input": "I love apples.", "output": "I love oranges.", "input_language": "English", "output_language": "English"},
]

# Create a prompt template for the examplesexample_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Create the FewShotChatMessagePromptTemplateew_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["input", "input_language", "output_language"],
    example_separator="\n\n",
)

# Create the final prompt templateinal_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    few_shot_prompt,
    ("human", "{input}"),
])

# Initialize LLM and create chainllm = ChatOpenAI(model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=final_prompt)

# Example usageesult = chain.invoke({"input": "I love bananas.", "input_language": "English", "output_language": "French"})
print(result)
```

----------------------------------------

TITLE: SemanticSimilarityExampleSelector Setup
DESCRIPTION: Demonstrates setting up SemanticSimilarityExampleSelector to dynamically select relevant few-shot examples based on semantic similarity. It uses OpenAIEmbeddings and Neo4jVector for example selection.

SOURCE: https://python.langchain.com/docs/tutorials/graph/

LANGUAGE: python
CODE:
```
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings

examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: 'Schindler's List'})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Neo4jVector,
    k=5,
    input_keys=["question"]
)
```

----------------------------------------

TITLE: How-to Guides Overview
DESCRIPTION: Lists various how-to guides for implementing specific features and patterns in LangChain, such as using tools, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/concepts/prompt_templates/

LANGUAGE: MARKDOWN
CODE:
```
How-to guides
*   How to use tools in a chain
*   How to use a vectorstore as a retriever
*   How to add memory to chatbots
*   How to use example selectors
*   How to add a semantic layer over graph database
*   How to invoke runnables in parallel
*   How to stream chat model responses
*   How to add default invocation args to a Runnable
*   How to add retrieval to chatbots
*   How to use few shot examples in chat models
*   How to do tool/function calling
*   How to install LangChain packages
*   How to add examples to the prompt for query analysis
*   How to use few shot examples
*   How to run custom functions
*   How to use output parsers to parse an LLM response into structured format
*   How to handle cases where no queries are generated
*   How to route between sub-chains
*   How to return structured data from a model
*   How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain. It details how to integrate tools, manage memory, handle streaming, and utilize various components for building robust LLM applications.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_length_based/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to include examples in prompts specifically for query analysis tasks. By providing sample queries and their desired analysis, you can guide the LLM to perform more accurate and consistent query interpretation.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate

# Define a prompt template with examples for query analysis
query_analysis_prompt = PromptTemplate(
    input_variables=["query"],
    template=[
        "Analyze the following user queries and categorize them.",
        "Query: 'What is the weather in London?' -> Category: Weather",
        "Query: 'Set a timer for 5 minutes' -> Category: Timer",
        "Query: 'Play some jazz music' -> Category: Music",
        "Query: '{query}' -> Category:"
    ].join("\n")
)

# Example usage
# formatted_prompt = query_analysis_prompt.format(query="What is the capital of France?")
# print(formatted_prompt)
```

----------------------------------------

TITLE: Install LangChain Packages
DESCRIPTION: Learn how to install the main LangChain package and various ecosystem packages like langchain-core, langchain-community, langchain-openai, and others. This covers the fundamental setup for using LangChain.

SOURCE: https://context7_llms

LANGUAGE: bash
CODE:
```
pip install langchain
pip install langchain-core langchain-community langchain-openai langchain-experimental langgraph langserve langchain-cli langsmith
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications, such as adding memory, using vectorstores, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/tool_runtime/

LANGUAGE: python
CODE:
```
# How to use tools in a chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

# Initialize LLM and Tool
llm = ChatOpenAI(model="gpt-3.5-turbo")
search = DuckDuckGoSearchRun()

# Define a prompt template that incorporates tool usage instructions
# This is a simplified example; actual tool integration often involves agent frameworks
# or specific chain types designed for tool use.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You can use the search tool."),
    ("user", "{input}"),
])

# A simple chain that might use a tool (conceptual example)
# In practice, tool use is often managed by agents or specific chain types.
# This example shows how a tool *could* be integrated if the LLM is instructed to use it.
# For direct tool invocation within a chain without an agent, you'd typically use LCEL's
# ability to pass tool outputs back into the chain.

# Conceptual chain structure:
# chain = prompt | llm_with_tool_calling_capability | output_parser

# Example of invoking a tool directly (not within a chain context here):
# print(search.run("LangChain documentation"))
print("Guide on using tools in a chain - conceptual example.")

```

LANGUAGE: python
CODE:
```
# How to add memory to chatbots
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.chains import ConversationChain

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Initialize memory
# Using ChatMessageHistory for simple in-memory storage
# For more complex scenarios, consider ConversationBufferMemory, etc.
message_history = ChatMessageHistory()

# Define a prompt template that includes memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You have a memory of the conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
])

# Create a conversation chain with memory
# Note: The exact way to integrate memory depends on the chain type.
# For LCEL, you'd typically pass the history as part of the input dict.

# Example using ConversationChain (a higher-level abstraction):
# conversation = ConversationChain(llm=llm, memory=ChatMessageHistory(), verbose=True)
# print(conversation.predict(input="Hi there!"))
# print(conversation.predict(input="I'm doing well, thank you!"))

# Example using LCEL with explicit history passing:
chain_with_memory = (
    {
        "history": RunnablePassthrough(),
        "input": lambda x: x["input"],
    }
    | prompt
    | llm
)

# To use this, you'd manage the history externally:
# history_input = {"history": message_history.messages, "input": "Hello!"}
# response = chain_with_memory.invoke(history_input)
# message_history.add_user_message(history_input["input"])
# message_history.add_ai_message(response.content)

print("Guide on adding memory to chatbots - conceptual example.")

```

LANGUAGE: python
CODE:
```
# How to stream chat model responses
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond concisely."),
    ("user", "Tell me a short story about a brave knight.")
])

# Create the chain
chain = prompt | llm

# Stream the response
print("Streaming response...")
for chunk in chain.stream({}):
    print(chunk.content, end="", flush=True)
print("\nStreaming complete.")

```

LANGUAGE: python
CODE:
```
# How to use output parsers to parse an LLM response into structured format
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the desired output structure using Pydantic
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Extract information about a person."),
    ("user", "Tell me about John Doe, who is 30 years old and lives in New York.")
])

# Create the chain with JsonOutputParser
# Note: For Pydantic models, use PydanticOutputParser
# json_chain = prompt | llm | JsonOutputParser()

# Using PydanticOutputParser for structured output
from langchain_core.output_parsers import PydanticOutputParser
pydantic_parser = PydanticOutputParser(pydantic_object=Person)

pydantic_chain = prompt | llm | pydantic_parser

# Run the chain
# response = pydantic_chain.invoke({})
# print(response)
print("Guide on using output parsers - conceptual example.")

```

LANGUAGE: python
CODE:
```
# How to return structured data from a model
# This is often achieved using output parsers, particularly PydanticOutputParser.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the desired output structure using Pydantic
class ProductInfo(BaseModel):
    product_name: str = Field(description="Name of the product")
    price: float = Field(description="Price of the product")
    in_stock: bool = Field(description="Whether the product is in stock")

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that extracts product information."),
    ("user", "Extract details for the 'SuperWidget' which costs $19.99 and is in stock.")
])

# Create the chain with PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=ProductInfo)

structured_chain = prompt | llm | output_parser

# Run the chain
# product_data = structured_chain.invoke({})
# print(product_data)
print("Guide on returning structured data - conceptual example.")

```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide covers the general usage of few-shot examples in LangChain, explaining how to provide examples to improve model performance on various tasks. It emphasizes the importance of well-crafted examples for better results.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Define example entries
examples = [
    {
        "input": "The weather is sunny.",
        "output": "positive"
    },
    {
        "input": "I am feeling sad today.",
        "output": "negative"
    }
]

# Define the template for each example
example_template = "Input: {input}\nOutput: {output}"

# Create the prompt template for examples
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)

# Define the overall prompt template
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify the sentiment of the following sentences.",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
    example_separator="\n"
)

# Example usage
input_text = "This is a fantastic day!"
formatted_prompt = prompt.format(input=input_text)
print(formatted_prompt)

```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on using specific LangChain features and functionalities, such as tools, vectorstores, memory, example selectors, parallel execution, streaming, function calling, and output parsing.

SOURCE: https://python.langchain.com/docs/how_to/prompts_composition/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: Guides for building various applications using LangChain, from simple LLM interactions to complex retrieval-augmented generation systems.

SOURCE: https://python.langchain.com/docs/how_to/migrate_agent/

LANGUAGE: python
CODE:
```
# Build a Question Answering application over a Graph Database
# ... (details omitted for brevity)

# Build a simple LLM application with chat models and prompt templates
# ... (details omitted for brevity)

# Build a Chatbot
# ... (details omitted for brevity)

# Build a Retrieval Augmented Generation (RAG) App: Part 2
# ... (details omitted for brevity)

# Build an Extraction Chain
# ... (details omitted for brevity)

# Build an Agent
# ... (details omitted for brevity)

# Tagging
# ... (details omitted for brevity)

# Build a Retrieval Augmented Generation (RAG) App: Part 1
# ... (details omitted for brevity)

# Build a semantic search engine
# ... (details omitted for brevity)

# Build a Question/Answering system over SQL data
# ... (details omitted for brevity)

# Summarize Text
# ... (details omitted for brevity)
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide explains how to use example selectors in LangChain, which help in dynamically selecting few-shot examples to include in prompts based on the input.

SOURCE: https://python.langchain.com/docs/how_to/extraction_parse/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define few-shot examples
examples = [
    {"input": "I am happy", "output": "I am sad"},
    {"input": "I am fast", "output": "I am slow"},
]

# Create a prompt template for the examples
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Create a FewShotChatMessagePromptTemplate
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Create the final prompt template
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides antonyms."),
    few_shot_prompt,
    ("human", "{input}")
])

# Initialize the language model and create the chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = final_prompt | llm | StrOutputParser()

# Example usage
response = chain.invoke({"input": "I am big"})
print(response)
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This how-to guide explains how to provide few-shot examples to chat models in LangChain to improve their performance and guide their responses.

SOURCE: https://python.langchain.com/docs/tutorials/rag/

LANGUAGE: python
CODE:
```
# This is a placeholder for the actual code in the how-to guide.
# The guide focuses on the concepts and steps involved in using few-shot examples.
# For specific code examples, please refer to the official LangChain documentation.

# Example conceptual steps:
# 1. Prepare a list of example input/output pairs
# 2. Format these examples into the chat model's message history
# 3. Send the examples along with the user's query to the chat model
```

----------------------------------------

TITLE: How to Install LangChain Packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for your Python environment. It covers the use of pip for managing LangChain dependencies.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: bash
CODE:
```
pip install langchain
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials demonstrating various use cases and functionalities within LangChain, from simple LLM applications to complex agents and RAG systems.

SOURCE: https://python.langchain.com/docs/how_to/installation/

LANGUAGE: python
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for your project.

SOURCE: https://python.langchain.com/docs/tutorials/extraction/

LANGUAGE: bash
CODE:
```
pip install langchain
# For specific integrations, e.g., OpenAI:
# pip install langchain-openai
# For LangChain community packages:
# pip install langchain-community
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides guides on how to implement specific features and functionalities within LangChain. It covers topics like using tools, memory, vectorstores, streaming, and more.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_office_file/

LANGUAGE: markdown
CODE:
```
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features within LangChain. It details how to use tools, vectorstores, memory, example selectors, parallel execution, streaming, function calling, and more.

SOURCE: https://python.langchain.com/docs/concepts/retrievers/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: How to Use Example Selectors
DESCRIPTION: This guide explains how to use example selectors in LangChain to dynamically select relevant examples for prompts. This is particularly useful in few-shot learning scenarios to provide the most pertinent examples to the language model.

SOURCE: https://python.langchain.com/docs/how_to/custom_tools/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Example dataexamples = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "What is the capital of Spain?", "answer": "Madrid"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"}
]

# Create embeddings and a vector store for examplesembeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts([ex['question'] for ex in examples], embeddings, metadatas=[{'answer': ex['answer']} for ex in examples])

# Create an example selectorexample_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Select top 2 similar examples
    input_keys=["question"], # Key to use for similarity search
    example_keys=["answer"], # Key to use for example output
)

# Define a prompt template for the examplesexample_prompt = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}\nAnswer: {answer}",
)

# Create a FewShotPromptTemplateew_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Here are some examples of question answering:",
    suffix="Question: {question}\nAnswer: "
)

# Use the prompt templateormatted_prompt = few_shot_prompt.format(question="What is the capital of Canada?")
print(formatted_prompt)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_web/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  - How to use tools in a chain
  - How to use a vectorstore as a retriever
  - How to add memory to chatbots
  - How to use example selectors
  - How to add a semantic layer over graph database
  - How to invoke runnables in parallel
  - How to stream chat model responses
  - How to add default invocation args to a Runnable
  - How to add retrieval to chatbots
  - How to use few shot examples in chat models
  - How to do tool/function calling
  - How to install LangChain packages
  - How to add examples to the prompt for query analysis
  - How to use few shot examples
  - How to run custom functions
  - How to use output parsers to parse an LLM response into structured format
  - How to handle cases where no queries are generated
  - How to route between sub-chains
  - How to return structured data from a model
  - How to summarize text through parallelization
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide explains how to effectively use few-shot examples in LangChain prompts to guide LLM behavior and improve response quality for specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_memory/

LANGUAGE: python
CODE:
```
from langchain.prompts import FewShotPromptTemplate

example_formatter_template = "Input: {input}\nOutput: {output}"

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_formatter_template,
)

```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide details how to use example selectors in LangChain, which dynamically select relevant examples to include in the prompt based on the input. This is useful for managing prompt length and improving efficiency when dealing with a large number of examples.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptExample
from langchain_core.example_selectors import LengthBasedExampleSelector

# Define example messages
example_1 = PromptExample(input="What is the capital of France?", output="Paris")
example_2 = PromptExample(input="What is the capital of Germany?", output="Berlin")
example_3 = PromptExample(input="What is the capital of Spain?", output="Madrid")

examples = [example_1, example_2, example_3]

# Create an example selector
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ]),
    maxLength=100  # Maximum length of the prompt
)

# Create a FewShotPromptTemplate using the selector
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ]),
    input_variables=["input"],
    prefix="Here are some examples:",
    suffix="\nQuestion: {input}\nAnswer:"
)

# Example usage
# formatted_prompt = few_shot_prompt.format(input="What is the capital of Italy?")
# print(formatted_prompt)
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: This guide details how to use example selectors in LangChain, which dynamically select relevant examples to include in the prompt based on the input. This is useful for managing prompt length and improving efficiency when dealing with a large number of examples.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptExample
from langchain_core.example_selectors import LengthBasedExampleSelector

# Define example messages
example_1 = PromptExample(input="What is the capital of France?", output="Paris")
example_2 = PromptExample(input="What is the capital of Germany?", output="Berlin")
example_3 = PromptExample(input="What is the capital of Spain?", output="Madrid")

examples = [example_1, example_2, example_3]

# Create an example selector
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ]),
    maxLength=100  # Maximum length of the prompt
)

# Create a FewShotPromptTemplate using the selector
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ]),
    input_variables=["input"],
    prefix="Here are some examples:",
    suffix="\nQuestion: {input}\nAnswer:"
)

# Example usage
# formatted_prompt = few_shot_prompt.format(input="What is the capital of Italy?")
# print(formatted_prompt)
```

----------------------------------------

TITLE: Example Selectors
DESCRIPTION: This guide explains example selectors in LangChain, which are used to dynamically select few-shot examples for prompts. It covers different strategies for choosing the most relevant examples based on the input.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_attach/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Assume 'examples' is a list of dictionaries, each with 'input' and 'output' keys
# Assume 'example_selector' is configured to use a vector store and embeddings

# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     examples, OpenAIEmbeddings(), Chroma, k=2
# )

# prompt_template = PromptTemplate(input_variables=["input"], template="...")
# few_shot_prompt = FewShotPromptTemplate(example_selector=example_selector, ...)

```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section lists various how-to guides for LangChain, covering practical implementation details for common tasks and features.

SOURCE: https://python.langchain.com/docs/how_to/sql_csv/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/binding/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for your project. It covers different installation methods and common dependencies.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_memory/

LANGUAGE: bash
CODE:
```
pip install langchain

```

----------------------------------------

TITLE: Getting Started with xAI Chat Models
DESCRIPTION: This page helps users get started with xAI chat models. It provides detailed documentation on how to integrate and use xAI models within the LangChain framework.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatXai

# Example usage (assuming you have xAI credentials configured)
# chat = ChatXai()
# response = chat.invoke("Hello, how are you?")
# print(response.content)
```

----------------------------------------

TITLE: Tutorials Overview
DESCRIPTION: Lists various tutorials available for building LLM applications with LangChain, covering topics from basic LLM applications to agents and RAG.

SOURCE: https://python.langchain.com/docs/concepts/prompt_templates/

LANGUAGE: MARKDOWN
CODE:
```
Tutorials
*   [Build a Question Answering application over a Graph Database](/docs/tutorials/graph/)
*   [Build a simple LLM application with chat models and prompt templates](/docs/tutorials/llm_chain/)
*   [Build a Chatbot](/docs/tutorials/chatbot/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 2](/docs/tutorials/qa_chat_history/)
*   [Build an Extraction Chain](/docs/tutorials/extraction/)
*   [Build an Agent](/docs/tutorials/agents/)
*   [Tagging](/docs/tutorials/classification/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 1](/docs/tutorials/rag/)
*   [Build a semantic search engine](/docs/tutorials/retrievers/)
*   [Build a Question/Answering system over SQL data](/docs/tutorials/sql_qa/)
*   [Summarize Text](/docs/tutorials/summarization/)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on how to implement specific features and functionalities within LangChain. It covers topics such as using tools, vectorstores, memory, parallel execution, streaming, and more.

SOURCE: https://python.langchain.com/docs/concepts/vectorstores/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for your project.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_tools/

LANGUAGE: bash
CODE:
```
pip install langchain
```

LANGUAGE: bash
CODE:
```
pip install langchain-community
```

LANGUAGE: bash
CODE:
```
pip install langchain-openai
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Details various how-to guides for implementing specific functionalities within LangChain, including tool usage, vectorstore retrieval, memory, prompt examples, and more.

SOURCE: https://python.langchain.com/docs/how_to/graph_semantic/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: Install Langchain Packages
DESCRIPTION: Installs the necessary Langchain and Langchain-Community packages for the project. Also includes optional setup for LangSmith tracing.

SOURCE: https://python.langchain.com/docs/how_to/tools_prompting/

LANGUAGE: python
CODE:
```
%pip install --upgrade --quiet langchain langchain-community

import getpass
import os
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: Details various how-to guides for implementing specific functionalities within LangChain applications, such as using tools, vectorstores, memory, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/tool_configure/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: LangChain Installation
DESCRIPTION: This section details how to install LangChain packages. It covers the general installation process and provides links to more specific guides for different functionalities and versions.

SOURCE: https://python.langchain.com/docs/how_to/installation/

LANGUAGE: python
CODE:
```
pip install langchain
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This how-to guide explains how to include examples within prompts for query analysis. This can help the language model better understand and process queries.

SOURCE: https://python.langchain.com/docs/tutorials/sql_qa/

LANGUAGE: python
CODE:
```
print("How-to: How to add examples to the prompt for query analysis")
# Further implementation details would follow here.
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: Offers practical guides on implementing specific functionalities within LangChain, such as using tools, managing memory, handling output parsing, and optimizing chains.

SOURCE: https://python.langchain.com/docs/concepts/callbacks/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to incorporate few-shot examples into prompts for query analysis, improving the model's understanding and performance.

SOURCE: https://python.langchain.com/docs/how_to/sql_query_checking/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import OpenAI

# Assume llm is an initialized language model instance
# Assume db_chain is a pre-configured chain for database interaction

# Example prompt with few-shot examples for query analysis
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that analyzes SQL queries."),
    ("human", "Analyze the following query: {query}\nExamples:\nQuery: SELECT * FROM users WHERE age > 30\nAnalysis: This query selects all columns from the users table for users older than 30.\n\nQuery: SELECT COUNT(*) FROM orders\nAnalysis: This query counts the total number of records in the orders table."),
    ("human", "Analyze the following query: {query}")
])

# Chain to analyze the query with few-shot examples
query_analysis_chain = prompt | llm | StrOutputParser()

# Example usage:
query_to_analyze = "SELECT name FROM products WHERE price < 100"
analysis_result = query_analysis_chain.invoke({"query": query_to_analyze})
print(f"Analysis: {analysis_result}")
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, output parsers, and parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/chat_models_universal_init/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  - How to use tools in a chain
    URL: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever
    URL: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots
    URL: /docs/how_to/chatbots_memory/
  - How to use example selectors
    URL: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database
    URL: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel
    URL: /docs/how_to/parallel/
  - How to stream chat model responses
    URL: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable
    URL: /docs/how_to/binding/
  - How to add retrieval to chatbots
    URL: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models
    URL: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling
    URL: /docs/how_to/function_calling/
  - How to install LangChain packages
    URL: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis
    URL: /docs/how_to/query_few_shot/
  - How to use few shot examples
    URL: /docs/how_to/few_shot_examples/
  - How to run custom functions
    URL: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format
    URL: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated
    URL: /docs/how_to/query_no_queries/
  - How to route between sub-chains
    URL: /docs/how_to/routing/
  - How to return structured data from a model
    URL: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: Lists various tutorials available for building different types of applications with LangChain, including QA, LLM chains, chatbots, agents, RAG, and more.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_mmr/

LANGUAGE: text
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: A list of how-to guides for implementing specific features and functionalities in LangChain.

SOURCE: https://python.langchain.com/docs/tutorials/summarization/

LANGUAGE: APIDOC
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This how-to guide provides instructions on how to install the necessary LangChain packages and their dependencies.

SOURCE: https://python.langchain.com/docs/tutorials/rag/

LANGUAGE: python
CODE:
```
# This is a placeholder for the actual code in the how-to guide.
# The guide focuses on the installation process.
# For specific commands, please refer to the official LangChain documentation.

# Example installation command:
# pip install langchain

# Installation of specific integrations might require additional packages:
# pip install langchain-openai
# pip install langchain-community
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: This section lists various tutorials available for LangChain, covering a wide range of LLM application development tasks.

SOURCE: https://python.langchain.com/docs/how_to/convert_runnable_to_tool/

LANGUAGE: python
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: Provides a collection of how-to guides for common tasks and advanced features in LangChain, such as using tools, memory, vectorstores, and handling specific output formats.

SOURCE: https://python.langchain.com/docs/concepts/few_shot_prompting/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  - How to use tools in a chain
  - How to use a vectorstore as a retriever
  - How to add memory to chatbots
  - How to use example selectors
  - How to add a semantic layer over graph database
  - How to invoke runnables in parallel
  - How to stream chat model responses
  - How to add default invocation args to a Runnable
  - How to add retrieval to chatbots
  - How to use few shot examples in chat models
  - How to do tool/function calling
  - How to install LangChain packages
  - How to add examples to the prompt for query analysis
  - How to use few shot examples
  - How to run custom functions
  - How to use output parsers to parse an LLM response into structured format
  - How to handle cases where no queries are generated
  - How to route between sub-chains
  - How to return structured data from a model
  - How to summarize text through parallelization

These guides offer practical solutions and best practices for integrating LangChain into your projects.
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: Lists various tutorials available for building different types of LLM applications with LangChain.

SOURCE: https://python.langchain.com/docs/how_to/summarize_refine/

LANGUAGE: python
CODE:
```
# Tutorials available:
# - Build a Question Answering application over a Graph Database
# - Build a simple LLM application with chat models and prompt templates
# - Build a Chatbot
# - Build a Retrieval Augmented Generation (RAG) App: Part 2
# - Build an Extraction Chain
# - Build an Agent
# - Tagging
# - Build a Retrieval Augmented Generation (RAG) App: Part 1
# - Build a semantic search engine
# - Build a Question/Answering system over SQL data
# - Summarize Text
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on specific functionalities and techniques within LangChain, such as using tools, vectorstores, memory, example selectors, and handling specific scenarios.

SOURCE: https://python.langchain.com/docs/how_to/query_high_cardinality/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to include few

SOURCE: https://python.langchain.com/docs/how_to/extraction_parse/



----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to include examples in prompts for query analysis, improving the accuracy of the analysis.

SOURCE: https://python.langchain.com/docs/tutorials/extraction/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Example prompt with few-shot examples
# prompt = PromptTemplate(
#     input_variables=["query"],
#     template="Analyze the following queries and categorize them:\n\nQuery: "What is the weather like?"\nCategory: Weather\n\nQuery: "Set a timer for 5 minutes"\nCategory: Timer\n\nQuery: {query}\nCategory: "
# )

# Initialize LLM
# llm = OpenAI()

# Create the chain
# chain = prompt | llm

# Example usage:
# response = chain.invoke({"query": "What is the capital of France?"})
# print(response)
```

----------------------------------------

TITLE: Pydantic Output Parsing with LCEL
DESCRIPTION: Illustrates parsing an LLM's response into a Pydantic object using PydanticOutputParser and LCEL. This example shows a chain that takes a query and returns a structured joke object with 'setup' and 'punchline' fields.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define a Pydantic model for the output
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")
    
# Assuming 'chain' is a pre-configured LangChain LCEL chain
# chain = prompt | model | PydanticOutputParser(pydantic_object=Joke)

# Example usage:
# print(list(chain.stream({"query": "Tell me a joke." })))

```

----------------------------------------

TITLE: Install LangChain and OpenAI
DESCRIPTION: Installs the necessary LangChain and OpenAI libraries, and prompts for the OpenAI API key if not already set.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

LANGUAGE: python
CODE:
```
%pip install -qU langchain langchain-openai langchain-chromaimport osfrom getpass import getpassif "OPENAI_API_KEY" not in os.environ:    os.environ["OPENAI_API_KEY"] = getpass()
```

----------------------------------------

TITLE: Improving Extraction with Reference Examples
DESCRIPTION: This guide explains how to structure example inputs and outputs for extraction tasks in LangChain. It details incorporating these examples into prompts to enhance extraction performance.

SOURCE: https://context7_llms

LANGUAGE: python
CODE:
```
# Conceptual example of structuring examples for LangChain's tool calling API
# examples = [
#     {"input": "Text 1", "output": "Extracted Data 1"},
#     {"input": "Text 2", "output": "Extracted Data 2"}
# ]
# prompt = f"Extract data from the following text: {text}\nExamples:\n{examples}"
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide covers the general approach to using few-shot examples in LangChain to enhance model performance.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_tools/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "apple", "output": "fruit"},
    {"input": "banana", "output": "fruit"},
    {"input": "carrot", "output": "vegetable"}
]

example_prompt = PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}")

final_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    prefix="Classify the input.",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to include examples within prompts for query analysis tasks. Providing examples helps the LLM understand the desired output format and context.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_memory/

LANGUAGE: python
CODE:
```
from langchain.prompts import PromptTemplate

example = "Input: What is the capital of France? Output: Paris"

prompt = PromptTemplate(
    input_variables=["input"],
    template=f"{example}\nInput: {{input}}\nOutput:"
)

```

----------------------------------------

TITLE: How to Use Few Shot Examples in Chat Models
DESCRIPTION: This guide covers the implementation of few-shot learning with LangChain chat models. It demonstrates how to provide example interactions within the prompt to guide the model's responses and improve accuracy for specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/custom_tools/

LANGUAGE: python
CODE:
```
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Initialize a chat modelllm = ChatOpenAI()

# Define few-shot examplesmessages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate: 'hello'"),
    AIMessage(content="Bonjour"),
    HumanMessage(content="Translate: 'goodbye'"),
    AIMessage(content="Au revoir"),
    HumanMessage(content="Translate: 'thank you'")
]

# Send the messages including examples to the chat modelesponse = llm(messages)
print(response.content)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, vectorstores, and handling different output formats.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_attach/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/installation/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Define Prompt Template with Examples
DESCRIPTION: Creates a chat prompt template that includes system instructions and a placeholder for few-shot examples to guide LLM extraction. It specifies how to handle missing information by returning null.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked "
            "to extract, return null for the attribute's value.",
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)
```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages and their dependencies using pip.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: bash
CODE:
```
# Install the core LangChain library
pip install langchain

# Install specific integrations, e.g., for OpenAI
pip install langchain-openai

# Install for vector stores, e.g., Chroma
pip install langchain-chroma

# Install for specific tools, e.g., DuckDuckGo
pip install duckduckgo-search

# Install for experimental features
pip install langchain-experimental
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as adding memory, using tools, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_json/

LANGUAGE: python
CODE:
```
How to use tools in a chain: /docs/how_to/tools_chain/
How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
How to add memory to chatbots: /docs/how_to/chatbots_memory/
How to use example selectors: /docs/how_to/example_selectors/
How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
How to invoke runnables in parallel: /docs/how_to/parallel/
How to stream chat model responses: /docs/how_to/chat_streaming/
How to add default invocation args to a Runnable: /docs/how_to/binding/
How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
How to do tool/function calling: /docs/how_to/function_calling/
How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
How to use few shot examples: /docs/how_to/few_shot_examples/
How to run custom functions: /docs/how_to/functions/
How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
How to route between sub-chains: /docs/how_to/routing/
How to return structured data from a model: /docs/how_to/structured_output/
How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: Install Main LangChain Package
DESCRIPTION: Installs the core LangChain package, serving as a starting point for using the framework. This package acts as a foundation, but additional dependencies for specific integrations are not included by default.

SOURCE: https://python.langchain.com/docs/how_to/installation/

LANGUAGE: bash
CODE:
```
pip install langchain
```

LANGUAGE: bash
CODE:
```
conda install langchain -c conda-forge
```

----------------------------------------

TITLE: Test Prompt Formatting
DESCRIPTION: Invokes the previously defined `example_prompt` with the first example from the `examples` list and prints the formatted output. This demonstrates how the prompt template processes the input data.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
print(example_prompt.invoke(examples[0]).to_string())
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide explains how to provide few-shot examples to models in LangChain to improve their performance on specific tasks.

SOURCE: https://python.langchain.com/docs/tutorials/extraction/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Example prompt with few-shot examples
# prompt = PromptTemplate(
#     input_variables=["input"],
#     template="Translate English to French:\nEnglish: Hello\nFrench: Bonjour\nEnglish: {input}\nFrench: "
# )

# Initialize LLM
# llm = OpenAI()

# Create the chain
# chain = prompt | llm

# Example usage:
# response = chain.invoke({"input": "How are you?"})
# print(response)
```

----------------------------------------

TITLE: End-to-End Example Setup
DESCRIPTION: Sets up an in-memory vector store with sample documents and initializes the LangChain components for an agentic RAG system. This includes embeddings, the vector store, and the agent creation.

SOURCE: https://python.langchain.com/docs/integrations/chat/anthropic/

LANGUAGE: python
CODE:
```
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# Set up vector store
embeddings = init_embeddings("openai:text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

document_1 = Document(
    id="1",
    page_content=(
        "To request vacation days, submit a leave request form through the "
        "HR portal. Approval will be sent by email."
    ),
    metadata={
        "category": "HR Policy",
        "doc_title": "Leave Policy",
        "provenance": "Leave Policy - page 1",
    },
)
document_2 = Document(id="2", page_content="Managers will review vacation requests within 3 business days.", metadata={
        "category": "HR Policy",
        "doc_title": "Leave Policy",
        "provenance": "Leave Policy - page 2",
    })
document_3 = Document(
    id="3",
    page_content=(
        "Employees with over 6 months tenure are eligible for 20 paid vacation days "
        "per year."
    ),
    metadata={
        "category": "Benefits Policy",
        "doc_title": "Benefits Guide 2025",
        "provenance": "Benefits Policy - page 1",
    },
)
documents = [document_1, document_2, document_3]
vector_store.add_documents(documents=documents)

# Define tool
async def retrieval_tool(
    query: str,
    category: Literal["HR Policy", "Benefits Policy"],
) -> list[dict]:
    """Access my knowledge base."""
    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("category") == category
    results = vector_store.similarity_search(
        query=query, k=2, filter=_filter_function
    )
    return [
        {
            "type": "search_result",
            "title": doc.metadata["doc_title"],
            "source": doc.metadata["provenance"],
            "citations": {"enabled": True},
            "content": [{"type": "text", "text": doc.page_content}],
        }
        for doc in results
    ]

# Create agent
llm = init_chat_model(
    "anthropic:claude-3-5-haiku-latest",
    betas=["search-results-2025-06-09"],
)
checkpointer = InMemorySaver()
agent = create_react_agent(llm, [retrieval_tool], checkpointer=checkpointer)

# Invoke on a query
config = {"configurable": {"thread_id": "session_1"}}
input_message = {
    "role": "user",
    "content": "How do I request vacation days?",
}

async for step in agent.astream(
    {"messages": [input_message]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials to guide users through building various applications with LangChain, from simple LLM apps to complex agents.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_async/

LANGUAGE: APIDOC
CODE:
```
LangChain Tutorials:
  - Build a Question Answering application over a Graph Database
    URL: /docs/tutorials/graph/
  - Build a simple LLM application with chat models and prompt templates
    URL: /docs/tutorials/llm_chain/
  - Build a Chatbot
    URL: /docs/tutorials/chatbot/
  - Build a Retrieval Augmented Generation (RAG) App: Part 2
    URL: /docs/tutorials/qa_chat_history/
  - Build an Extraction Chain
    URL: /docs/tutorials/extraction/
  - Build an Agent
    URL: /docs/tutorials/agents/
  - Tagging
    URL: /docs/tutorials/classification/
  - Build a Retrieval Augmented Generation (RAG) App: Part 1
    URL: /docs/tutorials/rag/
  - Build a semantic search engine
    URL: /docs/tutorials/retrievers/
  - Build a Question/Answering system over SQL data
    URL: /docs/tutorials/sql_qa/
  - Summarize Text
    URL: /docs/tutorials/summarization/
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides guides on how to perform specific tasks and implement features within LangChain. Topics include using tools, vectorstores, memory, example selectors, parallel execution, streaming, function calling, and more.

SOURCE: https://python.langchain.com/docs/how_to/custom_chat_model/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This how-to guide explains how to include examples within prompts to improve the accuracy and relevance of query analysis performed by LLMs.

SOURCE: https://python.langchain.com/docs/tutorials/rag/

LANGUAGE: python
CODE:
```
# This is a placeholder for the actual code in the how-to guide.
# The guide focuses on the concepts and steps involved in adding examples to prompts for query analysis.
# For specific code examples, please refer to the official LangChain documentation.

# Example conceptual steps:
# 1. Prepare a dataset of queries and their desired analysis or classification
# 2. Construct a prompt that includes these examples before the actual query
# 3. Use an LLM to analyze the query based on the provided examples
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to include few-shot examples within prompts to help language models better understand and analyze user queries.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Define few-shot examples for query analysisexamples = [
    {"query": "What is the weather today?", "analysis": "Weather query"},
    {"query": "Tell me a joke.", "analysis": "Entertainment query"},
    {"query": "What is the capital of France?", "analysis": "Factual query"},
]

# Create a prompt template for the examplesexample_prompt = ChatPromptTemplate.from_messages([
    ("human", "Analyze this query: {query}"),
    ("ai", "Analysis: {analysis}"),
])

# Create the FewShotChatMessagePromptTemplateew_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["query", "analysis"],
    example_separator="\n\n",
)

# Create the final prompt templateinal_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a query analysis assistant."),
    few_shot_prompt,
    ("human", "Analyze this query: {query}"),
])

# Initialize LLM and create chainllm = ChatOpenAI(model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=final_prompt)

# Example usageesult = chain.invoke({"query": "What is the latest stock price?"})
print(result)
```

----------------------------------------

TITLE: How to Install LangChain Packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for your Python environment. It covers the use of pip for installing the core library and optional dependencies.

SOURCE: https://python.langchain.com/docs/how_to/custom_tools/

LANGUAGE: bash
CODE:
```
# Install the core LangChain library
pip install langchain

# Install specific integrations (e.g., for OpenAI)
pip install langchain-openai

# Install for specific features like document loaders
pip install langchain-community

# Install all common integrations
pip install "langchain[all]"
```

----------------------------------------

TITLE: How to Use Few Shot Examples in Chat Models
DESCRIPTION: This guide explains how to effectively use few-shot learning with LangChain chat models. By providing example interactions within the prompt, you can guide the model to produce desired outputs for specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/custom_tools/

LANGUAGE: python
CODE:
```
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Initialize a chat modelllm = ChatOpenAI()

# Define few-shot examples for a classification taskmessages = [
    SystemMessage(content="Classify the following movie review as Positive, Negative, or Neutral."),
    HumanMessage(content="This movie was fantastic, I loved every minute!"),
    AIMessage(content="Positive"),
    HumanMessage(content="It was okay, nothing special."),
    AIMessage(content="Neutral"),
    HumanMessage(content="A complete waste of time and money.")
]

# Send the messages including examples to the chat modelesponse = llm(messages)
print(response.content)
```

----------------------------------------

TITLE: Setup and Environment Configuration
DESCRIPTION: Installs necessary LangChain packages and configures the OpenAI API key from environment variables or user input. It also initializes a ChatOpenAI model.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_memory/

LANGUAGE: python
CODE:
```
%pip install --upgrade --quiet langchain langchain-openai langgraph
import getpass
import os
if not os.environ.get("OPENAI_API_KEY"): os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
```

----------------------------------------

TITLE: Add Examples to Prompt for Tuning
DESCRIPTION: Shows how to add example input questions and their corresponding desired structured outputs (Search objects) to the prompt to improve the query generation results. This helps the LLM decompose questions more effectively.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
examples = []

question = "What's chat langchain, is it a langchain template?"
query = Search(
    query="What is chat langchain and is it a langchain template?",
    sub_queries=["What is chat langchain", "What is a langchain template"],
)
examples.append({"input": question, "tool_calls": [query]})

question = "How to build multi-agent system and stream intermediate steps from it"
query = Search(
    query="How to build multi-agent system and stream intermediate steps from it",
    sub_queries=[
        "How to build multi-agent system",
        "How to stream intermediate steps from multi-agent system",
        "How to stream intermediate steps",
    ],
)
examples.append({"input": question, "tool_calls": [query]})
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials to guide users through building various LLM applications with LangChain.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_runtime/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  - Build a Question Answering application over a Graph Database
  - Build a simple LLM application with chat models and prompt templates
  - Build a Chatbot
  - Build a Retrieval Augmented Generation (RAG) App: Part 2
  - Build an Extraction Chain
  - Build an Agent
  - Tagging
  - Build a Retrieval Augmented Generation (RAG) App: Part 1
  - Build a semantic search engine
  - Build a Question/Answering system over SQL data
  - Summarize Text
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for implementing specific functionalities within LangChain, including tool usage, vectorstore retrieval, memory, parallel execution, streaming, and more.

SOURCE: https://python.langchain.com/docs/how_to/agent_executor/

LANGUAGE: python
CODE:
```
How-to guides
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, including tool usage, vectorstores, memory, parallel execution, streaming, and more.

SOURCE: https://python.langchain.com/docs/how_to/parent_document_retriever/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: This section provides a list of tutorials for building various applications with LangChain, covering topics from basic LLM applications to complex agents and retrieval systems.

SOURCE: https://python.langchain.com/docs/how_to/vectorstores/

LANGUAGE: python
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_csv/

LANGUAGE: python
CODE:
```
How-to guides:
  How to use tools in a chain: /docs/how_to/tools_chain/
  How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  How to add memory to chatbots: /docs/how_to/chatbots_memory/
  How to use example selectors: /docs/how_to/example_selectors/
  How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  How to invoke runnables in parallel: /docs/how_to/parallel/
  How to stream chat model responses: /docs/how_to/chat_streaming/
  How to add default invocation args to a Runnable: /docs/how_to/binding/
  How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  How to do tool/function calling: /docs/how_to/function_calling/
  How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  How to use few shot examples: /docs/how_to/few_shot_examples/
  How to run custom functions: /docs/how_to/functions/
  How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  How to route between sub-chains: /docs/how_to/routing/
  How to return structured data from a model: /docs/how_to/structured_output/
  How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to incorporate few-shot examples into prompts for analyzing queries. It covers techniques for improving the accuracy and relevance of query analysis by providing the language model with illustrative examples.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Example data
examples = [
    {
        "query": "What is the capital of France?",
        "analysis": "This is a factual question about geography."
    },
    {
        "query": "Write a poem about a cat.",
        "analysis": "This is a creative writing request."
    }
]

# Define the template for each example
example_template = "Query: {query}\nAnalysis: {analysis}"

# Create the prompt template for examples
example_prompt = PromptTemplate(
    input_variables=["query", "analysis"],
    template=example_template
)

# Define the overall prompt template
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Analyze the following queries and categorize them.",
    suffix="Query: {query}\nAnalysis:",
    input_variables=["query"],
    example_separator="\n\n"
)

# Example usage
query = "Tell me a joke."
formatted_prompt = prompt.format(query=query)
print(formatted_prompt)

```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for your Python environment.

SOURCE: https://python.langchain.com/docs/concepts/rag/

LANGUAGE: python
CODE:
```
print('How-to: How to install LangChain packages')
```

----------------------------------------

TITLE: How to add examples to the prompt for query analysis
DESCRIPTION: This guide explains how to include examples within prompts for query analysis tasks. Providing examples helps the model better understand the structure and intent of the queries it needs to analyze.

SOURCE: https://python.langchain.com/docs/how_to/tool_choice/

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt with examples for query analysis
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a query analysis assistant. Classify the user query."),
    ("human", "Classify the following query: 'What is the weather in London?' -> WEATHER"),
    ("human", "Classify the following query: 'What is the stock price of AAPL?' -> STOCK"),
    ("human", "Classify the following query: '{query}' -> "),
])

# Initialize the chat model
llm = ChatOpenAI()

# Create a chain
chain = prompt | llm

# Invoke the chain with a new query
response = chain.invoke({"query": "Tell me about the latest news."})
print(response.content)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides guides on specific functionalities and implementation details within LangChain, such as using tools, managing memory, handling streaming, and implementing function calling.

SOURCE: https://python.langchain.com/docs/how_to/serialization/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for implementing specific functionalities within LangChain. It covers essential topics such as using tools, memory, prompt engineering techniques, and deployment considerations.

SOURCE: https://python.langchain.com/docs/how_to/sequence/

LANGUAGE: markdown
CODE:
```
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
*   [How to summarize text through parallelization](/docs/how_to/summarize_map_reduce/)
```

----------------------------------------

TITLE: Adding Examples to Prompts for Query Analysis
DESCRIPTION: This section covers how to guide an LLM to generate queries by incorporating examples into few-shot prompts. It's useful when fine-tuning an LLM for query generation or when needing to structure query output.

SOURCE: https://context7_llms

LANGUAGE: python
CODE:
```
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Example of a prompt with few-shot examples for query generation
# prompt_template = """
# Given the following context, generate a search query:
# Context: {context}
# Query:
# """
# prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
# llm = OpenAI(temperature=0)
# query_generation_chain = LLMChain(llm=llm, prompt=prompt)

# The actual implementation would involve providing examples within the prompt
# to guide the LLM's output format and content.
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as tool usage, memory, streaming, and structured output.

SOURCE: https://python.langchain.com/docs/how_to/tools_prompting/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Ollama Setup and Model Management
DESCRIPTION: Instructions for setting up a local Ollama instance, downloading models, and managing them via the command line.

SOURCE: https://python.langchain.com/docs/integrations/chat/ollama/

LANGUAGE: bash
CODE:
```
brew install ollama
brew services start ollama
ollama pull <name-of-model>
ollama pull gpt-oss:20b
ollama list
ollama run <name-of-model>
ollama help
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides guides on how to implement specific features and functionalities within LangChain. Topics include using tools, vectorstores, memory, example selectors, parallel execution, streaming, function calling, and more.

SOURCE: https://python.langchain.com/docs/how_to/merge_message_runs/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
    
    *   [How-to guides](/docs/how_to/)
    *   [How to use tools in a chain](/docs/how_to/tools_chain/)
    *   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
    *   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
    *   [How to use example selectors](/docs/how_to/example_selectors/)
    *   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
    *   [How to invoke runnables in parallel](/docs/how_to/parallel/)
    *   [How to stream chat model responses](/docs/how_to/chat_streaming/)
    *   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
    *   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
    *   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
    *   [How to do tool/function calling](/docs/how_to/function_calling/)
    *   [How to install LangChain packages](/docs/how_to/installation/)
    *   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
    *   [How to use few shot examples](/docs/how_to/few_shot_examples/)
    *   [How to run custom functions](/docs/how_to/functions/)
    *   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
    *   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
    *   [How to route between sub-chains](/docs/how_to/routing/)
    *   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tool_artifacts/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_runtime/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  - How to use tools in a chain
  - How to use a vectorstore as a retriever
  - How to add memory to chatbots
  - How to use example selectors
  - How to add a semantic layer over graph database
  - How to invoke runnables in parallel
  - How to stream chat model responses
  - How to add default invocation args to a Runnable
  - How to add retrieval to chatbots
  - How to use few shot examples in chat models
  - How to do tool/function calling
  - How to install LangChain packages
  - How to add examples to the prompt for query analysis
  - How to use few shot examples
  - How to run custom functions
  - How to use output parsers to parse an LLM response into structured format
  - How to handle cases where no queries are generated
  - How to route between sub-chains
  - How to return structured data from a model
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for implementing specific features and functionalities within LangChain, including tool usage, vectorstore retrievers, memory, streaming, and function calling.

SOURCE: https://python.langchain.com/docs/how_to/prompts_partial/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific functionalities within LangChain, such as using tools, memory, vectorstores, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/llm_caching/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on implementing specific functionalities within LangChain, such as using tools, memory, vectorstores, and handling structured output.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_langsmith/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as adding memory, using tools, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/concepts/output_parsers/

LANGUAGE: python
CODE:
```
# How-to guides:
# - How to use tools in a chain
# - How to use a vectorstore as a retriever
# - How to add memory to chatbots
# - How to use example selectors
# - How to add a semantic layer over graph database
# - How to invoke runnables in parallel
# - How to stream chat model responses
# - How to add default invocation args to a Runnable
# - How to add retrieval to chatbots
# - How to use few shot examples in chat models
# - How to do tool/function calling
# - How to install LangChain packages
# - How to add examples to the prompt for query analysis
# - How to use few shot examples
# - How to run custom functions
# - How to use output parsers to parse an LLM response into structured format
# - How to handle cases where no queries are generated
# - How to route between sub-chains
# - How to return structured data from a model
# - How to summarize text through parallelization
```

----------------------------------------

TITLE: Example Selectors
DESCRIPTION: Demonstrates various strategies for selecting examples, including using LangSmith datasets, selecting by length, maximal marginal relevance (MMR), n-gram overlap, and similarity.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import LengthBasedExampleSelector

# Example for selecting examples by length
# example_selector = LengthBasedExampleSelector(
#     examples=my_examples, 
#     example_prompt=example_prompt, 
#     max_length=100
# )

# Example for selecting examples by similarity
# example_selector = SemanticSimilarityExampleSelector(
#     vectorstore=Chroma.from_documents(my_examples, OpenAIEmbeddings()),
#     k=2,
#     example_prompt=example_prompt,
#     input_keys=["input"],
# )

```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages and their dependencies using pip.

SOURCE: https://python.langchain.com/docs/how_to/extraction_parse/

LANGUAGE: bash
CODE:
```
# Install the core LangChain library
pip install langchain

# Install specific integrations (e.g., OpenAI)
pip install langchain-openai

# Install other useful packages like LangChain Community
pip install langchain-community

# For specific features like agents or memory, you might need additional installs:
pip install langchain-core

# Example: Install everything needed for OpenAI and basic chains
pip install langchain langchain-openai langchain-core
```

----------------------------------------

TITLE: Install Dependencies
DESCRIPTION: Installs the necessary LangChain packages for Chroma vector store and OpenAI embeddings. This is a prerequisite for the subsequent code examples.

SOURCE: https://python.langchain.com/docs/how_to/multi_vector/

LANGUAGE: bash
CODE:
```
%pip install --upgrade --quiet  langchain-chroma langchain langchain-openai > /dev/null
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: Explains how to leverage few-shot examples to improve the performance of language models on specific tasks. This involves providing a few input-output pairs to guide the model.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Define the examples
examples = [
    {
        "input": "The weather is sunny and warm.",
        "output": "positive"
    },
    {
        "input": "I am feeling very sad today.",
        "output": "negative"
    },
    {
        "input": "This is an amazing experience!",
        "output": "positive"
    },
    {
        "input": "I hate this.",
        "output": "negative"
    }
]

# Create a prompt template for the examples
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Create the FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Classify the sentiment of the following text.",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
    example_separator="\n\n"
)

# Format the prompt with a new input
formatted_prompt = few_shot_prompt.format(input="I love this product!")

# print(formatted_prompt)
# You can then pass this formatted_prompt to an LLM
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide explains the usage of few-shot examples in LangChain to improve the performance of language models on specific tasks.

SOURCE: https://python.langchain.com/docs/concepts/rag/

LANGUAGE: python
CODE:
```
print('How-to: How to use few shot examples')
```

----------------------------------------

TITLE: Example Selectors
DESCRIPTION: Demonstrates various strategies for selecting examples, including using LangSmith datasets, selecting by length, maximal marginal relevance (MMR), n-gram overlap, and similarity.

SOURCE: https://python.langchain.com/docs/how_to/tool_artifacts/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import LengthBasedExampleSelector

# Example for selecting examples by length
# example_selector = LengthBasedExampleSelector(
#     examples=my_examples, 
#     example_prompt=example_prompt, 
#     max_length=100
# )

# Example for selecting examples by similarity
# example_selector = SemanticSimilarityExampleSelector(
#     vectorstore=Chroma.from_documents(my_examples, OpenAIEmbeddings()),
#     k=2,
#     example_prompt=example_prompt,
#     input_keys=["input"],
# )

```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for LangChain, detailing specific functionalities and implementation patterns for common tasks.

SOURCE: https://python.langchain.com/docs/how_to/qa_citations/

LANGUAGE: markdown
CODE:
```
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: This section lists various tutorials available for LangChain, covering a wide range of applications from simple LLM setups to complex agents and RAG systems.

SOURCE: https://python.langchain.com/docs/how_to/ensemble_retriever/

LANGUAGE: python
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: LangSmith Setup and API Key Configuration
DESCRIPTION: Sets up the LangSmith environment by configuring the API key and enabling tracing. It also installs necessary libraries like langsmith, langchain-core, langchain, langchain-openai, and langchain-benchmarks.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_langsmith/

LANGUAGE: python
CODE:
```
import getpass
import os

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Set LangSmith API key:\n\n")

os.environ["LANGSMITH_TRACING"] = "true"

# %pip install -qU "langsmith>=0.1.101" "langchain-core>=0.2.34" langchain langchain-openai langchain-benchmarks
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, vectorstores, and parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/message_history/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: How to Use Few Shot Examples in Chat Models
DESCRIPTION: This guide demonstrates how to provide few-shot examples to LangChain chat models to improve their performance on specific tasks. By including example input-output pairs in the prompt, the model can better understand the desired behavior.

SOURCE: https://python.langchain.com/docs/how_to/custom_tools/

LANGUAGE: python
CODE:
```
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Initialize a chat modelllm = ChatOpenAI()

# Define few-shot examples for sentiment analysismessages = [
    SystemMessage(content="You are a sentiment analysis assistant. Classify the sentiment of the following text."),
    HumanMessage(content="I love this new movie!"),
    AIMessage(content="Positive"),
    HumanMessage(content="The service was terrible."),
    AIMessage(content="Negative"),
    HumanMessage(content="It was an okay experience.")
]

# Send the messages including examples to the chat modelesponse = llm(messages)
print(response.content)
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: This section lists various tutorials available for LangChain, covering a wide range of applications from question answering to building agents and chatbots.

SOURCE: https://python.langchain.com/docs/how_to/contextual_compression/

LANGUAGE: markdown
CODE:
```
*   [Build a Question Answering application over a Graph Database](/docs/tutorials/graph/)
*   [Build a simple LLM application with chat models and prompt templates](/docs/tutorials/llm_chain/)
*   [Build a Chatbot](/docs/tutorials/chatbot/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 2](/docs/tutorials/qa_chat_history/)
*   [Build an Extraction Chain](/docs/tutorials/extraction/)
*   [Build an Agent](/docs/tutorials/agents/)
*   [Tagging](/docs/tutorials/classification/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 1](/docs/tutorials/rag/)
*   [Build a semantic search engine](/docs/tutorials/retrievers/)
*   [Build a Question/Answering system over SQL data](/docs/tutorials/sql_qa/)
*   [Summarize Text](/docs/tutorials/summarization/)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, output parsers, and more.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_json/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: A collection of tutorials demonstrating how to build various applications using LangChain, from simple LLM interactions to complex agents and retrieval systems.

SOURCE: https://python.langchain.com/docs/how_to/qa_per_user/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  - Build a Question Answering application over a Graph Database
  - Build a simple LLM application with chat models and prompt templates
  - Build a Chatbot
  - Build a Retrieval Augmented Generation (RAG) App: Part 2
  - Build an Extraction Chain
  - Build an Agent
  - Tagging
  - Build a Retrieval Augmented Generation (RAG) App: Part 1
  - Build a semantic search engine
  - Build a Question/Answering system over SQL data
  - Summarize Text
```

----------------------------------------

TITLE: AI21 Chat Model Integration
DESCRIPTION: This section covers the integration details, model features, setup (credentials and installation), instantiation, invocation, and chaining of AI21 chat models within LangChain. It serves as a comprehensive guide for developers looking to use AI21's capabilities through the LangChain framework.

SOURCE: https://python.langchain.com/docs/integrations/chat/ai21/

LANGUAGE: python
CODE:
```
# Conceptual guide for Chat Models
# import ChatModel from langchain_core.chat_models

# How-to guides for Chat Models
# from langchain_ai21 import ChatAI21

# Example Setup (Conceptual)
# Ensure you have the AI21 library installed:
# pip install langchain-ai21

# Set your AI21 API key as an environment variable:
# export AI21_API_KEY='YOUR_API_KEY'

# Example Instantiation
# from langchain_ai21 import ChatAI21
# chat = ChatAI21()

# Example Invocation
# from langchain_core.messages import HumanMessage
# response = chat.invoke([HumanMessage(content="Hello!")])
# print(response.content)

# Example Chaining (Conceptual)
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# 
# prompt = PromptTemplate.from_template("What is the capital of {country}?")
# chain = LLMChain(llm=chat, prompt=prompt)
# print(chain.run("France"))
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This guide demonstrates how to provide few-shot examples to chat models in LangChain. This technique helps improve the model's performance on specific tasks by showing it examples.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_memory/

LANGUAGE: python
CODE:
```
from langchain.prompts import FewShotChatMessagePromptTemplate

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific functionalities within LangChain, such as adding memory to chatbots, using vectorstores as retrievers, and handling tool/function calling.

SOURCE: https://python.langchain.com/docs/how_to/qa_per_user/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain
  - How to use a vectorstore as a retriever
  - How to add memory to chatbots
  - How to use example selectors
  - How to add a semantic layer over graph database
  - How to invoke runnables in parallel
  - How to stream chat model responses
  - How to add default invocation args to a Runnable
  - How to add retrieval to chatbots
  - How to use few shot examples in chat models
  - How to do tool/function calling
  - How to install LangChain packages
  - How to add examples to the prompt for query analysis
  - How to use few shot examples
  - How to run custom functions
  - How to use output parsers to parse an LLM response into structured format
  - How to handle cases where no queries are generated
  - How to route between sub-chains
  - How to return structured data from a model
  - How to summarize text through parallelization
```

----------------------------------------

TITLE: How to install LangChain packages
DESCRIPTION: This guide provides instructions on how to install the necessary LangChain packages for your Python environment. It covers the use of pip for managing dependencies.

SOURCE: https://python.langchain.com/docs/how_to/tool_choice/

LANGUAGE: bash
CODE:
```
pip install langchain
# For specific integrations, you might need to install additional packages:
pip install langchain-openai
pip install langchain-community
# Or install all core packages:
pip install "langchain[all]"
```

----------------------------------------

TITLE: Few-Shot Examples in Chat Models
DESCRIPTION: Demonstrates how to provide few-shot examples to chat models to guide their responses and improve accuracy for specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_fixing/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

# Define few-shot examples
example_1 = HumanMessage("I love dogs.")
example_1_output = AIMessage("I love dogs too! It's great you have a pet.")

example_2 = HumanMessage("The weather is nice today.")
example_2_output = AIMessage("Yes, it is! Perfect for a walk.")

# Create a prompt template with examples
prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant."),
    example_1,
    example_1_output,
    example_2,
    example_2_output,
    HumanMessagePromptTemplate.from_template("{user_input}")
])

# Initialize the chat model
model = ChatOpenAI()

# Create a chain
chain = prompt | model

# Invoke the chain with user input
response = chain.invoke({"user_input": "What do you think about AI?"})
print(response.content)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific functionalities within LangChain. These cover topics like using tools, vectorstores, memory, prompt selectors, parallel execution, streaming, and function calling.

SOURCE: https://python.langchain.com/docs/how_to/indexing/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: How to use few shot examples
DESCRIPTION: This guide provides a comprehensive explanation of how to use few-shot examples in LangChain to improve the performance and accuracy of language model responses.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Define few-shot examplesexamples = [
    {"input": "sea otter", "output": "otter"},
    {"input": "peppermint", "output": "mint"},
    {"input": "cheese", "output": "dairy product"},
]

# Create a prompt template for the examplesexample_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Create the FewShotChatMessagePromptTemplateew_shot_prompt = FewShotChatMessagePromptTemplate(
    examples
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This section details how to leverage few-shot examples when working with chat models in LangChain. It demonstrates how to structure prompts with example conversations to guide the chat model's responses.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# Define example messages
example_messages = [
    HumanMessage(content="Hi, I'm Bob!"),
    SystemMessage(content="Hi Bob! How can I help you today?"),
    HumanMessage(content="Can you tell me about LangChain?"),
    SystemMessage(content="LangChain is a framework for developing applications powered by language models.")
]

# Create the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="examples"),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

# Format the prompt with examples and user input
formatted_chat_prompt = chat_prompt.format_messages(
    examples=example_messages,
    user_input="What is RAG?"
)

# Print the formatted messages
for message in formatted_chat_prompt:
    print(f"{message.type}: {message.content}")

```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/contextual_compression/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This guide explains how to provide few-shot examples to chat models in LangChain to improve their performance on specific tasks. Including examples in the prompt helps the model understand the desired output format and behavior.

SOURCE: https://python.langchain.com/docs/how_to/tool_choice/

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.prompts import SystemMessagePromptTemplate

# Define system message and few-shot examples
system_template = "You are a helpful assistant."
example_human = "Translate English to French: 'I love programming.'"
example_ai = "J'adore la programmation."

# Create the prompt template with few-shot examples
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(example_human),
    AIMessagePromptTemplate.from_template(example_ai),
    HumanMessagePromptTemplate.from_template("Translate English to French: '{input}'"),
])

# Initialize the chat model
llm = ChatOpenAI()

# Create a chain
chain = chat_prompt | llm

# Invoke the chain with a new input
response = chain.invoke({"input": "Hello, how are you?"})
print(response.content)
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: Learn how to provide few-shot examples to chat models in LangChain to improve their performance and guide their responses.

SOURCE: https://python.langchain.com/docs/concepts/rag/

LANGUAGE: python
CODE:
```
print('How-to: How to use few shot examples in chat models')
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as using tools, vectorstores, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/how_to/tool_results_pass_to_model/

LANGUAGE: markdown
CODE:
```
- [How to use tools in a chain](/docs/how_to/tools_chain/)
- [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
- [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
- [How to use example selectors](/docs/how_to/example_selectors/)
- [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
- [How to invoke runnables in parallel](/docs/how_to/parallel/)
- [How to stream chat model responses](/docs/how_to/chat_streaming/)
- [How to add default invocation args to a Runnable](/docs/how_to/binding/)
- [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
- [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
- [How to do tool/function calling](/docs/how_to/function_calling/)
- [How to install LangChain packages](/docs/how_to/installation/)
- [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
- [How to use few shot examples](/docs/how_to/few_shot_examples/)
- [How to run custom functions](/docs/how_to/functions/)
- [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
- [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
- [How to route between sub-chains](/docs/how_to/routing/)
- [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: Setup and Environment Variables
DESCRIPTION: Installs necessary packages and sets up LangSmith environment variables for tracing. It requires the langchain-community and langgraph packages and optionally configures LangSmith API key and tracing.

SOURCE: https://python.langchain.com/docs/tutorials/sql_qa/

LANGUAGE: python
CODE:
```
%%capture --no-stderr
%pip install --upgrade --quiet langchain-community langgraph

# Comment out the below to opt-out of using LangSmith in this notebook. Not required.
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
    os.environ["LANGSMITH_TRACING"] = "true"
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: Lists various tutorials available for LangChain, covering different use cases and functionalities such as building QA applications, chatbots, agents, RAG, and more.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_constructor/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  - Build a Question Answering application over a Graph Database: /docs/tutorials/graph/
  - Build a simple LLM application with chat models and prompt templates: /docs/tutorials/llm_chain/
  - Build a Chatbot: /docs/tutorials/chatbot/
  - Build a Retrieval Augmented Generation (RAG) App: Part 2: /docs/tutorials/qa_chat_history/
  - Build an Extraction Chain: /docs/tutorials/extraction/
  - Build an Agent: /docs/tutorials/agents/
  - Tagging: /docs/tutorials/classification/
  - Build a Retrieval Augmented Generation (RAG) App: Part 1: /docs/tutorials/rag/
  - Build a semantic search engine: /docs/tutorials/retrievers/
  - Build a Question/Answering system over SQL data: /docs/tutorials/sql_qa/
  - Summarize Text: /docs/tutorials/summarization/
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on how to implement specific features and functionalities within LangChain, such as using tools, memory, output parsers, and more.

SOURCE: https://python.langchain.com/docs/how_to/ensemble_retriever/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Few-Shot Example Prompting
DESCRIPTION: Illustrates the concept of few-shot example prompting for AI agents, where past input-output examples are used to guide the model's behavior. This is a common technique for episodic memory implementation.

SOURCE: https://langchain-ai.github.io/langgraph/concepts/memory/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

# Define the examples
examples = [
    {
        "input": "Translate to French: Hello world",
        "output": "Bonjour le monde"
    },
    {
        "input": "Translate to French: How are you?",
        "output": "Comment allez-vous?"
    }
]

# Create a prompt template for the examples
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Create the FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate English sentences to French.",
    suffix="Translate English to French: {input}",
    input_variables=["input"],
    example_separator="\n\n",
)

# Initialize the language model
llm = ChatOpenAI()

# Format the prompt and invoke the LLM
formatted_prompt = prompt.format(input="What is your name?")
response = llm.invoke(formatted_prompt)

print(response.content)
```

----------------------------------------

TITLE: Install Dependencies
DESCRIPTION: Installs the necessary Python packages for the guide, including langchain-community, lxml, faiss-cpu, and langchain-openai.

SOURCE: https://python.langchain.com/docs/how_to/extraction_long_text/

LANGUAGE: python
CODE:
```
%pip install -qU langchain-community lxml faiss-cpu langchain-openai
```

----------------------------------------

TITLE: Maximal Marginal Relevance (MMR) Example Selector Setup
DESCRIPTION: Demonstrates how to initialize and use the MaxMarginalRelevanceExampleSelector. This selector picks examples that are semantically similar to the input while also ensuring diversity among the selected examples. It requires a list of examples, an embedding model (OpenAIEmbeddings), and a vector store (FAISS).

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_mmr/

LANGUAGE: python
CODE:
```
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # The number of examples to produce.
    k=2,
)

mmr_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

# Input is a feeling, so should select the happy/sad example as the first one
print(mmr_prompt.format(adjective="worried"))
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_async/

LANGUAGE: APIDOC
CODE:
```
LangChain How-to Guides:
  - How to use tools in a chain
    URL: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever
    URL: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots
    URL: /docs/how_to/chatbots_memory/
  - How to use example selectors
    URL: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database
    URL: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel
    URL: /docs/how_to/parallel/
  - How to stream chat model responses
    URL: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable
    URL: /docs/how_to/binding/
  - How to add retrieval to chatbots
    URL: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models
    URL: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling
    URL: /docs/how_to/function_calling/
  - How to install LangChain packages
    URL: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis
    URL: /docs/how_to/query_few_shot/
  - How to use few shot examples
    URL: /docs/how_to/few_shot_examples/
  - How to run custom functions
    URL: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format
    URL: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated
    URL: /docs/how_to/query_no_queries/
  - How to route between sub-chains
    URL: /docs/how_to/routing/
  - How to return structured data from a model
    URL: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials covering various LangChain use cases, from building simple LLM applications to complex agents and RAG systems.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_ngram/

LANGUAGE: APIDOC
CODE:
```
LangChain Tutorials:
  - Introduction: Getting started with LangChain.
  - LLM Chain: Building a simple LLM application.
  - Chatbot: Creating a chatbot with LangChain.
  - RAG: Building Retrieval Augmented Generation applications (Part 1 & 2).
  - Agents: Developing agents with LangChain.
  - SQL QA: Question answering over SQL data.
  - Summarization: Summarizing text using LangChain.
  - Semantic Search: Building a semantic search engine.
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This guide explains how to provide few-shot examples to chat models in LangChain to improve their performance on specific tasks.

SOURCE: https://python.langchain.com/docs/tutorials/extraction/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Example messages
# system_message = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
# human_message = HumanMessagePromptTemplate.from_template("User: {input}\nAssistant:")

# Create a chat prompt with few-shot examples
# chat_prompt = ChatPromptTemplate.from_messages([
#     system_message,
#     ("human", "What is the capital of France?"),
#     ("ai", "The capital of France is Paris."),
#     human_message
# ])

# Initialize the chat model
# llm = ChatOpenAI()

# Create the chain
# chain = chat_prompt | llm

# Example usage:
# response = chain.invoke({"input": "What is the capital of Germany?"})
# print(response.content)
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This guide explains how to provide few-shot examples to chat models in LangChain to improve their performance on specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/chatbots_tools/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory

chat = ChatOpenAI()

message_history = ChatMessageHistory()
message_history.add_user_message("Hi")
message_history.add_ai_message("Hello!")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | chat

# result = chain.invoke({"history": message_history.messages, "input": "How are you?"})
```

----------------------------------------

TITLE: Define Few-Shot Examples
DESCRIPTION: Creates a list of dictionaries, where each dictionary represents a question-answer pair for few-shot learning. These examples are used to train or guide the LLM.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """Are follow up questions needed here: Yes.Follow up: How old was Muhammad Ali when he died?Intermediate answer: Muhammad Ali was 74 years old when he died.Follow up: How old was Alan Turing when he died?Intermediate answer: Alan Turing was 41 years old when he died.So the final answer is: Muhammad Ali""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """Are follow up questions needed here: Yes.Follow up: Who was the founder of craigslist?Intermediate answer: Craigslist was founded by Craig Newmark.Follow up: When was Craig Newmark born?Intermediate answer: Craig Newmark was born on December 6, 1952.So the final answer is: December 6, 1952""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """Are follow up questions needed here: Yes.Follow up: Who was the mother of George Washington?Intermediate answer: The mother of George Washington was Mary Ball Washington.Follow up: Who was the father of Mary Ball Washington?Intermediate answer: The father of Mary Ball Washington was Joseph Ball.So the final answer is: Joseph Ball""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """Are follow up questions needed here: Yes.Follow up: Who is the director of Jaws?Intermediate Answer: The director of Jaws is Steven Spielberg.Follow up: Where is Steven Spielberg from?Intermediate Answer: The United States.Follow up: Who is the director of Casino Royale?Intermediate Answer: The director of Casino Royale is Martin Campbell.Follow up: Where is Martin Campbell from?Intermediate Answer: New Zealand.So the final answer is: No""",
    },
]
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: A collection of how-to guides for LangChain, offering practical instructions on implementing specific features and functionalities. These guides cover topics such as tool usage, memory, parallel execution, streaming, and more.

SOURCE: https://python.langchain.com/docs/how_to/sql_large_db/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain
  - How to use a vectorstore as a retriever
  - How to add memory to chatbots
  - How to use example selectors
  - How to add a semantic layer over graph database
  - How to invoke runnables in parallel
  - How to stream chat model responses
  - How to add default invocation args to a Runnable
  - How to add retrieval to chatbots
  - How to use few shot examples in chat models
  - How to do tool/function calling
  - How to install LangChain packages
  - How to add examples to the prompt for query analysis
  - How to use few shot examples
  - How to run custom functions
  - How to use output parsers to parse an LLM response into structured format
  - How to handle cases where no queries are generated
  - How to route between sub-chains
  - How to return structured data from a model
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_similarity/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_constructor/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as using tools, vectorstores, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/concepts/tools/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  /docs/how_to/

Guides cover:
- Installing LangChain packages
- Using tools and chains
- Integrating vectorstores as retrievers
- Adding memory to chatbots
- Implementing function calling and structured output
- Parallel execution and streaming responses
- Prompt engineering techniques (few-shot examples, example selectors)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/migrate_agent/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides guides on how to implement specific features and functionalities within LangChain applications. Topics include using tools, memory, vectorstores, streaming, and more.

SOURCE: https://python.langchain.com/docs/how_to/pydantic_compatibility/

LANGUAGE: markdown
CODE:
```
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features within LangChain, such as using tools, vectorstores, memory, prompt selectors, parallel execution, streaming, and function calling.

SOURCE: https://python.langchain.com/docs/how_to/tool_calling/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Create PromptTemplate Formatter
DESCRIPTION: Configures a PromptTemplate to format few-shot examples into a string. This template takes a 'question' and 'answer' as input.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
```

----------------------------------------

TITLE: Install Langchain Packages
DESCRIPTION: Installs or upgrades the necessary Langchain packages for SQL database interaction and OpenAI integration. Also includes optional setup for LangSmith tracing.

SOURCE: https://python.langchain.com/docs/how_to/sql_large_db/

LANGUAGE: python
CODE:
```
%pip install --upgrade --quiet  langchain langchain-community langchain-openai

# Uncomment the below to use LangSmith. Not required.
# import getpass
# import os
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
# os.environ["LANGSMITH_TRACING"] = "true"
```

----------------------------------------

TITLE: Prompt Engineering Example
DESCRIPTION: Provides an example of constructing and using prompts for LLMs. Effective prompt engineering is crucial for guiding LLM responses and achieving desired outcomes.

SOURCE: https://python.langchain.com/docs/how_to/summarize_refine/

LANGUAGE: python
CODE:
```
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a short joke about {topic}.",
)

formatted_prompt = prompt.format(topic="cats")
print(formatted_prompt)
```

----------------------------------------

TITLE: Using Toolkits
DESCRIPTION: Provides guidance on how to leverage pre-built toolkits within Langchain for various tasks. Toolkits bundle related tools and chains to simplify complex workflows.

SOURCE: https://python.langchain.com/docs/how_to/installation/

LANGUAGE: python
CODE:
```
from langchain.agents import load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
# Load tools for a specific toolkit, e.g., 'python-repl'
tools = load_tools(["python-repl"], llm=llm)
# Further steps to use these tools with an agent...
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This guide demonstrates how to provide few-shot examples directly to chat models to improve their performance on specific tasks by showing desired input-output patterns.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Define few-shot examples as chat messagessystem_message = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human_message1 = HumanMessagePromptTemplate.from_template("Input: What is 2+2?")
ai_message1 = AIMessagePromptTemplate.from_template("Output: 4")
human_message2 = HumanMessagePromptTemplate.from_template("Input: What is 5*3?")
ai_message2 = AIMessagePromptTemplate.from_template("Output: 15")

# Create the final prompt template including exampleschat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message1, ai_message1,
    human_message2, ai_message2,
    ("human", "{input}")
])

# Initialize LLM and create chainllm = ChatOpenAI(model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=chat_prompt)

# Example usageesult = chain.invoke({"input": "What is 10/2?"})
print(result)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on specific functionalities and implementation details within LangChain, such as using tools, vectorstores, memory, parallel execution, and streaming.

SOURCE: https://python.langchain.com/docs/concepts/retrieval/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain How-to Guides - Overview
DESCRIPTION: This section lists various practical guides for using LangChain, covering common tasks and functionalities.

SOURCE: https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  - Trimming Messages: Managing message length and content.
  - Vector Stores: Creating and querying vector databases.
  - Pydantic Compatibility: Ensuring compatibility with Pydantic models.
  - Migrating Chains: Guidance for upgrading from older chain implementations.
  - Migrating Memory: Adapting memory components for new versions or frameworks like LangGraph.

```

----------------------------------------

TITLE: openGauss VectorStore Integration
DESCRIPTION: This notebook covers how to get started with the openGauss VectorStore.

SOURCE: https://python.langchain.com/docs/integrations/vectorstores/

LANGUAGE: python
CODE:
```
from langchain_community.vectorstores import OpenGaussVectorStore

# Example usage (assuming openGauss connection)
# vector_store = OpenGaussVectorStore(table_name="my_table")
# results = vector_store.similarity_search("query text")
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides for common LangChain tasks, including how to trim messages and how to create and query vector stores.

SOURCE: https://python.langchain.com/docs/how_to/passthrough/

LANGUAGE: python
CODE:
```
## How-to Guide Topics:
- How to trim messages
- How to create and query vector stores
- How to pass through arguments from one step to the next
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tool_runtime/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain.

SOURCE: https://python.langchain.com/docs/how_to/MultiQueryRetriever/

LANGUAGE: markdown
CODE:
```
*   [How-to guides](/docs/how_to/)
    
    *   [How-to guides](/docs/how_to/)
    *   [How to use tools in a chain](/docs/how_to/tools_chain/)
    *   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
    *   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
    *   [How to use example selectors](/docs/how_to/example_selectors/)
    *   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
    *   [How to invoke runnables in parallel](/docs/how_to/parallel/)
    *   [How to stream chat model responses](/docs/how_to/chat_streaming/)
    *   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
    *   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
    *   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
    *   [How to do tool/function calling](/docs/how_to/function_calling/)
    *   [How to install LangChain packages](/docs/how_to/installation/)
    *   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
    *   [How to use few shot examples](/docs/how_to/few_shot_examples/)
    *   [How to run custom functions](/docs/how_to/functions/)
    *   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
    *   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
    *   [How to route between sub-chains](/docs/how_to/routing/)
    *   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: SemanticSimilarityExampleSelector Initialization and Usage
DESCRIPTION: Shows how to initialize a SemanticSimilarityExampleSelector using examples, an embedding model (OpenAIEmbeddings), and a vector store (Chroma), and then use it to select the most similar example to a given input question.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Assuming 'examples' is defined elsewhere
# examples = [...] 

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1,
)

# Select the most similar example to the input.
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})

print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: A collection of tutorials demonstrating how to build various LLM-powered applications with LangChain.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_web/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  - Build a Question Answering application over a Graph Database
  - Build a simple LLM application with chat models and prompt templates
  - Build a Chatbot
  - Build a Retrieval Augmented Generation (RAG) App: Part 2
  - Build an Extraction Chain
  - Build an Agent
  - Tagging
  - Build a Retrieval Augmented Generation (RAG) App: Part 1
  - Build a semantic search engine
  - Build a Question/Answering system over SQL data
  - Summarize Text
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for implementing specific features and functionalities within LangChain. It covers essential aspects for building robust LLM applications.

SOURCE: https://python.langchain.com/docs/how_to/configure/

LANGUAGE: markdown
CODE:
```
*   [How to use tools in a chain](/docs/how_to/tools_chain/)
*   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
*   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
*   [How to use example selectors](/docs/how_to/example_selectors/)
*   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
*   [How to invoke runnables in parallel](/docs/how_to/parallel/)
*   [How to stream chat model responses](/docs/how_to/chat_streaming/)
*   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
*   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
*   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
*   [How to do tool/function calling](/docs/how_to/function_calling/)
*   [How to install LangChain packages](/docs/how_to/installation/)
*   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
*   [How to use few shot examples](/docs/how_to/few_shot_examples/)
*   [How to run custom functions](/docs/how_to/functions/)
*   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
*   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
*   [How to route between sub-chains](/docs/how_to/routing/)
*   [How to return structured data from a model](/docs/how_to/structured_output/)
```

----------------------------------------

TITLE: Toolkit Usage Example
DESCRIPTION: Illustrates the general pattern for using LangChain toolkits by initializing a toolkit and retrieving its list of available tools.

SOURCE: https://python.langchain.com/docs/how_to/tools_builtin/

LANGUAGE: python
CODE:
```
# Initialize a toolkit
toolkit = ExampleTookit(...)
# Get list of tools
tools = toolkit.get_tools()
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/qa_sources/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain.

SOURCE: https://python.langchain.com/docs/versions/migrating_chains/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
  - How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain.

SOURCE: https://python.langchain.com/docs/how_to/multimodal_inputs/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  - How to use tools in a chain
    Description: Explains the integration and usage of tools within LangChain chains.
    Path: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever
    Description: Details on configuring and using vectorstores for retrieval.
    Path: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots
    Description: Covers methods for incorporating memory into chatbot applications.
    Path: /docs/how_to/chatbots_memory/
  - How to use example selectors
    Description: Guide on utilizing example selectors for prompt engineering.
    Path: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database
    Description: Steps for building semantic layers on top of graph databases.
    Path: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel
    Description: Demonstrates parallel execution of LangChain runnables.
    Path: /docs/how_to/parallel/
  - How to stream chat model responses
    Description: Explains how to stream responses from chat models.
    Path: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable
    Description: Shows how to set default arguments for Runnable invocations.
    Path: /docs/how_to/binding/
  - How to add retrieval to chatbots
    Description: Guide on integrating retrieval mechanisms into chatbots.
    Path: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models
    Description: Demonstrates using few-shot examples with chat models.
    Path: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling
    Description: Covers the implementation of tool and function calling capabilities.
    Path: /docs/how_to/function_calling/
  - How to install LangChain packages
    Description: Instructions for installing necessary LangChain packages.
    Path: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis
    Description: Techniques for adding examples to prompts for better query analysis.
    Path: /docs/how_to/query_few_shot/
  - How to use few shot examples
    Description: General guide on using few-shot examples in LangChain.
    Path: /docs/how_to/few_shot_examples/
  - How to run custom functions
    Description: Explains how to execute custom functions within LangChain.
    Path: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format
    Description: Details on using output parsers to structure LLM responses.
    Path: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated
    Description: Strategies for managing scenarios with no generated queries.
    Path: /docs/how_to/query_no_queries/
  - How to route between sub-chains
    Description: Guide on implementing routing logic between different sub-chains.
    Path: /docs/how_to/routing/
  - How to return structured data from a model
    Description: Techniques for obtaining structured output from language models.
    Path: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tool_choice/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This how-to guide explains how to provide few-shot examples to chat models in LangChain. This technique helps improve the model's performance on specific tasks.

SOURCE: https://python.langchain.com/docs/tutorials/sql_qa/

LANGUAGE: python
CODE:
```
print("How-to: How to use few shot examples in chat models")
# Further implementation details would follow here.
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on how to implement specific features and functionalities within LangChain. It covers topics like using tools, vectorstores, memory, parallel execution, and more.

SOURCE: https://python.langchain.com/docs/how_to/chat_model_caching/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Setup for Tool-Calling Models
DESCRIPTION: Provides guidance on setting up local LLMs, recommending a model fine-tuned for tool-calling, such as Hermes-2-Pro-Llama-3-8B-GGUF from NousResearch. It also links to further guides on running LLMs locally and using them with RAG.

SOURCE: https://python.langchain.com/docs/integrations/chat/llamacpp/

LANGUAGE: python
CODE:
```
Setup Recommendation:
Use a model fine-tuned for tool-calling.
Recommended Model: Hermes-2-Pro-Llama-3-8B-GGUF (NousResearch)

Further Guides:
- Run LLMs locally: https://python.langchain.com/v0.1/docs/guides/development/local_llms/
- Using local models with RAG: https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications, such as using tools, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/how_to/qa_chat_history_how_to/

LANGUAGE: APIDOC
CODE:
```
How to use tools in a chain:
  /docs/how_to/tools_chain/
How to use a vectorstore as a retriever:
  /docs/how_to/vectorstore_retriever/
How to add memory to chatbots:
  /docs/how_to/chatbots_memory/
How to use example selectors:
  /docs/how_to/example_selectors/
How to add a semantic layer over graph database:
  /docs/how_to/graph_semantic/
How to invoke runnables in parallel:
  /docs/how_to/parallel/
How to stream chat model responses:
  /docs/how_to/chat_streaming/
How to add default invocation args to a Runnable:
  /docs/how_to/binding/
How to add retrieval to chatbots:
  /docs/how_to/chatbots_retrieval/
How to use few shot examples in chat models:
  /docs/how_to/few_shot_examples_chat/
How to do tool/function calling:
  /docs/how_to/function_calling/
How to install LangChain packages:
  /docs/how_to/installation/
How to add examples to the prompt for query analysis:
  /docs/how_to/query_few_shot/
How to use few shot examples:
  /docs/how_to/few_shot_examples/
How to run custom functions:
  /docs/how_to/functions/
How to use output parsers to parse an LLM response into structured format:
  /docs/how_to/output_parser_structured/
How to handle cases where no queries are generated:
  /docs/how_to/query_no_queries/
How to route between sub-chains:
  /docs/how_to/routing/
How to return structured data from a model:
  /docs/how_to/structured_output/
How to summarize text through parallelization:
  /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as adding memory, using tools, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_html/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  /docs/how_to/

  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
  - How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: Set up threads
DESCRIPTION: Instructions on how to properly set up and trace operations involving threads within an application using LangSmith.

SOURCE: https://docs.smith.langchain.com/how_to_guides/tracing/

LANGUAGE: APIDOC
CODE:
```
/observability/how_to_guides/threads
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: Provides a list of tutorials for building various LLM applications with LangChain, covering topics from basic LLM applications to advanced agents and RAG systems.

SOURCE: https://python.langchain.com/docs/concepts/callbacks/

LANGUAGE: python
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, vectorstores, and parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/hybrid/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: Demonstrates how to incorporate few-shot examples when interacting with chat models. This is useful for guiding the model's responses and improving accuracy for specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Example messages for few-shot learning
example_messages = [
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': 'Paris'},
    {'role': 'user', 'content': 'What is the capital of Germany?'},
    {'role': 'assistant', 'content': 'Berlin'}
]

# Create a FewShotChatMessagePromptTemplate
example_prompt = FewShotChatMessagePromptTemplate.from_examples(
    examples=example_messages,
    input_variables=["input"],
    example_separator="\n---",
    suffix="User: {input}\nAssistant:",
    input_type="human"
)

# Create a ChatPromptTemplate with the few-shot examples
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    example_prompt
])

# You can then format this prompt with new user input
formatted_prompt = chat_prompt.format_messages(
    input="What is the capital of Spain?"
)

# The formatted_prompt can be passed to a chat model
# print(formatted_prompt)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/sql_csv/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain. It covers topics such as using tools, memory, vectorstores, parallel execution, streaming, and more.

SOURCE: https://python.langchain.com/docs/concepts/why_langchain/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/serialization/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/sql_query_checking/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This guide explains how to provide few-shot examples directly to chat models within LangChain, helping them understand the desired output format and behavior.

SOURCE: https://python.langchain.com/docs/how_to/extraction_parse/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Define few-shot examples as message lists
example_1 = [
    HumanMessage(content="Translate 'I am a student' to French."),
    AIMessage(content="Je suis un étudiant.")
]

example_2 = [
    HumanMessage(content="Translate 'Hello world' to Spanish."),
    AIMessage(content="Hola mundo.")
]

# Define the system prompt
system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful translation assistant.")

# Combine system prompt, examples, and the final human message
chat_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    *example_1,  # Unpack the first example
    *example_2,  # Unpack the second example
    HumanMessagePromptTemplate.from_template("Translate '{text}' to French.")
])

# Initialize the chat model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create the chain
chain = chat_prompt | llm

# Example usage
response = chain.invoke({"text": "Thank you very much"})
print(response.content)
```

----------------------------------------

TITLE: Setup and Data Loading with Pandas
DESCRIPTION: Installs necessary LangChain and Pandas libraries, sets up environment variables for LangSmith (optional), downloads the Titanic dataset, and loads it into a Pandas DataFrame, displaying its shape and columns.

SOURCE: https://python.langchain.com/docs/how_to/sql_csv/

LANGUAGE: python
CODE:
```
%pip install -qU langchain langchain-openai langchain-community langchain-experimental pandas

# Using LangSmith is recommended but not required. Uncomment below lines to use.
# import os
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

!wget https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv -O titanic.csv

import pandas as pd
df = pd.read_csv("titanic.csv")
print(df.shape)
print(df.columns.tolist())
```

----------------------------------------

TITLE: Set up threads
DESCRIPTION: Instructions on how to properly set up and trace operations involving threads within an application using LangSmith.

SOURCE: https://docs.smith.langchain.com/how_to_guides/tracing/trace_with_langchain/

LANGUAGE: APIDOC
CODE:
```
/observability/how_to_guides/threads
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: Lists various tutorials available for LangChain, covering a wide range of use cases from basic LLM applications to complex agents and retrieval-augmented generation (RAG) systems.

SOURCE: https://python.langchain.com/docs/concepts/few_shot_prompting/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  - Build a Question Answering application over a Graph Database
  - Build a simple LLM application with chat models and prompt templates
  - Build a Chatbot
  - Build a Retrieval Augmented Generation (RAG) App: Part 2
  - Build an Extraction Chain
  - Build an Agent
  - Tagging
  - Build a Retrieval Augmented Generation (RAG) App: Part 1
  - Build a semantic search engine
  - Build a Question/Answering system over SQL data
  - Summarize Text

These tutorials provide practical examples and step-by-step guidance on implementing various LLM-powered features using LangChain.
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, vectorstores, and parallel execution.

SOURCE: https://python.langchain.com/docs/tutorials/chatbot/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain
  - How to use a vectorstore as a retriever
  - How to add memory to chatbots
  - How to use example selectors
  - How to add a semantic layer over graph database
  - How to invoke runnables in parallel
  - How to stream chat model responses
  - How to add default invocation args to a Runnable
  - How to add retrieval to chatbots
  - How to use few shot examples in chat models
  - How to do tool/function calling
  - How to install LangChain packages
  - How to add examples to the prompt for query analysis
  - How to use few shot examples
  - How to run custom functions
  - How to use output parsers to parse an LLM response into structured format
  - How to handle cases where no queries are generated
  - How to route between sub-chains
  - How to return structured data from a model
  - How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, output parsers, and parallel execution.

SOURCE: https://python.langchain.com/docs/concepts/text_llms/

LANGUAGE: python
CODE:
```
How-to guides:
  How to use tools in a chain: /docs/how_to/tools_chain/
  How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  How to add memory to chatbots: /docs/how_to/chatbots_memory/
  How to use example selectors: /docs/how_to/example_selectors/
  How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  How to invoke runnables in parallel: /docs/how_to/parallel/
  How to stream chat model responses: /docs/how_to/chat_streaming/
  How to add default invocation args to a Runnable: /docs/how_to/binding/
  How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  How to do tool/function calling: /docs/how_to/function_calling/
  How to install LangChain packages: /docs/how_to/installation/
  How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  How to use few shot examples: /docs/how_to/few_shot_examples/
  How to run custom functions: /docs/how_to/functions/
  How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  How to route between sub-chains: /docs/how_to/routing/
  How to return structured data from a model: /docs/how_to/structured_output/
  How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials demonstrating how to build various LLM applications with LangChain, including QA, chatbots, RAG, and agents.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_similarity/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  - Build a Question Answering application over a Graph Database: /docs/tutorials/graph/
  - Build a simple LLM application with chat models and prompt templates: /docs/tutorials/llm_chain/
  - Build a Chatbot: /docs/tutorials/chatbot/
  - Build a Retrieval Augmented Generation (RAG) App: Part 2: /docs/tutorials/qa_chat_history/
  - Build an Extraction Chain: /docs/tutorials/extraction/
  - Build an Agent: /docs/tutorials/agents/
  - Tagging: /docs/tutorials/classification/
  - Build a Retrieval Augmented Generation (RAG) App: Part 1: /docs/tutorials/rag/
  - Build a semantic search engine: /docs/tutorials/retrievers/
  - Build a Question/Answering system over SQL data: /docs/tutorials/sql_qa/
  - Summarize Text: /docs/tutorials/summarization/
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/convert_runnable_to_tool/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/tutorials/sql_qa/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: How to use few shot examples in chat models
DESCRIPTION: This guide explains how to incorporate few-shot examples into chat models within LangChain. It covers techniques for providing context and examples to improve model performance and response quality. This is crucial for tasks requiring specific output formats or nuanced understanding.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Example messages
example_messages = [
    ("human", "What is the capital of France?"),
    ("ai", "The capital of France is Paris.")
]

# Create a FewShotChatMessagePromptTemplate
example_prompt = FewShotChatMessagePromptTemplate.from_messages(example_messages)

# Create a ChatPromptTemplate with the few-shot examples
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    example_prompt,
    ("human", "{input}")
])

# Example usage with a chat model (assuming 'chat_model' is initialized)
# formatted_prompt = chat_prompt.format_messages(input="What is the capital of Germany?")
# response = chat_model.invoke(formatted_prompt)
# print(response.content)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_runtime/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, vectorstores, and handling parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/debugging/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain. These cover topics such as using tools, managing memory, integrating vectorstores, and handling streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_custom_events/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/tutorials/rag/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section provides guides on how to implement specific features and functionalities within LangChain. It covers topics like using tools, memory, output parsers, and more.

SOURCE: https://python.langchain.com/docs/how_to/llm_token_usage_tracking/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Running LangChain Models Locally
DESCRIPTION: Instructions on how to configure and run LangChain models using local LLM providers.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors/

LANGUAGE: python
CODE:
```
from langchain_community.llms import Ollama

# Ensure Ollama is installed and a model (e.g., 'llama2') is pulled
# ollama pull llama2

llm = Ollama(model="llama2")

# You can then use this local LLM like any other LangChain LLM
# print(llm.invoke("Why is the sky blue?"))

# Other local LLM integrations include HuggingFaceHub, GPT4All, etc.
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/qa_citations/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Integrate Examples into Prompt
DESCRIPTION: Demonstrates how to integrate a list of formatted example messages into a LangChain prompt using `prompt.partial`. This allows the model to condition its responses on the provided examples, improving output quality for specific tasks.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import MessagesPlaceholder

# Assuming 'prompt' is a pre-defined PromptTemplate and 'example_msgs' is generated by tool_example_to_messages
query_analyzer_with_examples = (
    {"question": RunnablePassthrough()}
    | prompt.partial(examples=example_msgs)
    | structured_llm
)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities in LangChain, such as using tools, memory, output parsers, and parallel execution.

SOURCE: https://python.langchain.com/docs/concepts/tool_calling/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:

- **How to use tools in a chain**: Integrate external tools into LangChain workflows.
- **How to use a vectorstore as a retriever**: Configure vector stores for retrieval.
- **How to add memory to chatbots**: Implement stateful conversations.
- **How to use example selectors**: Manage few-shot examples for prompts.
- **How to add a semantic layer over graph database**: Enhance graph database querying.
- **How to invoke runnables in parallel**: Execute multiple components concurrently.
- **How to stream chat model responses**: Enable real-time streaming of LLM outputs.
- **How to add default invocation args to a Runnable**: Set default arguments for runnables.
- **How to add retrieval to chatbots**: Integrate retrieval mechanisms into chatbots.
- **How to use few shot examples in chat models**: Improve LLM performance with examples.
- **How to do tool/function calling**: Implement LLM function calling.
- **How to install LangChain packages**: Guide to installing LangChain components.
- **How to add examples to the prompt for query analysis**: Enhance query understanding with examples.
- **How to use few shot examples**: General guide to few-shot learning.
- **How to run custom functions**: Execute user-defined functions.
- **How to use output parsers to parse an LLM response into structured format**: Convert LLM output to structured data.
- **How to handle cases where no queries are generated**: Manage scenarios with no query output.
- **How to route between sub-chains**: Implement conditional logic in chains.
- **How to return structured data from a model**: Ensure models output structured formats.
- **How to summarize text through parallelization**: Efficient text summarization using parallel processing.
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on specific functionalities within LangChain, such as using tools, memory, output parsers, and parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_yaml/

LANGUAGE: python
CODE:
```
How to use tools in a chain: /docs/how_to/tools_chain/
How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
How to add memory to chatbots: /docs/how_to/chatbots_memory/
How to use example selectors: /docs/how_to/example_selectors/
How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
How to invoke runnables in parallel: /docs/how_to/parallel/
How to stream chat model responses: /docs/how_to/chat_streaming/
How to add default invocation args to a Runnable: /docs/how_to/binding/
How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
How to do tool/function calling: /docs/how_to/function_calling/
How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
How to use few shot examples: /docs/how_to/few_shot_examples/
How to run custom functions: /docs/how_to/functions/
How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
How to route between sub-chains: /docs/how_to/routing/
How to return structured data from a model: /docs/how_to/structured_output/
How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: Upstage Chat Integration
DESCRIPTION: Covers getting started with Upstage chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_upstage import ChatUpstage

# Example usage (requires Upstage API key)
# llm = ChatUpstage(api_key="YOUR_UPSTAGE_API_KEY")
# response = llm.invoke("What is the capital of Japan?")
# print(response)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/tool_configure/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/graph_constructing/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Trace without setting environment variables
DESCRIPTION: A guide on how to enable LangSmith tracing without explicitly setting environment variables, potentially simplifying setup in certain deployment scenarios.

SOURCE: https://docs.smith.langchain.com/how_to_guides/tracing/

LANGUAGE: APIDOC
CODE:
```
/observability/how_to_guides/trace_without_env_vars
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/sql_prompting/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/query_multiple_queries/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Redis Vector Store Integration
DESCRIPTION: This notebook covers how to get started with the Redis vector store.

SOURCE: https://python.langchain.com/docs/integrations/vectorstores/

LANGUAGE: python
CODE:
```
from langchain_community.vectorstores import Redis

# Example usage (assuming Redis client is configured)
# vector_store = Redis(index_name="my_index")
# results = vector_store.similarity_search("query text")
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_mmr/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: A collection of tutorials demonstrating how to build various applications using LangChain, from simple LLM applications to complex agents and RAG systems.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_string/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  /docs/tutorials/

This section offers practical guides for building different types of applications with LangChain. Each tutorial focuses on a specific use case and provides step-by-step instructions and code examples.

Featured Tutorials:
- Build a Question Answering application over a Graph Database
- Build a simple LLM application with chat models and prompt templates
- Build a Chatbot
- Build a Retrieval Augmented Generation (RAG) App: Part 2
- Build an Extraction Chain
- Build an Agent
- Tagging
- Build a Retrieval Augmented Generation (RAG) App: Part 1
- Build a semantic search engine
- Build a Question/Answering system over SQL data
- Summarize Text

These tutorials are designed to help users quickly get started with LangChain and explore its capabilities.
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/time_weighted_vectorstore/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Few-Shot Prompting for Tool Behavior
DESCRIPTION: Shows how to use few-shot examples to enhance a model's ability to call tools correctly. By including example queries and responses in the prompt, the model learns from demonstrations, improving its performance on tasks like math operations.

SOURCE: https://context7_llms

LANGUAGE: python
CODE:
```
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import tool
from langchain.schema import AIMessage, HumanMessage

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

llm = ChatOpenAI(temperature=0)

# Define few-shot examples
examples = [
    HumanMessage(content="What is 2 + 2?", additional_kwargs={'tool_calls': [{'name': 'add', 'arguments': {'a': 2, 'b': 2}, 'id': 'call_1'}]}),
    AIMessage(content="", additional_kwargs={'tool_calls': [{'name': 'add', 'arguments': {'a': 2, 'b': 2}, 'id': 'call_1'}]}),
    HumanMessage(content="What is 3 * 4?", additional_kwargs={'tool_calls': [{'name': 'multiply', 'arguments': {'a': 3, 'b': 4}, 'id': 'call_2'}]}),
    AIMessage(content="", additional_kwargs={'tool_calls': [{'name': 'multiply', 'arguments': {'a': 3, 'b': 4}, 'id': 'call_2'}]}),
]

# Create a prompt template with examples
# Note: The exact format for few-shot examples in prompts can vary. 
# This is a conceptual representation.
# A more robust approach might involve a dedicated FewShotPromptTemplate.

# Example using a simple list of messages:
# prompt_messages = [
#     SystemMessagePromptTemplate.from_template(
#         "You are a helpful assistant that can use tools."
#     )
# ]
# prompt_messages.extend(examples)
# prompt_messages.append(HumanMessagePromptTemplate.from_template("{input}"))

# chat_prompt = ChatPromptTemplate.from_messages(prompt_messages)

# formatted_prompt = chat_prompt.format_messages(input="What is 5 + 6?")

# response = llm.invoke(formatted_prompt)
# print(response)

```

----------------------------------------

TITLE: Oceanbase Vector Store Integration
DESCRIPTION: This notebook covers how to get started with the Oceanbase vector store.

SOURCE: https://python.langchain.com/docs/integrations/vectorstores/

LANGUAGE: python
CODE:
```
from langchain_community.vectorstores import OceanbaseVectorStore

# Example usage (assuming Oceanbase connection)
# vector_store = OceanbaseVectorStore(table_name="my_table")
# results = vector_store.similarity_search("query text")
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_length_based/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Few-shot Prompting in Chat Models
DESCRIPTION: Demonstrates how to use few-shot examples with chat models in LangChain. This guide explains how to provide example interactions to improve the model's understanding and response generation for specific tasks.

SOURCE: https://python.langchain.com/docs/concepts/few_shot_prompting/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Example of few-shot examples for a chat model
# This is a conceptual example, actual implementation might vary based on the specific use case.

# Define a prompt template that includes few-shot examples
# The examples guide the model on how to respond to user queries.
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    # Few-shot examples:
    ("human", "What is the capital of France?"),
    ("ai", "The capital of France is Paris."),
    ("human", "What is the capital of Japan?"),
    ("ai", "The capital of Japan is Tokyo."),
    # The actual user query:
    ("human", "{question}")
])

# Initialize the chat model
llm = ChatOpenAI()

# Create a chain that combines the prompt and the model
chain = chat_prompt_template | llm | StrOutputParser()

# Invoke the chain with a question
response = chain.invoke({"question": "What is the capital of Germany?"})
print(response)

```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific functionalities within LangChain, such as using tools, managing chat history, and parsing LLM outputs.

SOURCE: https://python.langchain.com/docs/how_to/migrate_agent/

LANGUAGE: python
CODE:
```
# How to use tools in a chain
# ... (details omitted for brevity)

# How to use a vectorstore as a retriever
# ... (details omitted for brevity)

# How to add memory to chatbots
# ... (details omitted for brevity)

# How to use example selectors
# ... (details omitted for brevity)

# How to add a semantic layer over graph database
# ... (details omitted for brevity)

# How to invoke runnables in parallel
# ... (details omitted for brevity)

# How to stream chat model responses
# ... (details omitted for brevity)

# How to add default invocation args to a Runnable
# ... (details omitted for brevity)

# How to add retrieval to chatbots
# ... (details omitted for brevity)

# How to use few shot examples in chat models
# ... (details omitted for brevity)

# How to do tool/function calling
# ... (details omitted for brevity)

# How to install LangChain packages
# ... (details omitted for brevity)

# How to add examples to the prompt for query analysis
# ... (details omitted for brevity)

# How to use few shot examples
# ... (details omitted for brevity)

# How to run custom functions
# ... (details omitted for brevity)

# How to use output parsers to parse an LLM response into structured format
# ... (details omitted for brevity)

# How to handle cases where no queries are generated
# ... (details omitted for brevity)

# How to route between sub-chains
# ... (details omitted for brevity)

# How to return structured data from a model
# ... (details omitted for brevity)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/graph_constructing/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/summarize_stuff/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: Provides a list of tutorials for building various applications with LangChain, such as question answering, chatbots, RAG, and agents.

SOURCE: https://python.langchain.com/docs/how_to/tools_few_shot/

LANGUAGE: markdown
CODE:
```
*   [Build a Question Answering application over a Graph Database](/docs/tutorials/graph/)
*   [Tutorials](/docs/tutorials/)
*   [Build a simple LLM application with chat models and prompt templates](/docs/tutorials/llm_chain/)
*   [Build a Chatbot](/docs/tutorials/chatbot/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 2](/docs/tutorials/qa_chat_history/)
*   [Build an Extraction Chain](/docs/tutorials/extraction/)
*   [Build an Agent](/docs/tutorials/agents/)
*   [Tagging](/docs/tutorials/classification/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 1](/docs/tutorials/rag/)
*   [Build a semantic search engine](/docs/tutorials/retrievers/)
*   [Build a Question/Answering system over SQL data](/docs/tutorials/sql_qa/)
*   [Summarize Text](/docs/tutorials/summarization/)
```

----------------------------------------

TITLE: Outlines Chat Integration
DESCRIPTION: Facilitates getting started with Outlines chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatOutlines

# Example usage (requires Outlines setup)
# llm = ChatOutlines()
# response = llm.invoke("Generate a list of potential blog post titles.")
# print(response)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_async/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/tool_configure/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/chat_models_universal_init/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/runnable_runtime_secrets/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, vectorstores, and parallel execution.

SOURCE: https://python.langchain.com/docs/concepts/text_splitters/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as adding memory, using tools, or streaming responses.

SOURCE: https://python.langchain.com/docs/how_to/streaming/

LANGUAGE: APIDOC
CODE:
```
LangChain How-to Guides:

This section provides practical instructions for implementing various LangChain features.

**Key How-to Guides:**
- **Using Tools in a Chain**: Integrating external tools (e.g., search engines, calculators) into LangChain workflows.
- **Vectorstore as Retriever**: Configuring and using vector stores for efficient retrieval.
- **Adding Memory to Chatbots**: Implementing conversational memory to maintain context.
- **Example Selectors**: Utilizing different strategies for selecting few-shot examples.
- **Semantic Layer over Graph Database**: Adding a semantic understanding layer to graph data.
- **Invoking Runnables in Parallel**: Executing multiple runnables concurrently for performance.
- **Streaming Chat Model Responses**: Handling real-time streaming of LLM outputs.
- **Default Invocation Args**: Setting default arguments for runnable invocations.
- **Adding Retrieval to Chatbots**: Integrating retrieval mechanisms into conversational agents.
- **Few-shot Examples in Chat Models**: Providing examples to guide chat model behavior.
- **Tool/Function Calling**: Enabling LLMs to call external functions or tools.
- **Installation**: Instructions for installing LangChain packages.
- **Query Few-shot**: Using few-shot examples for query analysis.
- **Few-shot Examples**: General usage of few-shot examples.
- **Running Custom Functions**: Executing user-defined functions within LangChain.
- **Output Parsers**: Parsing LLM responses into structured formats.
- **Handling No Queries**: Strategies for managing scenarios where no queries are generated.
- **Routing between Sub-chains**: Implementing logic to direct execution flow between different chains.
- **Structured Output**: Ensuring LLMs return data in a predefined structure.
- **Summarization through Parallelization**: Efficiently summarizing large texts using parallel processing.
```

----------------------------------------

TITLE: SQLiteVec VectorStore Integration
DESCRIPTION: This notebook covers how to get started with the SQLiteVec vector store.

SOURCE: https://python.langchain.com/docs/integrations/vectorstores/

LANGUAGE: python
CODE:
```
from langchain_community.vectorstores import SQLiteVec

# Example usage (assuming SQLite database with SQLiteVec)
# vector_store = SQLiteVec(table_name="my_table")
# results = vector_store.similarity_search("query text")
```

----------------------------------------

TITLE: Retrieval Context Example
DESCRIPTION: Provides an example of retrieved context, including document IDs, metadata (source, start index, section), and page content. This context is used to inform the LLM's response.

SOURCE: https://python.langchain.com/docs/tutorials/rag/

LANGUAGE: python
CODE:
```
{
    "retrieve": {
        "context": [
            {
                "id": "d6cef137-e1e8-4ddc-91dc-b62bd33c6020",
                "metadata": {
                    "source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    "start_index": 39221,
                    "section": "end"
                },
                "page_content": "Finite context length: The restricted context capacity limits the inclusion of historical information, detailed instructions, API call context, and responses. The design of the system has to work with this limited communication bandwidth, while mechanisms like self-reflection to learn from past mistakes would benefit a lot from long or infinite context windows. Although vector stores and retrieval can provide access to a larger knowledge pool, their representation power is not as powerful as full attention.\n\n\nChallenges in long-term planning and task decomposition: Planning over a lengthy history and effectively exploring the solution space remain challenging. LLMs struggle to adjust plans when faced with unexpected errors, making them less robust compared to humans who learn from trial and error."
            },
            {
                "id": "d1834ae1-eb6a-43d7-a023-08dfa5028799",
                "metadata": {
                    "source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    "start_index": 39086,
                    "section": "end"
                },
                "page_content": "}\n]\nChallenges#\nAfter going through key ideas and demos of building LLM-centered agents, I start to see a couple common limitations:"
            },
            {
                "id": "ca7f06e4-2c2e-4788-9a81-2418d82213d9",
                "metadata": {
                    "source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    "start_index": 32942,
                    "section": "end"
                },
                "page_content": "}\n]\nThen after these clarification, the agent moved into the code writing mode with a different system message.\nSystem message:"
            },
            {
                "id": "1fcc2736-30f4-4ef6-90f2-c64af92118cb",
                "metadata": {
                    "source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                    "start_index": 35127,
                    "section": "end"
                },
                "page_content": "\"content\": \"You will get instructions for code to write.\nYou will write a very long answer. Make sure that every detail of the architecture is, in the end, implemented as code.\nMake sure that every detail of the architecture is, in the end, implemented as code.\n\nThink step by step and reason yourself to the right decisions to make sure we get it right.\nYou will first lay out the names of the core classes, functions, methods that will be necessary, as well as a quick comment on their purpose.\n\nThen you will output the content of each file including ALL code.\nEach file must strictly follow a markdown code block format, where the following tokens must be replaced such that\nFILENAME is the lowercase file name including the file extension,\nLANG is the markup code block language for the code\'s language, and CODE is the code:\n\nFILENAME\n```LANG\nCODE\n```\n\nYou will start with the \"entrypoint\" file, then go to the ones that are imported by that file, and so on.\nPlease"
            }
        ]
    }
}
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain. It covers topics like using tools, vectorstores, memory, prompt selectors, and more.

SOURCE: https://python.langchain.com/docs/how_to/semantic-chunker/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tool_stream_events/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Offers practical guides on implementing specific features and functionalities within LangChain applications, such as using tools, memory, streaming, and structured output.

SOURCE: https://python.langchain.com/docs/how_to/tools_error/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_custom/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides Overview
DESCRIPTION: This section outlines various how-to guides for LangChain, demonstrating practical implementation of features such as tools, memory, output parsers, and parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/embed_text/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
How to summarize text through parallelization
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_directory/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/pydantic_compatibility/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_directory/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_attach/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides that explain how to perform specific tasks and implement features within LangChain.

SOURCE: https://python.langchain.com/docs/how_to/graph_constructing/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain, such as using tools, memory, output parsers, and parallel execution.

SOURCE: https://python.langchain.com/docs/how_to/code_splitter/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  - How to use tools in a chain
  - How to use a vectorstore as a retriever
  - How to add memory to chatbots
  - How to use example selectors
  - How to add a semantic layer over graph database
  - How to invoke runnables in parallel
  - How to stream chat model responses
  - How to add default invocation args to a Runnable
  - How to add retrieval to chatbots
  - How to use few shot examples in chat models
  - How to do tool/function calling
  - How to install LangChain packages
  - How to add examples to the prompt for query analysis
  - How to use few shot examples
  - How to run custom functions
  - How to use output parsers to parse an LLM response into structured format
  - How to handle cases where no queries are generated
  - How to route between sub-chains
  - How to return structured data from a model
  - How to summarize text through parallelization

Usage:
  - Implement specific LangChain features by following these guides.
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/sql_prompting/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/MultiQueryRetriever/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/concepts/async/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Trace without setting environment variables
DESCRIPTION: A guide on how to enable LangSmith tracing without explicitly setting environment variables, potentially simplifying setup in certain deployment scenarios.

SOURCE: https://docs.smith.langchain.com/how_to_guides/tracing/trace_with_langchain/

LANGUAGE: APIDOC
CODE:
```
/observability/how_to_guides/trace_without_env_vars
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tools_few_shot/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/structured_output/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: This section provides an overview of various tutorials available for LangChain, covering a wide range of applications from simple LLM interactions to complex agents and retrieval systems.

SOURCE: https://python.langchain.com/docs/how_to/multimodal_prompts/

LANGUAGE: python
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_langsmith/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Example Selectors
DESCRIPTION: Demonstrates various strategies for selecting examples, including using LangSmith datasets, selecting by length, maximal marginal relevance (MMR), n-gram overlap, and similarity.

SOURCE: https://python.langchain.com/docs/how_to/migrate_agent/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import LengthBasedExampleSelector

# Example for selecting examples by length
# example_selector = LengthBasedExampleSelector(
#     examples=my_examples, 
#     example_prompt=example_prompt, 
#     max_length=100
# )

# Example for selecting examples by similarity
# example_selector = SemanticSimilarityExampleSelector(
#     vectorstore=Chroma.from_documents(my_examples, OpenAIEmbeddings()),
#     k=2,
#     example_prompt=example_prompt,
#     input_keys=["input"],
# )

```

----------------------------------------

TITLE: RunPod Chat Model Integration
DESCRIPTION: Get started with RunPod chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatRunPod

# Example usage (requires RunPod API key and endpoint)
# llm = ChatRunPod(api_key="YOUR_RUNPOD_API_KEY", endpoint_url="YOUR_RUNPOD_ENDPOINT_URL")
# response = llm.invoke("Generate a creative story prompt.")
# print(response)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/query_high_cardinality/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/configure/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on performing specific tasks with LangChain, including trimming messages and creating/querying vector stores.

SOURCE: https://python.langchain.com/docs/how_to/sql_prompting/

LANGUAGE: python
CODE:
```
# How-to guide links:
# Trim messages: /docs/how_to/trim_messages/
# Create and query vector stores: /docs/how_to/vectorstores/
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/summarize_refine/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/query_constructing_filters/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: FewShotPromptTemplate Usage
DESCRIPTION: Demonstrates how to create a `FewShotPromptTemplate` using an `example_selector` and `example_prompt`, and then invoke it with an input.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# Assuming example_selector and example_prompt are defined elsewhere
# example_selector = SemanticSimilarityExampleSelector(...)
# example_prompt = PromptTemplate(...)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(
    prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)
```

----------------------------------------

TITLE: Python Langchain LLM Chatbot Example
DESCRIPTION: Provides an example of building a simple chatbot using Langchain LLMs in Python. This snippet demonstrates a conversational flow where the LLM responds to user input. Ensure 'langchain' and 'openai' are installed.

SOURCE: https://python.langchain.com/docs/tutorials/summarization/

LANGUAGE: python
CODE:
```
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize LLM and memory
llm = OpenAI(model_name="text-davinci-003", temperature=0.7)
memory = ConversationBufferMemory()

# Create the conversation chain
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Start the conversation
print("Chatbot: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = conversation.invoke(user_input)
    print(f"Chatbot: {response['response']}")
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications, such as using tools, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/how_to/custom_callbacks/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/tools_human/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: Example Selectors
DESCRIPTION: Demonstrates various strategies for selecting examples, including using LangSmith datasets, selecting by length, maximal marginal relevance (MMR), n-gram overlap, and similarity.

SOURCE: https://python.langchain.com/docs/how_to/query_multiple_queries/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import LengthBasedExampleSelector

# Example for selecting examples by length
# example_selector = LengthBasedExampleSelector(
#     examples=my_examples, 
#     example_prompt=example_prompt, 
#     max_length=100
# )

# Example for selecting examples by similarity
# example_selector = SemanticSimilarityExampleSelector(
#     vectorstore=Chroma.from_documents(my_examples, OpenAIEmbeddings()),
#     k=2,
#     example_prompt=example_prompt,
#     input_keys=["input"],
# )

```

----------------------------------------

TITLE: Symbl.ai Nebula Chat Integration
DESCRIPTION: Covers getting started with Nebula, Symbl.ai's chat integration.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatSymblaiNebula

# Example usage (requires Symbl.ai credentials)
# llm = ChatSymblaiNebula(app_id="YOUR_APP_ID", app_secret="YOUR_APP_SECRET")
# response = llm.invoke("Summarize our last meeting.")
# print(response)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/custom_llm/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: A list of available tutorials for building various LLM applications with LangChain.

SOURCE: https://python.langchain.com/docs/tutorials/summarization/

LANGUAGE: APIDOC
CODE:
```
Build a Question Answering application over a Graph Database
Build a simple LLM application with chat models and prompt templates
Build a Chatbot
Build a Retrieval Augmented Generation (RAG) App: Part 2
Build an Extraction Chain
Build an Agent
Tagging
Build a Retrieval Augmented Generation (RAG) App: Part 1
Build a semantic search engine
Build a Question/Answering system over SQL data
Summarize Text
```

----------------------------------------

TITLE: How to Build an Agent
DESCRIPTION: This guide provides a comprehensive overview of building agents in LangChain, which are systems that use an LLM to decide which actions to take and in what order. It covers agent types, tools, and execution.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_structured/

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

@tool
def get_current_weather(city: str) -> str:
    """Get the current weather in a given city."""
    return f"The weather in {city} is sunny and 25 degrees Celsius."

llm = ChatOpenAI()

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the get_current_weather tool."),
    ("human", "{input}"),
    # ... other message types ...
])

# Create the agent and executor
agent = create_tool_calling_agent(llm, [get_current_weather], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_current_weather])

# Invoke the agent
# print(agent_executor.invoke({"input": "What's the weather in Paris?"}))

```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tools_builtin/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Details various how-to guides for LangChain, covering practical implementation aspects and common tasks.

SOURCE: https://python.langchain.com/docs/concepts/tokens/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
- How to use tools in a chain
- How to use a vectorstore as a retriever
- How to add memory to chatbots
- How to use example selectors
- How to add a semantic layer over graph database
- How to invoke runnables in parallel
- How to stream chat model responses
- How to add default invocation args to a Runnable
- How to add retrieval to chatbots
- How to use few shot examples in chat models
- How to do tool/function calling
- How to install LangChain packages
- How to add examples to the prompt for query analysis
- How to use few shot examples
- How to run custom functions
- How to use output parsers to parse an LLM response into structured format
- How to handle cases where no queries are generated
- How to route between sub-chains
- How to return structured data from a model
- How to summarize text through parallelization
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_runtime/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Langchain Text Generation Example
DESCRIPTION: Demonstrates basic text generation using a Langchain LLM. This snippet requires the Langchain library to be installed.

SOURCE: https://python.langchain.com/docs/tutorials/graph/

LANGUAGE: Python
CODE:
```
from langchain_community.llms import OpenAI

# Initialize the LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0.7)

# Generate text
prompt = "Write a short story about a robot learning to love."
response = llm.invoke(prompt)

print(response)
```

----------------------------------------

TITLE: Extraction Techniques
DESCRIPTION: Guides on using reference examples and handling long text for extraction, as well as performing extraction using only prompting.

SOURCE: https://python.langchain.com/docs/how_to/chat_model_caching/

LANGUAGE: python
CODE:
```
# How to use reference examples when doing extraction
# /docs/how_to/extraction_examples/

# How to handle long text when doing extraction
# /docs/how_to/extraction_long_text/

# How to use prompting alone (no tool calling) to do extraction
# /docs/how_to/extraction_parse/
```

----------------------------------------

TITLE: How to use example selectors
DESCRIPTION: Details how to use example selectors in LangChain, which dynamically select relevant examples to include in the prompt based on the current input. This is crucial for efficient few-shot learning.

SOURCE: https://python.langchain.com/docs/how_to/few_shot_examples/

LANGUAGE: python
CODE:
```
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# Sample examples
examples = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "What is the capital of Spain?", "answer": "Madrid"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"}
]

# Create a prompt template for each example
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}",
)

# Set up a semantic similarity example selector
# This requires an embedding model and a vector store
similarity_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2,  # Number of examples to retrieve
    input_keys=["question"],
)

# Create the FewShotPromptTemplate using the selector
example_selector_prompt = FewShotPromptTemplate(
    example_selector=similarity_selector,
    example_prompt=example_prompt,
    prefix="Give the answer to the following question.",
    suffix="Question: {question}\nAnswer:",
    input_variables=["question"],
)

# Format the prompt with a new question
formatted_prompt = example_selector_prompt.format(question="What is the capital of Canada?")

# print(formatted_prompt)
# This formatted_prompt can be sent to an LLM
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/multi_vector/

LANGUAGE: APIDOC
CODE:
```
How to use tools in a chain:
  Guide on integrating and using tools within LangChain chains.

How to use a vectorstore as a retriever:
  Instructions for utilizing vector stores as retrieval mechanisms in LangChain.

How to add memory to chatbots:
  Guide on implementing memory functionalities for chatbots in LangChain.

How to use example selectors:
  Instructions on using example selectors for prompt engineering in LangChain.

How to add a semantic layer over graph database:
  Guide on adding semantic capabilities to graph databases with LangChain.

How to invoke runnables in parallel:
  Instructions for executing LangChain runnables concurrently.

How to stream chat model responses:
  Guide on enabling and handling streaming responses from chat models in LangChain.

How to add default invocation args to a Runnable:
  Instructions for setting default arguments for LangChain runnables.

How to add retrieval to chatbots:
  Guide on integrating retrieval mechanisms into LangChain chatbots.

How to use few shot examples in chat models:
  Instructions for using few-shot examples with chat models in LangChain.

How to do tool/function calling:
  Guide on implementing tool and function calling capabilities in LangChain.

How to install LangChain packages:
  Instructions for installing necessary LangChain packages.

How to add examples to the prompt for query analysis:
  Guide on incorporating examples into prompts for analyzing queries.

How to use few shot examples:
  Instructions for utilizing few-shot learning examples in LangChain.

How to run custom functions:
  Guide on executing custom functions within the LangChain framework.

How to use output parsers to parse an LLM response into structured format:
  Instructions for parsing LLM responses into structured formats using LangChain's output parsers.

How to handle cases where no queries are generated:
  Guide on managing scenarios where no queries are produced.

How to route between sub-chains:
  Instructions for implementing routing logic between different sub-chains in LangChain.

How to return structured data from a model:
  Guide on configuring models to return structured data using LangChain.
```

----------------------------------------

TITLE: OpenAI Chat Integration
DESCRIPTION: Provides a quick overview for getting started with OpenAI chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI

# Example usage (requires OpenAI API key)
# llm = ChatOpenAI(api_key="YOUR_OPENAI_API_KEY")
# response = llm.invoke("What is the capital of Canada?")
# print(response)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tool_calling_parallel/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tools_human/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Example Selectors
DESCRIPTION: Demonstrates various strategies for selecting examples, including using LangSmith datasets, selecting by length, maximal marginal relevance (MMR), n-gram overlap, and similarity.

SOURCE: https://python.langchain.com/docs/how_to/contextual_compression/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import LengthBasedExampleSelector

# Example for selecting examples by length
# example_selector = LengthBasedExampleSelector(
#     examples=my_examples, 
#     example_prompt=example_prompt, 
#     max_length=100
# )

# Example for selecting examples by similarity
# example_selector = SemanticSimilarityExampleSelector(
#     vectorstore=Chroma.from_documents(my_examples, OpenAIEmbeddings()),
#     k=2,
#     example_prompt=example_prompt,
#     input_keys=["input"],
# )

```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/recursive_text_splitter/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/pydantic_compatibility/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: How to Build an Agent
DESCRIPTION: This guide provides instructions on building agents in LangChain. It covers the core concepts of agents, including tools, language models, and agent executors, and how to combine them to create intelligent agents that can reason and act.

SOURCE: https://python.langchain.com/docs/how_to/tool_artifacts/

LANGUAGE: python
CODE:
```
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# tools = [
#     Tool(
#         name="Search",
#         func=lambda query: f"Results for {query}",
#         description="useful for when you need to answer questions about current events"
#     )
# ]

# prompt = PromptTemplate.from_template("...")
# agent = create_react_agent(llm=..., tools=tools, prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools)
# agent_executor.invoke({"input": "What is the weather today?"})

```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials demonstrating how to build various LLM applications with LangChain, from simple Q&A to complex agents.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_json/

LANGUAGE: python
CODE:
```
Build a Question Answering application over a Graph Database: /docs/tutorials/graph/
Build a simple LLM application with chat models and prompt templates: /docs/tutorials/llm_chain/
Build a Chatbot: /docs/tutorials/chatbot/
Build a Retrieval Augmented Generation (RAG) App: Part 2: /docs/tutorials/qa_chat_history/
Build an Extraction Chain: /docs/tutorials/extraction/
Build an Agent: /docs/tutorials/agents/
Tagging: /docs/tutorials/classification/
Build a Retrieval Augmented Generation (RAG) App: Part 1: /docs/tutorials/rag/
Build a semantic search engine: /docs/tutorials/retrievers/
Build a Question/Answering system over SQL data: /docs/tutorials/sql_qa/
Summarize Text: /docs/tutorials/summarization/
```

----------------------------------------

TITLE: Basic LLM Interaction Example
DESCRIPTION: Demonstrates a simple interaction with an LLM using Langchain. This snippet shows how to initialize an LLM and get a response.

SOURCE: https://python.langchain.com/docs/integrations/chat/ollama/

LANGUAGE: python
CODE:
```
from langchain_community.llms import OpenAI

# Initialize the LLM
llm = OpenAI(model_name="text-davinci-003", openai_api_key="YOUR_API_KEY")

# Get a response from the LLM
response = llm.invoke("What is Langchain?")
print(response)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/chat_streaming/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/document_loader_office_file/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific functionalities within LangChain, such as using tools, memory, and output parsers.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_retry/

LANGUAGE: python
CODE:
```
# How to use tools in a chain
# ... (implementation details)

# How to use a vectorstore as a retriever
# ... (implementation details)

# How to add memory to chatbots
# ... (implementation details)

# How to use example selectors
# ... (implementation details)

# How to add a semantic layer over graph database
# ... (implementation details)

# How to invoke runnables in parallel
# ... (implementation details)

# How to stream chat model responses
# ... (implementation details)

# How to add default invocation args to a Runnable
# ... (implementation details)

# How to add retrieval to chatbots
# ... (implementation details)

# How to use few shot examples in chat models
# ... (implementation details)

# How to do tool/function calling
# ... (implementation details)

# How to install LangChain packages
# ... (implementation details)

# How to add examples to the prompt for query analysis
# ... (implementation details)

# How to use few shot examples
# ... (implementation details)

# How to run custom functions
# ... (implementation details)

# How to use output parsers to parse an LLM response into structured format
# ... (implementation details)

# How to handle cases where no queries are generated
# ... (implementation details)

# How to route between sub-chains
# ... (implementation details)

# How to return structured data from a model
# ... (implementation details)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/llm_token_usage_tracking/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/prompts_partial/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Setup OpenAI Embeddings
DESCRIPTION: Installs the langchain-openai package and sets up the OpenAIEmbeddings model. Requires an OpenAI API key to be set as an environment variable.

SOURCE: https://python.langchain.com/docs/how_to/embed_text/

LANGUAGE: python
CODE:
```
pip install -qU langchain-openai

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
```

----------------------------------------

TITLE: SambaNovaCloud Chat Integration
DESCRIPTION: Helps users get started with SambaNovaCloud chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatSambaNovaCloud

# Example usage (requires SambaNovaCloud credentials)
# llm = ChatSambaNovaCloud(api_key="YOUR_SAMBANovacloud_API_KEY")
# response = llm.invoke("What are the latest advancements in AI?")
# print(response)
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials demonstrating how to build various applications using LangChain, from simple LLM applications to complex agents and retrieval systems.

SOURCE: https://python.langchain.com/docs/how_to/tool_runtime/

LANGUAGE: python
CODE:
```
# Example: Build a simple LLM application with chat models and prompt templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import OpenAI

# Initialize the LLM
llm = OpenAI()

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{topic}")
])

# Define a chain
chain = prompt | llm | StrOutputParser()

# Run the chain
response = chain.invoke({"topic": "What is LangChain?"})
print(response)

```

LANGUAGE: python
CODE:
```
# Example: Build a Question Answering application over a Graph Database
# This involves setting up a graph database, loading data, and creating a LangChain
# interface to query it using natural language.
# Specific implementation details would depend on the chosen graph database (e.g., Neo4j)
# and the graph-specific LangChain components.

# Placeholder for graph QA setup
print("Setting up Graph QA application...")

```

LANGUAGE: python
CODE:
```
# Example: Build a Retrieval Augmented Generation (RAG) App: Part 1
# This tutorial covers setting up a retriever, loading documents, and creating a basic RAG chain.
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.llms import OpenAI

# Load documents
loader = TextLoader("path/to/your/document.txt")
docs = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
)

# Initialize LLM
llm = OpenAI()

# Create the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
)

# Example usage
# query = "What is the main topic of the document?"
# print(rag_chain.invoke(query))
print("RAG setup complete. Ready for invocation.")

```

LANGUAGE: python
CODE:
```
# Example: Build an Agent
# This involves defining tools, creating an LLM agent, and running the agent.
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize LLM and Tool
llm = ChatOpenAI(model="gpt-3.5-turbo")
search = DuckDuckGoSearchRun()

# Define the agent's prompt
# Note: The prompt structure might vary based on the agent type and creation method.
# This is a simplified example.
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent (example using create_tool_calling_agent)
# This requires a list of tools and the LLM
tools = [search]
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# Create an Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
# agent_executor.invoke({"input": "What is the weather in San Francisco?"})
print("Agent setup complete. Ready for invocation.")

```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/streaming_llm/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/summarize_map_reduce/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/concepts/rag/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Perplexity Chat Integration
DESCRIPTION: Helps users get started with Perplexity chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatPerplexity

# Example usage (requires Perplexity API key)
# llm = ChatPerplexity(api_key="YOUR_PERPLEXITY_API_KEY")
# response = llm.invoke("What are the main causes of climate change?")
# print(response)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/tools_model_specific/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain.

SOURCE: https://python.langchain.com/docs/concepts/architecture/

LANGUAGE: APIDOC
CODE:
```
How-to guides:
  - How to use tools in a chain: /docs/how_to/tools_chain/
  - How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
  - How to add memory to chatbots: /docs/how_to/chatbots_memory/
  - How to use example selectors: /docs/how_to/example_selectors/
  - How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
  - How to invoke runnables in parallel: /docs/how_to/parallel/
  - How to stream chat model responses: /docs/how_to/chat_streaming/
  - How to add default invocation args to a Runnable: /docs/how_to/binding/
  - How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
  - How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
  - How to do tool/function calling: /docs/how_to/function_calling/
  - How to install LangChain packages: /docs/how_to/installation/
  - How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
  - How to use few shot examples: /docs/how_to/few_shot_examples/
  - How to run custom functions: /docs/how_to/functions/
  - How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
  - How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
  - How to route between sub-chains: /docs/how_to/routing/
  - How to return structured data from a model: /docs/how_to/structured_output/
  - How to summarize text through parallelization: /docs/how_to/summarize_map_reduce/
```

----------------------------------------

TITLE: OCIModelDeployment Chat Integration
DESCRIPTION: Helps users get started with OCIModelDeployment chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatOCIModelDeployment

# Example usage (requires OCI credentials)
# llm = ChatOCIModelDeployment(endpoint="your-oci-endpoint", deployment_id="your-deployment-id")
# response = llm.invoke("Write a Python function to calculate factorial.")
# print(response)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/vectorstore_retriever/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/merge_message_runs/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on how to perform specific tasks using LangChain. It includes instructions on trimming messages and creating and querying vector stores.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: markdown
CODE:
```
*   [How to trim messages](/docs/how_to/trim_messages/)
*   [How to create and query vector stores](/docs/how_to/vectorstores/)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_string/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Data Extraction with Examples
DESCRIPTION: Demonstrates data extraction using the configured runnable with provided few-shot examples. It invokes the runnable multiple times with the same text to show improved consistency in output.

SOURCE: https://python.langchain.com/docs/how_to/extraction_examples/

LANGUAGE: python
CODE:
```
for _ in range(5):
    text = "The solar system is large, but earth has only 1 moon."
    print(runnable.invoke({"text": text, "examples": messages}))
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/custom_callbacks/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Migrating from v0.0 Chains
DESCRIPTION: This guide provides instructions and examples for migrating from v0.0 chains to newer versions of LangChain. It covers specific chain types and their migration paths.

SOURCE: https://python.langchain.com/docs/how_to/query_high_cardinality/

LANGUAGE: python
CODE:
```
# Example: Migrating LLMChain
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Old v0.0 style (conceptual)
# llm = OpenAI(temperature=0)
# prompt = PromptTemplate(input_variables=["topic"], template="Tell me a joke about {topic}")
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# New style (conceptual)
# from langchain_core.runnables import RunnableSequence
# llm = OpenAI(temperature=0)
# prompt = PromptTemplate(input_variables=["topic"], template="Tell me a joke about {topic}")
# llm_chain = prompt | llm

# Specific migration examples for other chains like ConstitutionalChain, ConversationalChain, etc., would follow similar patterns.
```

----------------------------------------

TITLE: LangChain Example Selectors
DESCRIPTION: Dynamically select and format examples for few-shot prompting to enhance few-shot learning performance. Implements classes for example selection and formatting.

SOURCE: https://context7_llms

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Example using SemanticSimilarityExampleSelector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, 
    OpenAIEmbeddings(), 
    Chroma, 
    k=1
)

```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on specific functionalities and implementation details within LangChain.

SOURCE: https://python.langchain.com/docs/how_to/time_weighted_vectorstore/

LANGUAGE: APIDOC
CODE:
```
How to use tools in a chain: /docs/how_to/tools_chain/
How to use a vectorstore as a retriever: /docs/how_to/vectorstore_retriever/
How to add memory to chatbots: /docs/how_to/chatbots_memory/
How to use example selectors: /docs/how_to/example_selectors/
How to add a semantic layer over graph database: /docs/how_to/graph_semantic/
How to invoke runnables in parallel: /docs/how_to/parallel/
How to stream chat model responses: /docs/how_to/chat_streaming/
How to add default invocation args to a Runnable: /docs/how_to/binding/
How to add retrieval to chatbots: /docs/how_to/chatbots_retrieval/
How to use few shot examples in chat models: /docs/how_to/few_shot_examples_chat/
How to do tool/function calling: /docs/how_to/function_calling/
How to install LangChain packages: /docs/how_to/installation/
How to add examples to the prompt for query analysis: /docs/how_to/query_few_shot/
How to use few shot examples: /docs/how_to/few_shot_examples/
How to run custom functions: /docs/how_to/functions/
How to use output parsers to parse an LLM response into structured format: /docs/how_to/output_parser_structured/
How to handle cases where no queries are generated: /docs/how_to/query_no_queries/
How to route between sub-chains: /docs/how_to/routing/
How to return structured data from a model: /docs/how_to/structured_output/
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/multi_vector/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/extraction_parse/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Example Data Document Loader
DESCRIPTION: Loads example data for testing and demonstration. This loader provides sample documents.

SOURCE: https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/

LANGUAGE: python
CODE:
```
from langchain_community.document_loaders import ExampleLoader

loader = ExampleLoader(file_path="path/to/example_data.txt")
documents = loader.load()
```

----------------------------------------

TITLE: Setup and Database Connection
DESCRIPTION: Installs necessary packages, sets up environment variables (optional for LangSmith), and connects to a SQLite Chinook database using SQLDatabase. It then prints the dialect, usable table names, and a sample query result.

SOURCE: https://python.langchain.com/docs/how_to/sql_prompting/

LANGUAGE: python
CODE:
```
%pip install --upgrade --quiet  langchain langchain-community langchain-experimental langchain-openai

# Uncomment the below to use LangSmith. Not required.
# import getpass
# import os
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
# os.environ["LANGSMITH_TRACING"] = "true"

from langchain_community.utilities import SQLDatabasedb = SQLDatabase.from_uri("sqlite:///Chinook.db", sample_rows_in_table_info=3)
print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM Artist LIMIT 10;"))
```

----------------------------------------

TITLE: Dynamic Few-Shot Example Selection with FAISS
DESCRIPTION: Demonstrates how to use SemanticSimilarityExampleSelector with FAISS and OpenAIEmbeddings to dynamically select relevant examples for SQL query generation. This helps in providing contextually appropriate few-shot examples to the language model.

SOURCE: https://python.langchain.com/docs/how_to/sql_prompting/

LANGUAGE: python
CODE:
```
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Assuming 'examples' is a list of dictionaries with 'input' and 'query' keys
# examples = [
#     {'input': 'List all artists.', 'query': 'SELECT * FROM Artist;'},
#     {'input': 'How many employees are there', 'query': 'SELECT COUNT(*) FROM "Employee"'},
#     ...
# ]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

# Example usage of the selector:
# selected_examples = example_selector.select_examples({"input": "how many artists are there?"})
# print(selected_examples)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/runnable_runtime_secrets/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/query_multiple_retrievers/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/sql_large_db/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Example Selectors
DESCRIPTION: Demonstrates various strategies for selecting examples, including using LangSmith datasets, selecting by length, maximal marginal relevance (MMR), n-gram overlap, and similarity.

SOURCE: https://python.langchain.com/docs/how_to/callbacks_async/

LANGUAGE: python
CODE:
```
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import LengthBasedExampleSelector

# Example for selecting examples by length
# example_selector = LengthBasedExampleSelector(
#     examples=my_examples, 
#     example_prompt=example_prompt, 
#     max_length=100
# )

# Example for selecting examples by similarity
# example_selector = SemanticSimilarityExampleSelector(
#     vectorstore=Chroma.from_documents(my_examples, OpenAIEmbeddings()),
#     k=2,
#     example_prompt=example_prompt,
#     input_keys=["input"],
# )

```

----------------------------------------

TITLE: Qwen QwQ Chat Integration
DESCRIPTION: Helps users get started with QwQ chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatQwQ

# Example usage (requires QwQ API key)
# llm = ChatQwQ(api_key="YOUR_QWQ_API_KEY")
# response = llm.invoke("Write a haiku about nature.")
# print(response)
```

----------------------------------------

TITLE: LangChain Version Migration Guides
DESCRIPTION: Guides for migrating between different versions of LangChain, focusing on chain and memory upgrades.

SOURCE: https://python.langchain.com/docs/how_to/sql_prompting/

LANGUAGE: python
CODE:
```
# Version migration links:
# Pydantic compatibility: /docs/how_to/pydantic_compatibility/
# Migrating from v0.0 chains: /docs/versions/migrating_chains/
#   - ConstitutionalChain: /docs/versions/migrating_chains/constitutional_chain/
#   - ConversationalChain: /docs/versions/migrating_chains/conversation_chain/
#   - ConversationalRetrievalChain: /docs/versions/migrating_chains/conversation_retrieval_chain/
#   - LLMChain: /docs/versions/migrating_chains/llm_chain/
#   - LLMMathChain: /docs/versions/migrating_chains/llm_math_chain/
#   - LLMRouterChain: /docs/versions/migrating_chains/llm_router_chain/
#   - MapReduceDocumentsChain: /docs/versions/migrating_chains/map_reduce_chain/
#   - MapRerankDocumentsChain: /docs/versions/migrating_chains/map_rerank_docs_chain/
#   - MultiPromptChain: /docs/versions/migrating_chains/multi_prompt_chain/
#   - RefineDocumentsChain: /docs/versions/migrating_chains/refine_docs_chain/
#   - RetrievalQA: /docs/versions/migrating_chains/retrieval_qa/
#   - StuffDocumentsChain: /docs/versions/migrating_chains/stuff_docs_chain/
# Upgrading to LangGraph memory: /docs/versions/migrating_memory/
#   - BaseChatMessageHistory with LangGraph: /docs/versions/migrating_memory/chat_history/
#   - ConversationBufferMemory or ConversationStringBufferMemory: /docs/versions/migrating_memory/conversation_buffer_memory/
#   - ConversationBufferWindowMemory or ConversationTokenBufferMemory: /docs/versions/migrating_memory/conversation_buffer_window_memory/
#   - ConversationSummaryMemory or ConversationSummaryBufferMemory: /docs/versions/migrating_memory/conversation_summary_memory/
#   - Long-Term Memory Agent: /docs/versions/migrating_memory/long_term_memory_agent/
# Release policy: /docs/versions/release_policy/
```

----------------------------------------

TITLE: Langchain LLM Text Generation Example
DESCRIPTION: Demonstrates how to use Langchain to generate text using a specified LLM. This snippet shows the basic setup and usage for text generation.

SOURCE: https://python.langchain.com/docs/tutorials/summarization/

LANGUAGE: python
CODE:
```
from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0.7)

# Generate text
prompt = "Write a short story about a robot learning to love."
response = llm(prompt)

print(response)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/query_high_cardinality/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Reka Chat Integration
DESCRIPTION: Provides a quick overview for getting started with Reka chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatReka

# Example usage (requires Reka API key)
# llm = ChatReka(api_key="YOUR_REKA_API_KEY")
# response = llm.invoke("Explain the concept of quantum entanglement.")
# print(response)
```

----------------------------------------

TITLE: LangChain Tutorials Overview
DESCRIPTION: This section lists various tutorials available for LangChain, covering a wide range of applications from simple LLM chains to complex agents and RAG systems.

SOURCE: https://python.langchain.com/docs/how_to/qa_citations/

LANGUAGE: markdown
CODE:
```
*   [Build a Question Answering application over a Graph Database](/docs/tutorials/graph/)
*   [Build a simple LLM application with chat models and prompt templates](/docs/tutorials/llm_chain/)
*   [Build a Chatbot](/docs/tutorials/chatbot/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 2](/docs/tutorials/qa_chat_history/)
*   [Build an Extraction Chain](/docs/tutorials/extraction/)
*   [Build an Agent](/docs/tutorials/agents/)
*   [Tagging](/docs/tutorials/classification/)
*   [Build a Retrieval Augmented Generation (RAG) App: Part 1](/docs/tutorials/rag/)
*   [Build a semantic search engine](/docs/tutorials/retrievers/)
*   [Build a Question/Answering system over SQL data](/docs/tutorials/sql_qa/)
*   [Summarize Text](/docs/tutorials/summarization/)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Practical guides on implementing specific features and functionalities within LangChain.

SOURCE: https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/

LANGUAGE: python
CODE:
```
# How-to: Use tools in a chain
# Explains how to integrate external tools into LangChain chains.

# How-to: Use a vectorstore as a retriever
# Details on configuring and using vectorstores for retrieval.

# How-to: Add memory to chatbots
# Demonstrates methods for incorporating memory into chatbot applications.

# How-to: Use example selectors
# Covers the usage of example selectors for prompt engineering.

# How-to: Add a semantic layer over graph database
# Guides on building semantic interfaces for graph databases.

# How-to: Invoke runnables in parallel
# Shows how to execute LangChain runnables concurrently.

# How-to: Stream chat model responses
# Details on implementing streaming for chat model outputs.

# How-to: Add default invocation args to a Runnable
# Explains how to set default arguments for runnables.

# How-to: Add retrieval to chatbots
# Covers integrating retrieval mechanisms into chatbots.

# How-to: Use few shot examples in chat models
# Demonstrates using few-shot examples with chat models.

# How-to: Do tool/function calling
# Explains how to implement tool and function calling capabilities.

# How-to: Install LangChain packages
# Provides instructions for installing necessary LangChain packages.

# How-to: Add examples to the prompt for query analysis
# Shows how to enhance query analysis with few-shot examples.

# How-to: Use few shot examples
# General guide on utilizing few-shot examples in prompts.

# How-to: Run custom functions
# Details on executing custom functions within LangChain workflows.

# How-to: Use output parsers to parse an LLM response into structured format
# Covers parsing LLM outputs into structured data.

# How-to: Handle cases where no queries are generated
# Strategies for managing scenarios with no generated queries.

# How-to: Route between sub-chains
# Explains how to implement routing logic between different chains.

# How-to: Return structured data from a model
# Guides on configuring models to return structured output.
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_fixing/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides that explain how to implement specific features and functionalities within LangChain, such as adding memory to chatbots or using vectorstores as retrievers.

SOURCE: https://python.langchain.com/docs/how_to/output_parser_string/

LANGUAGE: APIDOC
CODE:
```
How-to Guides:
  /docs/how_to/

This section provides detailed instructions on implementing specific functionalities and patterns within LangChain. These guides are useful for understanding advanced concepts and customizing your LLM applications.

Key How-to Guides:
- How to use tools in a chain
- How to use a vectorstore as a retriever
- How to add memory to chatbots
- How to use example selectors
- How to add a semantic layer over graph database
- How to invoke runnables in parallel
- How to stream chat model responses
- How to add default invocation args to a Runnable
- How to add retrieval to chatbots
- How to use few shot examples in chat models
- How to do tool/function calling
- How to install LangChain packages
- How to add examples to the prompt for query analysis
- How to use few shot examples
- How to run custom functions
- How to use output parsers to parse an LLM response into structured format
- How to handle cases where no queries are generated
- How to route between sub-chains
- How to return structured data from a model

These guides offer practical solutions to common challenges encountered when building LLM applications.
```

----------------------------------------

TITLE: LangChain Tutorials
DESCRIPTION: A collection of tutorials demonstrating how to build various LLM applications using LangChain.

SOURCE: https://python.langchain.com/docs/how_to/multimodal_inputs/

LANGUAGE: APIDOC
CODE:
```
Tutorials:
  - Build a Question Answering application over a Graph Database
    Description: Guide on creating QA systems that query graph databases.
    Path: /docs/tutorials/graph/
  - Build a simple LLM application with chat models and prompt templates
    Description: Introduces basic LLM application development using chat models and prompt templates.
    Path: /docs/tutorials/llm_chain/
  - Build a Chatbot
    Description: Steps to create a conversational chatbot application.
    Path: /docs/tutorials/chatbot/
  - Build a Retrieval Augmented Generation (RAG) App: Part 2
    Description: Continues the RAG application development, focusing on advanced features.
    Path: /docs/tutorials/qa_chat_history/
  - Build an Extraction Chain
    Description: Demonstrates how to extract structured information using LangChain chains.
    Path: /docs/tutorials/extraction/
  - Build an Agent
    Description: Guide to building intelligent agents that can use tools.
    Path: /docs/tutorials/agents/
  - Tagging
    Description: Covers classification and tagging functionalities.
    Path: /docs/tutorials/classification/
  - Build a Retrieval Augmented Generation (RAG) App: Part 1
    Description: Introduces the fundamentals of building RAG applications.
    Path: /docs/tutorials/rag/
  - Build a semantic search engine
    Description: How to implement semantic search capabilities.
    Path: /docs/tutorials/retrievers/
  - Build a Question/Answering system over SQL data
    Description: Guide for creating QA systems that interact with SQL databases.
    Path: /docs/tutorials/sql_qa/
  - Summarize Text
    Description: Demonstrates text summarization techniques.
    Path: /docs/tutorials/summarization/
```

----------------------------------------

TITLE: NVIDIA AI Endpoints Chat Integration
DESCRIPTION: Facilitates getting started with NVIDIA chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

# Example usage (requires NVIDIA API key)
# llm = ChatNVIDIA(api_key="YOUR_NVIDIA_API_KEY", model="ai-endpoints/llama2-13b")
# response = llm.invoke("Describe the process of photosynthesis.")
# print(response)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: A collection of practical guides on implementing various features within LangChain, such as using tools, vectorstores, memory, and handling different output formats.

SOURCE: https://python.langchain.com/docs/tutorials/graph/

LANGUAGE: APIDOC
CODE:
```
LangChain How-to Guides:

Tools:
  - How to use tools in a chain: Integrates external tools (like search engines or calculators) into LangChain workflows.

Vectorstores:
  - How to use a vectorstore as a retriever: Configures and utilizes vector databases for efficient similarity search and retrieval.

Memory:
  - How to add memory to chatbots: Implements conversational memory to maintain context in chatbot applications.

Example Selectors:
  - How to use example selectors: Manages and selects few-shot examples for LLM prompts.

Graph Databases:
  - How to add a semantic layer over graph database: Connects LLMs to graph databases for semantic querying.

Parallel Invocation:
  - How to invoke runnables in parallel: Executes multiple LangChain components concurrently for improved performance.

Streaming:
  - How to stream chat model responses: Enables real-time streaming of responses from chat models.

Binding Arguments:
  - How to add default invocation args to a Runnable: Sets default parameters for runnable invocations.

Retrieval for Chatbots:
  - How to add retrieval to chatbots: Integrates retrieval mechanisms into chatbot functionalities.

Few-Shot Examples:
  - How to use few shot examples in chat models: Provides examples to LLMs for better response generation.
  - How to use few shot examples: General guide on using few-shot learning.
  - How to add examples to the prompt for query analysis: Structures prompts with examples for specific analytical tasks.

Function Calling:
  - How to do tool/function calling: Implements LLM-based function calling for structured output and actions.

Installation:
  - How to install LangChain packages: Instructions for installing the necessary LangChain libraries.

Custom Functions:
  - How to run custom functions: Executes user-defined functions within LangChain workflows.

Output Parsers:
  - How to use output parsers to parse an LLM response into structured format: Extracts and structures data from LLM outputs.
  - How to return structured data from a model: Ensures LLM outputs conform to a predefined structure.

Routing:
  - How to route between sub-chains: Directs execution flow between different chains based on conditions.

Query Handling:
  - How to handle cases where no queries are generated: Manages scenarios where LLMs fail to produce relevant queries.
```

----------------------------------------

TITLE: Example Data Document Loader
DESCRIPTION: Loads example data for testing and demonstration. This loader provides sample documents.

SOURCE: https://python.langchain.com/docs/integrations/chat/bedrock/

LANGUAGE: python
CODE:
```
from langchain_community.document_loaders import ExampleLoader

loader = ExampleLoader(file_path="path/to/example_data.txt")
documents = loader.load()
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/indexing/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: OCIGenAI Chat Integration
DESCRIPTION: Provides a quick overview for getting started with OCIGenAI chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatOCIGenAI

# Example usage (requires OCI credentials)
# llm = ChatOCIGenAI(compartment_id="your-compartment-id", model_id="your-model-id")
# response = llm.invoke("Explain the theory of relativity.")
# print(response)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/custom_chat_model/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Naver Chat Integration
DESCRIPTION: Provides a quick overview for getting started with Naver chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatNaver

# Example usage (requires Naver API key)
# llm = ChatNaver(api_key="YOUR_NAVER_API_KEY")
# response = llm.invoke("Translate 'hello' to Korean.")
# print(response)
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: This section provides practical guides on implementing specific features and functionalities within LangChain, such as using tools, vectorstores, memory, parallel execution, streaming, and function calling.

SOURCE: https://python.langchain.com/docs/how_to/fallbacks/

LANGUAGE: python
CODE:
```
How to use tools in a chain
How to use a vectorstore as a retriever
How to add memory to chatbots
How to use example selectors
How to add a semantic layer over graph database
How to invoke runnables in parallel
How to stream chat model responses
How to add default invocation args to a Runnable
How to add retrieval to chatbots
How to use few shot examples in chat models
How to do tool/function calling
How to install LangChain packages
How to add examples to the prompt for query analysis
How to use few shot examples
How to run custom functions
How to use output parsers to parse an LLM response into structured format
How to handle cases where no queries are generated
How to route between sub-chains
How to return structured data from a model
```

----------------------------------------

TITLE: Invoke Chain with Examples
DESCRIPTION: Shows how to invoke a LangChain chain that has been configured with few-shot examples. The input is processed by the chain, leveraging the provided examples for better context and output generation.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/

LANGUAGE: python
CODE:
```
query_analyzer_with_examples.invoke("what's the difference between web voyager and reflection agents? do both use langgraph?")
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: Guides on specific tasks within LangChain, including how to trim messages and how to create and query vector stores.

SOURCE: https://python.langchain.com/docs/how_to/query_few_shot/



----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/tool_calling/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Together AI Chat Integration
DESCRIPTION: Helps users get started with Together AI chat models.

SOURCE: https://python.langchain.com/docs/integrations/chat/

LANGUAGE: python
CODE:
```
from langchain_community.chat_models import ChatTogetherAI

# Example usage (requires Together AI API key)
# llm = ChatTogetherAI(api_key="YOUR_TOGETHERAI_API_KEY")
# response = llm.invoke("Write a poem about the sea.")
# print(response)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/agent_executor/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/chat_model_caching/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/tools_prompting/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```

----------------------------------------

TITLE: Use Few-Shot Prompting with Tool Calling
DESCRIPTION: Demonstrates how to use few-shot examples within prompts to guide an LLM in correctly calling tools. This improves the accuracy and reliability of tool usage.

SOURCE: https://python.langchain.com/docs/how_to/chat_token_usage_tracking/

LANGUAGE: python
CODE:
```
# Example of few-shot prompting with tool calling:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

# Define a tool
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"The weather in {city} is sunny."

# Convert the tool to a format compatible with OpenAI's function calling
weather_tool = convert_to_openai_tool(get_weather)

# Create a prompt with few-shot examples
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    # Few-shot example 1
    HumanMessage("What's the weather like in San Francisco?"),
    AIMessage("", tool_calls=[{"name": "get_weather", "arguments": {"city": "San Francisco"}, "id": "call_abc"}])
    # Few-shot example 2 (optional)
    # HumanMessage("Tell me about Paris."),
    # AIMessage("", tool_calls=[...])
    # The actual user query
    ("human", "{query}")
])

# Initialize the model with tool calling capabilities
model = ChatOpenAI(model="gpt-4", tools=[weather_tool])

# Create the chain
chain = prompt | model

# Example query
# query = "What is the weather in London?"
# response = chain.invoke({"query": query})
# print(response.tool_calls)
```

----------------------------------------

TITLE: Using Toolkits
DESCRIPTION: Provides guidance on how to leverage pre-built toolkits within Langchain for various tasks. Toolkits bundle related tools and chains to simplify complex workflows.

SOURCE: https://python.langchain.com/docs/how_to/example_selectors_similarity/

LANGUAGE: python
CODE:
```
from langchain.agents import load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
# Load tools for a specific toolkit, e.g., 'python-repl'
tools = load_tools(["python-repl"], llm=llm)
# Further steps to use these tools with an agent...
```

----------------------------------------

TITLE: LangChain How-to Guides
DESCRIPTION: A collection of how-to guides for implementing specific features and functionalities in LangChain.

SOURCE: https://python.langchain.com/docs/concepts/document_loaders/

LANGUAGE: MARKDOWN
CODE:
```
How-to guides
    *   [How-to guides](/docs/how_to/)
    *   [How to use tools in a chain](/docs/how_to/tools_chain/)
    *   [How to use a vectorstore as a retriever](/docs/how_to/vectorstore_retriever/)
    *   [How to add memory to chatbots](/docs/how_to/chatbots_memory/)
    *   [How to use example selectors](/docs/how_to/example_selectors/)
    *   [How to add a semantic layer over graph database](/docs/how_to/graph_semantic/)
    *   [How to invoke runnables in parallel](/docs/how_to/parallel/)
    *   [How to stream chat model responses](/docs/how_to/chat_streaming/)
    *   [How to add default invocation args to a Runnable](/docs/how_to/binding/)
    *   [How to add retrieval to chatbots](/docs/how_to/chatbots_retrieval/)
    *   [How to use few shot examples in chat models](/docs/how_to/few_shot_examples_chat/)
    *   [How to do tool/function calling](/docs/how_to/function_calling/)
    *   [How to install LangChain packages](/docs/how_to/installation/)
    *   [How to add examples to the prompt for query analysis](/docs/how_to/query_few_shot/)
    *   [How to use few shot examples](/docs/how_to/few_shot_examples/)
    *   [How to run custom functions](/docs/how_to/functions/)
    *   [How to use output parsers to parse an LLM response into structured format](/docs/how_to/output_parser_structured/)
    *   [How to handle cases where no queries are generated](/docs/how_to/query_no_queries/)
    *   [How to route between sub-chains](/docs/how_to/routing/)
    *   [How to return structured data from a model](/docs/how_to/structured_output/)
    *   [How to summarize text through parallelization](/docs/how_to/summarize_map_reduce/)
```

----------------------------------------

TITLE: Select Examples with Langchain (LangSmith, Length, MMR, N-gram, Similarity)
DESCRIPTION: Demonstrates how to select examples using Langchain, including selecting examples from a LangSmith dataset, by length, by maximal marginal relevance (MMR), by n-gram overlap, and by similarity. These examples showcase different strategies for example selection, allowing developers to optimize the performance of their LLM applications. The examples cover various selection criteria and techniques.

SOURCE: https://python.langchain.com/docs/how_to/multimodal_prompts/

LANGUAGE: Python
CODE:
```
# Placeholder for code examples for selecting examples with Langchain.
# Actual code examples would be placed here.
```

----------------------------------------

TITLE: LangChain Google Generative AI Setup
DESCRIPTION: Installs the LangChain Google Generative AI integration package, enabling the use of Google's Gemini models within LangChain applications.

SOURCE: https://python.langchain.com/docs/how_to/sql_csv/

LANGUAGE: python
CODE:
```
pip install -qU "langchain[google-genai]"
```

----------------------------------------

TITLE: LangChain Tool Calling - Few-Shot Prompting
DESCRIPTION: Demonstrates how to use few-shot examples within the prompt to guide the LLM on how and when to use tools effectively. This improves the accuracy and reliability of tool calls.

SOURCE: https://python.langchain.com/docs/how_to/multimodal_inputs/

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def get_time(city: str) -> str:
    """Get the current time for a city."""
    return f"The time in {city} is 10:00 AM."

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [get_weather, get_time]

# Few-shot examples embedded in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools."),
    ("human", "What is the weather in Paris?"),
    ("assistant", "{\"tool\": \"get_weather\", \"tool_input\": {\"city\": \"Paris\"}}"),
    ("human", "What time is it in Tokyo?"),
    ("assistant", "{\"tool\": \"get_time\", \"tool_input\": {\"city
```