# ============================================================================
# RAW REACT AGENT LOOP - WITHOUT LANGCHAIN OR LANGGRAPH
# ============================================================================
# This file demonstrates a complete ReAct (Reasoning + Acting) agent loop
# using ONLY the raw Ollama SDK - no LangChain, no LangGraph abstractions.
#
# ReAct Pattern Explained:
# ========================
# ReAct stands for "Reasoning + Acting" - a simple but powerful loop that:
#   1. REASON: AI thinks about the problem using the conversation history
#   2. ACT: AI decides which tool to use and what arguments to pass
#   3. OBSERVE: We execute the tool and see the result
#   4. LOOP: Add the result to history and go back to step 1
#   5. FINISH: When AI decides no more tools needed, return final answer
#
# This loop continues until either:
#   - The AI decides to stop using tools and gives a final answer, OR
#   - We hit MAX_ITERATIONS (safety limit to prevent infinite loops)
#
# What LangChain Hides From You:
# ===============================
# LangChain abstracts away complexity by automatically handling:
#   - Tool schema generation (we do it manually here)
#   - Message object conversions (we use raw dicts here)
#   - Tool invocation interfaces (we call functions directly here)
#   - LLM provider differences (we use raw Ollama here)
#   - Tracing and observability (we wrap calls manually here)
#
# By seeing the raw loop, you'll understand what's happening under the hood!

import ollama
from langsmith import traceable  # LangSmith for tracing tool and LLM calls

# ============================================================================
# CONFIGURATION - Settings for the agent
# ============================================================================
MAX_ITERATIONS = 10  # Safety limit: agent stops after 10 iterations (prevents infinite loops)
MODEL = "llama3.2"  # Which Ollama model to use for reasoning and tool selection


# ============================================================================
# TOOL DEFINITIONS - Regular Python functions that the agent can call
# ============================================================================
# These are simple Python functions that perform actual work.
# The agent will decide when and how to call these during the ReAct loop.
#
# In a real application, these would:
#   - Query databases
#   - Call APIs
#   - Perform calculations
#   - Fetch external data
#   - Update records
#
# The @traceable decorator is from LangSmith - it logs every time these
# functions are called, letting us see the agent's execution in the dashboard.
#
# WITHOUT @traceable: The agent calls these in a black box (no visibility)
# WITH @traceable: LangSmith records each tool call for debugging

@traceable(run_type="tool")  # Mark this as a tool for LangSmith tracing
def get_product_price(product_name: str) -> float:
    """
    TOOL 1: Look up a product's price in the catalog.
    
    The agent calls this when it needs to know "How much does a laptop cost?"
    This is the FIRST step in the ReAct loop when we have a discount question.
    
    Args:
        product_name: Name of product to look up (e.g., "laptop", "smartphone")
    
    Returns:
        The numeric price as a float
    """
    # Simulated product catalog
    # In production, this would query a database or API
    prices = {
        "laptop": 999.0,
        "smartphone": 499.0,
        "headphones": 199.0
    }
    # Return price if found, 0.0 if not
    return prices.get(product_name.lower(), 0.0)


@traceable(run_type="tool")  # Mark this as a tool for LangSmith tracing
def apply_discount(price: float, discount_tier: str) -> float:
    """
    TOOL 2: Apply a discount tier to a price.
    
    The agent calls this AFTER getting a price from get_product_price.
    This is the SECOND step in the ReAct loop for discount questions.
    
    Args:
        price: The original price to discount (must be numeric float)
        discount_tier: Which tier ("bronze", "silver", or "gold")
    
    Returns:
        The final price after discount is applied
    """
    # Define discount percentages for each tier
    # In real systems, this might be in a database or configuration service
    discounts = {
        "bronze": 0.05,   # 5% off
        "silver": 0.10,   # 10% off  
        "gold": 0.15      # 15% off
    }
    # Look up the discount for this tier (default to 0 if tier not found)
    discount = discounts.get(discount_tier.lower(), 0)
    # Calculate final price: price minus discount amount
    discounted_price = price * (1 - discount)
    # Round to 2 decimal places for currency
    return round(discounted_price, 2)


# ============================================================================
# CRITICAL: TOOL SCHEMA DEFINITION - JSON that describes tools to the LLM
# ============================================================================
# 
# THIS IS THE BIGGEST COMPLEXITY DIFFERENCE:
# 
# When we use LangChain's @tool decorator, it AUTOMATICALLY generates JSON
# schemas from Python docstrings. When using raw Ollama SDK, we must write
# these schemas by hand.
#
# WHY DO WE NEED THIS?
# ====================
# The LLM (llama3.2) doesn't know about our Python functions.
# We must describe them in JSON so the LLM can:
#   1. Understand what tools are available
#   2. Know what parameters each tool expects
#   3. Know what type each parameter should be (string, number, etc.)
#   4. Make good decisions about which tool to call
#
# THE JSON SCHEMA FORMAT:
# =======================
# Each tool is described with:
#   - name: What to call it (must match function name)
#   - description: What it does (helps LLM decide when to use it)
#   - parameters: 
#       - type: Always "object" for function parameters
#       - properties: Define each parameter with type and description
#       - required: List which parameters are mandatory
#
# VENDOR DIFFERENCES:
# ===================
# - Ollama format (what we're using): {"type": "function", "function": {...}}
# - OpenAI format: Different structure, different type names
# - Anthropic format: Different structure entirely
# This schema matching is why LangChain's abstraction layer is so valuable!
# LangChain automatically converts between vendor formats.
#
# When LangChain's @tool decorator processes our Python functions, it:
#   1. Reads the function name
#   2. Reads the docstring
#   3. Reads the type annotations (e.g., str, float)
#   4. Generates the appropriate JSON schema for your LLM provider
#   5. Stores it in a format that works with that vendor
#
# We're doing all of that manually here so you can see what's happening:

tools_for_llm = [
    # ====================================================================
    # TOOL 1 SCHEMA: Describes get_product_price to the LLM
    # ====================================================================
    # This tells the LLM: "You have a function called 'get_product_price' that
    # takes a string parameter called 'product_name' and returns a price."
    # The LLM will see this and know it can call this tool when it needs
    # to find out product prices.
    {
        "type": "function",  # Ollama format: "type" must be "function"
        "function": {
            "name": "get_product_price",  # Must match the actual Python function name
            "description": "Look up the price of a product in the catalog.",  # Helps LLM decide when to use this
            "parameters": {
                "type": "object",  # Always "object" - we're describing an object with properties
                "properties": {  # Each property is a parameter the function accepts
                    "product_name": {
                        "type": "string",  # This parameter must be a string
                        "description": "The product name (e.g., 'laptop', 'smartphone', 'headphones')",  # Tells LLM what to put here
                    },
                },
                "required": ["product_name"],  # This parameter is mandatory
            },
        },
    },
    # ====================================================================
    # TOOL 2 SCHEMA: Describes apply_discount to the LLM
    # ====================================================================
    # This tells the LLM: "You have a function called 'apply_discount' that
    # takes two parameters: a number (price) and a string (discount_tier)."
    # The LLM will see this and know it can call this tool when it needs
    # to calculate discounted prices.
    {
        "type": "function",  # Ollama format: "type" must be "function"
        "function": {
            "name": "apply_discount",  # Must match the actual Python function name
            "description": "Apply a discount tier to a price and return the final discounted price. Available tiers: bronze (5% off), silver (10% off), gold (15% off).",
            "parameters": {
                "type": "object",  # Always "object" - describing function parameters
                "properties": {  # Each property is a parameter the function accepts
                    "price": {
                        "type": "number",  # This parameter must be a number (int or float)
                        "description": "The original price before discount",  # Tells LLM what to put here
                    },
                    "discount_tier": {
                        "type": "string",  # This parameter must be a string
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",  # Tells LLM what to put here
                    },
                },
                "required": ["price", "discount_tier"],  # Both parameters are mandatory
            },
        },
    },
]


# ============================================================================
# LLM CALL WRAPPER - Making Ollama calls with tracing support
# ============================================================================
# 
# THE CHALLENGE WITH RAW OLLAMA SDK:
# ==================================
# When we call ollama.chat() directly, nobody is watching or recording what
# happens. We can't debug or understand what the LLM decided.
#
# THE SOLUTION:
# ==============
# We wrap the Ollama call in @traceable() from LangSmith.
# This decorator records:
#   - What messages we sent to the LLM
#   - What the LLM responded with
#   - How long the call took
#   - Which tools the LLM tried to call
#
# WITH LANGCHAIN: All this tracing happens automatically (built-in)
# WITHOUT LANGCHAIN: We must manually add @traceable wrappers
#
# Note: @traceable requires LANGSMITH_API_KEY environment variable set
# If not set, the code still runs but doesn't send traces to LangSmith dashboard

@traceable(name="Ollama Chat", run_type="llm")  # Mark this as an LLM call for tracing
def ollama_chat_traced(messages):
    """
    A wrapper around the raw Ollama SDK's chat function.
    
    This function:
    1. Takes conversation history (messages)
    2. Calls the Ollama API with those messages and available tools
    3. Returns the LLM's response (which may include tool calls)
    
    The @traceable decorator makes LangSmith record this call so you can
    debug what happened during the agent loop.
    
    Args:
        messages: List of message dicts in Ollama format. Each dict has:
                  - "role": "user", "assistant", "system", or "tool"
                  - "content": the actual text
                  - "tool_calls": (optional) list of tool calls the assistant wants to make
    
    Returns:
        ChatResponse object from Ollama containing:
        - .message: The assistant's message with tool_calls (if any)
        - .response: The text response
        - Other metadata like usage stats
    """
    # Call the raw Ollama SDK
    # Parameters:
    #   - model: Which LLM to use ("llama3.2")
    #   - tools: The tool schemas we defined above (tools_for_llm)
    #   - messages: The complete conversation history
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ============================================================================
# AGENT LOOP - The ReAct pattern using raw Ollama SDK
# ============================================================================
# ReAct = Reasoning + Acting in a loop
# The agent: Thinks -> Calls Tool -> Observes Result -> Repeats until done

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    """
    Runs an agent loop to answer a question using tools, with raw Ollama SDK.
    
    This demonstrates what happens when we move from LangChain abstractions
    to raw API calls - we need to handle:
    - Tool schema manually (JSON format varies by vendor)
    - Message formatting manually (each vendor has different conventions)
    - Tool invocation manually (no LangChain runnable interface)
    - Tracing manually (no automatic LangSmith integration)
    """
    
    # ========================================================================
    # STEP 0: SETUP - Prepare data structures before starting the ReAct loop
    # ========================================================================
    # Before we can run the loop, we need to set up the infrastructure
    # that will let us execute tool calls when the agent requests them.
    
    # Create a lookup table: tool name -> Python function
    # When the LLM says "call get_product_price with {'product_name': 'laptop'}",
    # we use this dict to find the actual Python function and call it.
    # The LLM only knows tool NAMES (strings), so we need this mapping.
    tools_dict = {
        "get_product_price": get_product_price,  # If LLM says "get_product_price", we call this function
        "apply_discount": apply_discount,        # If LLM says "apply_discount", we call this function
    }
    # In a real system, you might load this from a database or registry

    # Print the user's question so we can track what we're working on
    print(f"Question: {question}")
    print("Agent is thinking...")

    # ========================================================================
    # STATE MANAGEMENT: Track what has happened during the loop
    # ========================================================================
    # As the agent loop runs, we need to remember certain facts about what
    # has already been done. This prevents the AI from cheating.
    #
    # For example, in this case:
    # - If the LLM tries to calculate a discount before getting a real price,
    #   we should block it and force it to call get_product_price first.
    # - This is a GUARDRAIL - a rule we enforce in code to improve AI behavior.
    #
    # LangChain doesn't automatically provide guardrails - WE must implement
    # them. This is one place where raw code gives you control LangChain
    # abstracts away.
    
    price_retrieved = False  # Track: Has the agent called get_product_price yet?

    # ========================================================================
    # INITIALIZATION: Build the initial message history
    # ========================================================================
    # This is the most CRITICAL section - it's where we set up the context
    # that the AI will use to understand how to behave.
    #
    # THE CONVERSATION STRUCTURE:
    # ============================
    # A conversation with an LLM is just a list of messages.
    # Each message has:
    #   - "role": Who said it? "system", "user", "assistant", or "tool"
    #   - "content": What did they say?
    #   - "tool_calls": (optional) What tools does the assistant want to call?
    #
    # MESSAGE ROLES:
    # - "system": Background instructions for the AI (how to behave, what rules to follow)
    # - "user": Questions or statements from the human user
    # - "assistant": Responses or tool calls from the AI
    # - "tool": Results from running tools (feedback to the AI)
    #
    # THE FLOW IN THE AGENT LOOP:
    # ============================
    # 1. We send: [system message, few-shot examples, user question]
    # 2. Ollama responds: [assistant message WITH tool_calls]
    # 3. We execute the tool and collect the result
    # 4. We add to history: [assistant message, tool result message]
    # 5. Go back to step 1 - Ollama sees the tool result and responds again
    # 6. Repeat until Ollama stops calling tools (gives final answer)
    #
    # RAW MESSAGE FORMAT (what we're using here):
    # ============================================
    # Each message is a plain Python dictionary:
    #   {"role": "user", "content": "What is the price?"}
    #   {"role": "assistant", "content": "", "tool_calls": [...]}
    #   {"role": "tool", "content": "The price is $999"}
    #
    # LANGCHAIN MESSAGE FORMAT (what LangChain uses internally):
    # ============================================================
    # LangChain uses custom message classes:
    #   HumanMessage(content="What is the price?")
    #   AIMessage(content="", tool_calls=[...])
    #   ToolMessage(content="The price is $999", tool_call_id="...")
    #
    # LangChain's advantage: It handles vendor differences automatically.
    # Our challenge: We must manually format messages correctly for Ollama.

    # Start building the message history
    messages = [
        # ======================================================================
        # MESSAGE 1: SYSTEM MESSAGE - Background instructions for the AI
        # ======================================================================
        # This message tells the AI HOW TO BEHAVE.
        # It's not a question or a response - it's the "rules of the game".
        # The AI reads this first to understand:
        #   - What it can do (has tools available)
        #   - What it should NOT do (rules to follow)
        #   - What to do if something goes wrong (error recovery)
        {
            "role": "system",
            "content": (
                # PART 1: What is the AI's job?
                "You are a helpful assistant that can use tools to answer questions. "
                "You have access to a product catalog tool and can apply discounts based on customer tier. "
                # PART 2: What are the strict rules?
                # These are guardrails we're embedding in the prompt.
                # (We also check these in code, but prompt rules help the AI remember.)
                "\n\nSTRICT RULES:\n"
                "1. For discounted prices, ALWAYS call get_product_price FIRST\n"
                "2. Then call apply_discount with the numeric result\n"
                "3. apply_discount takes ONLY: price (float) and discount_tier (string)\n"
                "4. NEVER pass product_name to apply_discount\n"
                "5. NEVER skip tools or calculate yourself\n"
                "6. If you get an error, immediately call the correct tool - do NOT give up"
            ),
        },
        # ======================================================================
        # MESSAGES 2-8: FEW-SHOT EXAMPLE 1 - Correct tool-calling sequence
        # ======================================================================
        # "Few-shot" means we teach by example, not just with rules.
        # We show the AI an example of the CORRECT way to solve this problem.
        #
        # WHY FEW-SHOT EXAMPLES?
        # ======================
        # AI models learn from patterns. If we just say "call tools in order",
        # the AI doesn't really understand. But if we show an example of HOW
        # to call tools in order, the AI learns the pattern and copies it.
        #
        # The example shows:
        # 1. A user question (MESSAGE 2)
        # 2. The assistant deciding to call get_product_price (MESSAGE 3)
        # 3. The tool returning a price (MESSAGE 4)
        # 4. The assistant deciding to call apply_discount (MESSAGE 5)
        # 5. The tool returning a discounted price (MESSAGE 6)
        # 6. The assistant giving the final answer (MESSAGE 7)
        #
        # When the LLM sees this pattern before processing the actual user question,
        # it's more likely to follow the same pattern.
        #
        # This is what LangChain abstracts away with its prompt templates.
        # LangChain can generate few-shot examples automatically, but here
        # we're writing them by hand to show exactly what's happening.
        {
            # MESSAGE 2: A user question (part of the example)
            "role": "user",
            "content": "What is the price of headphones with silver discount?"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_product_price",
                        "arguments": {"product_name": "headphones"}
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "199.0"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "apply_discount",
                        "arguments": {"price": 199.0, "discount_tier": "silver"}
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "179.1"
        },
        {
            "role": "assistant",
            "content": "The price of headphones with silver discount is $179.10"
        },
        # ======================================================================
        # MESSAGES 8-15: FEW-SHOT EXAMPLE 2 - Error recovery pattern
        # ======================================================================
        # This second example is CRITICAL.
        # It teaches the AI: "If you make a mistake, don't give up - fix it!"
        #
        # The example shows:
        # 1. AI attempts to skip the first step (WRONG) (MESSAGE 8-9)
        # 2. AI gets an error message explaining what went wrong (MESSAGE 10)
        # 3. AI recovers by calling the right tool (MESSAGE 11-12)
        # 4. AI then calls the next tool with correct data (MESSAGE 13-14)
        # 5. AI gives the correct final answer (MESSAGE 15)
        #
        # Why is this important?
        # Because without this example, when the AI makes a mistake and gets an error,
        # it might give up and say "I can't answer this" instead of retrying.
        # This second example teaches it: "Errors are fixable, try again!"
        #
        # Few-shot learning is one of the most powerful prompting techniques.
        # One correct example might work. But one correct + one recovery example?
        # That often makes the difference between an AI that gives up and one that persists.
        {
            # MESSAGE 8: A user question (part of the example)
            "role": "user",
            "content": "What is the price of smartphone with bronze discount?"
        },
        {
            # MESSAGE 9: Assistant makes a MISTAKE - tries to call apply_discount without getting price first
            # This is intentional - we're showing what happens when AI gets it wrong.
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "apply_discount",  # WRONG! Should call get_product_price first
                        "arguments": {"price": 500, "discount_tier": "bronze"}  # And we're guessing the price!
                    }
                }
            ]
        },
        {
            # MESSAGE 10: Tool error - feedback explaining what went wrong
            # This is the key to error recovery: a clear error message that tells
            # the AI exactly what to do differently.
            "role": "tool",
            "content": "ERROR: You tried to call apply_discount without first calling get_product_price! You MUST call get_product_price first."
        },
        {
            # MESSAGE 11: Assistant recovers - calls the RIGHT tool now
            # The assistant got the error and learned from it.
            # Now it's calling get_product_price like it should have initially.
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_product_price",  # NOW it calls the right tool
                        "arguments": {"product_name": "smartphone"}
                    }
                }
            ]
        },
        {
            # MESSAGE 12: Tool result - got the real price
            "role": "tool",
            "content": "499.0"
        },
        {
            # MESSAGE 13: Assistant calls apply_discount with the REAL price
            # Now it has actual data to work with (499.0 not the guessed 500)
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "apply_discount",
                        "arguments": {"price": 499.0, "discount_tier": "bronze"}  # Using real price
                    }
                }
            ]
        },
        {
            # MESSAGE 14: Tool result - discounted price
            "role": "tool",
            "content": "474.05"
        },
        {
            # MESSAGE 15: Assistant gives the correct final answer
            # After recovering from its mistake, it provides the right answer.
            "role": "assistant",
            "content": "The price of smartphone with bronze discount is $474.05"
        },
        # ======================================================================
        # FINAL MESSAGE: The actual user question we're trying to answer
        # ======================================================================
        # After the system message and few-shot examples, now comes the REAL question.
        # The AI has been primed with:
        #   1. Instructions on how to behave (system message)
        #   2. Examples of correct behavior (few-shot 1)
        #   3. Examples of error recovery (few-shot 2)
        # Now it's ready to tackle the actual user question using what it learned.
        {
            "role": "user",
            "content": question  # The actual question from the user (e.g., "What is the price of a laptop after applying a gold discount?")
        },
    ]

    # ========================================================================
    # THE REACT LOOP - This is where the magic happens!
    # ========================================================================
    # ReAct = "Reasoning" + "Acting"
    # This loop implements the core agent pattern:
    #   1. REASON: AI reads the history and decides what to do
    #   2. ACT: AI decides which tool to call
    #   3. OBSERVE: We run the tool and see the result
    #   4. LOOP: Add result to history and go back to step 1
    #   5. EXIT: When AI gives final answer (no tools called), we stop
    #
    # Each iteration through this loop is one "turn" where:
    # - The AI thinks about the problem
    # - The AI chooses an action (call a tool)
    # - We execute that action
    # - We show the AI the result
    #
    # It's like a conversation:
    # User: "What's the price of a laptop with gold discount?"
    # AI: "I need to find out. Let me call get_product_price."
    # Us: "Here's the price: $999"
    # AI: "Now let me apply the gold discount."
    # Us: "The discounted price is $849.15"
    # AI: "The answer is $849.15"
    # (Loop stops because AI didn't call any tools in that last response)
    
    for i in range(1, MAX_ITERATIONS + 1):  # Loop up to MAX_ITERATIONS times
        print(f"\nIteration {i}")
        
        # ====================================================================
        # REACT STEP 1: REASONING - Ask the AI to think about what to do
        # ====================================================================
        # We send the entire conversation history to the LLM.
        # The LLM reads:
        #   - The system instructions
        #   - The few-shot examples
        #   - All previous tool calls and results
        #   - The user's question
        # Then it reasons: "What should I do next?"
        #
        # The LLM's response will either be:
        #   A) A tool call: "I want to call get_product_price with these arguments"
        #   B) A final answer: "I'm done thinking, here's the answer"
        #
        # This is where the "reasoning" happens. With context from the history,
        # the AI figures out the right next step.
        #
        # Note: This is all happening inside llama3.2 model running locally via Ollama.
        # No external API calls (unless you're using a cloud Ollama instance).
        
        response = ollama_chat_traced(messages=messages)  # Send conversation to LLM, get response
        
        # Extract the AI's message from the Ollama response
        # Response structure:
        #   - response.message: The message object
        #     - .content: Text of the message (if any)
        #     - .tool_calls: List of tool calls (if any) - each is an Ollama ToolCall object
        #
        # Difference from LangChain: LangChain wraps this in an AIMessage object.
        # With raw Ollama, we access the response object directly.
        ai_message = response.message  # Extract the message part of the response
        
        # Get the list of tool calls from the AI's message
        # tool_calls will be either:
        #   - An empty list [] if the AI isn't calling any tools
        #   - A list with one or more tool calls if the AI wants to use tools
        # Each tool call is an Ollama ToolCall object with:
        #   - .function.name: Name of the tool (string)
        #   - .function.arguments: Dictionary of arguments for the tool
        tool_calls = ai_message.tool_calls

        # ====================================================================
        # EXIT CONDITION: Check if we're done
        # ====================================================================
        # If tool_calls is an empty list, the AI has no more tools to call.
        # This means the AI believes it has enough information to answer the question.
        # When this happens, we should stop the loop and return the final answer.
        #
        # This is the natural exit point of the ReAct loop.
        # The loop can also exit if we hit MAX_ITERATIONS (safety limit).
        
        if not tool_calls:  # No tool calls means we're done
            print("Agent has stopped making tool calls. Final answer:")
            print(ai_message.content)  # Print what the AI concluded
            return ai_message.content  # Return and stop the loop

        # ====================================================================
        # REACT STEP 2: DECIDE - Determine which tool the AI wants to use
        # ====================================================================
        # The AI has told us it wants to call a tool. Now we extract the details:
        #   - Which tool? (by name)
        #   - What arguments? (the parameters for that tool)
        #
        # This is the "decision" phase - we figure out what action the AI selected.
        
        # Get the first tool call from the list
        # (Real systems might handle multiple tools per iteration,
        # but we keep it simple and process one at a time)
        tool_call = tool_calls[0]  # Select the first (and only in this simple version) tool call
        
        # Extract the tool name and arguments
        # Ollama ToolCall structure:
        #   tool_call.function.name:        The name of the tool (e.g., "get_product_price")
        #   tool_call.function.arguments:   Dictionary of arguments (e.g., {"product_name": "laptop"})
        tool_name = tool_call.function.name        # What tool does the AI want to call?
        tool_args = tool_call.function.arguments   # What arguments should we pass?
        
        print(f"Agent called tool: {tool_name} with args: {tool_args}")

        # Look up the actual Python function by name from our tools dictionary
        # We created tools_dict at the beginning with {"get_product_price": <func>, ...}
        # Now we use it to find the function: tool_name -> Python function
        tool_to_use = tools_dict.get(tool_name)  # Get the function object
        if not tool_to_use:  # Safety check: tool name not recognized
            raise ValueError(f"Tool {tool_name} not found")

        # ====================================================================
        # GUARDRAILS: Enforce business rules and safety constraints
        # ====================================================================
        # This is CODE-LEVEL ENFORCEMENT of rules.
        # Combined with PROMPT-LEVEL rules (in the system message),
        # guardrails make the AI behave the way we want.
        #
        # What's happening here:
        # We're saying: "If the AI tries to call apply_discount before
        # calling get_product_price, BLOCK IT and force it to retry."
        #
        # Why do we need this?
        # Because llama3.2 (and other LLMs) sometimes take shortcuts:
        #   - "Let me guess the price is $500, then apply the discount"
        #   - "Let me calculate it myself without calling tools"
        # These shortcuts bypass our tool infrastructure.
        #
        # GUARDRAILS are a key part of autonomous agent design.
        # They're your safety mechanism. Without them, agents will be unreliable.
        #
        # This particular guardrail: Enforce sequential tool calling.
        # State: price_retrieved (boolean) tracks if we've called get_product_price
        
        if tool_name == "apply_discount" and not price_retrieved:
            # VIOLATION: AI tried to call apply_discount without getting a price first
            error_msg = (
                f"ERROR: You tried to call apply_discount without first calling get_product_price!\n"
                f"You MUST call get_product_price first to get the product price, "
                f"then call apply_discount with that price.\n"
                f"Do NOT guess or hardcode prices. Call get_product_price now."
            )
            print(f"RULE VIOLATION: {error_msg}")
            
            # Add the attempted tool call to history so the AI can see what it tried
            messages.append(ai_message)  # Add the AI's invalid attempt
            
            # Add an error message as a tool response
            # This teaches the AI: "That didn't work, try something else"
            messages.append({
                "role": "tool",
                "content": error_msg,  # Specific error explaining what went wrong
            })
            
            # Continue to the next iteration without running the tool
            # The AI will see the error and (hopefully) try something different
            continue

        # ====================================================================
        # REACT STEP 3: EXECUTE - Run the tool and observe the result
        # ====================================================================
        # Now that we've decided which tool to use and what arguments to pass,
        # we EXECUTE that tool and collect the result.
        #
        # This is the "Act" in ReAct:
        #   1. Reason: (done above) AI thought about what to do
        #   2. Act: (happening now) We actually do it
        #   3. Observe: (happening below) We see what happened
        #   4. Loop: Go back to step 1 with new information
        #
        # Try-except: We handle both successful execution and errors
        # This is important because tools can fail (network errors, invalid data, etc.)
        
        try:
            # Actually call the Python function
            # **tool_args unpacks the dictionary: {"product_name": "laptop"} becomes product_name="laptop"
            observation = tool_to_use(**tool_args)  # Execute the tool
            print(f"Tool returned: {observation}")
            
            # UPDATE STATE: Track what we've done
            # We need to remember that we've called get_product_price so
            # later we can enforce the guardrail (don't call apply_discount without price)
            if tool_name == "get_product_price":
                price_retrieved = True  # Mark that we now have a real price from the tool
            
            # ================================================================
            # REACT STEP 4: OBSERVE - Add result to conversation history
            # ================================================================
            # Now we update the conversation history with:
            #   1. The AI's tool call (what it decided to do)
            #   2. The tool's result (what happened when we executed it)
            #
            # This is the "Observe" step - we observe the result and make it
            # visible to the AI by adding it to the history.
            # On the next loop iteration, the AI will see this result and
            # reason about what to do next.
            #
            # Conversation flow becomes:
            #   [previous history]
            #   User: "What's the price of a laptop with gold discount?"
            #   Assistant: [AI message with tool_calls]
            #   Tool: "The price is $999"
            #   (Now we loop back to Step 1)
            #   Assistant: [AI thinks and decides to call apply_discount]
            #   Tool: "The discounted price is $849.15"
            #   (Now loop back to Step 1)
            #   Assistant: [AI thinks and decides to stop, gives final answer]
            #   (Loop exits because no tool calls)
            
            # Append the AI's message (the one with tool_calls) to history
            messages.append(ai_message)
            
            # Append the tool's result as a tool response
            # Format: {"role": "tool", "content": "<result string>"}
            messages.append({
                "role": "tool",
                "content": str(observation),  # Convert result to string (even if it was a number)
            })
            
        except Exception as e:
            # ERROR HANDLING: Tool execution failed
            # This can happen if:
            #   - Tool received wrong types for arguments
            #   - Tool raised an exception
            #   - Tool couldn't complete (database down, API error, etc.)
            #
            # We handle this by:
            #   1. Creating a helpful error message
            #   2. Adding it to the conversation history
            #   3. Giving the AI a chance to recover on the next iteration
            #
            # The AI will read the error and (hopefully) understand
            # what went wrong and how to fix it.
            
            error_msg = f"Error calling {tool_name}: {str(e)}\n\nReminder: Follow the rules exactly. For discounted prices, call get_product_price FIRST with only product_name, then call apply_discount with the numeric price returned and the discount_tier."
            print(f"Tool error: {error_msg}")
            
            # Add both the failed attempt and the error to history
            messages.append(ai_message)  # Add what the AI tried to do
            messages.append({
                "role": "tool",
                "content": error_msg,  # Add the error explanation
            })
            # Loop continues - AI will see the error and retry
    
    # LOOP EXIT: Safety limit reached
    # If we get here, the AI didn't finish within MAX_ITERATIONS.
    # This is a safety mechanism to prevent infinite loops.
    #
    # In production, you might want to:
    #   - Return the best answer so far
    #   - Raise an exception
    #   - Timeout gracefully
    #   - Log the problem for analysis
    print("Error: MAX Iterations reached without a final answer.")
    print("The agent didn't complete within the iteration limit.")
    return None


# ============================================================================
# SUMMARY: What You've Just Seen
# ============================================================================
#
# YOU JUST BUILT A RAW REACT AGENT LOOP!
# Without LangChain, without LangGraph, using ONLY:
#   - Raw Ollama SDK (for LLM calls)
#   - Basic Python functions (for tools)
#   - Plain dictionaries and lists (for message history)
#   - A simple for-loop (for the ReAct cycle)
#
# KEY COMPONENTS OF ANY REACT AGENT:
# ===================================
# 1. TOOL DEFINITIONS: Functions that do actual work
# 2. TOOL SCHEMAS: JSON descriptions of tools for the LLM
# 3. SYSTEM MESSAGE: Instructions on how to behave
# 4. FEW-SHOT EXAMPLES: Teaching through examples
# 5. CONVERSATION HISTORY: Keep track of reasoning
# 6. THE LOOP:
#    - Call LLM with history
#    - Extract tool calls from response
#    - Execute tools
#    - Add results to history
#    - Repeat until done
#
# WHAT LANGCHAIN ABSTRACTS AWAY:
# ================================
# 1. Tool Schema Generation
#    - We: Write 50+ lines of JSON per tool
#    - LangChain: @tool decorator auto-generates
#
# 2. Message Formatting
#    - We: Manually create dicts with "role"/"content"
#    - LangChain: Message objects (HumanMessage, AIMessage, etc.)
#
# 3. Tool Invocation
#    - We: Direct function calls and manual execution
#    - LangChain: tool.invoke(args) - unified interface
#
# 4. Vendor Differences
#    - We: Handle Ollama format OR rewrite for OpenAI/Anthropic
#    - LangChain: Single code, works with any vendor
#
# 5. Tracing and Observability
#    - We: Manually add @traceable decorators everywhere
#    - LangChain: Automatic LangSmith integration
#
# 6. Guardrails and Safety
#    - We: Manually implement state tracking and rule enforcement
#    - LangChain: Can use built-in patterns and tools
#
# WHY LANGCHAIN IS VALUABLE:
# ============================
# Building this raw loop taught you what's REALLY happening.
# But in production, you don't want to:
#   - Rewrite schemas when switching LLM vendors
#   - Manually format messages for different APIs
#   - Recreate the same guardrails in every project
#   - Debug LLM calls without tracing infrastructure
#
# LangChain solves these problems.
# Now that you understand the raw loop, you can
# appreciate what LangChain provides and use it effectively.


# ============================================================================
# ENTRY POINT: Execute the agent
# ============================================================================
# This is where the program starts when you run it.
# We call run_agent with a sample question and watch the ReAct loop work.

if __name__ == "__main__":
    print("="*80)
    print("Welcome to the Raw Ollama Agent Loop!")
    print("No LangChain, no LangGraph - just raw Ollama SDK and pure Python logic")
    print("="*80)
    print()
    
    # Run the agent with a sample question
    # This will:
    #   1. Initialize the message history
    #   2. Start the ReAct loop (max 10 iterations)
    #   3. Each iteration: LLM thinks -> calls tool -> we execute -> repeat
    #   4. Exit when AI gives final answer or max iterations reached
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    
    print("\n" + "="*80)
    print("Agent execution complete!")
    print("="*80)