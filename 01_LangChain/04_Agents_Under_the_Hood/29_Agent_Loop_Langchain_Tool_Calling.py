# LangChain Code to Implement an Agent Loop with Tool Calling and Error Recovery, basically a ReAct pattern's manual implementation. This example uses a simple product pricing scenario to demonstrate how an agent can reason, call tools, observe results, and recover from errors in a loop.

from dotenv import load_dotenv
from langsmith import traceable  # LangSmith for tracing/debugging agent runs
from langchain.chat_models import init_chat_model  # Abstraction to switch models easily
from langchain.tools import tool  # Decorator to convert Python functions into LangChain tools
from langchain.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

# Configuration
MAX_ITERATIONS = 10  # Prevent infinite loops - agent stops after 10 iterations
MODEL = "llama3.2"  # Ollama model to use (can switch to "openai:gpt-4" easily)


# ============================================================================
# TOOLS - Functions the agent can call to perform actions
# ============================================================================
# The @tool decorator converts regular Python functions into LangChain tools
# that can be bound to an LLM and called during the agent loop

@tool
def get_product_price(product_name: str) -> float:
    """Gets the price of a product. Returns the numeric price value."""
    # Simulated product catalog - in production this would be a database call
    prices = {
        "laptop": 999.0,
        "smartphone": 499.0,
        "headphones": 199.0
    }
    # Return the price or 0.0 if product not found
    return prices.get(product_name.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Applies a discount to a price based on the discount tier."""
    # Discount percentages for each tier
    discounts = {
        "bronze": 0.05,   # 5% off
        "silver": 0.10,   # 10% off
        "gold": 0.15      # 15% off
    }
    discount = discounts.get(discount_tier.lower(), 0)
    discounted_price = price * (1 - discount)
    return round(discounted_price, 2)


# ============================================================================
# AGENT LOOP - The ReAct pattern implementation
# ============================================================================
# ReAct = Reasoning + Acting in a loop
# The agent: Thinks -> Calls Tool -> Observes Result -> Repeats until done

@traceable(name="LangChain Agent Loop")  # Enable LangSmith tracing for debugging
def run_agent(question: str):
    """
    Runs an agent loop to answer a question using tools.
    
    The agent follows this pattern:
    1. Receives a question
    2. Decides which tool to call (if any)
    3. Executes the tool and gets the result
    4. Uses the result to decide the next step
    5. Repeats until it has a final answer
    """
    
    # ========================================================================
    # SETUP: Prepare tools and LLM
    # ========================================================================
    
    # List of available tools the agent can use
    tools = [get_product_price, apply_discount]
    
    # Create a dictionary for quick tool lookup by name
    # e.g., {"get_product_price": <tool function>, "apply_discount": <tool function>}
    tools_dict = {t.name: t for t in tools}

    # Initialize the language model
    # temperature=0.0 makes it deterministic (less creative, more consistent)
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
    
    # Bind tools to the LLM - this teaches the model what tools are available
    # and how to call them (parameters, descriptions, etc.)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("Agent is thinking...")

    # ========================================================================
    # STATE TRACKING: Enforce sequential tool calling
    # ========================================================================
    # We track whether get_product_price has been called to prevent the model
    # from guessing prices or calling apply_discount without a real price
    
    price_retrieved = False  # Has get_product_price been called yet?
    retrieved_price = None   # The actual price returned (unused but available)

    # ========================================================================
    # CONVERSATION HISTORY: System prompt + Few-shot examples + User question
    # ========================================================================
    # Messages are the "memory" of the agent - everything it has said and heard
    
    messages = [
        # SYSTEM MESSAGE: The agent's instructions and rules
        SystemMessage(content=(
            "You are a helpful assistant that can use tools to answer questions. "
            "You have access to a product catalog tool and can apply discounts based on customer tier. "
            "\n\nSTRICT RULES:\n"
            "1. For discounted prices, ALWAYS call get_product_price FIRST\n"
            "2. Then call apply_discount with the numeric result\n"
            "3. apply_discount takes ONLY: price (float) and discount_tier (string)\n"
            "4. NEVER pass product_name to apply_discount\n"
            "5. NEVER skip tools or calculate yourself\n"
            "6. If you get an error, immediately call the correct tool - do NOT give up"
        )),
        
        # ====================================================================
        # FEW-SHOT EXAMPLE 1: Show the correct tool-calling sequence
        # ====================================================================
        # Teaching by example - this shows the model the exact pattern to follow
        
        HumanMessage(content="What is the price of headphones with silver discount?"),
        # Agent calls get_product_price first (correct!)
        AIMessage(content="", tool_calls=[{
            "name": "get_product_price",
            "args": {"product_name": "headphones"},
            "id": "call_example_1",
            "type": "tool_call"
        }]),
        # Tool returns the price
        ToolMessage(content="199.0", tool_call_id="call_example_1"),
        # Agent then calls apply_discount with the numeric price
        AIMessage(content="", tool_calls=[{
            "name": "apply_discount",
            "args": {"price": 199.0, "discount_tier": "silver"},
            "id": "call_example_2",
            "type": "tool_call"
        }]),
        # Tool returns the discounted price
        ToolMessage(content="179.1", tool_call_id="call_example_2"),
        # Agent gives final answer
        AIMessage(content="The price of headphones with silver discount is $179.10"),
        
        # ====================================================================
        # FEW-SHOT EXAMPLE 2: Show error recovery pattern
        # ====================================================================
        # This is CRITICAL - it teaches the model how to recover from mistakes
        # Many models (like llama3.2) need to see how to handle errors
        
        HumanMessage(content="What is the price of smartphone with bronze discount?"),
        # Agent makes a mistake - tries to call apply_discount first (wrong!)
        AIMessage(content="", tool_calls=[{
            "name": "apply_discount",
            "args": {"price": 500, "discount_tier": "bronze"},
            "id": "call_bad_1",
            "type": "tool_call"
        }]),
        # System returns an error
        ToolMessage(content="ERROR: You tried to call apply_discount without first calling get_product_price! You MUST call get_product_price first.", tool_call_id="call_bad_1", status="error"),
        # Agent RECOVERS by calling get_product_price (learns from error!)
        AIMessage(content="", tool_calls=[{
            "name": "get_product_price",
            "args": {"product_name": "smartphone"},
            "id": "call_fix_1",
            "type": "tool_call"
        }]),
        # Tool returns the correct price
        ToolMessage(content="499.0", tool_call_id="call_fix_1"),
        # Agent now calls apply_discount with the correct price
        AIMessage(content="", tool_calls=[{
            "name": "apply_discount",
            "args": {"price": 499.0, "discount_tier": "bronze"},
            "id": "call_fix_2",
            "type": "tool_call"
        }]),
        # Tool returns the discounted price
        ToolMessage(content="474.05", tool_call_id="call_fix_2"),
        # Agent gives final answer
        AIMessage(content="The price of smartphone with bronze discount is $474.05"),
        
        # ====================================================================
        # ACTUAL USER QUESTION
        # ====================================================================
        HumanMessage(content=question)
    ]

    # ========================================================================
    # AGENT LOOP: The ReAct cycle (Reason -> Act -> Observe -> Repeat)
    # ========================================================================
    
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {i}")
        
        # ====================================================================
        # STEP 1: AI REASONING - Model decides what to do next
        # ====================================================================
        # The model looks at the conversation history and decides:
        # - Should I call a tool? Which one? With what arguments?
        # - Or do I have enough information to give a final answer?
        
        ai_message = llm_with_tools.invoke(messages)
        
        # Uncomment these to see what the model is thinking:
        # print(f"DEBUG - tool_calls: {ai_message.tool_calls}")
        # print(f"DEBUG - content: {ai_message.content}")
        
        tool_calls = ai_message.tool_calls

        # ====================================================================
        # CHECK: Is this the final answer?
        # ====================================================================
        # If the model doesn't want to call any tools, it's giving a final answer
        
        if not tool_calls:
            print("Agent has stopped making tool calls. Final answer:")
            print(ai_message.content)
            return ai_message.content

        # ====================================================================
        # STEP 2: TOOL SELECTION - Extract tool call details
        # ====================================================================
        # The model may suggest multiple tools, but we force ONE tool per iteration
        # This makes the agent loop more predictable and easier to debug
        
        tool_call = tool_calls[0]  # Only process the first tool call
        tool_name = tool_call.get("name")        # e.g., "get_product_price"
        tool_args = tool_call.get("args", {})    # e.g., {"product_name": "laptop"}
        tool_call_id = tool_call.get("id")       # Unique ID to link call and result
        
        print(f"Agent called tool: {tool_name} with args: {tool_args}")

        # Look up the actual tool function from our tools dictionary
        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found")

        # ====================================================================
        # RULE ENFORCEMENT: Check sequential tool calling rules
        # ====================================================================
        # This is a GUARDRAIL we added because llama3.2 sometimes tries to cheat
        # by guessing prices instead of calling get_product_price
        # This code-level enforcement catches violations and forces correction
        
        if tool_name == "apply_discount" and not price_retrieved:
            # The model tried to skip get_product_price - BLOCK IT!
            error_msg = (
                f"ERROR: You tried to call apply_discount without first calling get_product_price!\n"
                f"You MUST call get_product_price first to get the product price, "
                f"then call apply_discount with that price.\n"
                f"Do NOT guess or hardcode prices. Call get_product_price now."
            )
            print(f"RULE VIOLATION: {error_msg}")
            
            # Add the error to conversation history so model can learn and fix it
            messages.append(ai_message)
            messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id, status="error"))
            continue  # Skip to next iteration - give model a chance to correct

        # ====================================================================
        # STEP 3: TOOL EXECUTION - Actually run the tool
        # ====================================================================
        
        try:
            # Execute the tool with the arguments the model provided
            observation = tool_to_use.invoke(tool_args)
            print(f"Tool returned: {observation}")
            
            # Track state: Did we just get a product price?
            if tool_name == "get_product_price":
                price_retrieved = True    # Mark that we now have a real price
                retrieved_price = observation  # Store it (for potential future use)
            
            # ================================================================
            # STEP 4: UPDATE CONVERSATION HISTORY
            # ================================================================
            # Add both the AI's tool call and the tool's result to history
            # This is how the agent "remembers" what happened
            
            messages.append(ai_message)  # What the AI said/did
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))  # Tool result
            
        except Exception as e:
            # Handle tool execution errors (e.g., wrong argument types)
            error_msg = f"Error calling {tool_name}: {str(e)}\n\nReminder: Follow the rules exactly. For discounted prices, call get_product_price FIRST with only product_name, then call apply_discount with the numeric price returned and the discount_tier."
            print(f"Tool error: {error_msg}")
            
            # Add error to conversation so model can learn and retry
            messages.append(ai_message)
            messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id, status="error"))
    
    # If we get here, we hit MAX_ITERATIONS without a final answer
    print("Error: MAX Iterations reached without a final answer.")
    return None


# ============================================================================
# ENTRY POINT: Run the agent
# ============================================================================

if __name__ == "__main__":
    print("Welcome to the Agent Loop!")
    result = run_agent("What is the price of a laptop after applying a gold discount?")