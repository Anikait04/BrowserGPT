import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated, Sequence
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
from browsertools import tools, get_browser

# Try different models based on availability
# Priority: tool-use-preview > llama-3.1 > llama-3.3
MODEL_OPTIONS = [
    "openai/gpt-oss-20b",  # Best for tools
              # Latest but may have issues
]

def create_llm(model_name: str):
    """Create LLM with proper tool binding"""
    return ChatGroq(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=2
    ).bind_tools(tools)

# Start with the best model for tool calling
llm = create_llm(MODEL_OPTIONS[0])

class AgentState(TypedDict):
    goal: str
    messages: Annotated[Sequence[BaseMessage], "Message history"]
    steps: int
    max_steps: int
    last_action: str
    page_content: str

async def agent_node(state: AgentState):
    """Agent decision node - decides which tool to call next"""
    if state["steps"] >= state["max_steps"]:
        print(f"\n⚠️ Max steps ({state['max_steps']}) reached")
        return {
            "messages": state["messages"] + [AIMessage(content="Max steps reached, stopping")],
            "steps": state["steps"] + 1,
            "last_action": "max_steps_reached"
        }
    
    # Build context-aware prompt with clear tool instructions
    system_prompt = """You are BrowserGPT, an expert web automation agent.

YOUR MISSION: Use browser tools to accomplish the user's goal step by step.

🔧 CRITICAL RULES:
1. You MUST call exactly ONE tool per response
2. NEVER write explanatory text without calling a tool
3. After navigate/click, ALWAYS call read_page next to see the page
4. Call finish_task ONLY when goal is 100% complete

📋 WORKFLOW PATTERN:
navigate → read_page → type_text → click_element → read_page → finish_task

🌐 GOOGLE SEARCH EXAMPLE:
1. navigate("https://google.com")
2. read_page()  ← See the search box
3. type_text("textarea[name='q']|your query|enter")
4. read_page()  ← Verify results loaded
5. finish_task("Search completed")

⚡ COMMON SELECTORS:
- Google search: textarea[name="q"] or input[name="q"]
- Submit buttons: button[type="submit"] or input[type="submit"]
- By ID: #element-id
- By class: .class-name

🎯 YOUR NEXT ACTION: Choose ONE tool that progresses toward the goal."""

    # Simplified user prompt
    recent_actions = state.get('last_action', 'none')
    page_preview = state.get('page_content', 'No content yet')[:400]
    
    user_prompt = f"""🎯 GOAL: {state['goal']}

📊 STATUS:
- Step: {state['steps']}/{state['max_steps']}
- Last action: {recent_actions}
- Page preview: {page_preview}

❓ What is the ONE tool you should call now to make progress?"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ] + list(state["messages"][-4:])  # Keep last 4 messages
    
    print(f"\n🤔 Agent thinking... (Step {state['steps'] + 1}/{state['max_steps']})")
    
    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        error_msg = str(e)
        print(f"❌ LLM Error: {error_msg}")
        
        # If tool calling fails, try alternative model
        if "tool_use_failed" in error_msg or "400" in error_msg:
            print("🔄 Trying alternative model...")
            alternative_llm = create_llm(MODEL_OPTIONS[1])
            try:
                response = await alternative_llm.ainvoke(messages)
            except Exception as e2:
                print(f"❌ Alternative model also failed: {e2}")
                raise
        else:
            raise
    
    return {
        "messages": state["messages"] + [response],
        "steps": state["steps"] + 1,
        "last_action": state.get("last_action", ""),
        "page_content": state.get("page_content", "")
    }

async def tool_execution_node(state: AgentState):
    """Execute tools and capture results"""
    last_msg = state["messages"][-1]
    
    if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
        return state
    
    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": [last_msg]})
    
    # Extract tool name and result
    tool_name = last_msg.tool_calls[0]['name']
    tool_result = result["messages"][-1].content if result["messages"] else ""
    
    print(f"   ✅ {tool_name}: {tool_result[:100]}")
    
    # Update page content if we read the page
    page_content = state.get("page_content", "")
    if tool_name == "read_page":
        page_content = tool_result
    
    return {
        "messages": state["messages"] + result["messages"],
        "steps": state["steps"],
        "last_action": tool_name,
        "page_content": page_content
    }

def router(state: AgentState):
    """Route based on last message"""
    last_msg = state["messages"][-1]
    
    # Check if we hit max steps
    if state["steps"] >= state["max_steps"]:
        print("\n🛑 Stopping: Max steps reached")
        return END
    
    # Check if task is finished
    if state.get("last_action") == "finish_task":
        print("\n✅ Task completed!")
        return END
    
    # Check if agent called a tool
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        tool_name = last_msg.tool_calls[0]['name']
        print(f"🔧 Executing: {tool_name}")
        return "tools"
    
    # If no tool call, end
    print("\n⚠️ No tool called, ending")
    return END

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", router, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

async def run_agent(goal: str, max_steps: int = 15):
    """Run the browser agent with a specific goal"""
    print(f"🚀 Starting browser agent")
    print(f"📋 Goal: {goal}")
    print(f"⚙️ Max steps: {max_steps}")
    print(f"🤖 Model: {llm.model_name}\n")
    
    try:
        initial_state = {
            "goal": goal,
            "messages": [],
            "steps": 0,
            "max_steps": max_steps,
            "last_action": "none",
            "page_content": ""
        }
        
        # Stream execution
        async for chunk in app.astream(initial_state, stream_mode="values"):
            pass  # Progress is printed in nodes
        
        print("\n" + "="*50)
        print("✅ Agent execution finished!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            browser = await get_browser()
            await browser.close()
            print("\n🔒 Browser closed")
        except:
            pass

async def main():
    """Main entry point with example tasks"""
    
    # Verify API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in .env file")
        return
    
    print("="*60)
    print("🌐 BROWSER AUTOMATION AGENT")
    print("="*60 + "\n")
    
    # TASK 1: Simple Google search
    await run_agent(
        goal="Go to google.com and search for 'LangGraph' and then click the first result",
        max_steps=12
    )
    
    # Uncomment to try other tasks:
    
    # TASK 2: Navigate and read
    # await run_agent(
    #     goal="Go to github.com and read the main page content",
    #     max_steps=8
    # )
    
    # TASK 3: Search with screenshot
    # await run_agent(
    #     goal="Search Google for 'Playwright Python' and take a screenshot",
    #     max_steps=12
    # )
    
    # TASK 4: Multiple searches
    # await run_agent(
    #     goal="Go to google.com, search for 'web scraping ethics', "
    #          "read the results, then take a screenshot",
    #     max_steps=15
    # )

if __name__ == "__main__":
    asyncio.run(main())