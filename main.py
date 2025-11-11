from langgraph.graph import StateGraph, END
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
import pandas as pd

ws_games = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KoEx6OWGb5iavmQWIQ6hMQ/world-series-games.csv")
playoff_games = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Nd4TGZ1HYlZc-p8s06KYCg/playoff-games.csv")
regular_games = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/coIqzejj3J9DlSftshDCqg/regular-season-games.csv")
team_stats = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_Rsg4b5HAFYefjMZ7pJaag/team-stats.csv")
pitching_stats = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/x1dk2AsQQ0COacggGBd68w/pitcher-stats.csv")
batting_stats = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/o4jTZU_aepNOBdYH6L32iw/player-batting-stats.csv")

print(f"World Series games:     {len(ws_games)}")
print(f"Playoff games:          {len(playoff_games)}")
print(f"Regular season games:   {len(regular_games)}")
print(f"Teams:                  {len(team_stats)}")
print(f"Pitchers:               {len(pitching_stats)}")
print(f"Batters:                {len(batting_stats)}")

# display(ws_games.head(3))
# print('='*200)
# display(playoff_games.head(3))

class State(TypedDict):
    question: str
    data_source: str
    answer: str

# llm = ChatOpenAI(
#     model = "gpt-4",
#     temperature = 0,
#     max_retries=2
# )

# llm = ChatOllama(model="llama3.2", temperature=0)

llm = ChatOllama(model="granite4:3b", temperature=0)

# Batting (player_batting_stats.csv)
batting_agent = create_pandas_dataframe_agent(
    llm, batting_stats,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# Pitching (pitcher_stats.csv)
pitching_agent = create_pandas_dataframe_agent(
    llm, pitching_stats,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# Teams (team_stats.csv)
team_agent = create_pandas_dataframe_agent(
    llm, team_stats,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# World Series games (world_series_games.csv)
games_ws_agent = create_pandas_dataframe_agent(
    llm, ws_games,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# Postseason games (playoff_games.csv)
games_playoffs_agent = create_pandas_dataframe_agent(
    llm, playoff_games,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# Regular season games (regular_season_games.csv)
games_regular_agent = create_pandas_dataframe_agent(
    llm, regular_games,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# Define the nodes for the StateGraph
def batting_node(state: State) -> State:
    """Query the batting statistics agent."""
    try:
        result = batting_agent.invoke({"input": state["question"]})
        answer = result.get("output") or result.get("output_text") or str(result)
    except Exception as e:
        answer = f"Error querying batting stats: {str(e)[:200]}"
        print(f"Error occurred: {answer}")
    
    return {
        "question": state["question"],
        "data_source": state["data_source"],
        "answer": answer
    }


def pitching_node(state: State) -> State:
    """Query the pitching statistics agent."""
    try:
        result = pitching_agent.invoke({"input": state["question"]})
        answer = result.get("output") or result.get("output_text") or str(result)
    except Exception as e:
        answer = f"Error querying pitching stats: {str(e)[:200]}"
        print(f"Error occurred: {answer}")
    
    return {
        "question": state["question"],
        "data_source": state["data_source"],
        "answer": answer
    }


def team_node(state: State) -> State:
    """Query the team statistics agent."""
    try:
        result = team_agent.invoke({"input": state["question"]})
        answer = result.get("output") or result.get("output_text") or str(result)
    except Exception as e:
        answer = f"Error querying team stats: {str(e)[:200]}"
        print(f"Error occurred: {answer}")
    
    return {
        "question": state["question"],
        "data_source": state["data_source"],
        "answer": answer
    }


def ws_games_node(state: State) -> State:
    """Query the World Series games agent."""
    try:
        result = games_ws_agent.invoke({"input": state["question"]})
        answer = result.get("output") or result.get("output_text") or str(result)
    except Exception as e:
        answer = f"Error querying World Series games: {str(e)[:200]}"
        print(f"Error occurred: {answer}")
    
    return {
        "question": state["question"],
        "data_source": state["data_source"],
        "answer": answer
    }


def playoff_games_node(state: State) -> State:
    """Query the playoff games agent."""
    try:
        result = games_playoffs_agent.invoke({"input": state["question"]})
        answer = result.get("output") or result.get("output_text") or str(result)
    except Exception as e:
        answer = f"Error querying playoff games: {str(e)[:200]}"
        print(f"Error occurred: {answer}")
    
    return {
        "question": state["question"],
        "data_source": state["data_source"],
        "answer": answer
    }


def regular_games_node(state: State) -> State:
    """Query the regular season games agent."""
    try:
        result = games_regular_agent.invoke({"input": state["question"]})
        answer = result.get("output") or result.get("output_text") or str(result)
    except Exception as e:
        answer = f"Error querying regular season games: {str(e)[:200]}"
        print(f"Error occurred: {answer}")
    
    return {
        "question": state["question"],
        "data_source": state["data_source"],
        "answer": answer
    }


# Routing

def route_question(state: State) -> State:
    """Use an LLM to intelligently route the question to the appropriate agent node."""
    
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant for a baseball data analysis system. 
        Your job is to analyze the user's question and determine which dataset would be most appropriate to answer it.
        
        Available datasets and their nodes:
        - batting_node: Player batting statistics (batting average, hits, home runs, RBIs, OPS, slugging)
        - pitching_node: Pitcher statistics (ERA, WHIP, strikeouts, saves, innings pitched)
        - team_node: Team-level statistics (wins, losses, win percentage, run differential, standings)
        - ws_games_node: World Series game results (scores, winners, game-by-game results)
        - playoff_games_node: Playoff games excluding World Series (Wild Card, Division Series, Championship Series, ALCS, NLCS, ALDS, NLDS)
        - regular_games_node: Regular season game results (scores, dates, matchups from April through September)
        
        Respond with ONLY the node name, nothing else. For example: "batting_node" or "pitching_node"
        
        If the question could apply to multiple datasets, choose the most specific one.
        If you're unsure, default to "ws_games_node"."""),
        ("user", "{question}")
    ])
    
    routing_chain = routing_prompt | llm
    
    try:
        response = routing_chain.invoke({"question": state["question"]})
        data_source = response.content.strip()
        
        valid_nodes = [
            "batting_node", "pitching_node", "team_node",
            "ws_games_node", "playoff_games_node", "regular_games_node"
        ]
        
        if data_source not in valid_nodes:
            print(f"LLM returned invalid node '{data_source}', defaulting to ws_games_node")
            data_source = "ws_games_node"
        else:
            print(f"LLM routing to: {data_source}")
            
    except Exception as e:
        print(f"Error in LLM routing: {e}, defaulting to ws_games_node")
        data_source = "ws_games_node"
    
    return {
        "question": state["question"],
        "data_source": data_source,
        "answer": "",
    }


# Routing node
def route_question(state: State) -> State:
    """Use an LLM to intelligently route the question to the appropriate agent node."""
    
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant for a baseball data analysis system. 
        Your job is to analyze the user's question and determine which dataset would be most appropriate to answer it.
        
        Available datasets and their nodes:
        - batting_node: Player batting statistics (batting average, hits, home runs, RBIs, OPS, slugging)
        - pitching_node: Pitcher statistics (ERA, WHIP, strikeouts, saves, innings pitched)
        - team_node: Team-level statistics (wins, losses, win percentage, run differential, standings)
        - ws_games_node: World Series game results (scores, winners, game-by-game results)
        - playoff_games_node: Playoff games excluding World Series (Wild Card, Division Series, Championship Series, ALCS, NLCS, ALDS, NLDS)
        - regular_games_node: Regular season game results (scores, dates, matchups from April through September)
        
        Respond with ONLY the node name, nothing else. For example: "batting_node" or "pitching_node"
        
        If the question could apply to multiple datasets, choose the most specific one.
        If you're unsure, default to "ws_games_node"."""),
        ("user", "{question}")
    ])
    
    routing_chain = routing_prompt | llm
    
    try:
        response = routing_chain.invoke({"question": state["question"]})
        data_source = response.content.strip()
        
        valid_nodes = [
            "batting_node", "pitching_node", "team_node",
            "ws_games_node", "playoff_games_node", "regular_games_node"
        ]
        
        if data_source not in valid_nodes:
            print(f"LLM returned invalid node '{data_source}', defaulting to ws_games_node")
            data_source = "ws_games_node"
        else:
            print(f"LLM routing to: {data_source}")
            
    except Exception as e:
        print(f"Error in LLM routing: {e}, defaulting to ws_games_node")
        data_source = "ws_games_node"
    
    return {
        "question": state["question"],
        "data_source": data_source,
        "answer": "",
    }

workflow = StateGraph(State)

workflow.add_node("route", route_question)

workflow.add_node("batting_node", batting_node)
workflow.add_node("pitching_node", pitching_node)
workflow.add_node("team_node", team_node)
workflow.add_node("ws_games_node", ws_games_node)
workflow.add_node("playoff_games_node", playoff_games_node)
workflow.add_node("regular_games_node", regular_games_node)

workflow.set_entry_point("route")


workflow.add_conditional_edges(
    "route",                
    lambda state: state["data_source"],
    {                         
        "batting_node": "batting_node",
        "pitching_node": "pitching_node",
        "team_node": "team_node",
        "ws_games_node": "ws_games_node",
        "playoff_games_node": "playoff_games_node",
        "regular_games_node": "regular_games_node"
    }
)

workflow.add_edge("batting_node", END)
workflow.add_edge("pitching_node", END)
workflow.add_edge("team_node", END)
workflow.add_edge("ws_games_node", END)
workflow.add_edge("playoff_games_node", END)
workflow.add_edge("regular_games_node", END)

GRAPH = workflow.compile()


def ask(question: str):
    state = {"question": question, "data_source": "", "answer": ""}
    print("\n" + "="*70)
    print("❓ Question:", question)
    print("="*70)

    result = GRAPH.invoke(state)

    print("\n" + "="*70)
    print("💡 ANSWER")
    print("="*70)
    print(result.get("answer", ""))
    print("="*70 + "\n")

    return result
    

# result=ask("Which player has the highest batting average?")

ask("Which pitcher has the lowest ERA?")

# ask("What is the current World Series score?")