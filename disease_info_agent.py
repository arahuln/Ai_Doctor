import os
# import os
import json
import time
from typing import List, Dict, Union
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import initialize_agent, AgentType
# from langchain.tools import DuckDuckGoSearchRun
# from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()


# NVIDIA API key (set in your environment)
# export NVIDIA_API_KEY="your_key_here"

# Initialize NVIDIA LLM
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# Add a search tool
# search = DuckDuckGoSearchRun()
search = TavilySearch(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    num_results=3,
    depth = "basic",
    topic = "general",
)

# Create agent with tools
# agent = initialize_agent(
#     tools=[search],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# def get_disease_info(diseases_with_probs):
#     """
#     Takes dict {disease: prob} and queries web for prevention & medications.
#     """
#     results = {}
#     for disease, prob in diseases_with_probs.items():
#         if prob >= 0.3:  # only consider significant diseases
#             query = f"What are the medications, dosage timings, prevention steps, and important care details for {disease}?"
#             response = agent.run(query)
#             results[disease] = response
#     return results
# if __name__ == "__main__":
#     # Example usage
#     example_diseases = {
#         "Diabetes": 0.85,
#         "Hypertension": 0.65,
#         "Common Cold": 0.25  # This one should be ignored
#     }
#     info = get_disease_info(example_diseases)
#     for disease, details in info.items():
#         print(f"Disease: {disease}\nDetails: {details}\n{'-'*40}\n")

# How many search results/snippets to fetch per disease
SEARCH_TOP_K = 5

# Pause between web calls to be polite (seconds)
SEARCH_SLEEP = 0.6

# Initialize LLM and search tool (singletons)
# _llm = ChatNVIDIA(model=NVIDIA_MODEL)
# _search = DuckDuckGoSearchRun()

# ---------- Helper prompt builder ----------
def _build_prompt(disease: str, snippets: List[Dict[str, str]]) -> str:
    """
    Build a careful prompt that instructs the LLM to use the provided snippets and
    return a strict JSON object with specific fields.
    Each snippet should be a dict: {"title": ..., "snippet": ..., "link": ...}
    """
    snippet_text = ""
    for i, s in enumerate(snippets, 1):
        title = s.get("title", "") or ""
        txt = s.get("snippet", "") or s.get("text", "") or ""
        link = s.get("link", "") or s.get("source", "") or ""
        snippet_text += f"Snippet {i} Title: {title}\nSnippet {i} Text: {txt}\nSnippet {i} URL: {link}\n\n"

    prompt = f"""
You are a careful medical-assistant style summarizer. Use ONLY the information present in the provided web snippets to prepare a concise JSON object describing the disease: "{disease}".

If the snippets don't clearly provide medication or dosing instructions, say "Not Found" for those fields rather than inventing them.

REQUIRED OUTPUT: Reply with a single valid JSON object (no extra commentary). The JSON must contain the following keys:
- disease (string)
- medication (string): recommended medications if present (comma-separated or concise list). If not present say "Not Found".
- when_to_take (string): timing / dosing guidance if present. If not present say "Not Found".
- prevention (string): prevention steps, vaccines, hygiene tips if present; else "Not Found".
- other (string): any other important points (complications, warnings). If none, "Not Found".
- sources (array of strings): include the URLs you used (from the snippets), up to {SEARCH_TOP_K}.

Snippets (use these, do NOT search again except to validate — use the snippet text as source):
{snippet_text}

Now produce the JSON object for the disease "{disease}".
"""
    return prompt.strip()

# ---------- Search wrapper ----------
def _run_search(query: str, top_k: int = SEARCH_TOP_K) -> List[Dict[str, str]]:
    """
    Run DuckDuckGo search and return a list of snippet dicts: {"title","snippet","link"}.
    The DuckDuckGo wrapper sometimes returns varied key names; normalize them.
    """
    try:
        results = search.run(query)  # Tools often return single string or list; handle both.
    except Exception as e:
        # fallback: return empty list
        print(f"Search failed for query '{query}': {e}")
        return []

    snippets = []

    # If the tool returned a JSON-like string or large text, we still try to parse sensible parts.
    # The duckduckgo tool usually returns a list of dicts or a single string with results.
    if isinstance(results, list):
        for item in results[:top_k]:
            if isinstance(item, dict):
                snippets.append({
                    "title": item.get("title") or item.get("heading") or "",
                    "snippet": item.get("snippet") or item.get("body") or item.get("text") or "",
                    "link": item.get("link") or item.get("url") or item.get("source") or ""
                })
            else:
                # non-dict entries: put as snippet text
                snippets.append({"title": "", "snippet": str(item), "link": ""})
    elif isinstance(results, str):
        # Try to split into lines and take first top_k meaningful lines
        lines = [l.strip() for l in results.splitlines() if l.strip()]
        for line in lines[:top_k]:
            snippets.append({"title": "", "snippet": line, "link": ""})
    else:
        # unknown shape
        snippets.append({"title": "", "snippet": str(results), "link": ""})

    # ensure we have at most top_k snippets
    return snippets[:top_k]

# ---------- Main function ----------
def get_disease_info(diseases: Union[List[str], Dict[str, float]]) -> Dict[str, Dict]:
    """
    Input:
        diseases: either a list of disease names or a dict {disease_name: prob}.
    Returns:
        dict: {disease_name: parsed_info_dict_or_raw_text}
    parsed_info_dict has keys: disease, medication, when_to_take, prevention, other, sources
    """
    # Normalize input
    if isinstance(diseases, dict):
        disease_items = [(d, float(p)) for d, p in diseases.items()]
    else:
        disease_items = [(d, 1.0) for d in diseases]

    results = {}
    for disease, prob in disease_items:
        # skip low-prob diseases here if desired (caller already filtered by >0.3)
        query = f"{disease} medications dosing prevention symptoms treatment guidelines"
        snippets = _run_search(query, top_k=SEARCH_TOP_K)
        time.sleep(SEARCH_SLEEP)  # be polite / rate limit

        prompt = _build_prompt(disease, snippets)

        try:
            # ChatNVIDIA usage: call with system/user messages or a simple prompt depending on wrapper.
            res = llm.call_as_llm(prompt) if hasattr(llm, "call_as_llm") else llm(prompt)
            # Some wrappers return an object, some return string; normalize
            if isinstance(res, dict) and "content" in res:
                llm_text = res["content"]
            elif hasattr(res, "content"):
                llm_text = res.content
            else:
                llm_text = str(res)
        except Exception as e:
            print(f"LLM call failed for {disease}: {e}")
            results[disease] = {"error": str(e), "raw": ""}
            continue

        # Try to parse JSON out of the model response
        parsed = None
        try:
            parsed = json.loads(llm_text)
        except Exception:
            # attempt to extract JSON substring
            start = llm_text.find("{")
            end = llm_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(llm_text[start:end+1])
                except Exception:
                    parsed = None

        if parsed and isinstance(parsed, dict):
            # ensure keys exist
            for k in ["disease", "medication", "when_to_take", "prevention", "other", "sources"]:
                if k not in parsed:
                    parsed[k] = "Not Found"
            # normalize sources to list
            if not isinstance(parsed.get("sources", []), list):
                parsed["sources"] = [parsed.get("sources")] if parsed.get("sources") else []
            results[disease] = parsed
        else:
            # fallback: save raw LLM text + snippets for transparency
            results[disease] = {
                "disease": disease,
                "medication": "Not Found",
                "when_to_take": "Not Found",
                "prevention": "Not Found",
                "other": "Not Found",
                "sources": [s.get("link") for s in snippets if s.get("link")],
                "raw": llm_text.strip()
            }

    return results

if __name__ == "__main__":
    print("Demo run (will try searching).")
    demo = get_disease_info(["Pneumonia", "Common cold"])
    print(json.dumps(demo, indent=2))