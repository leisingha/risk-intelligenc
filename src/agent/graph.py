"""LangGraph risk agent: plan → act → observe → reflect → (act | answer).

- plan:    scope guardrail, then choose tool calls (LLM via langchain-openai when
           OPENAI_API_KEY is set, otherwise a deterministic rule-based planner).
- act:     execute the next planned tool call.
- observe: record what came back (passages, scores) in the graph state.
- reflect: more planned calls? weak evidence worth one broadened retry? or answer.
- answer:  compose a cited answer, then the grounding guardrail approves it or the
           agent refuses.
The loop is hard-capped at AGENT_MAX_ITERATIONS tool calls; LangGraph's recursion
limit is a second backstop. Conversation history travels in the state object.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agent import guardrails as G
from src.agent.tools import TOOLS, corpus_info
from src.config import AGENT_MAX_ITERATIONS, CATEGORIES, EVAL_DIR, OPENAI_API_KEY, OPENAI_MODEL

CATEGORY_TERMS = {
    "cyber": ["cyber", "hack", "breach", "ransomware", "information security", "data security"],
    "climate": ["climate", "emission", "carbon", "greenhouse", "energy transition", "net zero"],
    "credit": ["credit", "default", "loan", "borrower", "counterparty"],
    "market": [
        "interest rate",
        "market risk",
        "volatility",
        "commodity price",
        "currency",
        "inflation",
        "oil price",
    ],
    "regulatory": [
        "regulat",
        "compliance",
        "legislation",
        "litigation",
        "antitrust",
        "export control",
        "sanction",
    ],
    "operational": ["operational", "supply chain", "outage", "disruption", "third-party", "vendor"],
}
SECTOR_TERMS = {
    "banking": ["bank", "banks", "banking", "lender"],
    "energy": ["energy", "oil", "oil and gas"],
    "technology": ["tech", "technology", "software", "semiconductor"],
}
QUOTED_RE = re.compile(r"[\"“](.+?)[\"”]", re.S)


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    planner: str
    intent: str
    planned_calls: list[dict]
    step: int
    iterations: int
    last_call: dict
    last_result: Any
    observations: list[dict]
    retrieved: dict[str, list[str]]
    best_score: float | None
    broadened: bool
    final_answer: str
    citations: list[str]
    refused: bool
    refusal_reason: str
    trace: list[str]


# ---------------------------------------------------------------- planning


def detect_categories(question: str) -> list[str]:
    q = question.lower()
    return [c for c in CATEGORIES if any(t in q for t in CATEGORY_TERMS[c])]


def detect_sector(question: str) -> str | None:
    q = question.lower()
    for sector, terms in SECTOR_TERMS.items():
        if any(re.search(rf"\b{re.escape(t)}\b", q) for t in terms):
            return sector
    return None


def rule_plan(question: str, info: dict) -> tuple[str, list[dict]]:
    quoted = QUOTED_RE.findall(question)
    if "classify" in question.lower() and quoted:
        return "classify", [{"tool": "classify_passage", "args": {"text": quoted[0]}}]
    companies = G.detect_companies(question, info["aliases"])
    categories = detect_categories(question)
    wants_compare = bool(re.search(r"\b(compare|comparison|versus|vs\.?)\b", question, re.I))
    if wants_compare and len(companies) >= 2 and categories:
        return "compare", [
            {
                "tool": "compare_companies",
                "args": {
                    "company_a": companies[0],
                    "company_b": companies[1],
                    "category": categories[0],
                },
            }
        ]
    sector = None if companies else detect_sector(question)
    if companies:
        return "search", [
            {"tool": "search_risk_disclosures", "args": {"query": question, "company": c}}
            for c in companies[:3]
        ]
    return "search", [
        {"tool": "search_risk_disclosures", "args": {"query": question, "sector": sector}}
    ]


PLANNER_SYSTEM = (
    "You plan tool calls for a risk-intelligence agent over 10-K Item 1A risk factors of "
    "these companies: {companies}. Call the tools needed to gather evidence; do not answer. "
    "Use search_risk_disclosures with a company filter when the user names a company; "
    "compare_companies when they compare two companies on one of {categories}; "
    "classify_passage only when they provide a passage to classify."
)


def llm_plan(question: str, info: dict, history: list[dict]) -> tuple[str, list[dict]]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from src.agent.tools import langchain_tools

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    messages = [
        SystemMessage(
            PLANNER_SYSTEM.format(
                companies=", ".join(info["tickers"]), categories=", ".join(CATEGORIES)
            )
        )
    ]
    for turn in history[-3:]:
        messages += [HumanMessage(turn["question"]), AIMessage(turn["answer"])]
    messages.append(HumanMessage(question))
    response = llm.bind_tools(langchain_tools()).invoke(messages)
    calls = [
        {"tool": tc["name"], "args": tc["args"]}
        for tc in response.tool_calls
        if tc["name"] in TOOLS
    ][:AGENT_MAX_ITERATIONS]
    tools_used = {c["tool"] for c in calls}
    intent = (
        "classify"
        if tools_used == {"classify_passage"}
        else "compare"
        if "compare_companies" in tools_used
        else "search"
    )
    return intent, calls


def plan_node(state: AgentState) -> AgentState:
    info = corpus_info()
    question = state["question"]
    trace = list(state.get("trace", []))
    # Quoted text to classify is user-supplied data, not a question about entities.
    scope = G.scope_check(QUOTED_RE.sub(" ", question), info["aliases"], info["max_filing_year"])
    if not scope.ok:
        trace.append(f"plan: scope guardrail refused ({scope.reason})")
        return {
            "refused": True,
            "refusal_reason": f"{scope.reason}: {scope.message}",
            "planned_calls": [],
            "trace": trace,
        }
    if OPENAI_API_KEY:
        intent, plan = llm_plan(question, info, state.get("history", []))
        planner = f"llm:{OPENAI_MODEL}"
    else:
        intent, plan = rule_plan(question, info)
        planner = "rule-based"
    trace.append(
        f"plan ({planner}): intent={intent}; calls="
        + "; ".join(f"{c['tool']}({_fmt_args(c['args'])})" for c in plan)
    )
    return {
        "planner": planner,
        "intent": intent,
        "planned_calls": plan,
        "step": 0,
        "trace": trace,
        "refused": False,
    }


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={str(v)[:50]!r}" for k, v in args.items() if v is not None)


# ---------------------------------------------------------------- act / observe / reflect


def act_node(state: AgentState) -> AgentState:
    call = state["planned_calls"][state["step"]]
    result = TOOLS[call["tool"]](**call["args"])
    trace = [*state.get("trace", []), f"act: {call['tool']}({_fmt_args(call['args'])})"]
    return {
        "last_call": call,
        "last_result": result,
        "step": state["step"] + 1,
        "iterations": state.get("iterations", 0) + 1,
        "trace": trace,
    }


def _hits_from(result: Any) -> list[dict]:
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "companies" in result:
        return [h for c in result["companies"].values() for h in c["top_passages"]]
    return []


def observe_node(state: AgentState) -> AgentState:
    result = state["last_result"]
    hits = _hits_from(result)
    retrieved = {k: list(v) for k, v in state.get("retrieved", {}).items()}
    for h in hits:
        chunks = retrieved.setdefault(h["passage_id"], [])
        if h["text"] not in chunks:
            chunks.append(h["text"])
    scores = [h["score"] for h in hits]
    prev = state.get("best_score")
    best = max([s for s in [prev, *scores] if s is not None], default=None)
    if hits:
        note = (
            f"observe: {len(hits)} passages, best {max(scores):.3f} "
            f"({hits[0]['passage_id']}, {hits[0]['company']})"
        )
    elif isinstance(result, dict) and "labels" in result:
        note = f"observe: classifier {result['model']} → {result['labels'] or 'no label ≥ 0.5'}"
    else:
        note = "observe: no passages returned"
    obs = [*state.get("observations", []), {"call": state["last_call"], "result": result}]
    return {
        "observations": obs,
        "retrieved": retrieved,
        "best_score": best,
        "trace": [*state.get("trace", []), note],
    }


def reflect_node(state: AgentState) -> AgentState:
    trace = list(state.get("trace", []))
    iterations = state.get("iterations", 0)
    if iterations >= AGENT_MAX_ITERATIONS:
        trace.append(f"reflect: iteration cap {AGENT_MAX_ITERATIONS} reached → answer")
        return {"trace": trace}
    if state["step"] < len(state["planned_calls"]):
        trace.append(
            f"reflect: {len(state['planned_calls']) - state['step']} planned call(s) left → act"
        )
        return {"trace": trace}
    weak = state.get("best_score") is None or state["best_score"] < G.MIN_RELEVANCE
    if state.get("intent") != "classify" and weak and not state.get("broadened"):
        last = state["last_call"]
        args = dict(last["args"])
        cats = detect_categories(state["question"])
        new_query = " ".join(CATEGORY_TERMS[cats[0]][:4]) + " risk" if cats else state["question"]
        if last["tool"] == "search_risk_disclosures":
            args = {"query": new_query, "company": args.get("company"), "sector": None}
        else:
            args = {"query": new_query}
        trace.append(
            "reflect: evidence weak → one broadened search (no sector filter, " "category keywords)"
        )
        return {
            "planned_calls": [
                *state["planned_calls"],
                {"tool": "search_risk_disclosures", "args": args},
            ],
            "broadened": True,
            "trace": trace,
        }
    trace.append("reflect: evidence gathered → answer")
    return {"trace": trace}


def route_after_reflect(state: AgentState) -> str:
    if state.get("iterations", 0) >= AGENT_MAX_ITERATIONS:
        return "answer"
    return "act" if state["step"] < len(state["planned_calls"]) else "answer"


def route_after_plan(state: AgentState) -> str:
    return "answer" if state.get("refused") or not state.get("planned_calls") else "act"


# ---------------------------------------------------------------- answering


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", re.sub(r"\s+", " ", text))
    return [
        p.strip()
        for p in parts
        if 8 <= len(p.split()) <= 60 and '"' not in p and "“" not in p and "”" not in p
    ]


def _best_sentence(question: str, text: str, aliases: dict[str, list[str]]) -> str | None:
    """The sentence sharing the most topic words with the question; None if it shares none,
    so a passage that is merely nearby in embedding space is never quoted as an answer.
    Requires at least MIN_SENTENCE_RELEVANCE of the question's topic words."""
    scored = [(G.query_relevance(question, s, aliases), s) for s in _sentences(text)]
    scored = [(r, s) for r, s in scored if r >= G.MIN_SENTENCE_RELEVANCE]
    if not scored:
        return None
    return max(scored, key=lambda rs: (rs[0], -abs(len(rs[1].split()) - 25)))[1]


def compose_rule_answer(state: AgentState) -> str:
    """Extractive: quote the most relevant sentence of the top passages, with citations."""
    question = state["question"]
    hits: list[dict] = []
    for obs in state.get("observations", []):
        hits.extend(_hits_from(obs["result"]))
    aliases = corpus_info()["aliases"]
    seen, used_sentences, per_company, lines = set(), set(), {}, []
    limit = 2 if state.get("intent") == "compare" or len({h["ticker"] for h in hits}) > 1 else 3
    for h in sorted(hits, key=lambda h: h["score"], reverse=True):
        if h["passage_id"] in seen or h["score"] < G.MIN_RELEVANCE:
            continue
        if per_company.get(h["ticker"], 0) >= limit:
            continue
        sentence = _best_sentence(question, h["text"], aliases)
        if sentence is None or sentence in used_sentences:
            continue
        used_sentences.add(sentence)
        seen.add(h["passage_id"])
        per_company[h["ticker"]] = per_company.get(h["ticker"], 0) + 1
        year = str(h["filing_date"])[:4]
        lines.append(
            f'{h["company"]} ({year} 10-K) states: "{sentence.rstrip(".")}." '
            f'[{h["passage_id"]}]'
        )
        if len(lines) >= 6:
            break
    return " ".join(lines)


ANSWER_SYSTEM = (
    "Answer the question using ONLY the passages below from 10-K risk factors. Rules: "
    "every sentence must contain a short verbatim quote in double quotes copied exactly "
    "from a passage, and end with that passage's id in square brackets, e.g. [XOM-2024-007]. "
    "No forecasts, no outside knowledge. If the passages do not answer the question, reply "
    "exactly INSUFFICIENT_EVIDENCE."
)


def compose_llm_answer(state: AgentState) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    passages = "\n\n".join(
        f"[{pid}]\n{' '.join(chunks)}" for pid, chunks in state.get("retrieved", {}).items()
    )
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    msg = llm.invoke(
        [
            SystemMessage(ANSWER_SYSTEM),
            HumanMessage(f"Passages:\n{passages}\n\nQuestion: {state['question']}"),
        ]
    )
    return str(msg.content).strip()


def answer_node(state: AgentState) -> AgentState:
    trace = list(state.get("trace", []))
    if state.get("refused"):
        return _finish(state, "", [], True, state["refusal_reason"], trace)

    if state.get("intent") == "classify":
        res = state["observations"][-1]["result"]
        ranked = sorted(res["scores"].items(), key=lambda kv: kv[1], reverse=True)
        labels = ", ".join(res["labels"]) or "none above 0.5"
        text = (
            f"Predicted risk categories ({res['model']}): {labels}. Scores: "
            + ", ".join(f"{c} {p:.2f}" for c, p in ranked)
            + "."
        )
        trace.append("answer: classifier output (no corpus claims to ground)")
        return _finish(state, text, [], False, "", trace)

    rel = G.check_relevance(state.get("best_score"))
    if not rel.ok:
        trace.append(f"answer: relevance guardrail refused ({rel.reason})")
        return _finish(state, "", [], True, f"{rel.reason}: {rel.message}", trace)

    draft = compose_llm_answer(state) if OPENAI_API_KEY else compose_rule_answer(state)
    if not draft or draft == "INSUFFICIENT_EVIDENCE":
        trace.append("answer: no supportable sentence → refuse")
        return _finish(
            state,
            "",
            [],
            True,
            "insufficient_evidence: the retrieved passages "
            "do not contain a statement that answers the question.",
            trace,
        )
    retrieved = {pid: " ".join(chunks) for pid, chunks in state.get("retrieved", {}).items()}
    grounding = G.verify_grounding(draft, retrieved)
    if not grounding.ok:
        trace.append(f"answer: grounding guardrail refused ({grounding.reason})")
        detail = "; ".join(grounding.unsupported[:3])
        return _finish(state, "", [], True, f"ungrounded: {grounding.reason}. {detail}", trace)
    trace.append(f"answer: grounded, {len(grounding.citations)} citation(s)")
    return _finish(state, draft, grounding.citations, False, "", trace)


def _finish(
    state: AgentState,
    answer: str,
    citations: list[str],
    refused: bool,
    reason: str,
    trace: list[str],
) -> AgentState:
    shown = answer if not refused else f"I can't answer this from the corpus. {reason}"
    history = [*state.get("history", []), {"question": state["question"], "answer": shown}]
    return {
        "final_answer": shown,
        "citations": citations,
        "refused": refused,
        "refusal_reason": reason,
        "trace": trace,
        "history": history,
    }


# ---------------------------------------------------------------- graph


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("observe", observe_node)
    g.add_node("reflect", reflect_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "answer": "answer"})
    g.add_edge("act", "observe")
    g.add_edge("observe", "reflect")
    g.add_conditional_edges("reflect", route_after_reflect, {"act": "act", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


_GRAPH = None


def ask(question: str, history: list[dict] | None = None) -> dict:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    final = _GRAPH.invoke(
        {"question": question, "history": history or [], "trace": []},
        config={"recursion_limit": 4 * AGENT_MAX_ITERATIONS + 6},
    )
    out = {
        k: final.get(k)
        for k in [
            "question",
            "citations",
            "refused",
            "refusal_reason",
            "planner",
            "intent",
            "iterations",
            "trace",
            "history",
        ]
    }
    out["answer"] = final.get("final_answer", "")
    return out


def format_transcript(res: dict) -> str:
    lines = [f"Q: {res['question']}", f"A: {res['answer']}"]
    if res["citations"]:
        lines.append(f"Citations: {', '.join(res['citations'])}")
    lines.append("Trace:")
    lines += [f"  - {t}" for t in res["trace"]]
    return "\n".join(lines)


def run_adversarial() -> dict:
    items = [
        json.loads(line)
        for line in (EVAL_DIR / "adversarial.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows, md = [], ["# Adversarial transcript", ""]
    for item in items:
        res = ask(item["question"])
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "refused": res["refused"],
                "reason": (res["refusal_reason"] or "").split(":")[0] or "-",
            }
        )
        md += [f"## {item['id']} ({item['type']})", "", "```", format_transcript(res), "```", ""]
        print(format_transcript(res), "\n")
    refused = sum(r["refused"] for r in rows)
    (EVAL_DIR / "adversarial_transcript.md").write_text("\n".join(md))
    return {
        "adversarial_questions": len(rows),
        "adversarial_refused": refused,
        "adversarial_refusal_rate": round(refused / len(rows), 4),
        "adversarial": rows,
    }


def run_demo() -> dict:
    questions = json.loads((EVAL_DIR / "demo_questions.json").read_text())
    rows, md = [], ["# Demo transcript", ""]
    history: list[dict] = []
    for q in questions:
        res = ask(q, history)
        history = res["history"]
        print(format_transcript(res), "\n")
        md += ["```", format_transcript(res), "```", ""]
        rows.append(
            {
                "question": q,
                "refused": res["refused"],
                "citations": len(res["citations"] or []),
                "iterations": res["iterations"] or 0,
            }
        )
    (EVAL_DIR / "demo_transcript.md").write_text("\n".join(md))
    answered = [r for r in rows if not r["refused"]]
    return {
        "demo_questions": len(rows),
        "demo_answered": len(answered),
        "demo_answered_with_citations": sum(r["citations"] > 0 for r in answered),
        "demo": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="5 scripted questions")
    parser.add_argument("--adversarial", action="store_true", help="10 should-refuse questions")
    parser.add_argument("--ask")
    args = parser.parse_args()
    from src.results import update

    if args.ask:
        print(format_transcript(ask(args.ask)))
    if args.demo:
        update("agent", {"planner": "llm" if OPENAI_API_KEY else "rule-based", **run_demo()})
    if args.adversarial:
        update("agent", run_adversarial())


if __name__ == "__main__":
    main()
