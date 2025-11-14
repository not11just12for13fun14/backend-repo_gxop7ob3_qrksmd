import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import re
import requests

# Optional heavy imports guarded inside functions to speed cold start

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    question: str


class SolveResponse(BaseModel):
    answer: str
    explanation: Optional[str] = None
    qtype: str = "general"
    sources: List[str] = []
    created_at: datetime = datetime.utcnow()


@app.get("/")
def read_root():
    return {"message": "Homework Solver API is running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


# --------------------------
# Solver utilities
# --------------------------

MATH_PATTERN = re.compile(r"^[\s\d\+\-\*/\^\(\)xX=\.]+$")


def is_math_question(q: str) -> bool:
    q = q.strip()
    if any(word in q.lower() for word in ["solve", "simplify", "evaluate", "derivative", "integrate", "factor", "expand"]):
        return True
    return bool(MATH_PATTERN.match(q))


def solve_math(question: str) -> SolveResponse:
    # Defer import to speed startup
    import sympy as sp

    expr_text = question.replace("^", "**").strip()

    # Try to detect equation and solve for x
    if "=" in expr_text:
        left, right = expr_text.split("=", 1)
        # allow common symbols
        symbols = list({ch for ch in re.findall(r"[a-zA-Z]", expr_text)}) or ['x']
        syms = sp.symbols(' '.join(symbols))
        # Map single 'x' for convenience
        x = syms[0] if isinstance(syms, (list, tuple)) else syms
        try:
            sol = sp.solve(sp.Eq(sp.sympify(left), sp.sympify(right)), x)
            answer = f"{symbols[0]} = {sol}"
            explanation = f"Solved the equation symbolically for {symbols[0]}. Solutions: {sol}"
            return SolveResponse(answer=str(answer), explanation=explanation, qtype="math")
        except Exception:
            # Fall back to evaluation if possible
            pass

    try:
        sym = sp.sympify(expr_text)
        val = sp.simplify(sym)
        return SolveResponse(answer=str(val), explanation="Simplified the expression using symbolic math.", qtype="math")
    except Exception as e:
        return SolveResponse(answer="Could not parse math expression.", explanation=str(e), qtype="math")


WIKI_SEARCH_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_OPENSEARCH = "https://en.wikipedia.org/w/api.php"


def wiki_answer(question: str) -> Optional[SolveResponse]:
    try:
        # First, try direct summary by title guess
        title_guess = question.strip().rstrip('?').title()
        r = requests.get(WIKI_SEARCH_URL + requests.utils.quote(title_guess), timeout=6)
        if r.status_code == 200:
            data = r.json()
            if data.get("extract"):
                url = data.get("content_urls", {}).get("desktop", {}).get("page")
                return SolveResponse(
                    answer=data.get("title", title_guess),
                    explanation=data.get("extract"),
                    qtype="factual",
                    sources=[url] if url else []
                )
        # Fallback: opensearch
        params = {
            "action": "opensearch",
            "search": question,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }
        r2 = requests.get(WIKI_OPENSEARCH, params=params, timeout=6)
        if r2.status_code == 200:
            arr = r2.json()
            if len(arr) >= 4 and arr[1]:
                title = arr[1][0]
                desc = arr[2][0] if arr[2] else None
                link = arr[3][0] if arr[3] else None
                return SolveResponse(
                    answer=title,
                    explanation=desc,
                    qtype="factual",
                    sources=[link] if link else []
                )
    except Exception:
        pass
    return None


def generic_answer(question: str) -> SolveResponse:
    # Always provide a helpful, structured response and give quick search links
    q_lower = question.strip().rstrip('.').rstrip('?').lower()
    templates = []

    if q_lower.startswith("who is") or q_lower.startswith("who was"):
        templates.append("Try clarifying the person's full name and key role or era (e.g., 'Who is Ada Lovelace, the 19th‑century mathematician?').")
    elif q_lower.startswith("what is") or q_lower.startswith("what are"):
        templates.append("Add the domain for precision (e.g., 'What is entropy in thermodynamics?').")
    elif q_lower.startswith("when "):
        templates.append("Include the event and region (e.g., 'When did the Meiji Restoration begin in Japan?').")
    elif q_lower.startswith("where "):
        templates.append("Specify the context or field (e.g., geography, history, biology).")
    elif q_lower.startswith("why "):
        templates.append("Mention the mechanism or theory you expect (cause/effect, model names).")

    if not templates:
        templates.append("If it's factual, include specific keywords (names, dates, field). If it's math, include an explicit expression or equation.")

    # Provide quick search links so the user can open authoritative results
    try:
        google = f"https://www.google.com/search?q={requests.utils.quote(question)}"
        wiki = f"https://en.wikipedia.org/w/index.php?search={requests.utils.quote(question)}"
        sources = [google, wiki]
    except Exception:
        sources = []

    return SolveResponse(
        answer="Here's a concise explanation:",
        explanation=(
            "I couldn't find a direct factual summary. "
            + " ".join(templates)
        ),
        qtype="general",
        sources=sources,
    )


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Decide path
    if is_math_question(q):
        result = solve_math(q)
    else:
        result = wiki_answer(q) or generic_answer(q)

    # Persist to DB (best-effort)
    try:
        from database import create_document
        # Derive collection from schema name: "homeworkquery"
        doc = {
            "question": q,
            "answer": result.answer,
            "explanation": result.explanation,
            "qtype": result.qtype,
            "sources": result.sources,
        }
        create_document("homeworkquery", doc)
    except Exception:
        # Database might not be configured; ignore errors for functionality
        pass

    return result


@app.get("/history")
def history(limit: int = 10):
    try:
        from database import get_documents
        docs = get_documents("homeworkquery", {}, limit=limit)
        # Convert ObjectId and datetime to strings
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])
            for k, v in list(d.items()):
                if hasattr(v, "isoformat"):
                    d[k] = v.isoformat()
        return {"items": docs}
    except Exception:
        return {"items": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
