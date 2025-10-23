from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
from typing import Optional, List, Dict
from dotenv import load_dotenv
import os
import httpx
import logging

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Boot + ENV
# ---------------------------
if os.getenv("RENDER") is None:
    print("Loading .env file for local development...")
    load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASS = os.getenv("NEO4J_PASS")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

# NEW: Gemini config (explanations are enabled only if API key exists)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")  # you can override

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://your-client-app-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

# Log on startup
logger.info("🚀 FastAPI server started. Routes loaded. Neo4j DB=%s | GeminiModel=%s | GeminiKey=%s",
            NEO4J_DB, GEMINI_MODEL, "SET" if GEMINI_API_KEY else "MISSING")

# ---------------------------
# Models
# ---------------------------
class Stmt(BaseModel):
    statement: str
    parameters: dict | None = None

class CypherBody(BaseModel):
    statements: list[Stmt]

class Recommendation(BaseModel):
    trackId: str
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    source: str

# NEW: v2 response adds an explanation (backward-compatible for clients that ignore extra fields)
class RecommendationResponseV2(RecommendationResponse):
    explanation: Optional[str] = None

class TrackLite(BaseModel):
    trackId: str
    name: str

class TrackDetail(BaseModel):
    trackId: str
    name: str
    durationSec: Optional[int] = None
    released: Optional[str] = None
    genre: Optional[str] = None
    subgenre: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[TrackLite]

# ---------------------------
# Helpers
# ---------------------------
@app.get("/")
def health():
    logger.info("💓 Health check hit")
    return {"ok": True}

def run_statements(statements: list[Stmt]):
    with GraphDatabase.driver(URI, auth=(USER, PASS)) as d:
        with d.session(database=NEO4J_DB) as s:
            out = []
            for st in statements:
                res = s.run(st.statement, st.parameters or {})
                out.append([r.data() for r in res])
            return out

@app.post("/cypher")
def cypher(body: CypherBody, authorization: str = Header(default="")):
    if not API_TOKEN or authorization != f"Bearer {API_TOKEN}":
        logger.warning("🔐 /cypher unauthorized attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        logger.info("🧠 /cypher executing %d statement(s)", len(body.statements))
        return {"results": run_statements(body.statements)}
    except (SessionExpired, ServiceUnavailable, OSError) as e:
        logger.warning("♻️  /cypher retry after exception: %s", e)
        return {"results": run_statements(body.statements)}

def require_auth(authorization: Optional[str] = Header(None, alias="Authorization")):
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

def run_query(statement: str, params: dict) -> List[dict]:
    with driver.session(database=NEO4J_DB) as s:
        res = s.run(statement, params)
        return res.data()

# ---------------------------
# Cypher
# ---------------------------
TRACK_SEARCH = """
MATCH (t:Track)
WHERE toLower(t.name) CONTAINS toLower($q)
   OR toLower(t.trackId) CONTAINS toLower($q)
RETURN t.trackId AS trackId, t.name AS name
ORDER BY name
LIMIT $limit
"""

TRACK_META = """
MATCH (t:Track {trackId:$tid})
OPTIONAL MATCH (t)-[:BelongsTo]->(alb:Category {level:'Album'})
OPTIONAL MATCH (alb)<-[:IsParentOf]-(art:Category {level:'Artist'})
OPTIONAL MATCH (art)<-[:IsParentOf]-(sub:Category {level:'Subgenre'})
OPTIONAL MATCH (sub)<-[:IsParentOf]-(gen:Category {level:'Genre'})
RETURN
  t.trackId AS trackId,
  t.name     AS name,
  t.durationSec AS durationSec,
  t.released AS released,
  gen.name   AS genre,
  sub.name   AS subgenre,
  art.name   AS artist,
  alb.name   AS album
"""

# NEW: batched version for many trackIds
TRACK_META_MANY = """
UNWIND $tids AS tid
MATCH (t:Track {trackId:tid})
OPTIONAL MATCH (t)-[:BelongsTo]->(alb:Category {level:'Album'})
OPTIONAL MATCH (alb)<-[:IsParentOf]-(art:Category {level:'Artist'})
OPTIONAL MATCH (art)<-[:IsParentOf]-(sub:Category {level:'Subgenre'})
OPTIONAL MATCH (sub)<-[:IsParentOf]-(gen:Category {level:'Genre'})
RETURN
  t.trackId AS trackId,
  t.name     AS name,
  t.durationSec AS durationSec,
  t.released AS released,
  gen.name   AS genre,
  sub.name   AS subgenre,
  art.name   AS artist,
  alb.name   AS album
ORDER BY name
"""

@app.get("/tracks/search", response_model=SearchResponse)
def tracks_search(
    q: str = Query(..., min_length=1, description="Search by name or trackId"),
    limit: int = Query(20, ge=1, le=50),
    _=Depends(require_auth)
):
    rows = run_query(TRACK_SEARCH, {"q": q, "limit": limit})
    logger.info("🔎 Search q='%s' -> %d rows", q, len(rows))
    return {"results": [TrackLite(**r).model_dump() for r in rows]}

@app.get("/tracks/{track_id}", response_model=TrackDetail)
def track_detail(
    track_id: str,
    _=Depends(require_auth)
):
    rows = run_query(TRACK_META, {"tid": track_id})
    if not rows:
        logger.info("❌ Track not found: %s", track_id)
        raise HTTPException(status_code=404, detail="Track not found")
    logger.info("ℹ️ Track detail fetched: %s", track_id)
    return TrackDetail(**rows[0]).model_dump()

# ---------------------------
# Recs (unchanged logic) + additive LLM explanation
# ---------------------------
DECAY_RATE = 0.98

TRACK_RECOMMENDATION_CYPHER = """
WITH $trackId AS tid, toInteger($k) AS k, toFloat($decay) AS d

// decayed RecIndex
CALL {
  WITH tid, d
  OPTIONAL MATCH (idx:RecIndex {pid: tid})
  WITH tid, d, CASE WHEN idx IS NULL THEN [] ELSE idx.recIds END AS ids
  UNWIND ids AS rid
  OPTIONAL MATCH (:Track {trackId: tid})-[r:InSameSession]-(rec:Track {trackId: rid})
  WITH d, rid,
       coalesce(r.popularity, 0) AS pop,
       coalesce(duration.between(datetime({epochMillis:r.last_seen}), datetime()).days, 0) AS ageDays
  WITH collect({ rid: rid, score: round(pop * (d ^ ageDays), 3) }) AS rows
  RETURN [x IN rows WHERE x.score > 0] AS idxList
}

// decayed edges
CALL {
  WITH tid, d
  OPTIONAL MATCH (:Track {trackId: tid})-[r:InSameSession]-(rec:Track)
  WHERE rec.trackId <> tid
  WITH rec.trackId AS rid,
       max(r.popularity) AS pop,
       max(coalesce(duration.between(datetime({epochMillis:r.last_seen}), datetime()).days, 0)) AS ageDays,
       d
  WITH rid, round(pop * (d ^ ageDays), 3) AS score
  ORDER BY score DESC
  RETURN [x IN collect({trackId: rid, score: score}) WHERE x.score > 0] AS edgeList
}

// album fallback
OPTIONAL MATCH (t:Track {trackId: tid})-[:BelongsTo]->(alb:Category)<-[:BelongsTo]-(albRec:Track)
WHERE albRec.trackId <> tid
WITH k, idxList, edgeList, collect({trackId: albRec.trackId, score: 0.5}) AS albumList

WITH k, idxList, edgeList, albumList,
     CASE WHEN size(idxList) > 0 THEN 'recindex+decay'
          WHEN size(edgeList) > 0 THEN 'edges+decay'
          ELSE 'album' END AS source,
     CASE WHEN size(idxList) > 0
          THEN [x IN idxList | {trackId: x.rid, score: x.score}]
          ELSE edgeList END AS baseList

WITH k, source, baseList, edgeList, albumList, [x IN baseList | x.trackId] AS have
WITH k, source, baseList + [e IN edgeList WHERE NOT e.trackId IN have] AS b2, albumList
WITH k, source, b2, [x IN b2 | x.trackId] AS have2, albumList
RETURN (b2 + [a IN albumList WHERE NOT a.trackId IN have2])[..k] AS recommendations, source
"""

# ---------------------------
# Tiny LLM explainer
# ---------------------------
def _by_id(rows: List[Dict]) -> Dict[str, Dict]:
    return {r["trackId"]: r for r in rows}

async def llm_explain_recs(
    seed: Dict,
    recs: List[Dict],
    source: str
) -> Optional[str]:
    """
    Ask Gemini for a short, human-friendly explanation of the top recs.
    Returns None on any error or if GEMINI_API_KEY is absent.
    """
    if not GEMINI_API_KEY:
        logger.warning("🟡 Gemini skipped — GEMINI_API_KEY missing")
        return None

    # Keep it brief: top 5 for the prompt
    top = recs[:5]
    lines = []
    for r in top:
        lines.append(
            f"- {r.get('name') or r['trackId']} "
            f"(id={r['trackId']}, artist={r.get('artist')}, album={r.get('album')}, "
            f"genre={r.get('genre')}, score={r.get('score')})"
        )
    seed_line = (
        f"{seed.get('name') or seed['trackId']} (id={seed['trackId']}, "
        f"artist={seed.get('artist')}, album={seed.get('album')}, genre={seed.get('genre')})"
    )

    prompt = (
        "You are a recommender **explainer**. "
        "Given a seed music track and a scored list of recommended tracks, "
        "write 3–5 concise sentences that tell a user why the *top items* are good picks. "
        "Prefer reasons like strong co-listening signals, same album/artist/genre, and recency (higher score). "
        "Do NOT include JSON—return plain text only.\n\n"
        f"Seed track:\n{seed_line}\n\n"
        f"Source of candidates: {source}\n\n"
        "Top recommendations (descending by score):\n"
        + "\n".join(lines)
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        logger.info("💬 Calling Gemini model=%s", GEMINI_MODEL)
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            # Best-effort extract
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
            )
            logger.info("✅ Gemini response received (%d chars)", len(text))
            return text or None
    except Exception as e:
        logger.error("🔴 Gemini request failed: %s - %s", type(e).__name__, e)
        return None

# ---------------------------
# Endpoint
# ---------------------------
@app.get("/tracks/{track_id}/recommendations", response_model=RecommendationResponseV2)
async def get_track_recommendations(
    track_id: str,
    k: int = Query(8, ge=1, le=50, alias="k"),
    _=Depends(require_auth)
):
    logger.info("🎧 Recs request | track_id=%s | k=%d", track_id, k)

    # Seed exists?
    rows = run_query("MATCH (t:Track {trackId:$tid}) RETURN t", {"tid": track_id})
    if not rows:
        logger.info("❌ Track not found for recs: %s", track_id)
        raise HTTPException(status_code=404, detail="Track not found")

    # Run recs via internal /cypher (same as before)
    body = CypherBody(
        statements=[
            Stmt(
                statement=TRACK_RECOMMENDATION_CYPHER,
                parameters={"trackId": track_id, "k": k, "decay": DECAY_RATE}
            )
        ]
    )

    cypher_url = "https://neo4j-cypher-proxy.onrender.com/cypher"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        logger.info("➡️  Posting to cypher proxy: %s", cypher_url)
        resp = await client.post(cypher_url, json=body.model_dump(), headers=headers)
        resp.raise_for_status()

    results = resp.json().get("results", [])
    if not results or not results[0]:
        logger.info("⚠️  Empty recs result for %s", track_id)
        return RecommendationResponseV2(recommendations=[], source="none", explanation=None)

    first = results[0][0]
    recs = first.get("recommendations", [])
    source = first.get("source", "unknown")
    logger.info("✅ Recs computed | count=%d | source=%s", len(recs), source)

    # ---- LLM explanation (non-blocking on failure) ----
    explanation = None
    try:
        seed_meta = run_query(TRACK_META, {"tid": track_id})
        seed_info = seed_meta[0] if seed_meta else {"trackId": track_id, "name": track_id}

        rec_ids = [r["trackId"] for r in recs[:5]]
        rec_meta_rows = run_query(TRACK_META_MANY, {"tids": rec_ids})
        meta_by_id = _by_id(rec_meta_rows)

        enriched = []
        for r in recs[:5]:
            meta = dict(meta_by_id.get(r["trackId"], {"trackId": r["trackId"], "name": r["trackId"]}))
            meta["score"] = r.get("score", 0.0)
            enriched.append(meta)

        explanation = await llm_explain_recs(seed_info, enriched, source)
    except Exception as e:
        logger.error("🔴 Explanation step failed but continuing: %s - %s", type(e).__name__, e)
        explanation = None  # never break the main response

    return RecommendationResponseV2(
        recommendations=recs,
        source=source,
        explanation=explanation
    )
