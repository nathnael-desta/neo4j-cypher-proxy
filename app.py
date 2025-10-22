from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
from typing import Optional, List
from dotenv import load_dotenv
import os
import httpx

if os.getenv("RENDER") is None:
    # This block runs ONLY on your local machine
    print("Loading .env file for local development...") # Optional: good for debugging
    load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASS = os.getenv("NEO4J_PASS")
API_TOKEN  = os.getenv("API_TOKEN")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")


driver = GraphDatabase.driver(URI, auth=(USER, PASS))

app = FastAPI()

# Define the origins that are allowed to access your API
# Replace "http://localhost:3000" with the actual URL of your client app
origins = [
    # Allow your local development server
    "http://localhost:8080",
    # Allow a specific production client URL (if applicable)
    "https://your-client-app-domain.com",
    # Optional: You can allow all origins in development (NOT RECOMMENDED FOR PRODUCTION)
    # "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # List of allowed origins
    allow_credentials=True,             # Allow cookies/authorization headers to be passed
    allow_methods=["*"],                # Allow all methods (GET, POST, etc.)
    allow_headers=["*", "Authorization"], # Allow all headers, explicitly including Authorization
)

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

@app.get("/")
def health():
    return {"ok": True}

def run_statements(statements: list[Stmt]):
    # Build a fresh driver per request -> avoids stale/defunct connections
    with GraphDatabase.driver(URI, auth=(USER, PASS)) as driver:
        with driver.session(database="neo4j") as s:
            out = []
            for st in statements:
                res = s.run(st.statement, st.parameters or {})
                out.append([r.data() for r in res])
            return out

@app.post("/cypher")
def cypher(body: CypherBody, authorization: str = Header(default="")):
    if not API_TOKEN or authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return {"results": run_statements(body.statements)}
    except (SessionExpired, ServiceUnavailable, OSError):
        # One quick retry in case Aura just rotated/closed connections
        return {"results": run_statements(body.statements)}


def require_auth(authorization: Optional[str] = Header(None, alias="Authorization")):
    if not API_TOKEN:  # no auth configured → allow
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

def run_query(statement: str, params: dict) -> List[dict]:
    with driver.session(database=NEO4J_DB) as s:
        res = s.run(statement, params)
        return res.data()  # list[dict]


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


@app.get("/tracks/search", response_model=SearchResponse)
def tracks_search(
    q: str = Query(..., min_length=1, description="Search by name or trackId"),
    limit: int = Query(20, ge=1, le=50),
    _=Depends(require_auth)
):
    rows = run_query(TRACK_SEARCH, {"q": q, "limit": limit})
    return {"results": [TrackLite(**r).model_dump() for r in rows]}

@app.get("/tracks/{track_id}", response_model=TrackDetail)
def track_detail(
    track_id: str,
    _=Depends(require_auth)
):
    rows = run_query(TRACK_META, {"tid": track_id})
    if not rows:
        raise HTTPException(status_code=404, detail="Track not found")
    return TrackDetail(**rows[0]).model_dump()

DECAY_RATE = 0.98

TRACK_RECOMMENDATION_CYPHER = """
WITH $trackId AS tid, toInteger($k) AS k, toFloat($decay) AS d

// --- RecIndex (decayed by joining the real edge for last_seen)
OPTIONAL MATCH (idx:RecIndex {pid: tid})
WITH tid, k, d, idx,
     CASE WHEN idx IS NULL THEN [] ELSE idx.recIds END AS ids
UNWIND ids AS rid
OPTIONAL MATCH (:Track {trackId: tid})-[r:InSameSession]-(rec:Track {trackId: rid})
WITH tid, k, d, rid,
     coalesce(r.popularity, 0) AS pop,
     coalesce(duration.between(datetime({epochMillis:r.last_seen}), datetime()).days, 0) AS ageDays
WITH tid, k, d,
     [x IN collect({rid: rid, score: round(pop * pow(d, ageDays), 3)})
        WHERE x.score > 0] AS idxList

// --- Live edge list (decayed)
OPTIONAL MATCH (:Track {trackId: tid})-[r:InSameSession]-(rec:Track)
WHERE rec.trackId <> tid
WITH tid, k, d, idxList, rec, r,
     coalesce(duration.between(datetime({epochMillis:r.last_seen}), datetime()).days, 0) AS ageDays
WITH tid, k, idxList,
     rec.trackId AS rid,
     round(max(r.popularity * pow(d, ageDays)), 3) AS score
ORDER BY score DESC
WITH tid, k, idxList,
     [x IN collect({trackId: rid, score: score}) WHERE x.score > 0] AS edgeList

// --- Album fallback
OPTIONAL MATCH (t:Track {trackId: tid})-[:BelongsTo]->(alb:Category)<-[:BelongsTo]-(albRec:Track)
WHERE albRec.trackId <> tid
WITH k, idxList, edgeList, collect({trackId: albRec.trackId, score: 0.5}) AS albumList

// --- Choose base & fill to K
WITH k, idxList, edgeList, albumList,
     CASE WHEN size(idxList) > 0 THEN 'recindex+decay'
          WHEN size(edgeList) > 0 THEN 'edges+decay'
          ELSE 'album' END AS source

WITH k, source,
     CASE WHEN source STARTS WITH 'recindex' THEN
          // project idxList into uniform shape
          [ {trackId: x.rid, score: x.score} IN idxList ]
     ELSE edgeList END AS baseList,
     edgeList, albumList

WITH k, source, baseList, edgeList, [x IN baseList | x.trackId] AS have
WITH k, source,
     baseList + [e IN edgeList WHERE NOT e.trackId IN have] AS basePlusEdges,
     albumList

WITH k, source, basePlusEdges, [x IN basePlusEdges | x.trackId] AS have2, albumList
RETURN (basePlusEdges + [a IN albumList WHERE NOT a.trackId IN have2])[..k] AS recommendations,
       source
"""


@app.get("/tracks/{track_id}/recommendations", response_model=RecommendationResponse)
async def get_track_recommendations(
    track_id: str,
    k: int = Query(8, ge=1, le=50, alias="k"),
    _=Depends(require_auth)
):
    rows = run_query("MATCH (t:Track {trackId:$tid}) RETURN t", {"tid": track_id})
    if not rows:
        raise HTTPException(status_code=404, detail="Track not found")

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
    async with httpx.AsyncClient() as client:
        resp = await client.post(cypher_url, json=body.model_dump(), headers=headers)
        resp.raise_for_status()

    results = resp.json().get("results", [])
    if not results or not results[0]:
        return RecommendationResponse(recommendations=[], source="none")
    first = results[0][0]
    return RecommendationResponse(
        recommendations=first.get("recommendations", []),
        source=first.get("source", "unknown")
    )
