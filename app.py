from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
import os

API_TOKEN = os.getenv("API_TOKEN")
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASS = os.getenv("NEO4J_PASS")
API_TOKEN  = os.getenv("API_TOKEN")

driver = GraphDatabase.driver(URI, auth=(USER, PASS))

app = FastAPI()

class Stmt(BaseModel):
    statement: str
    parameters: dict | None = None

class CypherBody(BaseModel):
    statements: list[Stmt]

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
