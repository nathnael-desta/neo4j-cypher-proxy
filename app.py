from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
import os

API_TOKEN = os.getenv("API_TOKEN")
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASS = os.getenv("NEO4J_PASS")

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
