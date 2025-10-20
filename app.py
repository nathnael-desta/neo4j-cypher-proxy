from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
import os

API_TOKEN = os.getenv("API_TOKEN")
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASS = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
app = FastAPI()

class Stmt(BaseModel):
    statement: str
    parameters: dict | None = None

class CypherBody(BaseModel):
    statements: list[Stmt]

@app.post("/cypher")
def cypher(body: CypherBody, authorization: str = Header(default="")):
    # simple bearer check
    if not API_TOKEN or authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    out = []
    with driver.session(database="neo4j") as s:
        for st in body.statements:
            res = s.run(st.statement, st.parameters or {})
            out.append([r.data() for r in res])
    return {"results": out}
