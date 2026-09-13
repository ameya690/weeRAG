from typing import Optional, Any, Callable
import time, json, sqlite3, hashlib
def _now(): return time.time()
def _sha256(s: str)->str: return hashlib.sha256(s.encode("utf-8")).hexdigest()
class Cache:
    def __init__(self, path: str = ".wee_cache.sqlite"):
        self.path=path; con=sqlite3.connect(self.path); 
        con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL, ttl REAL)"); con.commit(); con.close()
    def set(self, key_signature: str, value: Any, ttl_seconds: Optional[float]=None):
        k=_sha256(key_signature); v=json.dumps(value, ensure_ascii=False); created=_now(); ttl=float(ttl_seconds) if ttl_seconds is not None else None
        con=sqlite3.connect(self.path); con.execute("REPLACE INTO cache (key,value,created,ttl) VALUES (?,?,?,?)", (k,v,created,ttl)); con.commit(); con.close()
    def get(self, key_signature: str):
        k=_sha256(key_signature); con=sqlite3.connect(self.path); row=con.execute("SELECT value,created,ttl FROM cache WHERE key=?",(k,)).fetchone(); 
        if not row: con.close(); return None
        value, created, ttl=row
        if ttl is not None and (_now()>created+ttl): con.execute("DELETE FROM cache WHERE key=?",(k,)); con.commit(); con.close(); return None
        con.close(); return json.loads(value)
def cached(cache: Cache, namespace: str, key_fn: Callable[..., str]):
    def _wrap(fn):
        def wrapper(*args, **kwargs):
            sig=f"{namespace}|{key_fn(*args, **kwargs)}"; hit=cache.get(sig)
            if hit is not None: return hit
            out=fn(*args, **kwargs); cache.set(sig, out); return out
        return wrapper
    return _wrap
