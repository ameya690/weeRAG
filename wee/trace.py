from typing import List, Dict, Any, Optional
import time, json, contextlib, threading
class Span:
    __slots__=("name","start","end","attrs","children")
    def __init__(self, name: str, attrs: Optional[Dict[str,Any]]=None):
        self.name=name; self.start=time.time(); self.end=None; self.attrs=dict(attrs or {}); self.children=[]
    def close(self): 
        if self.end is None: self.end=time.time()
    @property
    def duration(self)->float: return (self.end or time.time())-self.start
class Tracer:
    def __init__(self): self.root_spans=[]; self._local=threading.local()
    def _stack(self):
        if not hasattr(self._local,"stack"): self._local.stack=[]
        return self._local.stack
    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        s=Span(name, attrs); st=self._stack()
        if st: st[-1].children.append(s)
        else: self.root_spans.append(s)
        st.append(s)
        try: yield s
        finally: s.close(); st.pop()
    def export_json(self)->Dict[str,Any]:
        def to_dict(sp: Span)->Dict[str,Any]:
            return {"name":sp.name,"start":sp.start,"end":sp.end,"duration":sp.duration,"attrs":sp.attrs,"children":[to_dict(c) for c in sp.children]}
        return {"spans": [to_dict(s) for s in self.root_spans]}
    def export_html(self)->str:
        spans=[]
        def collect(sp,depth): spans.append((sp,depth)); [collect(c,depth+1) for c in sp.children]
        [collect(r,0) for r in self.root_spans]
        if not spans: return "<html><body><p>No spans.</p></body></html>"
        t0=min(sp.start for sp,_ in spans); t1=max((sp.end or time.time()) for sp,_ in spans); total=max(t1-t0,1e-9)
        rows=[]
        for sp,depth in spans:
            left=100*(sp.start-t0)/total; width=100*sp.duration/total
            rows.append(f'<div style="position:relative;margin-left:{depth*16}px;height:22px;"><div style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;height:16px;background:#9ecae1;border-radius:4px;"></div><div style="position:absolute;left:0;top:0;height:22px;line-height:22px;font-family:monospace;font-size:12px;">{sp.name} <span style="color:#555">({sp.duration:.3f}s)</span></div></div>')
        return "<html><body><h3>wee trace</h3>"+"\n".join(rows)+"</body></html>"
