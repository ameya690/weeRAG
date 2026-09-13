import contextlib
import os
import threading
import time
from typing import Any


class Span:
    __slots__ = ("name", "start", "end", "attrs", "children")

    def __init__(self, name: str, attrs: dict[str, Any] | None = None):
        self.name = name
        self.start = time.time()
        self.end = None
        self.attrs = dict(attrs or {})
        self.children = []

    def close(self):
        if self.end is None:
            self.end = time.time()

    @property
    def duration(self) -> float:
        return (self.end or time.time()) - self.start


class Tracer:
    # Mapping from friendly wee attr names to OpenTelemetry GenAI semantic
    # convention attribute names.  Attributes whose keys appear here are
    # translated; all others are passed through with a ``wee.`` prefix.
    ATTR_MAP: dict[str, str] = {
        "model": "gen_ai.request.model",
        "input_tokens": "gen_ai.usage.input_tokens",
        "output_tokens": "gen_ai.usage.output_tokens",
        "temperature": "gen_ai.request.temperature",
        "system": "gen_ai.system",
    }

    def __init__(self):
        self.root_spans = []
        self._local = threading.local()

    def _stack(self):
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        s = Span(name, attrs)
        st = self._stack()
        if st:
            st[-1].children.append(s)
        else:
            self.root_spans.append(s)
        st.append(s)
        try:
            yield s
        finally:
            s.close()
            st.pop()

    def export_json(self) -> dict[str, Any]:
        def to_dict(sp: Span) -> dict[str, Any]:
            return {
                "name": sp.name,
                "start": sp.start,
                "end": sp.end,
                "duration": sp.duration,
                "attrs": sp.attrs,
                "children": [to_dict(c) for c in sp.children],
            }
        return {"spans": [to_dict(s) for s in self.root_spans]}

    def export_html(self) -> str:
        spans = []

        def collect(sp, depth):
            spans.append((sp, depth))
            for c in sp.children:
                collect(c, depth + 1)

        for r in self.root_spans:
            collect(r, 0)

        if not spans:
            return "<html><body><p>No spans.</p></body></html>"

        t0 = min(sp.start for sp, _ in spans)
        t1 = max((sp.end or time.time()) for sp, _ in spans)
        total = max(t1 - t0, 1e-9)

        rows = []
        for sp, depth in spans:
            left = 100 * (sp.start - t0) / total
            width = 100 * sp.duration / total
            bar = (
                f'<div style="position:relative;margin-left:{depth * 16}px;height:22px;">'
                f'<div style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;'
                f'height:16px;background:#9ecae1;border-radius:4px;"></div>'
                f'<div style="position:absolute;left:0;top:0;height:22px;line-height:22px;'
                f'font-family:monospace;font-size:12px;">'
                f'{sp.name} <span style="color:#555">({sp.duration:.3f}s)</span>'
                f"</div></div>"
            )
            rows.append(bar)

        return "<html><body><h3>wee trace</h3>\n" + "\n".join(rows) + "</body></html>"

    # ------------------------------------------------------------------
    # OpenTelemetry GenAI semantic-convention export
    # ------------------------------------------------------------------

    def _map_attrs(self, attrs: dict[str, Any]) -> dict[str, Any]:
        """Translate friendly attr names to OTel GenAI convention names.

        Keys present in ``ATTR_MAP`` are renamed; all others are kept but
        prefixed with ``wee.`` so they live in a custom namespace.
        """
        mapped: dict[str, Any] = {}
        for key, value in attrs.items():
            otel_key = self.ATTR_MAP.get(key)
            if otel_key is not None:
                mapped[otel_key] = value
            else:
                mapped[f"wee.{key}"] = value
        return mapped

    def export_otlp(self) -> list[dict]:
        """Return spans formatted per OTel GenAI semantic conventions.

        The output is a list of span dicts (one per root span) with
        recursively nested ``children``.  No external dependencies are
        required -- the dicts are plain Python and can be serialised to
        JSON then forwarded to any OTLP-compatible collector.
        """

        def _to_otlp(sp: Span) -> dict[str, Any]:
            return {
                "name": sp.name,
                "start_time_unix_nano": int(sp.start * 1e9),
                "end_time_unix_nano": int((sp.end or time.time()) * 1e9),
                "attributes": self._map_attrs(sp.attrs),
                "children": [_to_otlp(c) for c in sp.children],
            }

        return [_to_otlp(s) for s in self.root_spans]

    # ------------------------------------------------------------------
    # W3C Trace Context helper (educational)
    # ------------------------------------------------------------------

    def to_trace_context(self) -> dict[str, str]:
        """Return a W3C ``traceparent`` header value.

        Generates random 16-byte *trace-id* and 8-byte *span-id* encoded
        as lowercase hex, then assembles them into the ``traceparent``
        format: ``00-{trace_id}-{span_id}-01``.

        This is intentionally self-contained (uses ``os.urandom``) so the
        module stays free of external dependencies.
        """
        trace_id = os.urandom(16).hex()
        span_id = os.urandom(8).hex()
        return {"traceparent": f"00-{trace_id}-{span_id}-01"}
