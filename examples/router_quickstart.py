from wee import Router

endpoints = [
    {"name": "mini", "price_in": 0.02, "price_out": 0.08, "latency": 0.2, "quality": 0.62, "max_tokens": 8192, "tags": ["fast","cheap"]},
    {"name": "pro",  "price_in": 0.10, "price_out": 0.40, "latency": 0.6, "quality": 0.86, "max_tokens": 16384, "tags": ["quality"]},
    {"name": "ultra","price_in": 0.25, "price_out": 0.80, "latency": 1.0, "quality": 0.92, "max_tokens": 32768, "tags": ["quality"]},
]

def main():
    rt = Router(endpoints)
    # Quality-first within $0.005 budget for 300-in/200-out tokens
    decision = rt.route(prompt_len=300, max_output_tokens=200, budget_usd=0.03, priority="quality")
    print("Decision (quality):", decision)

    # Speed-first, no strict budget, but require 'fast'
    decision = rt.route(prompt_len=300, max_output_tokens=200, latency_target_s=None, priority="speed", required_tags=["fast"])
    print("Decision (speed):", decision)

    # Cost-first
    decision = rt.route(prompt_len=600, max_output_tokens=400, priority="cost")
    print("Decision (cost):", decision)

if __name__ == "__main__":
    main()
