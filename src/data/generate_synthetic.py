"""
Synthetic CRE Underwriting Data Generator
Run tonight: python -m src.data.generate_synthetic --output data/
"""
import json, os, random, math, argparse
from typing import Dict, List, Tuple

PROPERTY_TYPES = ["multifamily", "office", "retail", "industrial", "mixed-use", "hotel", "self-storage", "medical office"]
LOCATIONS = ["Dallas, TX", "Miami, FL", "Phoenix, AZ", "Denver, CO", "Atlanta, GA", "Chicago, IL",
             "Los Angeles, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Nashville, TN",
             "Charlotte, NC", "Tampa, FL", "Houston, TX", "Portland, OR", "San Diego, CA",
             "Raleigh, NC", "Salt Lake City, UT", "Minneapolis, MN", "Columbus, OH"]
PREFIXES = ["Oakwood", "Riverstone", "Parkview", "Summit", "Heritage", "Lakeside", "Crossroads",
            "Horizon", "Gateway", "Pinnacle", "Meridian", "Sterling", "Granite", "Cedar",
            "Magnolia", "Falcon", "Eagle", "Harbor", "Midtown", "Uptown", "Westgate", "Northstar"]
SUFFIXES = {"multifamily": ["Apartments", "Residences", "Village", "Gardens", "Heights"],
            "office": ["Center", "Tower", "Plaza", "Campus"], "retail": ["Shopping Center", "Plaza", "Marketplace"],
            "industrial": ["Business Park", "Logistics Center"], "mixed-use": ["Village", "District"],
            "hotel": ["Hotel", "Suites", "Resort"], "self-storage": ["Self Storage", "Storage Center"],
            "medical office": ["Medical Center", "Medical Plaza"]}


def _compute_ads(principal, rate, years):
    mr = rate / 12; n = years * 12
    if mr > 1e-6:
        monthly = principal * (mr * (1 + mr)**n) / ((1 + mr)**n - 1)
    else:
        monthly = principal / n
    return round(monthly * 12, 2)


def generate_scenario(sid: int, complexity: int = None) -> Dict:
    if complexity is None:
        complexity = random.choices([1, 2, 3, 4], weights=[0.15, 0.25, 0.35, 0.25])[0]

    pt = random.choice(PROPERTY_TYPES)
    loc = random.choice(LOCATIONS)
    name = f"{random.choice(PREFIXES)} {random.choice(SUFFIXES.get(pt, ['Center']))}"

    # Financials
    price = random.choice([random.randint(1,10), random.randint(10,50), random.randint(50,200)]) * 1_000_000
    ltv_t = random.uniform(0.55, 0.80)
    loan = round(price * ltv_t / 100_000) * 100_000
    rate = round(random.uniform(0.045, 0.085), 4)
    amort = random.choice([25, 30])

    cap_market = random.uniform(0.04, 0.10)
    noi = round(price * cap_market / 1000) * 1000
    exp_ratio = random.uniform(0.30, 0.55)
    egi = round(noi / (1 - exp_ratio) / 1000) * 1000
    opex = egi - noi
    vac = round(random.uniform(0.03, 0.10), 3)
    gpr = round(egi / (1 - vac) / 1000) * 1000
    growth = round(random.uniform(0.01, 0.05), 3)

    units = random.choice(range(20, 500, 10)) if pt in ["multifamily", "hotel"] else None
    sqft = random.choice(range(10000, 500000, 5000)) if pt in ["office", "retail", "industrial", "medical office", "mixed-use"] else None

    # Ground truth
    ads = _compute_ads(loan, rate, amort)
    dscr = round(noi / ads, 3) if ads > 0 else 0
    cap_rate = round(noi / price * 100, 2)
    ltv = round(loan / price * 100, 2)
    dy = round(noi / loan * 100, 2) if loan > 0 else 0
    noi_y3 = round(noi * (1 + growth) ** 3, 2)
    noi_y5 = round(noi * (1 + growth) ** 5, 2)

    scenario = {
        "id": f"cre_{sid:05d}", "complexity": complexity,
        "property": {"name": name, "type": pt, "location": loc, "units": units, "sqft": sqft},
        "financials": {
            "purchase_price": price, "loan_amount": loan, "interest_rate": rate,
            "amortization_years": amort, "gpr": gpr, "vacancy_rate": vac,
            "egi": egi, "opex": opex, "noi": noi, "noi_growth_rate": growth,
        },
        "ground_truth": {
            "annual_debt_service": ads, "dscr": dscr, "cap_rate": cap_rate,
            "ltv": ltv, "debt_yield": dy, "noi_year_3": noi_y3, "noi_year_5": noi_y5,
        },
    }

    scenario["input_prompt"] = _gen_prompt(scenario, complexity)
    scenario["computation_markers"] = _gen_markers(scenario, complexity)
    return scenario


def _gen_prompt(s, c):
    p, f = s["property"], s["financials"]
    gt = s["ground_truth"]
    if c == 1:
        metric = random.choice(["dscr", "ltv", "cap_rate", "debt_yield"])
        if metric == "dscr":
            return f"What is the DSCR for a property with NOI of ${f['noi']:,.0f} and annual debt service of ${gt['annual_debt_service']:,.0f}?"
        elif metric == "ltv":
            return f"Calculate the LTV for a ${f['loan_amount']:,.0f} loan on a ${f['purchase_price']:,.0f} property."
        elif metric == "cap_rate":
            return f"What cap rate does a ${f['purchase_price']:,.0f} property with ${f['noi']:,.0f} NOI trade at?"
        else:
            return f"What is the debt yield on a ${f['loan_amount']:,.0f} loan with ${f['noi']:,.0f} NOI?"
    elif c == 2:
        return (f"Compute key metrics for {p['name']}, a {p['type']} in {p['location']}. "
                f"Price: ${f['purchase_price']:,.0f}. Loan: ${f['loan_amount']:,.0f}. "
                f"NOI: ${f['noi']:,.0f}. ADS: ${gt['annual_debt_service']:,.0f}.")
    elif c == 3:
        return (f"Analyze: {p['name']} ({p['type']}) in {p['location']}. "
                f"Price ${f['purchase_price']:,.0f}, loan ${f['loan_amount']:,.0f} at {f['interest_rate']*100:.2f}% / {f['amortization_years']}yr. "
                f"NOI ${f['noi']:,.0f}. Calculate ADS, then DSCR, LTV, cap rate.")
    else:
        u = f" ({p['units']} units)" if p['units'] else (f" ({p['sqft']:,} SF)" if p['sqft'] else "")
        return (f"Full underwriting analysis:\n\nProperty: {p['name']}{u}\nType: {p['type'].title()}\n"
                f"Location: {p['location']}\nPrice: ${f['purchase_price']:,.0f}\n"
                f"Loan: ${f['loan_amount']:,.0f}\nRate: {f['interest_rate']*100:.2f}%\n"
                f"Amort: {f['amortization_years']}yr\nGPR: ${f['gpr']:,.0f}\n"
                f"Vacancy: {f['vacancy_rate']*100:.1f}%\nOpEx: ${f['opex']:,.0f}\n"
                f"NOI: ${f['noi']:,.0f}\nGrowth: {f['noi_growth_rate']*100:.1f}%/yr\n\n"
                f"Provide DSCR, LTV, cap rate, debt yield, 3-year NOI projection, and recommendation.")


def _gen_markers(s, c):
    f, gt = s["financials"], s["ground_truth"]
    m = []
    if c == 1:
        prompt = s["input_prompt"].lower()
        if "dscr" in prompt:
            m.append({"expert": "dscr", "expert_id": 32, "inputs": {"noi": f["noi"], "ads": gt["annual_debt_service"]}, "output": gt["dscr"]})
        elif "ltv" in prompt:
            m.append({"expert": "ltv", "expert_id": 33, "inputs": {"loan": f["loan_amount"], "value": f["purchase_price"]}, "output": gt["ltv"]})
        elif "cap rate" in prompt:
            m.append({"expert": "cap_rate", "expert_id": 34, "inputs": {"noi": f["noi"], "price": f["purchase_price"]}, "output": gt["cap_rate"]})
        else:
            m.append({"expert": "debt_yield", "expert_id": 35, "inputs": {"noi": f["noi"], "loan": f["loan_amount"]}, "output": gt["debt_yield"]})
    elif c == 2:
        m = [
            {"expert": "dscr", "expert_id": 32, "inputs": {"noi": f["noi"], "ads": gt["annual_debt_service"]}, "output": gt["dscr"]},
            {"expert": "ltv", "expert_id": 33, "inputs": {"loan": f["loan_amount"], "value": f["purchase_price"]}, "output": gt["ltv"]},
            {"expert": "cap_rate", "expert_id": 34, "inputs": {"noi": f["noi"], "price": f["purchase_price"]}, "output": gt["cap_rate"]},
            {"expert": "debt_yield", "expert_id": 35, "inputs": {"noi": f["noi"], "loan": f["loan_amount"]}, "output": gt["debt_yield"]},
        ]
    elif c >= 3:
        m = [
            {"expert": "annual_debt_service", "expert_id": 36, "inputs": {"principal": f["loan_amount"], "rate": f["interest_rate"], "years": f["amortization_years"]}, "output": gt["annual_debt_service"], "order": 1},
            {"expert": "dscr", "expert_id": 32, "inputs": {"noi": f["noi"], "ads": gt["annual_debt_service"]}, "output": gt["dscr"], "order": 2, "depends_on": "annual_debt_service"},
            {"expert": "ltv", "expert_id": 33, "inputs": {"loan": f["loan_amount"], "value": f["purchase_price"]}, "output": gt["ltv"], "order": 2},
            {"expert": "cap_rate", "expert_id": 34, "inputs": {"noi": f["noi"], "price": f["purchase_price"]}, "output": gt["cap_rate"], "order": 2},
            {"expert": "debt_yield", "expert_id": 35, "inputs": {"noi": f["noi"], "loan": f["loan_amount"]}, "output": gt["debt_yield"], "order": 2},
        ]
        if c == 4:
            m.append({"expert": "noi_projection", "expert_id": 37, "inputs": {"noi": f["noi"], "growth": f["noi_growth_rate"], "years": 3}, "output": gt["noi_year_3"], "order": 3})
    return m


def validate_dataset(scenarios):
    errors = []
    for s in scenarios:
        f, gt = s["financials"], s["ground_truth"]
        ads_check = _compute_ads(f["loan_amount"], f["interest_rate"], f["amortization_years"])
        if abs(gt["annual_debt_service"] - ads_check) > 1.0:
            errors.append(f"{s['id']}: ADS {gt['annual_debt_service']} vs {ads_check}")
        dscr_check = round(f["noi"] / ads_check, 3) if ads_check > 0 else 0
        if abs(gt["dscr"] - dscr_check) > 0.002:
            errors.append(f"{s['id']}: DSCR {gt['dscr']} vs {dscr_check}")
    return len(scenarios) - len(errors), errors


def print_stats(scenarios):
    comps = [s["complexity"] for s in scenarios]
    prices = [s["financials"]["purchase_price"] for s in scenarios]
    dscrs = [s["ground_truth"]["dscr"] for s in scenarios]
    markers = sum(len(s["computation_markers"]) for s in scenarios)
    print(f"  Scenarios: {len(scenarios)}")
    for c in [1,2,3,4]: print(f"    Complexity {c}: {comps.count(c)}")
    print(f"  Price range: ${min(prices):,.0f} – ${max(prices):,.0f}")
    print(f"  DSCR range: {min(dscrs):.3f}x – {max(dscrs):.3f}x")
    print(f"  Total markers: {markers} (avg {markers/len(scenarios):.1f}/scenario)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=3000)
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nGenerating {args.n_train} train + {args.n_eval} eval scenarios...\n")

    # Training
    random.seed(args.seed)
    train = [generate_scenario(i) for i in range(args.n_train)]
    valid, errs = validate_dataset(train)
    print(f"Training data: {valid}/{args.n_train} valid")
    if errs: print(f"  Errors: {errs[:3]}")
    print_stats(train)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "train.json"), "w") as f:
        json.dump(train, f, indent=2)
    print(f"  → Saved to {args.output}train.json\n")

    # Eval
    random.seed(args.seed + 1)
    eval_data = [generate_scenario(10000 + i) for i in range(args.n_eval)]
    valid, errs = validate_dataset(eval_data)
    print(f"Eval data: {valid}/{args.n_eval} valid")
    print_stats(eval_data)
    with open(os.path.join(args.output, "eval.json"), "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"  → Saved to {args.output}eval.json\n")

    # Debug (10 quick-test scenarios)
    random.seed(0)
    debug = [generate_scenario(99000 + i) for i in range(10)]
    with open(os.path.join(args.output, "debug.json"), "w") as f:
        json.dump(debug, f, indent=2)
    print(f"Debug: 10 scenarios → {args.output}debug.json")

    # Print a sample
    print(f"\n{'='*60}\nSample scenario:\n{'='*60}")
    s = train[0]
    print(f"  {s['id']} | {s['property']['name']} | {s['property']['type']} | {s['property']['location']}")
    print(f"  Price: ${s['financials']['purchase_price']:,.0f} | Loan: ${s['financials']['loan_amount']:,.0f}")
    print(f"  NOI: ${s['financials']['noi']:,.0f} | Rate: {s['financials']['interest_rate']*100:.2f}%")
    print(f"  GT: DSCR={s['ground_truth']['dscr']}x LTV={s['ground_truth']['ltv']}% Cap={s['ground_truth']['cap_rate']}%")
    print(f"  Complexity: {s['complexity']} | Markers: {len(s['computation_markers'])}")
    print(f"\n  Prompt: {s['input_prompt'][:200]}...")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
