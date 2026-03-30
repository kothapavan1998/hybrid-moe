"""
CRE Underwriting Computation Experts.

6 deterministic PyTorch functions for commercial real estate underwriting.
All differentiable (basic arithmetic ops), all exact, no neural approximation.

Active in experiments: DSCR, LTV, Cap Rate, Debt Yield (4 experts, 36 total with 32 neural).
Defined but not active: Annual Debt Service, NOI Projection.
"""
import torch
import math


def compute_dscr(params: torch.Tensor) -> torch.Tensor:
    """DSCR = NOI / Annual Debt Service. params[:, 0]=NOI, [:, 1]=ADS"""
    noi = params[:, 0]
    ads = params[:, 1]
    return (noi / (ads + 1e-8)).unsqueeze(-1)


def compute_ltv(params: torch.Tensor) -> torch.Tensor:
    """LTV% = Loan / Value * 100. params[:, 0]=loan, [:, 1]=value"""
    loan = params[:, 0]
    value = params[:, 1]
    return (loan / (value + 1e-8) * 100.0).unsqueeze(-1)


def compute_cap_rate(params: torch.Tensor) -> torch.Tensor:
    """Cap Rate% = NOI / Price * 100. params[:, 0]=NOI, [:, 1]=price"""
    noi = params[:, 0]
    price = params[:, 1]
    return (noi / (price + 1e-8) * 100.0).unsqueeze(-1)


def compute_debt_yield(params: torch.Tensor) -> torch.Tensor:
    """Debt Yield% = NOI / Loan * 100. params[:, 0]=NOI, [:, 1]=loan"""
    noi = params[:, 0]
    loan = params[:, 1]
    return (noi / (loan + 1e-8) * 100.0).unsqueeze(-1)


def compute_annual_debt_service(params: torch.Tensor) -> torch.Tensor:
    """Standard amortization. params[:, 0]=principal, [:, 1]=annual_rate, [:, 2]=term_years"""
    principal = params[:, 0]
    annual_rate = params[:, 1]
    term_years = params[:, 2]
    monthly_rate = annual_rate / 12.0
    n_payments = term_years * 12.0
    numerator = monthly_rate * torch.pow(1.0 + monthly_rate, n_payments)
    denominator = torch.pow(1.0 + monthly_rate, n_payments) - 1.0 + 1e-8
    monthly_payment = principal * (numerator / denominator)
    return (monthly_payment * 12.0).unsqueeze(-1)


def compute_noi_projection(params: torch.Tensor) -> torch.Tensor:
    """Projected NOI = NOI * (1+g)^n. params[:, 0]=NOI, [:, 1]=growth_rate, [:, 2]=years"""
    noi = params[:, 0]
    growth = params[:, 1]
    years = params[:, 2]
    return (noi * torch.pow(1.0 + growth, years)).unsqueeze(-1)


# ============================================================================
# Expert Registry
# ============================================================================
EXPERT_REGISTRY = {
    "dscr": {
        "expert_id": 32,
        "function": compute_dscr,
        "n_input_params": 2,
        "n_output_params": 1,
        "input_names": ["noi", "annual_debt_service"],
        "output_names": ["dscr"],
        "description": "Debt Service Coverage Ratio",
    },
    "ltv": {
        "expert_id": 33,
        "function": compute_ltv,
        "n_input_params": 2,
        "n_output_params": 1,
        "input_names": ["loan_amount", "property_value"],
        "output_names": ["ltv_percent"],
        "description": "Loan-to-Value Ratio",
    },
    "cap_rate": {
        "expert_id": 34,
        "function": compute_cap_rate,
        "n_input_params": 2,
        "n_output_params": 1,
        "input_names": ["noi", "purchase_price"],
        "output_names": ["cap_rate_percent"],
        "description": "Capitalization Rate",
    },
    "debt_yield": {
        "expert_id": 35,
        "function": compute_debt_yield,
        "n_input_params": 2,
        "n_output_params": 1,
        "input_names": ["noi", "loan_amount"],
        "output_names": ["debt_yield_percent"],
        "description": "Debt Yield",
    },
    "annual_debt_service": {
        "expert_id": 36,
        "function": compute_annual_debt_service,
        "n_input_params": 3,
        "n_output_params": 1,
        "input_names": ["principal", "annual_rate", "term_years"],
        "output_names": ["annual_debt_service"],
        "description": "Annual Debt Service (Amortization)",
    },
    "noi_projection": {
        "expert_id": 37,
        "function": compute_noi_projection,
        "n_input_params": 3,
        "n_output_params": 1,
        "input_names": ["current_noi", "growth_rate", "years"],
        "output_names": ["projected_noi"],
        "description": "NOI Projection (Compound Growth)",
    },
}

NUM_COMPUTATION_EXPERTS = len(EXPERT_REGISTRY)
FIRST_COMPUTATION_EXPERT_ID = 32
ACTIVE_EXPERT_NAMES = ["dscr", "ltv", "cap_rate", "debt_yield"]
NUM_ACTIVE_EXPERTS = len(ACTIVE_EXPERT_NAMES)


# ============================================================================
# Tests
# ============================================================================
def test_all_experts():
    print("=" * 60)
    print("  Testing CRE Computation Experts")
    print("=" * 60)
    t = torch.tensor
    all_passed = True

    # DSCR
    r = compute_dscr(t([[1026000.0, 796044.0]]))
    e = 1026000.0 / 796044.0
    ok = abs(r.item() - e) < 0.001
    print(f"  {'✓' if ok else '✗'} DSCR: {r.item():.4f}x (expected {e:.4f}x)")
    all_passed &= ok

    # LTV
    r = compute_ltv(t([[10500000.0, 15000000.0]]))
    e = 70.0
    ok = abs(r.item() - e) < 0.01
    print(f"  {'✓' if ok else '✗'} LTV: {r.item():.2f}% (expected {e:.2f}%)")
    all_passed &= ok

    # Cap Rate
    r = compute_cap_rate(t([[1026000.0, 15000000.0]]))
    e = 1026000.0 / 15000000.0 * 100
    ok = abs(r.item() - e) < 0.01
    print(f"  {'✓' if ok else '✗'} Cap Rate: {r.item():.2f}% (expected {e:.2f}%)")
    all_passed &= ok

    # Debt Yield
    r = compute_debt_yield(t([[1026000.0, 10500000.0]]))
    e = 1026000.0 / 10500000.0 * 100
    ok = abs(r.item() - e) < 0.01
    print(f"  {'✓' if ok else '✗'} Debt Yield: {r.item():.2f}% (expected {e:.2f}%)")
    all_passed &= ok

    # Annual Debt Service
    r = compute_annual_debt_service(t([[10500000.0, 0.065, 30.0]]))
    mr = 0.065 / 12; n = 360
    monthly = 10500000 * (mr * (1 + mr)**n) / ((1 + mr)**n - 1)
    e = monthly * 12
    ok = abs(r.item() - e) < 5.0  # PyTorch float32 vs Python float64 precision diff
    print(f"  {'✓' if ok else '✗'} ADS: ${r.item():,.2f} (expected ${e:,.2f}, diff=${abs(r.item()-e):.2f})")
    all_passed &= ok

    # NOI Projection
    r = compute_noi_projection(t([[1026000.0, 0.03, 3.0]]))
    e = 1026000.0 * (1.03 ** 3)
    ok = abs(r.item() - e) < 1.0
    print(f"  {'✓' if ok else '✗'} NOI Proj Y3: ${r.item():,.2f} (expected ${e:,.2f})")
    all_passed &= ok

    # Batch test
    batch = t([[500000.0, 400000.0], [1000000.0, 750000.0], [2000000.0, 1800000.0]])
    r = compute_dscr(batch)
    ok = r.shape == (3, 1)
    print(f"  {'✓' if ok else '✗'} Batch: shape {r.shape}")
    all_passed &= ok

    # Gradient test
    params = t([[1026000.0, 796044.0]], requires_grad=True)
    r = compute_dscr(params)
    r.backward()
    ok = params.grad is not None and params.grad.shape == (1, 2)
    print(f"  {'✓' if ok else '✗'} Gradient flows: grad shape {params.grad.shape if ok else 'NONE'}")
    all_passed &= ok

    print("=" * 60)
    print(f"  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    test_all_experts()
