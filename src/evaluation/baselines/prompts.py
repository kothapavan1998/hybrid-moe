"""Baseline prompt templates for comparison experiments."""

SYSTEM_BASE = "You are a commercial real estate underwriting analyst. Provide precise calculations."

ZEROSHOT_TEMPLATE = """{system}

{prompt}"""

COT_TEMPLATE = """{system}
Think step by step. Show all calculations clearly before giving final answers.

{prompt}

Let me work through this step by step:"""

REACT_TOOLS = """You have access to these calculation tools:
- calculate_dscr(noi, annual_debt_service) → DSCR ratio
- calculate_ltv(loan_amount, property_value) → LTV percentage  
- calculate_cap_rate(noi, purchase_price) → Cap rate percentage
- calculate_debt_yield(noi, loan_amount) → Debt yield percentage
- calculate_ads(principal, annual_rate, term_years) → Annual debt service
- calculate_noi_projection(current_noi, growth_rate, years) → Projected NOI

When you need a calculation, output:
Action: tool_name(arg1, arg2, ...)
Then wait for the Observation before continuing."""

REACT_TEMPLATE = """{system}
{tools}

{prompt}

Thought: Let me analyze the financials step by step."""

TOOLKEN_TEMPLATE = """{system}
When you need to calculate a metric, emit a special token:
[DSCR noi=VALUE ads=VALUE] → system computes DSCR
[LTV loan=VALUE value=VALUE] → system computes LTV
[CAP_RATE noi=VALUE price=VALUE] → system computes cap rate
[DEBT_YIELD noi=VALUE loan=VALUE] → system computes debt yield
[ADS principal=VALUE rate=VALUE years=VALUE] → system computes annual debt service
[NOI_PROJ noi=VALUE growth=VALUE years=VALUE] → system computes projected NOI

The system will replace the token with the computed result. Continue writing after each computation.

{prompt}"""


def format_prompt(method: str, scenario: dict) -> str:
    prompt = scenario["input_prompt"]
    if method == "zeroshot":
        return ZEROSHOT_TEMPLATE.format(system=SYSTEM_BASE, prompt=prompt)
    elif method == "cot":
        return COT_TEMPLATE.format(system=SYSTEM_BASE, prompt=prompt)
    elif method == "react":
        return REACT_TEMPLATE.format(system=SYSTEM_BASE, tools=REACT_TOOLS, prompt=prompt)
    elif method == "toolken":
        return TOOLKEN_TEMPLATE.format(system=SYSTEM_BASE, prompt=prompt)
    else:
        return prompt
