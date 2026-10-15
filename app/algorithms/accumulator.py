"""Stateful Benefit Accumulator Algorithm (Section 5.1 Deductible & Section 5.2 Out-of-Pocket Max)."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class AccumulatorState(BaseModel):
    policyholder_id: str
    is_family_plan: bool = False
    deductible_limit: float = 1000.0  # $1,000 individual, $2,500 family
    oop_max_limit: float = 4500.0     # $4,500 individual, $9,000 family
    deductible_satisfied: float = 0.0
    oop_accumulated: float = 0.0

    @property
    def deductible_remaining(self) -> float:
        return max(0.0, self.deductible_limit - self.deductible_satisfied)

    @property
    def oop_remaining(self) -> float:
        return max(0.0, self.oop_max_limit - self.oop_accumulated)

    @property
    def is_deductible_met(self) -> bool:
        return self.deductible_remaining <= 0.0

    @property
    def is_oop_max_reached(self) -> bool:
        return self.oop_remaining <= 0.0


class BenefitAccumulator:
    """
    Simulates cumulative policy benefit accumulation and catastrophic coverage triggering.
    Governed by Section 5.1 (Deductibles) & Section 5.2 (Out-of-Pocket Maximum).
    """

    def __init__(self):
        self._member_states: Dict[str, AccumulatorState] = {}

    def get_or_create_state(self, policyholder_id: str, is_family: bool = False) -> AccumulatorState:
        if policyholder_id not in self._member_states:
            ded = 2500.0 if is_family else 1000.0
            oop = 9000.0 if is_family else 4500.0
            self._member_states[policyholder_id] = AccumulatorState(
                policyholder_id=policyholder_id,
                is_family_plan=is_family,
                deductible_limit=ded,
                oop_max_limit=oop
            )
        return self._member_states[policyholder_id]

    def adjudicate_and_accumulate(
        self,
        policyholder_id: str,
        eligible_charges: float,
        in_network: bool = True
    ) -> Dict[str, Any]:
        """
        Adjudicates a claim through the accumulator state machine:
        1. Satisfies remaining deductible first.
        2. Applies co-insurance rate (80% in-net / 70% out-of-net) on the remaining amount.
        3. Protects the member: if patient coinsurance would exceed remaining OOP Max,
           caps patient out-of-pocket and pays the rest at 100%!
        """
        state = self.get_or_create_state(policyholder_id)

        # 1. Deductible portion
        deductible_portion = min(eligible_charges, state.deductible_remaining)
        amount_after_deductible = max(0.0, eligible_charges - deductible_portion)

        # 2. Co-insurance calculation
        co_rate = 0.80 if in_network else 0.70
        tentative_patient_coinsurance = amount_after_deductible * (1.0 - co_rate)

        # 3. Check Out-of-Pocket Max protection
        # Total patient obligation before OOP cap = deductible_portion + tentative_patient_coinsurance
        raw_patient_cost = deductible_portion + tentative_patient_coinsurance

        if state.is_oop_max_reached:
            # Plan covers 100% per Section 5.2!
            actual_patient_cost = 0.0
            plan_payout = eligible_charges
            oop_max_triggered = True
        elif raw_patient_cost > state.oop_remaining:
            actual_patient_cost = state.oop_remaining
            plan_payout = eligible_charges - actual_patient_cost
            oop_max_triggered = True
        else:
            actual_patient_cost = raw_patient_cost
            plan_payout = eligible_charges - actual_patient_cost
            oop_max_triggered = False

        # Update accumulator state
        state.deductible_satisfied += deductible_portion
        state.oop_accumulated += actual_patient_cost

        return {
            "policyholder_id": policyholder_id,
            "eligible_charges": round(eligible_charges, 2),
            "deductible_applied": round(deductible_portion, 2),
            "deductible_remaining_after": round(state.deductible_remaining, 2),
            "plan_payout": round(plan_payout, 2),
            "patient_responsibility": round(actual_patient_cost, 2),
            "oop_accumulated_total": round(state.oop_accumulated, 2),
            "oop_max_remaining": round(state.oop_remaining, 2),
            "oop_max_reached": state.is_oop_max_reached,
            "catastrophic_100_percent_active": oop_max_triggered,
            "citations": [
                "[Section 5.1] Annual Deductible",
                "[Section 5.2] Out-of-Pocket Maximum",
                "[Section 5.3] Reimbursement Calculation"
            ]
        }
