"""Budget pacing: a per-campaign PI controller that throttles bids to spend evenly over the day.

`multiplier` in [0, 1] scales the candidate's value. The controller compares the share of the
daily budget spent with the share of the day elapsed (a linear target; a traffic-shaped curve is
a drop-in replacement) and integrates the error. State is tiny and lives in Redis next to counters.
"""

from pydantic import BaseModel, Field


class Pacer(BaseModel):
    """Proportional-integral throttle for one campaign."""

    daily_budget: float = Field(gt=0)
    kp: float = 2.0
    ki: float = 0.5
    spent: float = 0.0
    integral: float = 0.0
    multiplier: float = 1.0

    def update(self, day_fraction: float, spend: float) -> float:
        """Record `spend` since the last update and return the new multiplier."""
        self.spent += spend
        error = day_fraction - self.spent / self.daily_budget  # > 0: under-spending
        self.integral = min(max(self.integral + error, -2.0), 2.0)
        self.multiplier = min(max(1.0 + self.kp * error + self.ki * self.integral, 0.0), 1.0)
        if self.spent >= self.daily_budget:
            self.multiplier = 0.0
        return self.multiplier

    @property
    def exhausted(self) -> bool:
        """True once the daily budget is spent (the policy gates the campaign)."""
        return self.spent >= self.daily_budget
