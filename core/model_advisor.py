"""Grounds a language model's proposal into a concrete, allowed action.

The model is an untrusted advisor: it receives the current facts and the exact
allow-list of action names, and may reply with one action name. This bridge
accepts the reply only when it names an action from that allow-list; otherwise
it declines and the caller keeps its own decision. The model never executes
anything -- selection still flows through the normal executor and precondition
checks.
"""

from __future__ import annotations

import re

from core.action_policy import ActionPolicy
from core.model_interface import CognitiveModel, ModelRequest

_ADVISOR_INSTRUCTION = (
    "You choose one action for a symbolic task. Reply with only the action name "
    "from the allowed list, nothing else."
)


class ModelActionAdvisor(ActionPolicy):
    """An :class:`ActionPolicy` whose suggestion comes from a language model."""

    def __init__(self, model: CognitiveModel, *, allow_abstain: bool = True) -> None:
        self.model = model
        self.allow_abstain = allow_abstain
        self.last_reply: str | None = None

    def select(self, facts: frozenset[str], actions: tuple[str, ...]) -> str | None:
        if not actions:
            return None
        prompt = (
            f"Current facts: {', '.join(sorted(facts)) or 'none'}\n"
            f"Allowed actions: {', '.join(actions)}\n"
            "Which single action do you choose?"
        )
        request = ModelRequest(
            inputs=(),
            system_instruction=_ADVISOR_INSTRUCTION,
            context=(prompt,),
        )
        try:
            response = self.model.generate(request)
        except Exception:
            # A failing advisor must never break the loop; fall back to caller.
            return None
        self.last_reply = response.text
        return self._match_action(response.text, actions)

    @staticmethod
    def _match_action(text: str, actions: tuple[str, ...]) -> str | None:
        lowered = text.lower()
        for name in actions:
            if re.search(rf"(?<![a-z0-9_]){re.escape(name.lower())}(?![a-z0-9_])", lowered):
                return name
        return None
