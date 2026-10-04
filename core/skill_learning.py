from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Skill:
    name: str
    proficiency: float = 0.0
    practice_count: int = 0


class SkillLearning:
    """Updates declared skill proficiency from bounded practice outcomes."""
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def practice(self, name: str, outcome: float, learning_rate: float = .2) -> Skill:
        if not name.strip() or not 0.0 <= outcome <= 1.0 or not 0.0 < learning_rate <= 1.0:
            raise ValueError("Skill name, outcome, and learning rate are invalid.")
        skill = self.skills.setdefault(name, Skill(name))
        skill.proficiency = round(skill.proficiency + learning_rate * (outcome - skill.proficiency), 6)
        skill.practice_count += 1
        return skill
