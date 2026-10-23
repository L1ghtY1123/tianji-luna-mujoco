"""Executable requirements live in contracts; instructions come from SKILL.md."""
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    inputs: list[str] = Field(default_factory=list)
    requires: dict[str, bool]
    provides: list[str]


class SkillSelectionError(ValueError):
    def __init__(self, code, skill, unmet_requirements=None):
        super().__init__(code + ":" + skill)
        self.code = code
        self.details = {"selected_skill": skill, "unmet_requirements": unmet_requirements or {}}


class Registry:
    def __init__(self, root=ROOT / "demo_skills"):
        self.contracts = {}
        self.documents = {}
        for path in sorted(Path(root).glob("*/contract.json")):
            contract = Contract.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if contract.id in self.contracts or path.parent.name != contract.id:
                raise ValueError("invalid_skill_identity")
            self.contracts[contract.id] = contract
            self.documents[contract.id] = (path.parent / "SKILL.md").read_text(encoding="utf-8")
        if not self.contracts:
            raise ValueError("no_skill_contracts")

    def eligible(self, facts):
        return {name: contract for name, contract in self.contracts.items()
                if all(bool(facts.get(key, False)) == value for key, value in contract.requires.items())}

    def describe(self):
        """The model gets original contracts, without computed eligibility hints."""
        return {name: contract.model_dump() for name, contract in self.contracts.items() if name != 'observe'}

    def validate_selection(self, skill, facts):
        """Validate the next program skill without selecting a replacement."""
        if skill not in self.contracts:
            raise SkillSelectionError("unknown_skill", skill)
        unmet = {}
        for fact, expected in self.contracts[skill].requires.items():
            actual = bool(facts.get(fact, False))
            if actual != expected:
                unmet[fact] = {"expected": expected, "actual": actual}
        if unmet:
            raise SkillSelectionError("skill_contract_not_satisfied", skill, unmet)
