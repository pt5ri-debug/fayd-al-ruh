from core.concept import Concept
from core.meaning_model import MeaningModel
from core.ontology import Ontology


def test_meaning_model_resolves_alias_in_text():
    ontology = Ontology()
    car = Concept("car", frozenset({"automobile"}))
    ontology.add_concept(car)
    assert MeaningModel().resolve("The automobile moves", ontology) == [car]
