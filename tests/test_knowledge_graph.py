import pytest

from core.entity import Entity, EntityType
from core.evidence import Evidence
from core.belief import Belief, BeliefStatus
from core.relation import Relation
from core.knowledge_graph import KnowledgeGraph


def create_basic_graph():
    graph = KnowledgeGraph()

    robot = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    ball = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    graph.add_entity(robot)
    graph.add_entity(ball)

    return graph, robot, ball


def test_create_empty_graph():

    graph = KnowledgeGraph()

    assert graph.entity_count == 0
    assert graph.relation_count == 0
    assert graph.evidence_count == 0
    assert graph.belief_count == 0


def test_add_entities():

    graph, robot, ball = create_basic_graph()

    assert graph.entity_count == 2

    assert graph.get_entity(robot.id) is robot
    assert graph.get_entity(ball.id) is ball


def test_duplicate_entity_is_rejected():

    graph = KnowledgeGraph()

    robot = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    graph.add_entity(robot)

    with pytest.raises(ValueError):
        graph.add_entity(robot)


def test_add_evidence():

    graph = KnowledgeGraph()

    evidence = Evidence(
        description="The robot detected the ball.",
        source="vision_sensor",
        supports=True,
        strength=0.95,
    )

    graph.add_evidence(evidence)

    assert graph.evidence_count == 1
    assert graph.get_evidence(evidence.id) is evidence


def test_add_belief_with_evidence():

    graph = KnowledgeGraph()

    evidence = Evidence(
        description="The ball was detected.",
        source="vision_sensor",
        supports=True,
        strength=0.9,
    )

    graph.add_evidence(evidence)

    belief = Belief(
        statement="The ball exists.",
        confidence=0.9,
    )

    belief.add_evidence(evidence)

    graph.add_belief(belief)

    assert graph.belief_count == 1

    assert graph.get_belief(
        belief.id
    ) is belief

    evidence_list = graph.get_belief_evidence(
        belief.id
    )

    assert evidence in evidence_list


def test_belief_with_missing_evidence_is_rejected():

    graph = KnowledgeGraph()

    belief = Belief(
        statement="The ball exists.",
    )

    belief.supporting_evidence.append(
        "EV-does-not-exist"
    )

    with pytest.raises(ValueError):

        graph.add_belief(belief)


def test_complete_knowledge_chain():

    graph, robot, ball = create_basic_graph()

    evidence = Evidence(
        description=(
            "Vision sensor detected the robot "
            "looking at the ball."
        ),
        source="vision_sensor",
        supports=True,
        strength=0.95,
    )

    graph.add_evidence(evidence)

    belief = Belief(
        statement="The robot sees the ball.",
        confidence=0.9,
        status=BeliefStatus.PROVISIONAL,
    )

    belief.add_evidence(evidence)

    graph.add_belief(belief)

    relation = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
        confidence=0.9,
    )

    relation.attach_belief(
        belief.id
    )

    relation.add_evidence(
        evidence.id
    )

    graph.add_relation(relation)

    assert graph.entity_count == 2
    assert graph.evidence_count == 1
    assert graph.belief_count == 1
    assert graph.relation_count == 1

    stored_belief = graph.get_relation_belief(
        relation.id
    )

    assert stored_belief is belief

    relation_evidence = (
        graph.get_relation_evidence(
            relation.id
        )
    )

    assert evidence in relation_evidence


def test_relation_requires_existing_entities():

    graph = KnowledgeGraph()

    relation = Relation(
        source="E-unknown",
        target="E-ball",
        relation_type="SEES",
    )

    with pytest.raises(ValueError):

        graph.add_relation(relation)


def test_relation_requires_existing_belief():

    graph, robot, ball = create_basic_graph()

    relation = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
    )

    relation.attach_belief(
        "BL-does-not-exist"
    )

    with pytest.raises(ValueError):

        graph.add_relation(relation)


def test_relation_requires_existing_evidence():

    graph, robot, ball = create_basic_graph()

    relation = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
    )

    relation.add_evidence(
        "EV-does-not-exist"
    )

    with pytest.raises(ValueError):

        graph.add_relation(relation)


def test_find_relations():

    graph, robot, ball = create_basic_graph()

    relation = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
    )

    graph.add_relation(relation)

    relations = graph.find_relations(
        robot.id
    )

    assert len(relations) == 1
    assert relations[0] is relation


def test_find_relations_by_type():

    graph, robot, ball = create_basic_graph()

    sees = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
    )

    likes = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="LIKES",
    )

    graph.add_relation(sees)
    graph.add_relation(likes)

    relations = graph.find_relations(
        robot.id,
        relation_type="SEES",
    )

    assert len(relations) == 1
    assert relations[0] is sees


def test_get_neighbors():

    graph = KnowledgeGraph()

    robot = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    ball = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    box = Entity(
        entity_type=EntityType.OBJECT,
        name="Box",
    )

    graph.add_entity(robot)
    graph.add_entity(ball)
    graph.add_entity(box)

    graph.add_relation(
        Relation(
            source=robot.id,
            target=ball.id,
            relation_type="SEES",
        )
    )

    graph.add_relation(
        Relation(
            source=robot.id,
            target=box.id,
            relation_type="NEAR",
        )
    )

    neighbors = graph.get_neighbors(
        robot.id
    )

    neighbor_ids = {
        entity.id
        for entity in neighbors
    }

    assert neighbor_ids == {
        ball.id,
        box.id,
    }


def test_find_path():

    graph = KnowledgeGraph()

    robot = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    ball = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    box = Entity(
        entity_type=EntityType.OBJECT,
        name="Box",
    )

    graph.add_entity(robot)
    graph.add_entity(ball)
    graph.add_entity(box)

    graph.add_relation(
        Relation(
            source=robot.id,
            target=ball.id,
            relation_type="SEES",
        )
    )

    graph.add_relation(
        Relation(
            source=ball.id,
            target=box.id,
            relation_type="NEAR",
        )
    )

    path = graph.find_path(
        robot.id,
        box.id,
    )

    assert path == [
        robot.id,
        ball.id,
        box.id,
    ]


def test_find_path_when_no_path_exists():

    graph, robot, ball = create_basic_graph()

    path = graph.find_path(
        robot.id,
        ball.id,
    )

    assert path == []


def test_remove_entity_removes_relations():

    graph, robot, ball = create_basic_graph()

    relation = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
    )

    graph.add_relation(relation)

    assert graph.relation_count == 1

    graph.remove_entity(robot.id)

    assert graph.entity_count == 1
    assert graph.relation_count == 0


def test_remove_evidence():

    graph = KnowledgeGraph()

    evidence = Evidence(
        description="Test evidence.",
        source="test",
    )

    graph.add_evidence(evidence)

    belief = Belief(
        statement="Test belief.",
    )

    belief.add_evidence(evidence)

    graph.add_belief(belief)

    graph.remove_evidence(
        evidence.id
    )

    assert graph.evidence_count == 0

    assert (
        evidence.id
        not in belief.supporting_evidence
    )


def test_validate_clean_graph():

    graph, robot, ball = create_basic_graph()

    evidence = Evidence(
        description="Robot sees the ball.",
        source="vision",
    )

    graph.add_evidence(evidence)

    belief = Belief(
        statement="Robot sees the ball.",
    )

    belief.add_evidence(evidence)

    graph.add_belief(belief)

    relation = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
    )

    relation.attach_belief(
        belief.id
    )

    relation.add_evidence(
        evidence.id
    )

    graph.add_relation(relation)

    errors = graph.validate()

    assert errors == []


def test_statistics():

    graph, robot, ball = create_basic_graph()

    evidence = Evidence(
        description="The ball exists.",
        source="vision",
    )

    graph.add_evidence(evidence)

    belief = Belief(
        statement="The ball exists.",
    )

    belief.add_evidence(evidence)

    graph.add_belief(belief)

    relation = Relation(
        source=robot.id,
        target=ball.id,
        relation_type="SEES",
    )

    graph.add_relation(relation)

    assert graph.entity_count == 2
    assert graph.evidence_count == 1
    assert graph.belief_count == 1
    assert graph.relation_count == 1