from core.procedural_memory import Procedure, ProceduralMemory


def test_procedure_is_stored_not_executed():
    memory = ProceduralMemory(); memory.learn(Procedure("observe", ("read sensor",)))
    assert memory.retrieve("observe").steps == ("read sensor",)
