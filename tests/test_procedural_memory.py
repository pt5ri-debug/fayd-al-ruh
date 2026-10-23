from core.procedural_memory import ProceduralMemory, Procedure


def test_procedure_is_stored_not_executed():
    memory = ProceduralMemory()
    memory.learn(Procedure("observe", ("read sensor",)))
    assert memory.retrieve("observe").steps == ("read sensor",)
