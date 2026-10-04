from core.memory import CognitiveMemory


def test_memory_contract_promotes_observation_to_episode():
    memory = CognitiveMemory(); memory.store("OBS-1", "observation", "rain")
    assert "OBS-1" in memory.episodic_memory.episodes
