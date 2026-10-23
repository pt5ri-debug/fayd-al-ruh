from core.episodic_memory import EpisodicMemory


def test_episode_is_timestamped_and_recallable():
    memory = EpisodicMemory()
    memory.store("E1", "rain", .8)
    assert memory.recall("E1").access_count == 1
