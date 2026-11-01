"""The Autonomous Cognitive System: a Proactive Mind, cognitive learning, and staged training.

This package is the project's **autonomous cognitive layer**, added beside the
existing cognitive core rather than inside it. It is composed of three parts, and
they share one rule: *nothing here widens the model's own authority*. The mind may
think, propose, criticise, and learn, but a generated idea is never a fact, a
teacher is never trusted blindly, and a trained version is never adopted without
an evaluation that shows it did not regress.

* :mod:`cognition.proactive_mind` -- the eight engines composed into one bounded
  thinking cycle (:class:`ProactiveMind`).
* :mod:`cognition.metacognition` -- the mind's model of its own thinking: which
  strategies pay off, where it stagnates, and how it escalates
  (:class:`MetacognitiveController`).
* :mod:`cognition.learning_system` -- six memories and the learning operations
  over them (:class:`CognitiveLearningSystem`).
* :mod:`cognition.consolidation` -- turning raw experience into durable, reviewed
  knowledge (:class:`Consolidator`).
* :mod:`cognition.teacher_student` -- the student-first exchange that turns a
  teacher's answer into a *qualified* example (:class:`TeacherStudentSystem`).
* :mod:`cognition.training` -- staged, resumable training with a candidate /
  version / rollback gate (:class:`IncrementalTrainer`).
* :mod:`cognition.curriculum` -- adaptive gradual training that chooses its own
  next step and stops itself (:class:`AutonomousCurriculum`).
* :mod:`cognition.evaluation` -- the ten cognitive indicators and a before/after
  comparison (:class:`CognitiveEvaluator`).
* :mod:`cognition.runner` -- background operation under three resource modes
  (:class:`CognitiveRunner`).

The records (:mod:`cognition.records`), storage (:mod:`cognition.storage`), and
reliability rule (:mod:`cognition.reliability`) are the shared foundation.
"""

from __future__ import annotations

from cognition.consolidation import (
    ConsolidationReport,
    Consolidator,
    Pattern,
    ReviewSchedule,
    SkillRegistry,
)
from cognition.curriculum import (
    AdaptivePolicy,
    AutonomousCurriculum,
    CurriculumReport,
    MultiSeedEvaluator,
    TrainingMemory,
)
from cognition.evaluation import (
    INDICATORS,
    LABELS,
    CognitiveEvaluator,
    CognitiveReport,
    Comparison,
)
from cognition.learning_system import CognitiveLearningSystem, ErrorAnalysis, LearningStats
from cognition.metacognition import (
    CyclePlan,
    Escalation,
    EscalationPolicy,
    FocusBuffer,
    MetacognitiveController,
    PersistentNovelty,
    SelfModel,
    StagnationDetector,
    StagnationReading,
    StrategyYield,
)
from cognition.proactive_mind import ProactiveMind, load_scheduler_state
from cognition.records import (
    Experience,
    Idea,
    IdeaKind,
    Reliability,
    Source,
    SourceKind,
    ThoughtCycle,
    TrainingExample,
    TrainingExampleKind,
    promote,
)
from cognition.runner import CognitiveRunner, ResourceMode
from cognition.scheduler import (
    SchedulerSettings,
    SchedulerState,
    ThoughtScheduler,
)
from cognition.teacher_student import (
    AnswerQuality,
    CallableBackend,
    ModelBackend,
    StudentProfile,
    TeacherStudentSystem,
    TeachingOutcome,
)
from cognition.training import (
    STAGE_SETTINGS,
    IncrementalTrainer,
    TrainingOutcome,
    TrainingPlan,
    TrainingStage,
)
from cognition.versions import Evaluation, ModelRegistry, ModelVersion

__all__ = [
    "INDICATORS",
    "LABELS",
    "STAGE_SETTINGS",
    "AdaptivePolicy",
    "AnswerQuality",
    "AutonomousCurriculum",
    "CallableBackend",
    "CognitiveEvaluator",
    "CognitiveLearningSystem",
    "CognitiveReport",
    "CognitiveRunner",
    "Comparison",
    "ConsolidationReport",
    "Consolidator",
    "CurriculumReport",
    "CyclePlan",
    "ErrorAnalysis",
    "Escalation",
    "EscalationPolicy",
    "Evaluation",
    "Experience",
    "FocusBuffer",
    "Idea",
    "IdeaKind",
    "IncrementalTrainer",
    "LearningStats",
    "MetacognitiveController",
    "ModelBackend",
    "ModelRegistry",
    "ModelVersion",
    "MultiSeedEvaluator",
    "Pattern",
    "PersistentNovelty",
    "ProactiveMind",
    "Reliability",
    "ResourceMode",
    "ReviewSchedule",
    "SchedulerSettings",
    "SchedulerState",
    "SelfModel",
    "SkillRegistry",
    "Source",
    "SourceKind",
    "StagnationDetector",
    "StagnationReading",
    "StrategyYield",
    "StudentProfile",
    "TeacherStudentSystem",
    "TeachingOutcome",
    "ThoughtCycle",
    "ThoughtScheduler",
    "TrainingExample",
    "TrainingExampleKind",
    "TrainingMemory",
    "TrainingOutcome",
    "TrainingPlan",
    "TrainingStage",
    "load_scheduler_state",
    "promote",
]
