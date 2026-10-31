"""The Autonomous Cognitive System: a Proactive Mind, cognitive learning, and staged training.

This package is the project's **autonomous cognitive layer**, added beside the
existing cognitive core rather than inside it. It is composed of three parts, and
they share one rule: *nothing here widens the model's own authority*. The mind may
think, propose, criticise, and learn, but a generated idea is never a fact, a
teacher is never trusted blindly, and a trained version is never adopted without
an evaluation that shows it did not regress.

* :mod:`cognition.proactive_mind` -- the eight engines composed into one bounded
  thinking cycle (:class:`ProactiveMind`).
* :mod:`cognition.learning_system` -- six memories and the learning operations
  over them (:class:`CognitiveLearningSystem`).
* :mod:`cognition.teacher_student` -- the student-first exchange that turns a
  teacher's answer into a *qualified* example (:class:`TeacherStudentSystem`).
* :mod:`cognition.training` -- staged, resumable training with a candidate /
  version / rollback gate (:class:`IncrementalTrainer`).
* :mod:`cognition.evaluation` -- the ten cognitive indicators and a before/after
  comparison (:class:`CognitiveEvaluator`).
* :mod:`cognition.runner` -- background operation under three resource modes
  (:class:`CognitiveRunner`).

The records (:mod:`cognition.records`), storage (:mod:`cognition.storage`), and
reliability rule (:mod:`cognition.reliability`) are the shared foundation.
"""

from __future__ import annotations

from cognition.evaluation import (
    INDICATORS,
    LABELS,
    CognitiveEvaluator,
    CognitiveReport,
    Comparison,
)
from cognition.learning_system import CognitiveLearningSystem, ErrorAnalysis, LearningStats
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
    "AnswerQuality",
    "CallableBackend",
    "CognitiveEvaluator",
    "CognitiveLearningSystem",
    "CognitiveReport",
    "CognitiveRunner",
    "Comparison",
    "ErrorAnalysis",
    "Evaluation",
    "Experience",
    "Idea",
    "IdeaKind",
    "IncrementalTrainer",
    "LearningStats",
    "ModelBackend",
    "ModelRegistry",
    "ModelVersion",
    "ProactiveMind",
    "Reliability",
    "ResourceMode",
    "SchedulerSettings",
    "SchedulerState",
    "Source",
    "SourceKind",
    "StudentProfile",
    "TeacherStudentSystem",
    "TeachingOutcome",
    "ThoughtCycle",
    "ThoughtScheduler",
    "TrainingExample",
    "TrainingExampleKind",
    "TrainingOutcome",
    "TrainingPlan",
    "TrainingStage",
    "load_scheduler_state",
    "promote",
]
