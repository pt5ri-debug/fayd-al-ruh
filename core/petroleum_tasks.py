"""Petroleum-engineering sandbox tasks, one or more per discipline.

These tasks give the project's petroleum identity something to *learn*, not just
talk about. Each task is a short, explicit action chain whose final fact is a
real petroleum milestone -- a reserve estimate, a drilled well, restored
production, a verified cement bond, a computed net present value. The tasks run
inside the same safe, replayable symbolic sandbox as the toy tasks, so the
learning machinery is exercised without any external tool.

The catalog is split the same way as the toy one: ``PETROLEUM_TASKS`` are trained
on, ``PETROLEUM_HELD_OUT_TASKS`` are never trained on and are used to check that
the same machinery generalises to a discipline it has not seen.
"""

from __future__ import annotations

from core.sandbox_environment import SandboxTask
from core.symbolic_simulation import ActionSpec


def _step(name: str, requires: str, produces: str, *, consumes: str | None = None) -> ActionSpec:
    return ActionSpec(
        name,
        frozenset({requires}),
        frozenset({produces}),
        frozenset({consumes}) if consumes else frozenset(),
    )


def _reservoir_appraisal_task() -> SandboxTask:
    return SandboxTask(
        name="reservoir-appraisal",
        description="Appraise a structure: drill, characterise the reservoir, estimate reserves.",
        discipline="reservoir",
        allowed_actions=(
            _step("drill_exploration", "undrilled_structure", "discovery"),
            _step("log_reservoir", "discovery", "reservoir_characterized", consumes="discovery"),
            _step("estimate_reserves", "reservoir_characterized", "reserves_estimated"),
            _step("review_seismic", "undrilled_structure", "seismic_reviewed"),
        ),
        initial_facts=frozenset({"undrilled_structure"}),
        max_steps=5,
    )


def _drill_well_task() -> SandboxTask:
    return SandboxTask(
        name="drill-well",
        description="Drill a well: set surface casing, then drill to the target.",
        discipline="drilling",
        allowed_actions=(
            _step("drill_surface", "surface_ready", "surface_hole"),
            _step("run_casing", "surface_hole", "casing_set"),
            _step("drill_target", "casing_set", "well_drilled"),
            _step("circulate_mud", "surface_ready", "mud_circulated"),
        ),
        initial_facts=frozenset({"surface_ready"}),
        max_steps=5,
    )


def _restore_production_task() -> SandboxTask:
    return SandboxTask(
        name="restore-production",
        description="Restore a declining well: support pressure, then stimulate.",
        discipline="production",
        allowed_actions=(
            _step("water_injection", "rate_declining", "pressure_supported"),
            _step("stimulate", "pressure_supported", "production_restored"),
            _step("monitor", "rate_declining", "rate_monitored"),
        ),
        initial_facts=frozenset({"rate_declining"}),
        max_steps=5,
    )


def _core_analysis_task() -> SandboxTask:
    return SandboxTask(
        name="core-analysis",
        description="Analyse a core: porosity, permeability, then saturation.",
        discipline="petrophysics",
        allowed_actions=(
            _step("measure_porosity", "core_sample", "porosity_measured"),
            _step("measure_permeability", "porosity_measured", "permeability_measured"),
            _step("compute_saturation", "permeability_measured", "saturation_known"),
            _step("photograph_core", "core_sample", "core_photographed"),
        ),
        initial_facts=frozenset({"core_sample"}),
        max_steps=5,
    )


def _log_interpretation_task() -> SandboxTask:
    return SandboxTask(
        name="log-interpretation",
        description="Interpret logs: calibrate, estimate saturation, identify pay.",
        discipline="well_logging",
        allowed_actions=(
            _step("calibrate_log", "raw_log", "calibrated_log"),
            _step("interpret_resistivity", "calibrated_log", "saturation_estimated"),
            _step("identify_pay", "saturation_estimated", "pay_identified"),
            _step("print_log", "raw_log", "log_printed"),
        ),
        initial_facts=frozenset({"raw_log"}),
        max_steps=5,
    )


def _complete_well_task() -> SandboxTask:
    return SandboxTask(
        name="complete-well",
        description="Complete a well: set casing, perforate, install completion.",
        discipline="completions",
        allowed_actions=(
            _step("run_casing", "well_drilled", "casing_set"),
            _step("perforate", "casing_set", "perforated"),
            _step("install_completion", "perforated", "well_completed"),
            _step("inspect_equipment", "well_drilled", "equipment_inspected"),
        ),
        initial_facts=frozenset({"well_drilled"}),
        max_steps=5,
    )


def _enhanced_recovery_task() -> SandboxTask:
    return SandboxTask(
        name="enhanced-recovery",
        description="Enhanced recovery: polymer flood, improve sweep, raise recovery.",
        discipline="eor",
        allowed_actions=(
            _step("polymer_injection", "mature_field", "polymer_flood"),
            _step("mobility_control", "polymer_flood", "sweep_improved"),
            _step("recover_incremental", "sweep_improved", "recovery_improved"),
            _step("sample_fluids", "mature_field", "fluids_sampled"),
        ),
        initial_facts=frozenset({"mature_field"}),
        max_steps=5,
    )


def _pressure_transient_test_task() -> SandboxTask:
    return SandboxTask(
        name="pressure-transient-test",
        description="Well test: record buildup, identify flow regime, estimate skin.",
        discipline="well_testing",
        allowed_actions=(
            _step("record_buildup", "well_shut_in", "buildup_recorded"),
            _step("analyze_derivative", "buildup_recorded", "flow_regime_identified"),
            _step("estimate_skin", "flow_regime_identified", "skin_estimated"),
            _step("check_gauge", "well_shut_in", "gauge_checked"),
        ),
        initial_facts=frozenset({"well_shut_in"}),
        max_steps=5,
    )


def _install_gas_lift_task() -> SandboxTask:
    return SandboxTask(
        name="install-gas-lift",
        description="Gas lift: install valves, inject gas, stabilise the rate.",
        discipline="artificial_lift",
        allowed_actions=(
            _step("install_valves", "low_bhp", "valves_installed"),
            _step("inject_gas", "valves_installed", "gas_lifted"),
            _step("stabilize_rate", "gas_lifted", "rate_restored"),
            _step("survey_well", "low_bhp", "well_surveyed"),
        ),
        initial_facts=frozenset({"low_bhp"}),
        max_steps=5,
    )


def _matrix_acidizing_task() -> SandboxTask:
    return SandboxTask(
        name="matrix-acidizing",
        description="Stimulate: pump acid, let it react, flow back to restore permeability.",
        discipline="stimulation",
        allowed_actions=(
            _step("pump_acid", "formation_damaged", "acid_placed"),
            _step("soak", "acid_placed", "reaction_complete"),
            _step("flow_back", "reaction_complete", "permeability_restored"),
            _step("sample_returns", "formation_damaged", "returns_sampled"),
        ),
        initial_facts=frozenset({"formation_damaged"}),
        max_steps=5,
    )


def _primary_cementing_task() -> SandboxTask:
    return SandboxTask(
        name="primary-cementing",
        description="Cement: place slurry, bump the plug, verify the bond.",
        discipline="cementing",
        allowed_actions=(
            _step("pump_cement", "casing_set", "cement_placed"),
            _step("displace_plug", "cement_placed", "plug_bumped"),
            _step("verify_bond", "plug_bumped", "bond_verified"),
            _step("mix_slurry", "casing_set", "slurry_mixed"),
        ),
        initial_facts=frozenset({"casing_set"}),
        max_steps=5,
    )


def _control_kick_task() -> SandboxTask:
    return SandboxTask(
        name="control-kick",
        description="Well control: close the BOP, circulate kill mud, verify stability.",
        discipline="well_control",
        allowed_actions=(
            _step("close_bop", "kick_detected", "well_shut_in"),
            _step("circulate_kill_mud", "well_shut_in", "kill_mud_circulated"),
            _step("verify_stable", "kill_mud_circulated", "well_stable"),
            _step("record_pressures", "kick_detected", "pressures_recorded"),
        ),
        initial_facts=frozenset({"kick_detected"}),
        max_steps=5,
    )


def _field_economics_task() -> SandboxTask:
    return SandboxTask(
        name="field-economics",
        description="Evaluate economics: estimate revenue, estimate costs, compute NPV.",
        discipline="economics",
        allowed_actions=(
            _step("estimate_revenue", "production_profile", "revenue_estimated"),
            _step("estimate_costs", "revenue_estimated", "costs_estimated"),
            _step("compute_npv", "costs_estimated", "npv_computed"),
            _step("build_profile", "production_profile", "profile_built"),
        ),
        initial_facts=frozenset({"production_profile"}),
        max_steps=5,
    )


def _well_integrity_task() -> SandboxTask:
    return SandboxTask(
        name="well-integrity-check",
        description="Verify integrity: inspect, test barriers, certify.",
        discipline="hse",
        allowed_actions=(
            _step("run_inspection", "well_operating", "inspection_done"),
            _step("test_barriers", "inspection_done", "barriers_tested"),
            _step("certify", "barriers_tested", "integrity_certified"),
            _step("file_report", "well_operating", "report_filed"),
        ),
        initial_facts=frozenset({"well_operating"}),
        max_steps=5,
    )


def _gas_processing_task() -> SandboxTask:
    return SandboxTask(
        name="gas-processing",
        description="Treat gas: dehydrate, sweeten, recover NGL.",
        discipline="facilities",
        allowed_actions=(
            _step("dehydrate_gas", "wet_gas", "gas_dehydrated"),
            _step("sweeten_gas", "gas_dehydrated", "gas_sweetened"),
            _step("recover_ngl", "gas_sweetened", "ngl_recovered"),
            _step("sample_stream", "wet_gas", "stream_sampled"),
        ),
        initial_facts=frozenset({"wet_gas"}),
        max_steps=5,
    )


def _process_safety_task() -> SandboxTask:
    return SandboxTask(
        name="process-safety-review",
        description="Manage process safety: run a HAZOP, define barriers, verify protection.",
        discipline="process_safety",
        allowed_actions=(
            _step("run_hazop", "operating_plant", "hazards_identified"),
            _step("define_barriers", "hazards_identified", "barriers_defined"),
            _step("verify_layers", "barriers_defined", "protection_verified"),
            _step("review_procedures", "operating_plant", "procedures_reviewed"),
        ),
        initial_facts=frozenset({"operating_plant"}),
        max_steps=5,
    )


def _unconventional_completion_task() -> SandboxTask:
    return SandboxTask(
        name="complete-shale-well",
        description="Complete a shale well: land the lateral, pump stages, place proppant.",
        discipline="unconventional",
        allowed_actions=(
            _step("identify_sweet_spot", "lateral_drilled", "landing_confirmed"),
            _step("pump_fracture_stages", "landing_confirmed", "stages_pumped"),
            _step("place_proppant", "stages_pumped", "proppant_placed"),
            _step("take_core", "lateral_drilled", "core_taken"),
        ),
        initial_facts=frozenset({"lateral_drilled"}),
        max_steps=5,
    )


def _seismic_interpretation_task() -> SandboxTask:
    """Held-out: geoscience, never trained on."""
    return SandboxTask(
        name="seismic-interpretation",
        description="Interpret seismic: pick a horizon, map the structure, locate a prospect.",
        discipline="geoscience",
        allowed_actions=(
            _step("interpret_horizon", "seismic_volume", "horizon_interpreted"),
            _step("map_structure", "horizon_interpreted", "structure_mapped"),
            _step("locate_prospect", "structure_mapped", "prospect_located"),
            _step("reprocess_volume", "seismic_volume", "volume_reprocessed"),
        ),
        initial_facts=frozenset({"seismic_volume"}),
        max_steps=5,
    )


def _flow_assurance_task() -> SandboxTask:
    """Held-out: flow assurance, never trained on."""
    return SandboxTask(
        name="flow-assurance",
        description="Prevent a hydrate plug: inhibit, monitor, keep the line flowing.",
        discipline="flow_assurance",
        allowed_actions=(
            _step("inject_inhibitor", "cold_line", "inhibitor_injected"),
            _step("monitor_conditions", "inhibitor_injected", "conditions_monitored"),
            _step("confirm_flow", "conditions_monitored", "flow_assured"),
            _step("sample_gas", "cold_line", "gas_sampled"),
        ),
        initial_facts=frozenset({"cold_line"}),
        max_steps=5,
    )


PETROLEUM_TASKS: tuple[SandboxTask, ...] = (
    _reservoir_appraisal_task(),
    _drill_well_task(),
    _restore_production_task(),
    _core_analysis_task(),
    _log_interpretation_task(),
    _complete_well_task(),
    _enhanced_recovery_task(),
    _pressure_transient_test_task(),
    _install_gas_lift_task(),
    _matrix_acidizing_task(),
    _primary_cementing_task(),
    _control_kick_task(),
    _field_economics_task(),
    _well_integrity_task(),
    _gas_processing_task(),
    _process_safety_task(),
    _unconventional_completion_task(),
)

# Disciplines the learning agents never train on; used to measure generalisation.
PETROLEUM_HELD_OUT_TASKS: tuple[SandboxTask, ...] = (
    _seismic_interpretation_task(),
    _flow_assurance_task(),
)

# The fact whose presence marks success, keyed by task name.
PETROLEUM_SUCCESS_FACTS: dict[str, str] = {
    "reservoir-appraisal": "reserves_estimated",
    "drill-well": "well_drilled",
    "restore-production": "production_restored",
    "core-analysis": "saturation_known",
    "log-interpretation": "pay_identified",
    "complete-well": "well_completed",
    "enhanced-recovery": "recovery_improved",
    "pressure-transient-test": "skin_estimated",
    "install-gas-lift": "rate_restored",
    "matrix-acidizing": "permeability_restored",
    "primary-cementing": "bond_verified",
    "control-kick": "well_stable",
    "field-economics": "npv_computed",
    "well-integrity-check": "integrity_certified",
    "gas-processing": "ngl_recovered",
    "process-safety-review": "protection_verified",
    "complete-shale-well": "proppant_placed",
    "seismic-interpretation": "prospect_located",
    "flow-assurance": "flow_assured",
}


def petroleum_tasks() -> tuple[SandboxTask, ...]:
    """Every petroleum task, trained and held-out."""
    return (*PETROLEUM_TASKS, *PETROLEUM_HELD_OUT_TASKS)


def petroleum_task_by_name(name: str) -> SandboxTask:
    for task in petroleum_tasks():
        if task.name == name:
            return task
    raise KeyError(f"Unknown petroleum task: {name}")


def success_fact_for(name: str) -> str:
    try:
        return PETROLEUM_SUCCESS_FACTS[name]
    except KeyError as error:
        raise KeyError(f"Unknown petroleum task: {name}") from error


def tasks_by_discipline() -> dict[str, tuple[SandboxTask, ...]]:
    """Trained tasks grouped by their discipline key."""
    grouped: dict[str, list[SandboxTask]] = {}
    for task in PETROLEUM_TASKS:
        grouped.setdefault(task.discipline, []).append(task)
    return {key: tuple(tasks) for key, tasks in grouped.items()}


__all__ = [
    "PETROLEUM_HELD_OUT_TASKS",
    "PETROLEUM_SUCCESS_FACTS",
    "PETROLEUM_TASKS",
    "petroleum_task_by_name",
    "petroleum_tasks",
    "success_fact_for",
    "tasks_by_discipline",
]
