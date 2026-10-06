"""The petroleum-engineering curriculum: seed knowledge for the whole degree.

Where ``core.domain_knowledge`` gives the project its petroleum *frame* (the
handful of relations a practitioner reasons over every day), this module gives it
the *curriculum* -- the concepts and relations a student of petroleum engineering
meets across the degree, organized by the course that teaches them.

It is a compact, evidence-aware map of the program, not a textbook: each course
contributes the concepts it is built on and the relations between them, all at a
modest seeded confidence that later evidence can revise. The map spans the four
blocks of a real petroleum-engineering degree:

- foundations: mathematics, physics, chemistry, statistics, computing, numerical
  methods;
- geoscience: general, structural, and petroleum geology, sedimentology,
  geophysics;
- engineering science: fluid mechanics, thermodynamics, mechanics of materials,
  rock mechanics;
- the petroleum disciplines: petrophysics, reservoir fluids and engineering,
  drilling, production, well logging, completions, cementing, stimulation,
  artificial lift, well testing, enhanced recovery, well control, facilities,
  offshore, economics, project management, HSE, and professional ethics.

``seed_curriculum`` plants it into an ontology (idempotently) alongside the domain
seed, so the project's knowledge is genuinely the knowledge of the field.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.concept import Concept
from core.ontology import Ontology
from core.semantic_relation import SemanticRelation


@dataclass(frozen=True)
class CourseKnowledge:
    """The concepts and relations one course of the degree contributes."""

    key: str
    label: str
    category: str
    concepts: tuple[tuple[str, tuple[str, ...]], ...]
    relations: tuple[tuple[str, str, str, float], ...]

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.label.strip():
            raise ValueError("Course key and label cannot be empty.")
        if not self.category.strip():
            raise ValueError(f"Course '{self.key}' needs a category.")
        if not self.concepts:
            raise ValueError(f"Course '{self.key}' needs at least one concept.")
        if not self.relations:
            raise ValueError(f"Course '{self.key}' needs at least one relation.")


# Concepts that recur across many courses. Seeded once and shared, so a relation
# in one course can lean on the same node another course introduced.
_SHARED_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mathematical_model", ("mathematical model",)),
    ("numerical_model", ("numerical model",)),
    ("energy", ("energy",)),
    ("force", ("force",)),
    ("work", ("work",)),
    ("heat", ("heat",)),
    ("temperature", ("temperature",)),
    ("fluid", ("fluid",)),
    ("flow", ("flow",)),
    ("experiment", ("experiment",)),
    ("data", ("data",)),
    ("measurement", ("measurement",)),
    ("instrument", ("instrument", "device")),
    ("error", ("error", "uncertainty")),
    ("estimation", ("estimation",)),
    ("risk", ("risk",)),
    ("decision", ("decision",)),
    ("cost", ("cost",)),
    ("law", ("law", "principle")),
    ("equation", ("equation",)),
    ("algorithm", ("algorithm",)),
    ("program", ("program", "software")),
    ("simulation", ("simulation",)),
    ("result", ("result",)),
    ("mineral", ("mineral",)),
    ("sedimentary_rock", ("sedimentary rock",)),
    ("reservoir_rock", ("reservoir rock", "carrier bed")),
    ("sedimentary_basin", ("sedimentary basin", "basin")),
    ("migration", ("migration",)),
    ("trap", ("trap",)),
    ("liquid", ("liquid",)),
    ("gas", ("gas",)),
    ("oil", ("oil", "crude")),
    ("contamination", ("contamination", "pollution")),
    ("environment", ("environment",)),
)


_COURSES: tuple[CourseKnowledge, ...] = (
    CourseKnowledge(
        key="math",
        label="Mathematics",
        category="Foundations",
        concepts=(
            ("calculus", ("calculus", "differentiation")),
            ("integration", ("integration", "integral")),
            ("differential_equations", ("differential equations",)),
            ("linear_algebra", ("linear algebra",)),
            ("matrix", ("matrix", "matrices")),
            ("vectors", ("vectors",)),
            ("series", ("series",)),
        ),
        relations=(
            ("calculus", "differential_equations", "used_in", 0.7),
            ("integration", "differential_equations", "used_in", 0.7),
            ("matrix", "linear_algebra", "part_of", 0.7),
            ("vectors", "linear_algebra", "part_of", 0.7),
            ("differential_equations", "mathematical_model", "builds", 0.6),
        ),
    ),
    CourseKnowledge(
        key="physics",
        label="Physics",
        category="Foundations",
        concepts=(
            ("mechanics", ("mechanics",)),
            ("electricity", ("electricity",)),
            ("magnetism", ("magnetism",)),
            ("waves", ("waves",)),
            ("light", ("light", "optics")),
            ("kinetic_energy", ("kinetic energy",)),
        ),
        relations=(
            ("force", "mechanics", "part_of", 0.7),
            ("mechanics", "energy", "conserves", 0.6),
            ("electricity", "magnetism", "related_to", 0.7),
            ("waves", "energy", "transports", 0.6),
            ("light", "waves", "is_a", 0.7),
            ("force", "kinetic_energy", "produces", 0.6),
        ),
    ),
    CourseKnowledge(
        key="chemistry",
        label="Chemistry",
        category="Foundations",
        concepts=(
            ("atom", ("atom",)),
            ("molecule", ("molecule",)),
            ("chemical_bond", ("chemical bond",)),
            ("chemical_reaction", ("chemical reaction",)),
            ("acidity", ("acidity", "pH")),
            ("solution", ("solution",)),
            ("chemical_analysis", ("chemical analysis",)),
        ),
        relations=(
            ("atom", "molecule", "forms", 0.7),
            ("molecule", "chemical_bond", "held_by", 0.7),
            ("chemical_reaction", "acidity", "affects", 0.6),
            ("solution", "chemical_reaction", "hosts", 0.6),
            ("chemical_analysis", "solution", "measures", 0.6),
        ),
    ),
    CourseKnowledge(
        key="statistics",
        label="Statistics and Data Analysis",
        category="Foundations",
        concepts=(
            ("probability", ("probability",)),
            ("distribution", ("distribution",)),
            ("mean", ("mean", "average")),
            ("variance", ("variance",)),
            ("regression", ("regression",)),
            ("correlation", ("correlation",)),
            ("sample", ("sample", "dataset")),
        ),
        relations=(
            ("probability", "distribution", "describes", 0.7),
            ("distribution", "mean", "has", 0.6),
            ("distribution", "variance", "has", 0.6),
            ("regression", "correlation", "measures", 0.6),
            ("sample", "regression", "trains", 0.6),
            ("sample", "data", "is_a", 0.7),
        ),
    ),
    CourseKnowledge(
        key="computing",
        label="Computing and Programming",
        category="Foundations",
        concepts=(
            ("computer_algorithm", ("computer algorithm",)),
            ("programming_language", ("programming language",)),
            ("data_structure", ("data structure",)),
            ("database", ("database",)),
            ("numerical_computing", ("numerical computing",)),
            ("engineering_software", ("engineering software",)),
        ),
        relations=(
            ("computer_algorithm", "program", "implemented_by", 0.7),
            ("programming_language", "program", "writes", 0.6),
            ("data_structure", "computer_algorithm", "supports", 0.6),
            ("database", "data", "stores", 0.7),
            ("engineering_software", "simulation", "runs", 0.6),
            ("numerical_computing", "numerical_model", "computes", 0.6),
        ),
    ),
    CourseKnowledge(
        key="numerical_methods",
        label="Numerical Methods",
        category="Foundations",
        concepts=(
            ("numerical_approximation", ("numerical approximation",)),
            ("numerical_analysis", ("numerical analysis",)),
            ("iteration", ("iteration",)),
            ("grid", ("grid", "mesh")),
            ("stability", ("stability",)),
            ("convergence", ("convergence",)),
        ),
        relations=(
            ("numerical_approximation", "numerical_model", "builds", 0.6),
            ("numerical_analysis", "error", "bounds", 0.6),
            ("iteration", "convergence", "seeks", 0.6),
            ("grid", "differential_equations", "discretizes", 0.6),
            ("stability", "result", "qualifies", 0.6),
        ),
    ),
    CourseKnowledge(
        key="geology",
        label="General Geology",
        category="Earth Sciences",
        concepts=(
            ("crust", ("crust",)),
            ("rocks", ("rocks",)),
            ("igneous_rock", ("igneous rock",)),
            ("metamorphic_rock", ("metamorphic rock",)),
            ("rock_cycle", ("rock cycle",)),
            ("geologic_time", ("geologic time",)),
            ("strata", ("strata", "layers")),
        ),
        relations=(
            ("igneous_rock", "rocks", "is_a", 0.7),
            ("metamorphic_rock", "rocks", "is_a", 0.7),
            ("sedimentary_rock", "rocks", "is_a", 0.7),
            ("rocks", "rock_cycle", "undergo", 0.6),
            ("strata", "geologic_time", "dated_by", 0.6),
            ("crust", "rocks", "composed_of", 0.7),
        ),
    ),
    CourseKnowledge(
        key="structural_geology",
        label="Structural Geology",
        category="Earth Sciences",
        concepts=(
            ("fold", ("fold",)),
            ("fault", ("fault",)),
            ("fracture", ("fracture",)),
            ("tectonic_stress", ("tectonic stress",)),
            ("structural_accumulation", ("structural trap",)),
            ("cross_section", ("cross-section",)),
        ),
        relations=(
            ("tectonic_stress", "fold", "forms", 0.6),
            ("tectonic_stress", "fault", "forms", 0.6),
            ("fault", "fracture", "creates", 0.6),
            ("structural_accumulation", "trap", "is_a", 0.7),
            ("cross_section", "strata", "shows", 0.6),
        ),
    ),
    CourseKnowledge(
        key="sedimentology",
        label="Sedimentology",
        category="Earth Sciences",
        concepts=(
            ("deposition", ("deposition",)),
            ("grains", ("grains", "clasts")),
            ("grain_sorting", ("grain sorting",)),
            ("facies", ("facies",)),
            ("sandstone", ("sandstone",)),
            ("limestone", ("limestone",)),
            ("primary_porosity", ("primary porosity",)),
        ),
        relations=(
            ("deposition", "sedimentary_rock", "forms", 0.7),
            ("grains", "grain_sorting", "sorted_by", 0.6),
            ("sandstone", "sedimentary_rock", "is_a", 0.7),
            ("limestone", "sedimentary_rock", "is_a", 0.7),
            ("facies", "deposition", "records", 0.6),
            ("primary_porosity", "deposition", "results_from", 0.6),
        ),
    ),
    CourseKnowledge(
        key="petroleum_geology",
        label="Petroleum Geology",
        category="Earth Sciences",
        concepts=(
            ("source_rock", ("source rock",)),
            ("organic_matter", ("organic matter",)),
            ("thermal_maturation", ("thermal maturation",)),
            ("primary_migration", ("primary migration",)),
            ("secondary_migration", ("secondary migration",)),
            ("structural_trap", ("structural trap",)),
            ("stratigraphic_trap", ("stratigraphic trap",)),
            ("cap_rock", ("cap rock", "seal")),
        ),
        relations=(
            ("source_rock", "organic_matter", "contains", 0.7),
            ("organic_matter", "thermal_maturation", "transformed_by", 0.7),
            ("thermal_maturation", "oil", "generates", 0.7),
            ("thermal_maturation", "gas", "generates", 0.7),
            ("oil", "primary_migration", "undergoes", 0.6),
            ("primary_migration", "secondary_migration", "followed_by", 0.6),
            ("secondary_migration", "trap", "accumulates_in", 0.7),
            ("cap_rock", "trap", "seals", 0.7),
            ("structural_trap", "trap", "is_a", 0.7),
            ("stratigraphic_trap", "trap", "is_a", 0.7),
        ),
    ),
    CourseKnowledge(
        key="geophysics",
        label="Geophysics",
        category="Earth Sciences",
        concepts=(
            ("seismic_wave", ("seismic wave",)),
            ("seismic_survey", ("seismic survey",)),
            ("seismic_reflection", ("seismic reflection",)),
            ("wave_velocity", ("wave velocity",)),
            ("gravity_survey", ("gravity survey",)),
            ("geomagnetism", ("geomagnetism",)),
            ("seismic_section", ("seismic section",)),
        ),
        relations=(
            ("seismic_survey", "seismic_wave", "generates", 0.7),
            ("seismic_reflection", "seismic_wave", "is_a", 0.6),
            ("seismic_wave", "wave_velocity", "has", 0.6),
            ("seismic_section", "seismic_reflection", "built_from", 0.6),
            ("gravity_survey", "rocks", "images", 0.5),
            ("geomagnetism", "rocks", "images", 0.5),
            ("seismic_survey", "trap", "locates", 0.6),
        ),
    ),
    CourseKnowledge(
        key="fluid_mechanics",
        label="Fluid Mechanics",
        category="Engineering Science",
        concepts=(
            ("viscosity", ("viscosity",)),
            ("density", ("density",)),
            ("fluid_pressure", ("fluid pressure",)),
            ("bernoulli", ("Bernoulli",)),
            ("darcy_s_law", ("Darcy's law",)),
            ("reynolds_number", ("Reynolds number",)),
            ("laminar_flow", ("laminar flow",)),
            ("turbulent_flow", ("turbulent flow",)),
        ),
        relations=(
            ("viscosity", "flow", "resists", 0.7),
            ("density", "fluid_pressure", "affects", 0.6),
            ("bernoulli", "fluid_pressure", "relates", 0.6),
            ("darcy_s_law", "flow", "governs", 0.7),
            ("reynolds_number", "turbulent_flow", "predicts", 0.6),
            ("laminar_flow", "turbulent_flow", "transition_to", 0.6),
            ("darcy_s_law", "permeability", "depends_on", 0.6),
        ),
    ),
    CourseKnowledge(
        key="thermodynamics",
        label="Thermodynamics",
        category="Engineering Science",
        concepts=(
            ("first_law", ("first law",)),
            ("second_law", ("second law",)),
            ("entropy", ("entropy",)),
            ("enthalpy", ("enthalpy",)),
            ("phase_behavior", ("phase behavior",)),
            ("thermal_equilibrium", ("thermal equilibrium",)),
            ("equation_of_state", ("equation of state",)),
        ),
        relations=(
            ("first_law", "energy", "conserves", 0.7),
            ("second_law", "entropy", "introduces", 0.7),
            ("enthalpy", "energy", "measures", 0.6),
            ("phase_behavior", "equation_of_state", "described_by", 0.6),
            ("thermal_equilibrium", "temperature", "equalizes", 0.6),
            ("equation_of_state", "oil", "applies_to", 0.5),
        ),
    ),
    CourseKnowledge(
        key="mechanics_of_materials",
        label="Mechanics of Materials",
        category="Engineering Science",
        concepts=(
            ("stress", ("stress",)),
            ("strain", ("strain",)),
            ("young_s_modulus", ("Young's modulus",)),
            ("bending", ("bending",)),
            ("torsion", ("torsion",)),
            ("failure", ("failure",)),
            ("safety_factor", ("safety factor",)),
        ),
        relations=(
            ("stress", "strain", "causes", 0.7),
            ("young_s_modulus", "stress", "relates", 0.7),
            ("bending", "stress", "induces", 0.6),
            ("torsion", "stress", "induces", 0.6),
            ("failure", "stress", "results_from", 0.6),
            ("safety_factor", "failure", "prevents", 0.6),
            ("safety_factor", "engineering_design", "guards", 0.6),
        ),
    ),
    CourseKnowledge(
        key="rock_mechanics",
        label="Rock Mechanics",
        category="Engineering Science",
        concepts=(
            ("in_situ_stress", ("in-situ stress", "overburden")),
            ("pore_pressure", ("pore pressure",)),
            ("tensile_strength", ("tensile strength",)),
            ("compressive_strength", ("compressive strength",)),
            ("natural_fracture", ("natural fracture",)),
            ("rock_failure", ("rock failure",)),
            ("fracture_pressure", ("fracture pressure",)),
        ),
        relations=(
            ("in_situ_stress", "pore_pressure", "opposed_by", 0.6),
            ("fracture_pressure", "in_situ_stress", "depends_on", 0.6),
            ("natural_fracture", "tensile_strength", "opens_under", 0.6),
            ("rock_failure", "compressive_strength", "governed_by", 0.6),
            ("pore_pressure", "rock_failure", "triggers", 0.5),
            ("compressive_strength", "safety_factor", "checked_by", 0.5),
        ),
    ),
    CourseKnowledge(
        key="petrophysics",
        label="Petrophysics",
        category="Petroleum Disciplines",
        concepts=(
            ("effective_porosity", ("effective porosity",)),
            ("absolute_permeability", ("absolute permeability",)),
            ("water_saturation", ("water saturation",)),
            ("hydrocarbon_saturation", ("hydrocarbon saturation",)),
            ("formation_factor", ("formation factor",)),
            ("electrical_conductivity", ("electrical conductivity",)),
            ("compressibility", ("compressibility",)),
            ("petrophysical_properties", ("petrophysical properties",)),
        ),
        relations=(
            ("effective_porosity", "porosity", "is_a", 0.7),
            ("absolute_permeability", "permeability", "is_a", 0.7),
            ("water_saturation", "water", "measures", 0.7),
            ("hydrocarbon_saturation", "hydrocarbon", "measures", 0.7),
            ("formation_factor", "effective_porosity", "depends_on", 0.6),
            ("electrical_conductivity", "water_saturation", "indicates", 0.6),
            ("compressibility", "porosity", "affects", 0.5),
            ("petrophysical_properties", "reservoir_rock", "describe", 0.7),
        ),
    ),
    CourseKnowledge(
        key="reservoir_fluids",
        label="Reservoir Fluid Properties",
        category="Petroleum Disciplines",
        concepts=(
            ("reservoir_fluid", ("reservoir fluid",)),
            ("gas_oil_ratio", ("gas-oil ratio", "GOR")),
            ("formation_volume_factor", ("formation volume factor",)),
            ("bubble_point", ("bubble point",)),
            ("dew_point", ("dew point",)),
            ("fluid_viscosity", ("fluid viscosity",)),
            ("thermal_expansion", ("thermal expansion",)),
        ),
        relations=(
            ("reservoir_fluid", "oil", "includes", 0.7),
            ("reservoir_fluid", "gas", "includes", 0.7),
            ("reservoir_fluid", "water", "includes", 0.6),
            ("gas_oil_ratio", "reservoir_fluid", "characterizes", 0.6),
            ("formation_volume_factor", "reservoir_fluid", "characterizes", 0.6),
            ("bubble_point", "gas", "releases", 0.6),
            ("dew_point", "liquid", "condenses", 0.5),
            ("fluid_viscosity", "viscosity", "is_a", 0.6),
        ),
    ),
    CourseKnowledge(
        key="reservoir_engineering",
        label="Reservoir Engineering",
        category="Petroleum Disciplines",
        concepts=(
            ("material_balance", ("material balance",)),
            ("productivity_index", ("productivity index",)),
            ("drive_mechanism", ("drive mechanism",)),
            ("solution_gas_drive", ("solution gas drive",)),
            ("natural_water_drive", ("natural water drive",)),
            ("pressure_distribution", ("pressure distribution",)),
            ("reservoir_connectivity", ("reservoir connectivity",)),
            ("reservoir_study", ("reservoir study",)),
        ),
        relations=(
            ("material_balance", "reserve", "estimates", 0.7),
            ("productivity_index", "production_rate", "predicts", 0.6),
            ("drive_mechanism", "reservoir_pressure", "maintains", 0.7),
            ("solution_gas_drive", "drive_mechanism", "is_a", 0.6),
            ("natural_water_drive", "drive_mechanism", "is_a", 0.6),
            ("reservoir_connectivity", "reservoir_simulation", "modeled_by", 0.6),
            ("pressure_distribution", "reservoir_pressure", "maps", 0.6),
            ("reservoir_study", "reserve", "verifies", 0.6),
        ),
    ),
    CourseKnowledge(
        key="drilling_engineering",
        label="Drilling Engineering",
        category="Petroleum Disciplines",
        concepts=(
            ("drilling_rig", ("drilling rig",)),
            ("drill_string", ("drill string",)),
            ("weight_on_bit", ("weight on bit",)),
            ("rate_of_penetration", ("rate of penetration",)),
            ("well_casing", ("well casing",)),
            ("formation_pressure", ("formation pressure", "pore pressure")),
            ("mud_engineering", ("mud engineering",)),
            ("well_trajectory", ("well trajectory",)),
        ),
        relations=(
            ("drilling_rig", "well", "drills", 0.7),
            ("drill_string", "drill_bit", "connects", 0.6),
            ("weight_on_bit", "rate_of_penetration", "affects", 0.6),
            ("rate_of_penetration", "drilling", "measures", 0.6),
            ("well_casing", "casing", "is_a", 0.6),
            ("mud_engineering", "drilling_fluid", "designs", 0.7),
            ("drilling_fluid", "formation_pressure", "balances", 0.7),
            ("well_trajectory", "directional_well", "plans", 0.6),
            ("rate_of_penetration", "cost", "affects", 0.5),
        ),
    ),
    CourseKnowledge(
        key="production_engineering",
        label="Production Engineering",
        category="Petroleum Disciplines",
        concepts=(
            ("well_flow", ("well flow",)),
            ("inflow_performance", ("inflow performance", "IPR")),
            ("bottom_hole_pressure", ("bottom-hole pressure",)),
            ("wellhead_pressure", ("wellhead pressure",)),
            ("flowline", ("flowline",)),
            ("flow_control", ("flow control",)),
            ("production_treatment", ("production treatment",)),
        ),
        relations=(
            ("well_flow", "flow", "is_a", 0.7),
            ("inflow_performance", "well_flow", "predicts", 0.7),
            ("bottom_hole_pressure", "well_flow", "drives", 0.7),
            ("wellhead_pressure", "bottom_hole_pressure", "depends_on", 0.6),
            ("flowline", "flow", "carries", 0.6),
            ("flow_control", "production_rate", "regulates", 0.6),
            ("production_treatment", "gas", "separates", 0.6),
            ("production_treatment", "water", "separates", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_logging",
        label="Well Logging",
        category="Petroleum Disciplines",
        concepts=(
            ("gamma_ray_log", ("gamma ray log",)),
            ("resistivity_log", ("resistivity log",)),
            ("density_log", ("density log",)),
            ("sonic_log", ("sonic log",)),
            ("porosity_log", ("porosity log",)),
            ("log_interpretation", ("log interpretation",)),
            ("log_calibration", ("log calibration",)),
        ),
        relations=(
            ("gamma_ray_log", "well_log", "is_a", 0.7),
            ("resistivity_log", "well_log", "is_a", 0.7),
            ("density_log", "well_log", "is_a", 0.7),
            ("sonic_log", "well_log", "is_a", 0.7),
            ("porosity_log", "porosity", "measures", 0.7),
            ("resistivity_log", "water_saturation", "estimates", 0.7),
            ("log_interpretation", "reservoir_rock", "identifies", 0.6),
            ("log_calibration", "log_interpretation", "improves", 0.6),
            ("density_log", "porosity", "measures", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_completions",
        label="Well Completions",
        category="Petroleum Disciplines",
        concepts=(
            ("cased_hole_completion", ("cased-hole completion",)),
            ("open_hole_completion", ("open-hole completion",)),
            ("perforation", ("perforation",)),
            ("screen", ("screen",)),
            ("gravel_pack", ("gravel pack",)),
            ("production_valve", ("production valve",)),
            ("well_equipment", ("well equipment",)),
        ),
        relations=(
            ("cased_hole_completion", "completion", "is_a", 0.7),
            ("open_hole_completion", "completion", "is_a", 0.7),
            ("perforation", "perforating", "is_a", 0.7),
            ("perforation", "cased_hole_completion", "enables", 0.6),
            ("screen", "gravel", "controls", 0.7),
            ("gravel_pack", "gravel", "controls", 0.6),
            ("production_valve", "well_flow", "regulates", 0.6),
            ("well_equipment", "completion", "supports", 0.6),
        ),
    ),
    CourseKnowledge(
        key="cementing",
        label="Well Cementing",
        category="Petroleum Disciplines",
        concepts=(
            ("cement_slurry", ("cement slurry",)),
            ("setting_time", ("setting time",)),
            ("cement_additives", ("cement additives",)),
            ("cement_plug", ("cement plug",)),
            ("bond_quality", ("bond quality",)),
            ("lightweight_cement", ("lightweight cement",)),
            ("retarder", ("retarder",)),
        ),
        relations=(
            ("cement_slurry", "cement", "is_a", 0.7),
            ("cement_slurry", "setting_time", "has", 0.6),
            ("cement_additives", "setting_time", "control", 0.6),
            ("cement_plug", "cement", "is_a", 0.6),
            ("bond_quality", "cement", "verifies", 0.6),
            ("lightweight_cement", "light_cement", "is_a", 0.6),
            ("retarder", "cement_additives", "is_a", 0.6),
            ("cement_slurry", "casing", "fills", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_stimulation",
        label="Well Stimulation",
        category="Petroleum Disciplines",
        concepts=(
            ("matrix_acidizing", ("matrix acidizing",)),
            ("massive_hydraulic_fracturing", ("hydraulic fracturing",)),
            ("fracturing_fluid", ("fracturing fluid",)),
            ("proppant", ("proppant",)),
            ("breakdown_pressure", ("breakdown pressure",)),
            ("formation_damage", ("formation damage",)),
        ),
        relations=(
            ("matrix_acidizing", "acidizing", "is_a", 0.7),
            ("massive_hydraulic_fracturing", "hydraulic_fracturing", "is_a", 0.7),
            ("fracturing_fluid", "hydraulic_fracturing", "used_in", 0.6),
            ("fracturing_fluid", "fracture_pressure", "requires", 0.6),
            ("proppant", "fracturing_fluid", "carried_by", 0.6),
            ("breakdown_pressure", "hydraulic_fracturing", "initiates", 0.6),
            ("matrix_acidizing", "formation_damage", "removes", 0.6),
        ),
    ),
    CourseKnowledge(
        key="artificial_lift",
        label="Artificial Lift",
        category="Petroleum Disciplines",
        concepts=(
            ("continuous_gas_lift", ("gas lift",)),
            ("electrical_submersible_pump", ("electrical submersible pump", "ESP")),
            ("sucker_rod_pump", ("sucker rod pump",)),
            ("plunger_lift", ("plunger lift",)),
            ("operating_pressure", ("operating pressure",)),
            ("pump_efficiency", ("pump efficiency",)),
            ("lift_design", ("lift design",)),
        ),
        relations=(
            ("continuous_gas_lift", "gas_lift", "is_a", 0.7),
            ("electrical_submersible_pump", "esp", "is_a", 0.7),
            ("sucker_rod_pump", "pump", "is_a", 0.7),
            ("plunger_lift", "Artificial Lift", "is_a", 0.6),
            ("operating_pressure", "Artificial Lift", "drives", 0.6),
            ("pump_efficiency", "pump", "measures", 0.6),
            ("lift_design", "Artificial Lift", "optimizes", 0.6),
            ("Artificial Lift", "production_rate", "increases", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_testing",
        label="Well Testing",
        category="Petroleum Disciplines",
        concepts=(
            ("buildup_test", ("buildup test",)),
            ("drawdown_test", ("drawdown test",)),
            ("pressure_derivative", ("pressure derivative",)),
            ("reservoir_model", ("reservoir model",)),
            ("skin_factor", ("skin factor",)),
            ("reservoir_transmissibility", ("reservoir transmissibility",)),
            ("reservoir_boundary", ("reservoir boundary",)),
        ),
        relations=(
            ("buildup_test", "pressure_transient_test", "is_a", 0.7),
            ("drawdown_test", "pressure_transient_test", "is_a", 0.7),
            ("pressure_derivative", "buildup_test", "analyzed_from", 0.6),
            ("reservoir_model", "pressure_transient_test", "interpreted_with", 0.6),
            ("skin_factor", "reservoir_transmissibility", "affects", 0.6),
            ("reservoir_boundary", "reservoir_model", "detected_by", 0.6),
            ("reservoir_transmissibility", "permeability", "depends_on", 0.6),
        ),
    ),
    CourseKnowledge(
        key="eor",
        label="Enhanced Oil Recovery",
        category="Petroleum Disciplines",
        concepts=(
            ("chemical_flooding", ("chemical flooding",)),
            ("polymer_flooding", ("polymer flooding",)),
            ("alkaline_flooding", ("alkaline flooding",)),
            ("miscible_gas_injection", ("miscible gas injection",)),
            ("thermal_recovery", ("thermal recovery",)),
            ("water_flooding", ("water flooding",)),
            ("sweep_efficiency", ("sweep efficiency",)),
            ("interfacial_tension", ("interfacial tension",)),
        ),
        relations=(
            ("chemical_flooding", "Enhanced Oil Recovery", "is_a", 0.7),
            ("polymer_flooding", "chemical_flooding", "is_a", 0.6),
            ("alkaline_flooding", "chemical_flooding", "is_a", 0.6),
            ("miscible_gas_injection", "Enhanced Oil Recovery", "is_a", 0.6),
            ("thermal_recovery", "Enhanced Oil Recovery", "is_a", 0.6),
            ("water_flooding", "waterflood", "is_a", 0.6),
            ("sweep_efficiency", "interfacial_tension", "depends_on", 0.6),
            ("polymer_flooding", "sweep_efficiency", "improves", 0.6),
            ("thermal_recovery", "steam_flooding", "uses", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_control",
        label="Well Control",
        category="Petroleum Disciplines",
        concepts=(
            ("blowout_preventer", ("blowout preventer", "BOP")),
            ("pressure_kick", ("pressure kick",)),
            ("weighted_kill_mud", ("weighted kill mud",)),
            ("wait_and_weight", ("wait and weight",)),
            ("shut_in_pressure", ("shut-in pressure",)),
            ("well_barrier", ("well barrier",)),
            ("pressure_test", ("pressure test",)),
        ),
        relations=(
            ("blowout_preventer", "gas_kick", "controls", 0.7),
            ("pressure_kick", "gas_kick", "is_a", 0.6),
            ("weighted_kill_mud", "kill_mud", "is_a", 0.7),
            ("wait_and_weight", "weighted_kill_mud", "uses", 0.6),
            ("shut_in_pressure", "shut_in", "measures", 0.6),
            ("well_barrier", "pressure_barrier", "is_a", 0.6),
            ("pressure_test", "well_barrier", "verifies", 0.6),
            ("blowout_preventer", "safety_valve", "complements", 0.5),
        ),
    ),
    CourseKnowledge(
        key="facilities",
        label="Facilities, Transport and Storage",
        category="Petroleum Disciplines",
        concepts=(
            ("wellhead_equipment", ("wellhead equipment",)),
            ("separator", ("separator",)),
            ("gas_processing", ("gas processing",)),
            ("pipeline", ("pipeline",)),
            ("pump_station", ("pump station",)),
            ("storage_tank", ("storage tank",)),
            ("production_metering", ("production metering",)),
            ("pipeline_integrity", ("pipeline integrity",)),
        ),
        relations=(
            ("wellhead_equipment", "wellhead_pressure", "handles", 0.6),
            ("separator", "oil", "separates", 0.6),
            ("separator", "gas", "separates", 0.6),
            ("separator", "water", "separates", 0.6),
            ("gas_processing", "gas", "treats", 0.6),
            ("pipeline", "oil", "transports", 0.7),
            ("pump_station", "pipeline", "drives", 0.6),
            ("storage_tank", "oil", "stores", 0.6),
            ("production_metering", "production_rate", "measures", 0.6),
            ("pipeline_integrity", "inspection", "requires", 0.6),
        ),
    ),
    CourseKnowledge(
        key="offshore",
        label="Offshore Drilling",
        category="Petroleum Disciplines",
        concepts=(
            ("offshore_platform", ("offshore platform",)),
            ("jack_up_rig", ("jack-up rig",)),
            ("drillship", ("drillship",)),
            ("subsea_wellhead", ("subsea wellhead",)),
            ("subsea_christmas_tree", ("subsea christmas tree",)),
            ("subsea_flowline", ("subsea flowline",)),
            ("subsea_bop", ("subsea BOP",)),
        ),
        relations=(
            ("offshore_platform", "drilling_rig", "is_a", 0.6),
            ("jack_up_rig", "offshore_platform", "is_a", 0.6),
            ("drillship", "offshore_platform", "is_a", 0.6),
            ("subsea_wellhead", "wellhead_equipment", "is_a", 0.6),
            ("subsea_christmas_tree", "wellhead_equipment", "is_a", 0.6),
            ("subsea_flowline", "pipeline", "is_a", 0.6),
            ("subsea_bop", "blowout_preventer", "is_a", 0.6),
            ("offshore_platform", "water", "stands_in", 0.5),
        ),
    ),
    CourseKnowledge(
        key="petroleum_economics",
        label="Petroleum Economics",
        category="Petroleum Disciplines",
        concepts=(
            ("project_net_present_value", ("net present value", "NPV")),
            ("discount_rate", ("discount rate",)),
            ("internal_rate_of_return", ("internal rate of return", "IRR")),
            ("payback_period", ("payback period",)),
            ("cash_flow_analysis", ("cash flow analysis",)),
            ("oil_price", ("oil price",)),
            ("royalty", ("royalty", "production tax")),
            ("sensitivity_analysis", ("sensitivity analysis",)),
        ),
        relations=(
            ("project_net_present_value", "net_present_value", "is_a", 0.7),
            ("discount_rate", "project_net_present_value", "discounts", 0.7),
            ("internal_rate_of_return", "cost", "compares_to", 0.6),
            ("payback_period", "cost", "recovers", 0.6),
            ("cash_flow_analysis", "revenue", "projects", 0.6),
            ("oil_price", "revenue", "determines", 0.7),
            ("royalty", "cost", "adds_to", 0.6),
            ("sensitivity_analysis", "risk", "quantifies", 0.6),
            ("project_net_present_value", "decision", "informs", 0.6),
        ),
    ),
    CourseKnowledge(
        key="project_management",
        label="Project Management",
        category="Petroleum Disciplines",
        concepts=(
            ("project_life_cycle", ("project life cycle",)),
            ("project_scope", ("project scope",)),
            ("schedule", ("schedule",)),
            ("budget", ("budget",)),
            ("risk_management", ("risk management",)),
            ("contracting", ("contracting",)),
            ("quality_assurance", ("quality assurance",)),
            ("stakeholders", ("stakeholders",)),
        ),
        relations=(
            ("project_life_cycle", "project_scope", "defines", 0.6),
            ("schedule", "budget", "balanced_with", 0.6),
            ("risk_management", "risk", "manages", 0.7),
            ("contracting", "cost", "sets", 0.6),
            ("quality_assurance", "decision", "supports", 0.5),
            ("stakeholders", "decision", "influence", 0.6),
            ("budget", "cost", "allocates", 0.6),
        ),
    ),
    CourseKnowledge(
        key="hse",
        label="Health, Safety and Environment",
        category="Petroleum Disciplines",
        concepts=(
            ("risk_assessment", ("risk assessment",)),
            ("personal_protective_equipment", ("personal protective equipment", "PPE")),
            ("industrial_accidents", ("industrial accidents",)),
            ("emissions", ("emissions",)),
            ("waste_management", ("waste management",)),
            ("emergency_response", ("emergency response",)),
            ("groundwater_protection", ("groundwater protection",)),
            ("occupational_health", ("occupational health",)),
        ),
        relations=(
            ("risk_assessment", "risk", "assesses", 0.7),
            ("personal_protective_equipment", "industrial_accidents", "reduces", 0.6),
            ("industrial_accidents", "well_integrity", "threaten", 0.6),
            ("emissions", "environment", "pollutes", 0.7),
            ("waste_management", "contamination", "prevents", 0.6),
            ("emergency_response", "gas_kick", "mitigates", 0.6),
            ("groundwater_protection", "water", "protects", 0.6),
            ("occupational_health", "industrial_accidents", "prevents", 0.6),
        ),
    ),
    CourseKnowledge(
        key="professional_ethics",
        label="Ethics and Professionalism",
        category="Petroleum Disciplines",
        concepts=(
            ("professional_responsibility", ("professional responsibility",)),
            ("scientific_integrity", ("scientific integrity",)),
            ("safety_first", ("safety first",)),
            ("transparency", ("transparency",)),
            ("conflict_of_interest", ("conflict of interest",)),
            ("data_ethics", ("data ethics",)),
            ("social_responsibility", ("social responsibility",)),
        ),
        relations=(
            ("professional_responsibility", "decision", "guides", 0.7),
            ("scientific_integrity", "data", "protects", 0.6),
            ("safety_first", "well_integrity", "prioritizes", 0.7),
            ("transparency", "data", "requires", 0.6),
            ("conflict_of_interest", "decision", "threatens", 0.6),
            ("data_ethics", "data", "governs", 0.6),
            ("social_responsibility", "environment", "protects", 0.6),
        ),
    ),
    # ---------------------------------------------------- advanced: Earth Sciences
    CourseKnowledge(
        key="basin_analysis",
        label="Basin Analysis and Petroleum Systems",
        category="Earth Sciences",
        concepts=(
            ("burial_history", ("burial history", "subsidence history")),
            ("geothermal_gradient", ("geothermal gradient",)),
            ("hydrocarbon_charge", ("hydrocarbon charge", "charge")),
            ("play_fairway", ("play fairway",)),
            ("prospect", ("prospect", "lead")),
        ),
        relations=(
            ("burial_history", "thermal_maturation", "controls", 0.7),
            ("geothermal_gradient", "thermal_maturation", "drives", 0.7),
            ("hydrocarbon_charge", "trap", "fills", 0.7),
            ("play_fairway", "prospect", "contains", 0.6),
            ("sedimentary_basin", "play_fairway", "hosts", 0.6),
        ),
    ),
    CourseKnowledge(
        key="seismic_interpretation",
        label="Seismic Interpretation",
        category="Earth Sciences",
        concepts=(
            ("seismic_attribute", ("seismic attribute",)),
            ("horizon", ("horizon",)),
            ("fault_interpretation", ("fault interpretation",)),
            ("amplitude_anomaly", ("amplitude anomaly",)),
            ("inversion_model", ("seismic inversion", "inversion")),
        ),
        relations=(
            ("seismic_section", "horizon", "reveals", 0.6),
            ("fault_interpretation", "trap", "defines", 0.6),
            ("amplitude_anomaly", "hydrocarbon", "indicates", 0.5),
            ("inversion_model", "seismic_attribute", "uses", 0.6),
            ("seismic_attribute", "seismic_section", "derived_from", 0.6),
        ),
    ),
    # ----------------------------------------------- advanced: Engineering Science
    CourseKnowledge(
        key="petroleum_geomechanics",
        label="Petroleum Geomechanics",
        category="Engineering Science",
        concepts=(
            ("in_situ_stress", ("in situ stress", "stress state")),
            ("fracture_gradient", ("fracture gradient",)),
            ("wellbore_stability", ("wellbore stability",)),
            ("sand_production", ("sand production",)),
            ("stress_orientation", ("stress orientation",)),
        ),
        relations=(
            ("in_situ_stress", "wellbore_stability", "governs", 0.7),
            ("fracture_gradient", "hydraulic_fracturing", "limits", 0.6),
            ("sand_production", "wellbore_stability", "threatens", 0.6),
            ("stress_orientation", "in_situ_stress", "describes", 0.6),
            ("wellbore_stability", "drilling", "constrains", 0.6),
        ),
    ),
    CourseKnowledge(
        key="multiphase_flow",
        label="Multiphase Flow",
        category="Engineering Science",
        concepts=(
            ("flow_regime", ("flow regime",)),
            ("slip_velocity", ("slip velocity",)),
            ("pressure_gradient", ("pressure gradient",)),
            ("holdup", ("liquid holdup", "holdup")),
            ("flow_pattern_map", ("flow pattern map",)),
        ),
        relations=(
            ("flow_regime", "pressure_gradient", "determines", 0.6),
            ("holdup", "slip_velocity", "caused_by", 0.6),
            ("flow_pattern_map", "flow_regime", "predicts", 0.6),
            ("multiphase_flow", "pressure_gradient", "has", 0.6),
            ("gas", "liquid", "flow_together_as", 0.5),
        ),
    ),
    # -------------------------------------------- advanced: Petroleum Disciplines
    CourseKnowledge(
        key="reservoir_simulation_advanced",
        label="Reservoir Simulation and Modelling",
        category="Petroleum Disciplines",
        concepts=(
            ("history_matching", ("history matching",)),
            ("grid_discretization", ("grid discretization", "gridding")),
            ("upscaling", ("upscaling",)),
            ("equation_of_state", ("equation of state",)),
            ("compositional_model", ("compositional model",)),
            ("simulator", ("reservoir simulator", "simulator")),
        ),
        relations=(
            ("compositional_model", "equation_of_state", "uses", 0.7),
            ("history_matching", "simulator", "calibrates", 0.7),
            ("upscaling", "grid_discretization", "precedes", 0.6),
            ("simulator", "reservoir_simulation", "is_a", 0.7),
        ),
    ),
    CourseKnowledge(
        key="advanced_eor",
        label="Advanced Enhanced Oil Recovery",
        category="Petroleum Disciplines",
        concepts=(
            ("polymer_flooding", ("polymer flooding",)),
            ("surfactant_flooding", ("surfactant flooding",)),
            ("miscible_gas_injection", ("miscible gas injection",)),
            ("thermal_recovery", ("thermal recovery",)),
            ("mobility_ratio", ("mobility ratio",)),
            ("residual_oil_saturation", ("residual oil saturation",)),
            ("conformance_control", ("conformance control",)),
        ),
        relations=(
            ("polymer_flooding", "mobility_ratio", "improves", 0.7),
            ("surfactant_flooding", "residual_oil_saturation", "reduces", 0.7),
            ("miscible_gas_injection", "oil", "mixes_with", 0.6),
            ("thermal_recovery", "heavy_oil", "recovers", 0.6),
            ("conformance_control", "mobility_ratio", "manages", 0.6),
        ),
    ),
    CourseKnowledge(
        key="unconventional_resources",
        label="Unconventional Resources Engineering",
        category="Petroleum Disciplines",
        concepts=(
            ("hydraulic_fracturing", ("hydraulic fracturing", "fracking")),
            ("proppant", ("proppant",)),
            ("fracture_conductivity", ("fracture conductivity",)),
            ("stimulated_rock_volume", ("stimulated rock volume",)),
            ("cluster_spacing", ("cluster spacing",)),
            ("refracturing", ("refracturing", "refrac")),
        ),
        relations=(
            ("hydraulic_fracturing", "stimulated_rock_volume", "creates", 0.7),
            ("proppant", "fracture_conductivity", "maintains", 0.7),
            ("cluster_spacing", "stimulated_rock_volume", "controls", 0.6),
            ("refracturing", "recovery_factor", "improves", 0.6),
            ("shale_play", "hydraulic_fracturing", "developed_by", 0.6),
        ),
    ),
    CourseKnowledge(
        key="subsea_engineering",
        label="Subsea and Deepwater Engineering",
        category="Petroleum Disciplines",
        concepts=(
            ("subsea_well", ("subsea well",)),
            ("subsea_tree", ("subsea tree", "christmas tree")),
            ("production_riser", ("production riser",)),
            ("umbilical", ("umbilical",)),
            ("floating_production", ("floating production", "FPSO")),
            ("subsea_manifold", ("subsea manifold",)),
        ),
        relations=(
            ("subsea_well", "subsea_tree", "controlled_by", 0.7),
            ("subsea_tree", "subsea_manifold", "connects_to", 0.6),
            ("production_riser", "floating_production", "links_to", 0.7),
            ("umbilical", "subsea_tree", "controls", 0.6),
            ("floating_production", "hydrocarbon", "processes", 0.6),
        ),
    ),
    CourseKnowledge(
        key="flow_assurance",
        label="Flow Assurance",
        category="Petroleum Disciplines",
        concepts=(
            ("hydrate", ("hydrate", "gas hydrate")),
            ("wax_deposition", ("wax deposition", "wax")),
            ("asphaltene", ("asphaltene",)),
            ("scale_deposition", ("scale deposition", "scale")),
            ("slugging", ("slugging", "slug flow")),
            ("thermal_insulation", ("thermal insulation",)),
        ),
        relations=(
            ("hydrate", "flow_restriction", "causes", 0.7),
            ("wax_deposition", "flow_restriction", "causes", 0.7),
            ("asphaltene", "flow_restriction", "causes", 0.6),
            ("scale_deposition", "flow_restriction", "causes", 0.6),
            ("thermal_insulation", "hydrate", "prevents", 0.6),
            ("slugging", "flow_restriction", "causes", 0.5),
        ),
    ),
    CourseKnowledge(
        key="process_safety_engineering",
        label="Process Safety Engineering",
        category="Petroleum Disciplines",
        concepts=(
            ("bow_tie", ("bow-tie", "bow tie model")),
            ("ignition_source", ("ignition source",)),
            ("safety_instrumented_system", ("safety instrumented system", "SIS")),
            ("safety_case", ("safety case",)),
            ("barrier_management", ("barrier management",)),
        ),
        relations=(
            ("safety_instrumented_system", "layers_of_protection", "is_a", 0.7),
            ("bow_tie", "barrier_management", "maps", 0.6),
            ("hazard_analysis", "ignition_source", "identifies", 0.6),
            ("process_safety_event", "safety_case", "governed_by", 0.6),
            ("barrier_management", "process_safety_event", "prevents", 0.7),
        ),
    ),
    CourseKnowledge(
        key="petroleum_data_analytics",
        label="Petroleum Data Analytics and Machine Learning",
        category="Petroleum Disciplines",
        concepts=(
            ("feature_engineering", ("feature engineering",)),
            ("machine_learning_model", ("machine learning model", "ML model")),
            ("training_data", ("training data",)),
            ("validation_split", ("validation split", "holdout")),
            ("predictive_maintenance", ("predictive maintenance",)),
            ("decline_curve_analysis", ("decline curve analysis",)),
        ),
        relations=(
            ("feature_engineering", "machine_learning_model", "feeds", 0.7),
            ("training_data", "machine_learning_model", "trains", 0.7),
            ("validation_split", "machine_learning_model", "evaluates", 0.6),
            ("decline_curve_analysis", "production", "forecasts", 0.6),
            ("predictive_maintenance", "pump", "protects", 0.6),
        ),
    ),
    CourseKnowledge(
        key="carbon_capture_storage",
        label="Carbon Capture, Utilisation and Storage",
        category="Petroleum Disciplines",
        concepts=(
            ("co2_capture", ("CO2 capture", "carbon capture")),
            ("co2_plume", ("CO2 plume",)),
            ("storage_complex", ("storage complex", "storage site")),
            ("monitoring_well", ("monitoring well",)),
            ("leakage_risk", ("leakage risk",)),
            ("caprock_integrity", ("caprock integrity",)),
        ),
        relations=(
            ("co2_capture", "storage_complex", "feeds", 0.7),
            ("co2_plume", "storage_complex", "migrates_within", 0.6),
            ("monitoring_well", "leakage_risk", "detects", 0.6),
            ("caprock_integrity", "leakage_risk", "prevents", 0.7),
            ("co2_plume", "enhanced_oil_recovery", "supports", 0.5),
        ),
    ),
    CourseKnowledge(
        key="well_decommissioning",
        label="Well Plugging and Decommissioning",
        category="Petroleum Disciplines",
        concepts=(
            ("plug_and_abandonment", ("plug and abandonment", "P&A")),
            ("abandonment_barrier", ("abandonment barrier",)),
            ("cement_plug", ("cement plug",)),
            ("platform_removal", ("platform removal",)),
            ("permanent_barrier", ("permanent barrier",)),
        ),
        relations=(
            ("plug_and_abandonment", "abandonment_barrier", "requires", 0.7),
            ("cement_plug", "abandonment_barrier", "forms", 0.7),
            ("abandonment_barrier", "well_integrity", "ensures", 0.7),
            ("platform_removal", "offshore", "concludes", 0.6),
            ("permanent_barrier", "abandonment_barrier", "is_a", 0.7),
        ),
    ),
    CourseKnowledge(
        key="integrated_field_management",
        label="Integrated Field Management",
        category="Petroleum Disciplines",
        concepts=(
            ("integrated_asset_model", ("integrated asset model",)),
            ("production_forecast", ("production forecast",)),
            ("opportunity_identification", ("opportunity identification",)),
            ("value_optimization", ("value optimization", "value optimisation")),
            ("decision_gate", ("decision gate", "stage gate")),
        ),
        relations=(
            ("integrated_asset_model", "production_forecast", "produces", 0.7),
            ("opportunity_identification", "value_optimization", "enables", 0.6),
            ("decision_gate", "value_optimization", "governs", 0.6),
            ("production_forecast", "decision", "informs", 0.6),
            ("value_optimization", "net_present_value", "maximizes", 0.6),
        ),
    ),
)


def courses() -> tuple[CourseKnowledge, ...]:
    """The courses of the petroleum-engineering degree the seed spans."""
    return _COURSES


def course_keys() -> tuple[str, ...]:
    return tuple(course.key for course in _COURSES)


def course_labels() -> dict[str, str]:
    return {course.key: course.label for course in _COURSES}


def course_categories() -> dict[str, str]:
    return {course.key: course.category for course in _COURSES}


def curriculum_summary() -> str:
    """One-line description of the curriculum frame."""
    labels = ", ".join(course.label for course in _COURSES)
    return f"Petroleum-engineering curriculum: {labels}."


def seed_curriculum(ontology: Ontology) -> list[Concept]:
    """Add the curriculum concepts and relations, reusing any that already exist.

    Idempotent, and shares nodes with the domain seed: a concept already planted
    by ``core.domain_knowledge`` (e.g. ``reservoir`` or ``porosity``) is reused rather
    than duplicated, so the curriculum and the frame form one graph.
    """
    concepts: dict[str, Concept] = {}

    def concept_for(label: str, aliases: tuple[str, ...]) -> Concept:
        if label not in concepts:
            existing = ontology.find_concepts(label)
            concepts[label] = existing[0] if existing else _add_concept(ontology, label, aliases)
        return concepts[label]

    for label, aliases in _SHARED_CONCEPTS:
        concept_for(label, aliases)
    for course in _COURSES:
        for label, aliases in course.concepts:
            concept_for(label, aliases)

    for course in _COURSES:
        for source, target, relation_type, confidence in course.relations:
            _add_relation(
                ontology,
                concept_for(source, ()),
                concept_for(target, ()),
                relation_type,
                confidence,
            )

    return list(concepts.values())


def curriculum_coverage(ontology: Ontology) -> dict[str, dict[str, int]]:
    """Per-course counts of seeded concepts and relations present.

    A relation counts when both endpoints resolve to concepts the ontology holds,
    so a partially-restored graph is reported honestly.
    """
    coverage: dict[str, dict[str, int]] = {}
    for course in _COURSES:
        concept_hits = sum(1 for label, _ in course.concepts if ontology.find_concepts(label))
        relation_hits = 0
        for source, target, relation_type, _ in course.relations:
            sources = ontology.find_concepts(source)
            targets = ontology.find_concepts(target)
            if not sources or not targets:
                continue
            source_ids = {concept.id for concept in sources}
            target_ids = {concept.id for concept in targets}
            if any(
                relation.source_id in source_ids
                and relation.target_id in target_ids
                and relation.relation_type == relation_type
                for relation in ontology.relations.values()
            ):
                relation_hits += 1
        coverage[course.key] = {
            "concepts": concept_hits,
            "relations": relation_hits,
            "expected_concepts": len(course.concepts),
            "expected_relations": len(course.relations),
        }
    return coverage


def curriculum_coverage_score(ontology: Ontology) -> float:
    """Fraction of the whole curriculum's concepts and relations present.

    A single number over every course, used as an evaluation metric so that
    losing any part of the seeded degree shows up as a regression.
    """
    expected = 0
    present = 0
    for counts in curriculum_coverage(ontology).values():
        expected += counts["expected_concepts"] + counts["expected_relations"]
        present += counts["concepts"] + counts["relations"]
    if expected == 0:
        return 0.0
    return present / expected


def _add_concept(ontology: Ontology, label: str, aliases: tuple[str, ...]) -> Concept:
    concept = Concept(label, frozenset(aliases))
    ontology.add_concept(concept)
    return concept


def _add_relation(
    ontology: Ontology,
    source: Concept,
    target: Concept,
    relation_type: str,
    confidence: float,
) -> None:
    if any(
        relation.source_id == source.id
        and relation.target_id == target.id
        and relation.relation_type == relation_type
        for relation in ontology.relations.values()
    ):
        return
    ontology.add_relation(
        SemanticRelation(source.id, target.id, relation_type, confidence)
    )


__all__ = [
    "CourseKnowledge",
    "course_categories",
    "course_keys",
    "course_labels",
    "courses",
    "curriculum_coverage",
    "curriculum_coverage_score",
    "curriculum_summary",
    "seed_curriculum",
]
