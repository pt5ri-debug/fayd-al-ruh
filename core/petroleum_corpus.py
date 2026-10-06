"""The petroleum-engineering corpus: the knowledge the project is seeded with.

Where :mod:`core.domain_knowledge` gives the frame (the handful of relations a
practitioner reasons over) and :mod:`core.curriculum_knowledge` gives the degree
map (courses and the concepts they teach), this module holds the *content*: the
definitions, relations, formulas, procedures, standards, and glossary terms that
make the project's petroleum identity concrete.

Every entry is inspectable and evidence-aware, exactly like the other seeds:

* its concepts and relations are planted into the ontology at a modest seeded
  confidence that later evidence can revise;
* its prose becomes a citable document, so a retrieved answer can quote it and
  point back to it;
* its sentences become part of the language-model training corpus, so the model
  is trained on the same knowledge the graph carries rather than a hand-written
  sample.

``seed_petroleum_corpus`` plants it idempotently and shares nodes with the frame
and the curriculum, so all three layers form one graph. ``corpus_coverage``
reports, per discipline, how much of the corpus an ontology currently holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.concept import Concept
from core.ontology import Ontology
from core.semantic_relation import SemanticRelation

if TYPE_CHECKING:
    from core.documents import Document

_CONFIDENCE = 0.6


@dataclass(frozen=True)
class CorpusEntry:
    """One unit of petroleum-engineering knowledge, organized by discipline."""

    key: str
    discipline: str
    title: str
    definition: str
    concepts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    facts: tuple[tuple[str, str, str, float], ...] = ()
    formulas: tuple[tuple[str, str], ...] = ()
    procedures: tuple[str, ...] = ()
    standards: tuple[str, ...] = ()
    glossary: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.discipline.strip():
            raise ValueError("Corpus entry key and discipline cannot be empty.")
        if not self.title.strip() or not self.definition.strip():
            raise ValueError(f"Corpus entry '{self.key}' needs a title and a definition.")


_ENTRIES: tuple[CorpusEntry, ...] = (
    # ------------------------------------------------------------------ reservoir
    CorpusEntry(
        key="reservoir_rock_properties",
        discipline="reservoir",
        title="Reservoir rock properties",
        definition=(
            "A reservoir rock stores hydrocarbons in its pore space and lets them "
            "flow to a well. Porosity is the fraction of the bulk volume that is "
            "void; effective porosity counts only the connected pores. Permeability "
            "measures how easily a fluid moves through those pores, and it depends "
            "more on pore connectivity and grain size than on porosity alone. A "
            "rock can be very porous yet nearly impermeable when its pores are "
            "isolated, which is why both properties are always reported together."
        ),
        concepts=(
            ("effective_porosity", ("effective porosity", "connected porosity")),
            ("absolute_permeability", ("absolute permeability",)),
            ("grain_size", ("grain size", "sorting")),
            ("pore_throat", ("pore throat", "pore connectivity")),
        ),
        facts=(
            ("effective_porosity", "permeability", "influences", 0.6),
            ("grain_size", "permeability", "influences", 0.6),
            ("pore_throat", "permeability", "controls", 0.7),
            ("reservoir_rock", "effective_porosity", "has_property", 0.7),
        ),
        formulas=(
            ("Porosity", "phi = Vpore / Vbulk"),
            ("Darcy's law (linear)", "q = -k A / mu * dp/dL"),
        ),
        procedures=(
            "Measure porosity and permeability on a core sample.",
            "Compare core values with well-log estimates to calibrate the model.",
        ),
        standards=("Core analysis follows API RP 40.",),
        glossary=(("porosity", "مسامية"), ("permeability", "نفاذية")),
    ),
    CorpusEntry(
        key="reservoir_drive_mechanisms",
        discipline="reservoir",
        title="Reservoir drive mechanisms",
        definition=(
            "A drive mechanism is the source of energy that pushes hydrocarbons "
            "toward the wellbore. Solution-gas drive expands dissolved gas as "
            "pressure falls; gas-cap drive expands a free gas cap; water drive "
            "supports pressure as an aquifer advances; gravity drainage lets oil "
            "drain downward. Recovery factor is highest for strong water and "
            "gas-cap drives and lowest for solution-gas drive, so identifying the "
            "drive early guides how the field is developed."
        ),
        concepts=(
            ("solution_gas_drive", ("solution gas drive", "dissolved gas drive")),
            ("gas_cap_drive", ("gas cap drive",)),
            ("water_drive", ("water drive", "aquifer drive")),
            ("gravity_drainage", ("gravity drainage",)),
        ),
        facts=(
            ("solution_gas_drive", "recovery_factor", "limits", 0.6),
            ("water_drive", "reservoir_pressure", "maintains", 0.7),
            ("gas_cap_drive", "reservoir_pressure", "maintains", 0.7),
            ("water_drive", "recovery_factor", "increases", 0.6),
        ),
        formulas=(("Recovery factor", "RF = produced volume / original volume in place"),),
        procedures=(
            "Identify the drive from pressure and production trends.",
            "Estimate the recovery factor for that drive.",
        ),
        standards=("Reserve definitions follow the SPE-PRMS system.",),
    ),
    CorpusEntry(
        key="volumetric_reserves",
        discipline="reservoir",
        title="Volumetric reserves",
        definition=(
            "Original oil in place is the volume of oil a reservoir holds before "
            "production. It is computed volumetrically from the bulk rock volume, "
            "the average porosity, the oil saturation, and the formation volume "
            "factor. Reserves are that volume multiplied by the recovery factor, so "
            "a small error in porosity or saturation changes the estimate a great "
            "deal; reserves are therefore reported as a range, not one number."
        ),
        concepts=(
            ("original_oil_in_place", ("original oil in place", "ooip")),
            ("bulk_volume", ("bulk volume",)),
            ("formation_volume_factor", ("formation volume factor", "bo")),
        ),
        facts=(
            ("original_oil_in_place", "recovery_factor", "multiplied_by", 0.7),
            ("bulk_volume", "original_oil_in_place", "influences", 0.6),
            ("saturation", "original_oil_in_place", "influences", 0.6),
        ),
        formulas=(
            ("Original oil in place", "OOIP = 7758 * A * h * phi * (1 - Sw) / Bo"),
            ("Reserves", "N = OOIP * RF"),
        ),
        procedures=(
            "Map the structure and average the reservoir properties.",
            "Compute original oil in place, then apply the recovery factor.",
        ),
        standards=("Report reserves with SPE-PRMS categories (1P, 2P, 3P).",),
        glossary=(("reserve", "احتياطي"), ("oil in place", "النفط الموجود في المكان")),
    ),
    CorpusEntry(
        key="material_balance",
        discipline="reservoir",
        title="Material balance and reservoir simulation",
        definition=(
            "The material-balance equation states that the original hydrocarbons "
            "equal what remains plus what has been produced. It is used to estimate "
            "original oil in place and to detect water influx without a full "
            "simulation. Reservoir simulation extends the same conservation idea to "
            "a grid of cells over time, solving flow equations numerically to "
            "forecast production under different development plans."
        ),
        concepts=(
            ("material_balance", ("material balance",)),
            ("water_influx", ("water influx",)),
            ("history_matching", ("history match", "history matching")),
        ),
        facts=(
            ("material_balance", "original_oil_in_place", "estimates", 0.6),
            ("water_influx", "material_balance", "detected_by", 0.6),
            ("reservoir_simulation", "history_matching", "calibrated_by", 0.7),
        ),
        formulas=(("Material balance (simplified)", "Np * Bt = N * (Bt - Bti) + We"),),
        procedures=(
            "Assemble pressure, production, and PVT data.",
            "Match history, then forecast the development plan.",
        ),
    ),
    # ------------------------------------------------------------------ petrophysics
    CorpusEntry(
        key="formation_evaluation",
        discipline="petrophysics",
        title="Formation evaluation",
        definition=(
            "Formation evaluation turns well-log and core measurements into the "
            "quantities a reservoir engineer needs: porosity, water saturation, "
            "lithology, and net pay. Porosity comes from density, neutron, and "
            "sonic logs; water saturation from resistivity logs using Archie's "
            "equation; lithology from a combination of gamma-ray, density, and "
            "neutron responses. Core data calibrate the log-derived values, so the "
            "two are always interpreted together."
        ),
        concepts=(
            ("water_saturation", ("water saturation", "sw")),
            ("lithology", ("lithology", "rock type")),
            ("archie_equation", ("archie equation", "archie")),
            ("net_to_gross", ("net to gross", "ntg")),
        ),
        facts=(
            ("electrical_log", "water_saturation", "measures", 0.7),
            ("density_log", "porosity", "measures", 0.7),
            ("core_sample", "lithology", "verifies", 0.6),
            ("net_to_gross", "net_pay", "determines", 0.6),
        ),
        formulas=(
            ("Archie water saturation", "Sw = (a * Rw / (phi^m * Rt))^(1/n)"),
            ("Net pay", "h_net = h_gross * NTG"),
        ),
        procedures=(
            "Pick formation tops from logs.",
            "Compute porosity, then water saturation, then net pay.",
        ),
        standards=("Log calibration follows API and SPWLA recommended practices.",),
        glossary=(("water saturation", "التشبع المائي"), ("net pay", "السمك المنتج")),
    ),
    CorpusEntry(
        key="saturation_height",
        discipline="petrophysics",
        title="Capillary pressure and saturation-height",
        definition=(
            "Capillary pressure is the pressure difference across a curved "
            "fluid interface in a pore. It controls how fluids distribute in a "
            "reservoir: water wets the small pores, and hydrocarbons enter the "
            "larger pores above the free-water level. A saturation-height function "
            "built from capillary-pressure data predicts water saturation at any "
            "height above the contact, which is used to initialize the reservoir "
            "model."
        ),
        concepts=(
            ("capillary_pressure", ("capillary pressure", "pc")),
            ("free_water_level", ("free water level", "fwl")),
            ("wettability", ("wettability", "wetting")),
            ("transition_zone", ("transition zone",)),
        ),
        facts=(
            ("capillary_pressure", "water_saturation", "controls", 0.7),
            ("free_water_level", "transition_zone", "defines", 0.6),
            ("wettability", "capillary_pressure", "influences", 0.7),
        ),
        formulas=(("Capillary pressure", "Pc = 2 * sigma * cos(theta) / r"),),
        procedures=(
            "Measure capillary pressure on representative core plugs.",
            "Fit a saturation-height function and initialize the model.",
        ),
    ),
    # ------------------------------------------------------------------ well_logging
    CorpusEntry(
        key="logging_tools",
        discipline="well_logging",
        title="Well-logging tools",
        definition=(
            "A well log is a continuous record of a rock property along the "
            "wellbore. The gamma-ray log measures natural radioactivity and "
            "separates shale from clean reservoir rock. The density log measures "
            "bulk density and gives porosity; the neutron log responds to hydrogen "
            "and gives porosity in clean formations; the sonic log measures "
            "acoustic travel time. Resistivity logs respond to conductive formation "
            "water and are the basis for water-saturation calculation."
        ),
        concepts=(
            ("gamma_ray_log", ("gamma ray log", "gr log", "gamma ray")),
            ("density_log", ("density log", "rhob")),
            ("sonic_log", ("sonic log", "acoustic log")),
            ("neutron_log", ("neutron log",)),
        ),
        facts=(
            ("gamma_ray_log", "lithology", "indicates", 0.7),
            ("density_log", "porosity", "measures", 0.7),
            ("sonic_log", "porosity", "measures", 0.6),
            ("resistivity_log", "water_saturation", "indicates", 0.7),
        ),
        formulas=(("Density porosity", "phi_D = (rho_ma - rho_b) / (rho_ma - rho_f)"),),
        procedures=(
            "Run a logging suite after drilling the interval.",
            "Correct the logs for borehole and environmental effects.",
            "Interpret porosity, saturation, and lithology.",
        ),
        glossary=(("well log", "سجل البئر"), ("gamma ray", "أشعة جاما")),
    ),
    CorpusEntry(
        key="log_interpretation_workflow",
        discipline="well_logging",
        title="Log interpretation workflow",
        definition=(
            "Interpretation proceeds in a fixed order because each step feeds the "
            "next. First, quality-control the logs and correct them. Second, pick "
            "lithology from the gamma-ray, density, and neutron responses. Third, "
            "compute porosity from the density and neutron logs. Fourth, compute "
            "water saturation from resistivity and Archie's equation. Fifth, apply "
            "cut-offs to find net pay. Skipping a step, or using an uncorrected "
            "log, propagates error into the reserves."
        ),
        concepts=(
            ("log_qc", ("log quality control", "log qc")),
            ("cutoff", ("cut-off", "cutoff")),
            ("shale_volume", ("shale volume", "vsh")),
        ),
        facts=(
            ("log_qc", "log_interpretation_workflow", "precedes", 0.6),
            ("cutoff", "net_pay", "defines", 0.7),
            ("shale_volume", "effective_porosity", "reduces", 0.6),
        ),
        procedures=(
            "QC and correct the logs.",
            "Pick lithology, then porosity, then saturation.",
            "Apply cut-offs to define net pay.",
        ),
    ),
    # ------------------------------------------------------------------ drilling
    CorpusEntry(
        key="drilling_fluids",
        discipline="drilling",
        title="Drilling fluids",
        definition=(
            "Drilling fluid, or mud, is pumped down the drill string and back up "
            "the annulus. It cools and lubricates the bit, carries cuttings to the "
            "surface, and — most important for safety — its hydrostatic pressure "
            "holds back formation fluids. Mud weight is chosen to stay above pore "
            "pressure and below fracture pressure; that window is the basis of well "
            "control. Mud is described by its density, viscosity, and filtrate "
            "properties."
        ),
        concepts=(
            ("mud_weight", ("mud weight", "mud density")),
            ("hydrostatic_pressure", ("hydrostatic pressure",)),
            ("pore_pressure", ("pore pressure", "formation pressure")),
            ("fracture_pressure", ("fracture pressure", "fracture gradient")),
        ),
        facts=(
            ("drilling_fluid", "hydrostatic_pressure", "provides", 0.7),
            ("mud_weight", "hydrostatic_pressure", "controls", 0.7),
            ("hydrostatic_pressure", "pore_pressure", "overbalances", 0.7),
            ("drilling_fluid", "drill_bit", "lubricates", 0.6),
        ),
        formulas=(
            ("Hydrostatic pressure", "Ph = 0.052 * MW * TVD"),
            ("Mud window", "Ppore < Ph < Pfracture"),
        ),
        procedures=(
            "Design mud weight from pore and fracture pressures.",
            "Monitor returns and pit level for a kick.",
        ),
        standards=("Mud programs follow API RP 13B testing practice.",),
        glossary=(("drilling fluid", "سائل الحفر"), ("mud weight", "وزن الطين")),
    ),
    CorpusEntry(
        key="casing_and_cementing_design",
        discipline="drilling",
        title="Casing design",
        definition=(
            "Casing is steel pipe set in the well to keep the hole open and to "
            "isolate formations. A well is drilled in sections: conductor, surface, "
            "intermediate, and production casing, each set before the next, smaller "
            "hole is drilled. Each string is designed to resist burst, collapse, "
            "and tensile loads with a safety factor. Cement is pumped behind the "
            "casing to support it and to seal off permeable zones."
        ),
        concepts=(
            ("surface_casing", ("surface casing",)),
            ("intermediate_casing", ("intermediate casing",)),
            ("production_casing", ("production casing",)),
            ("burst_load", ("burst load", "collapse load")),
        ),
        facts=(
            ("casing", "well", "supports", 0.8),
            ("cement", "casing", "supports", 0.7),
            ("production_casing", "reservoir", "isolates", 0.6),
        ),
        formulas=(("Burst design", "P_burst = 0.052 * MW_grad * TVD * SF"),),
        procedures=(
            "Set each casing string before drilling the next section.",
            "Cement the casing and verify the bond with a cement bond log.",
        ),
        standards=("Casing design follows API 5CT and API RP 65.",),
    ),
    CorpusEntry(
        key="directional_drilling",
        discipline="drilling",
        title="Directional and horizontal drilling",
        definition=(
            "Directional drilling steers the bit along a planned path rather than "
            "straight down. It is used to reach a reservoir under an obstacle, to "
            "drill several wells from one pad, and to place a horizontal well along "
            "a thin pay zone. The path is described by measured depth, true "
            "vertical depth, inclination, and azimuth. Horizontal wells expose far "
            "more reservoir to the wellbore and often produce at a much higher rate."
        ),
        concepts=(
            ("inclination", ("inclination", "inclination angle")),
            ("azimuth", ("azimuth",)),
            ("measured_depth", ("measured depth", "md")),
            ("true_vertical_depth", ("true vertical depth", "tvd")),
            ("horizontal_well", ("horizontal well",)),
        ),
        facts=(
            ("horizontal_well", "production_rate", "increases", 0.7),
            ("inclination", "directional_well", "describes", 0.6),
            ("measured_depth", "true_vertical_depth", "differs_from", 0.6),
        ),
        procedures=(
            "Plan the well path from surface to target.",
            "Steer with a downhole motor and survey the path.",
        ),
    ),
    # ------------------------------------------------------------------ production
    CorpusEntry(
        key="production_optimisation",
        discipline="production",
        title="Production optimisation and decline",
        definition=(
            "Production optimisation seeks the highest safe rate from each well. "
            "Rate is governed by the pressure drawdown between reservoir and "
            "wellbore and by inflow performance. As a field produces, pressure "
            "falls and rate declines; decline-curve analysis fits the trend and "
            "forecasts future production. When the reservoir can no longer lift its "
            "own fluids, artificial lift is installed to restore the rate."
        ),
        concepts=(
            ("drawdown", ("drawdown",)),
            ("inflow_performance", ("inflow performance", "ipr")),
            ("decline_curve", ("decline curve", "decline curve analysis")),
            ("nodal_analysis", ("nodal analysis",)),
        ),
        facts=(
            ("drawdown", "production_rate", "increases", 0.7),
            ("inflow_performance", "production_rate", "governs", 0.7),
            ("decline_curve", "production_decline", "forecasts", 0.6),
        ),
        formulas=(
            ("Productivity index", "PI = q / (Pr - Pwf)"),
            ("Exponential decline", "q(t) = qi * exp(-D * t)"),
        ),
        procedures=(
            "Build an inflow-performance relationship for the well.",
            "Select the lift method that keeps the well above its economic limit.",
        ),
        glossary=(("production", "الإنتاج"), ("decline", "الانحدار")),
    ),
    CorpusEntry(
        key="artificial_lift_selection",
        discipline="artificial_lift",
        title="Artificial-lift selection",
        definition=(
            "Artificial lift adds energy to the wellbore so fluids reach the "
            "surface. Gas lift injects gas into the tubing to lighten the fluid "
            "column; an electrical submersible pump (ESP) uses a downhole motor and "
            "pump; a sucker-rod pump uses surface reciprocating motion; a "
            "progressing-cavity pump handles heavy oil and solids. Selection depends "
            "on rate, depth, fluid properties, and the presence of sand or gas."
        ),
        concepts=(
            ("gas_lift_valve", ("gas lift valve", "gas lift mandrel")),
            ("sucker_rod_pump", ("sucker rod pump", "beam pump")),
            ("progressing_cavity_pump", ("progressing cavity pump", "pcp")),
        ),
        facts=(
            ("gas_lift", "production_rate", "increases", 0.7),
            ("esp", "production_rate", "increases", 0.7),
            ("sucker_rod_pump", "artificial_lift", "implements", 0.6),
        ),
        procedures=(
            "Screen lift methods against rate, depth, and fluid properties.",
            "Design the selected system and set it below the working fluid level.",
        ),
        glossary=(("artificial lift", "الرفع الصناعي"),),
    ),
    # ------------------------------------------------------------------ completions
    CorpusEntry(
        key="well_completions_design",
        discipline="completions",
        title="Well completions",
        definition=(
            "A completion is the equipment and operations that connect the "
            "reservoir to the wellbore so fluids can be produced safely. The well "
            "is perforated through casing at the pay zone, or completed open-hole "
            "with sand control. A packer isolates the producing interval, tubing "
            "carries the fluids, and a Christmas tree controls the flow at surface. "
            "The completion is designed for the life of the well, so it must allow "
            "future intervention."
        ),
        concepts=(
            ("perforation_tunnel", ("perforation tunnel", "perforation")),
            ("packer", ("packer",)),
            ("sand_control", ("sand control", "gravel pack")),
            ("christmas_tree", ("christmas tree", "wellhead")),
        ),
        facts=(
            ("perforating", "well", "connects", 0.8),
            ("packer", "production_casing", "isolates", 0.7),
            ("sand_control", "production", "stabilizes", 0.6),
        ),
        procedures=(
            "Select the completion type from reservoir and sand behaviour.",
            "Perforate, then run the completion string and test the well.",
        ),
        glossary=(("completion", "إكمال البئر"), ("perforation", "تثقيب")),
    ),
    # ------------------------------------------------------------------ stimulation
    CorpusEntry(
        key="hydraulic_fracturing",
        discipline="stimulation",
        title="Hydraulic fracturing",
        definition=(
            "Hydraulic fracturing creates a conductive fracture in the reservoir to "
            "bypass near-wellbore damage and to increase the contact between the "
            "well and the formation. Fluid is pumped above the fracture pressure to "
            "open the rock, and proppant is carried into the fracture to keep it "
            "open after pumping stops. Fracture conductivity — proppant "
            "concentration and placement — determines how much the treatment "
            "improves productivity."
        ),
        concepts=(
            ("proppant", ("proppant", "propping agent")),
            ("fracture_conductivity", ("fracture conductivity",)),
            ("breakdown_pressure", ("breakdown pressure", "fracture gradient")),
            ("fracture_geometry", ("fracture geometry", "fracture half-length")),
        ),
        facts=(
            ("hydraulic_fracturing", "permeability", "increases", 0.7),
            ("proppant", "fracture_conductivity", "provides", 0.7),
            ("hydraulic_fracturing", "production_rate", "increases", 0.7),
        ),
        formulas=(("Fracture gradient", "FG = P_breakdown / TVD"),),
        procedures=(
            "Design the fracture geometry and fluid schedule.",
            "Pump the pad, slurry, and flush, then flow the well back.",
        ),
        standards=("Treatments are designed and reported to API RP 19C.",),
        glossary=(("hydraulic fracturing", "التكسير الهيدروليكي"),),
    ),
    CorpusEntry(
        key="matrix_acidizing",
        discipline="stimulation",
        title="Matrix acidizing",
        definition=(
            "Matrix acidizing dissolves near-wellbore damage to restore "
            "permeability without fracturing the rock. Hydrochloric acid dissolves "
            "carbonate rock and the fines that plug the pores; a mud-acid or "
            "organic-acid blend is used for sandstone. The treatment is pumped "
            "below the fracture pressure so the acid penetrates the matrix. "
            "Success is measured by the change in skin factor before and after."
        ),
        concepts=(
            ("skin_factor", ("skin factor", "skin")),
            ("formation_damage", ("formation damage", "damage")),
            ("acid_recipe", ("acid recipe", "acid system")),
        ),
        facts=(
            ("acid_stimulation", "formation_damage", "removes", 0.7),
            ("acid_stimulation", "skin_factor", "reduces", 0.7),
            ("acid_stimulation", "permeability", "restores", 0.7),
        ),
        procedures=(
            "Diagnose the damage from a pressure-transient test.",
            "Pump the acid below fracture pressure and flow the well back.",
        ),
    ),
    # ------------------------------------------------------------------ cementing
    CorpusEntry(
        key="primary_cementing",
        discipline="cementing",
        title="Primary cementing",
        definition=(
            "Primary cementing places a cement sheath in the annulus between the "
            "casing and the formation. The sheath supports the casing, protects it "
            "from corrosion, and — critically — provides zonal isolation so fluids "
            "cannot move between formations behind the pipe. A two-plug method "
            "separates the cement from the drilling fluid. A cement bond log "
            "verifies that the sheath is continuous and well bonded."
        ),
        concepts=(
            ("cement_sheath", ("cement sheath", "cement annulus")),
            ("zonal_isolation", ("zonal isolation",)),
            ("cement_bond_log", ("cement bond log", "cbl")),
            ("cement_slurry_design", ("cement slurry design", "slurry")),
        ),
        facts=(
            ("cement", "zonal_isolation", "provides", 0.8),
            ("cement_bond_log", "cement", "evaluates", 0.7),
            ("cement_sheath", "well_integrity", "supports", 0.7),
        ),
        procedures=(
            "Design the slurry for the hole and the formations.",
            "Displace the cement with plugs and verify the bond with a log.",
        ),
        standards=("Cementing follows API RP 10B and API RP 65.",),
        glossary=(("cement", "الأسمنت"), ("zonal isolation", "العزل بين الطبقات")),
    ),
    # ------------------------------------------------------------------ well_control
    CorpusEntry(
        key="well_control_kicks",
        discipline="well_control",
        title="Well control and kicks",
        definition=(
            "A kick is an unwanted influx of formation fluid into the wellbore, "
            "caused when the hydrostatic pressure of the mud falls below pore "
            "pressure. If it is not controlled, a kick becomes a blowout. The first "
            "response is to shut in the well with the blowout preventer and read "
            "the shut-in pressures. The driller's method and the wait-and-weight "
            "method then circulate the influx out and replace the mud with a "
            "heavier kill fluid that balances pore pressure."
        ),
        concepts=(
            ("kick", ("kick", "influx", "gas kick")),
            ("shut_in_pressure", ("shut-in pressure", "sicp")),
            ("kill_mud_weight", ("kill mud weight", "kill fluid")),
            ("drillers_method", ("driller's method", "wait and weight")),
        ),
        facts=(
            ("safety_valve", "gas_kick", "prevents", 0.8),
            ("shut_in", "gas_kick", "controls", 0.7),
            ("kill_mud", "pore_pressure", "balances", 0.7),
            ("hydrostatic_pressure", "gas_kick", "prevents", 0.7),
        ),
        formulas=(("Kill mud weight", "KMW = MW + (SIDPP / (0.052 * TVD))"),),
        procedures=(
            "Shut in the well and record the shut-in pressures.",
            "Circulate the kick out and weight up the mud to kill weight.",
        ),
        standards=("Well control follows API Standard 53 and IADC guidelines.",),
        glossary=(("kick", "اندفاع"), ("blowout", "انفجار البئر")),
    ),
    # ------------------------------------------------------------------ eor
    CorpusEntry(
        key="enhanced_oil_recovery",
        discipline="eor",
        title="Enhanced oil recovery",
        definition=(
            "Enhanced oil recovery (EOR) recovers oil that waterflooding leaves "
            "behind. Thermal methods, such as steam flooding and cyclic steam "
            "stimulation, lower oil viscosity so it flows. Chemical methods inject "
            "polymers to improve sweep or surfactants to lower interfacial tension. "
            "Gas methods inject carbon dioxide or hydrocarbon gas, which mixes with "
            "the oil and mobilises it. Each method targets a different trapping "
            "mechanism, so selection follows a screening study."
        ),
        concepts=(
            ("thermal_eor", ("thermal eor", "steam flooding")),
            ("chemical_eor", ("chemical eor", "polymer flooding")),
            ("miscible_gas", ("miscible gas", "co2 flooding")),
            ("screening_study", ("eor screening", "screening study")),
        ),
        facts=(
            ("thermal_eor", "recovery_factor", "increases", 0.7),
            ("chemical_eor", "recovery_factor", "increases", 0.6),
            ("miscible_gas", "recovery_factor", "increases", 0.7),
        ),
        procedures=(
            "Screen EOR methods against reservoir and fluid properties.",
            "Run a pilot, then scale the method to the field.",
        ),
        glossary=(("enhanced oil recovery", "الاستخلاص المعزز للنفط"),),
    ),
    # ------------------------------------------------------------------ well_testing
    CorpusEntry(
        key="pressure_transient_testing",
        discipline="well_testing",
        title="Pressure-transient testing",
        definition=(
            "A pressure-transient test measures how pressure changes with time "
            "while a well is produced or shut in. A drawdown or buildup test "
            "reveals permeability, reservoir pressure, and the skin factor that "
            "quantifies near-wellbore damage. The data are interpreted by matching "
            "the measured response to a model. The result guides stimulation: a "
            "positive skin means the well is damaged and will benefit from "
            "acidizing or fracturing."
        ),
        concepts=(
            ("buildup_test", ("buildup test", "build-up test")),
            ("drawdown_test", ("drawdown test",)),
            ("skin_factor", ("skin factor", "skin")),
            ("wellbore_storage", ("wellbore storage",)),
        ),
        facts=(
            ("pressure_buildup", "permeability", "estimates", 0.7),
            ("pressure_transient_test", "skin_factor", "measures", 0.7),
            ("skin_factor", "formation_damage", "quantifies", 0.7),
        ),
        formulas=(("Skin and drawdown", "delta_P_skin = (q * mu * s) / (2 * pi * k * h)"),),
        procedures=(
            "Run a buildup test after a period of stable production.",
            "Match the response to a model and report k, p*, and skin.",
        ),
    ),
    # ------------------------------------------------------------------ economics
    CorpusEntry(
        key="petroleum_economics_npv",
        discipline="economics",
        title="Petroleum project economics",
        definition=(
            "Petroleum projects are judged by discounted cash flow. Net present "
            "value discounts future cash flows at a chosen rate and sums them; a "
            "positive NPV means the project adds value. Other measures are the "
            "internal rate of return, the payback period, and the profit-to-"
            "investment ratio. Because prices, costs, and volumes are uncertain, "
            "the evaluation is repeated across scenarios and the project is "
            "reported with its sensitivities."
        ),
        concepts=(
            ("discount_rate", ("discount rate", "discount factor")),
            ("cash_flow", ("cash flow",)),
            ("payback_period", ("payback period", "payback")),
            ("sensitivity_analysis", ("sensitivity analysis", "sensitivity")),
        ),
        facts=(
            ("production", "revenue", "generates", 0.8),
            ("revenue", "net_present_value", "increases", 0.7),
            ("cost", "net_present_value", "reduces", 0.7),
            ("discount_rate", "net_present_value", "reduces", 0.7),
        ),
        formulas=(
            ("Net present value", "NPV = sum( CF_t / (1 + r)^t )"),
            ("Profit to investment", "PIR = NPV / investment"),
        ),
        procedures=(
            "Build a cash-flow model from the production forecast.",
            "Compute NPV, IRR, and payback, then run sensitivities.",
        ),
        standards=("Evaluations follow SPE guidance on project economics.",),
        glossary=(("net present value", "صافي القيمة الحالية"),),
    ),
    CorpusEntry(
        key="field_development_planning",
        discipline="economics",
        title="Field development planning",
        definition=(
            "Field development planning chooses how to deplete a field: the number "
            "and type of wells, the recovery mechanism, the facilities, and the "
            "schedule. Options are compared on value, risk, and robustness. A "
            "development plan is a living document — it is revised as new wells "
            "produce and as prices and costs change. Reservoir, drilling, "
            "production, facilities, and economics must agree on one plan."
        ),
        concepts=(
            ("development_concept", ("development concept", "development plan")),
            ("well_count", ("well count", "well density")),
            ("facilities_capacity", ("facilities capacity", "capacity")),
        ),
        facts=(
            ("development_concept", "reserve", "determines", 0.6),
            ("well_count", "production", "increases", 0.6),
        ),
        procedures=(
            "Generate development options.",
            "Evaluate each option, then select and schedule the chosen plan.",
        ),
    ),
    # ------------------------------------------------------------------ hse
    CorpusEntry(
        key="well_integrity_management",
        discipline="hse",
        title="Well integrity management",
        definition=(
            "Well integrity is the ability of a well to contain formation fluids "
            "throughout its life without leaks. It is built from two independent "
            "barriers: a primary barrier, usually the hydrostatic column of mud or "
            "the completion, and a secondary barrier, the casing and cement. "
            "Integrity is maintained by testing barriers, monitoring annulus "
            "pressures, and repairing faults. A well without two barriers is shut "
            "in until integrity is restored."
        ),
        concepts=(
            ("primary_barrier", ("primary barrier",)),
            ("secondary_barrier", ("secondary barrier",)),
            ("annulus_pressure", ("annulus pressure", "sustained casing pressure")),
            ("integrity_test", ("integrity test", "pressure test")),
        ),
        facts=(
            ("pressure_barrier", "well_integrity", "provides", 0.8),
            ("inspection", "well_integrity", "verifies", 0.7),
            ("cement", "well_integrity", "supports", 0.7),
            ("annulus_pressure", "well_integrity", "indicates", 0.6),
        ),
        procedures=(
            "Define the two barrier envelopes for the well.",
            "Test the barriers, monitor annulus pressure, and repair faults.",
        ),
        standards=("Well integrity follows ISO 16530 and NORSOK D-010.",),
        glossary=(("well integrity", "سلامة البئر"), ("barrier", "حاجز")),
    ),
    CorpusEntry(
        key="process_safety",
        discipline="hse",
        title="Process safety",
        definition=(
            "Process safety protects people and the environment from the release of "
            "hazardous fluids and energy. It rests on identifying hazards, "
            "designing barriers against them, and verifying that the barriers work. "
            "A hazard and operability study, a layer-of-protection analysis, and a "
            "safety-instrumented-system review are the standard tools. A "
            "safety case collects the evidence that the risks are as low as "
            "reasonably practicable."
        ),
        concepts=(
            ("hazard_analysis", ("hazard analysis", "hazop")),
            ("safety_instrumented_system", ("safety instrumented system", "sis")),
            ("safety_case", ("safety case",)),
            ("as_low_as_reasonably_practicable", ("alarp",)),
        ),
        facts=(
            ("hazard_analysis", "risk", "identifies", 0.7),
            ("safety_instrumented_system", "risk", "reduces", 0.7),
            ("inspection", "safety_instrumented_system", "verifies", 0.6),
        ),
        procedures=(
            "Identify hazards and the barriers that control them.",
            "Verify the barriers and document the safety case.",
        ),
        standards=("Process safety follows API RP 14C and IEC 61511.",),
    ),
    # ------------------------------------------------------------------ foundations (engineering science)
    CorpusEntry(
        key="reservoir_fluid_properties",
        discipline="reservoir",
        title="Reservoir fluid properties",
        definition=(
            "Reservoir fluids are described by their pressure-volume-temperature "
            "(PVT) behaviour. Oil is characterised by its formation volume factor, "
            "its gas-oil ratio, and its viscosity; gas by its compressibility and "
            "deviation factor. Above the bubble point the oil is undersaturated and "
            "holds all its gas in solution; below it, free gas appears. These "
            "properties convert surface volumes to reservoir volumes and drive "
            "every material-balance and simulation calculation."
        ),
        concepts=(
            ("bubble_point", ("bubble point", "saturation pressure")),
            ("gas_oil_ratio", ("gas oil ratio", "gor")),
            ("viscosity", ("viscosity",)),
            ("dew_point", ("dew point",)),
        ),
        facts=(
            ("bubble_point", "gas_oil_ratio", "determines", 0.6),
            ("viscosity", "production_rate", "reduces", 0.6),
            ("formation_volume_factor", "original_oil_in_place", "affects", 0.6),
        ),
        formulas=(
            ("Formation volume factor", "Bo = Vreservoir / Vsurface"),
            ("Gas deviation factor", "PV = Z n R T"),
        ),
        procedures=(
            "Sample the fluid at reservoir conditions.",
            "Measure the PVT behaviour in the laboratory.",
        ),
    ),
    CorpusEntry(
        key="rock_mechanics",
        discipline="reservoir",
        title="Rock mechanics",
        definition=(
            "Rock mechanics describes how rock deforms and fails under stress. "
            "In-situ stress has a vertical component from the overburden and two "
            "horizontal components that differ where the basin has a tectonic "
            "history. Pore pressure reduces the effective stress that the rock "
            "framework carries. Rock strength and stress control wellbore "
            "stability, fracture orientation, and sand production."
        ),
        concepts=(
            ("in_situ_stress", ("in-situ stress", "stress state")),
            ("effective_stress", ("effective stress",)),
            ("overburden", ("overburden", "overburden stress")),
            ("rock_strength", ("rock strength", "compressive strength")),
        ),
        facts=(
            ("in_situ_stress", "hydraulic_fracturing", "controls", 0.7),
            ("effective_stress", "rock_strength", "affects", 0.6),
            ("overburden", "in_situ_stress", "contributes", 0.7),
        ),
        formulas=(("Terzaghi effective stress", "sigma' = sigma - alpha * Pp"),),
        procedures=(
            "Estimate the stress state from logs and tests.",
            "Apply it to wellbore stability and fracture design.",
        ),
    ),
    CorpusEntry(
        key="fluid_flow_in_porous_media",
        discipline="reservoir",
        title="Fluid flow in porous media",
        definition=(
            "Flow through porous rock is described by Darcy's law: rate is "
            "proportional to permeability and the pressure gradient, and inversely "
            "proportional to fluid viscosity. In radial flow toward a well, the "
            "same law gives the productivity index. Multiphase flow adds relative "
            "permeability: the effective permeability of each phase falls as the "
            "other phases occupy pore space, which is why water production reduces "
            "oil rate."
        ),
        concepts=(
            ("darcy_law", ("darcy law", "darcy's law")),
            ("relative_permeability", ("relative permeability", "kr")),
            ("radial_flow", ("radial flow",)),
            ("productivity_index", ("productivity index", "pi")),
        ),
        facts=(
            ("darcy_law", "permeability", "relates_to", 0.7),
            ("relative_permeability", "production_rate", "reduces", 0.6),
            ("radial_flow", "productivity_index", "gives", 0.6),
        ),
        formulas=(
            ("Radial flow", "q = 2 * pi * k * h * dP / (mu * ln(re/rw))"),
            ("Productivity index", "PI = q / (Pr - Pwf)"),
        ),
        procedures=(
            "Estimate permeability and skin from a well test.",
            "Forecast rate with the productivity index and decline.",
        ),
    ),
    CorpusEntry(
        key="offshore_operations",
        discipline="hse",
        title="Offshore operations",
        definition=(
            "Offshore wells are drilled from a fixed platform, a jack-up, or a "
            "floating rig. A floating rig needs a marine riser to connect the "
            "wellhead on the seabed to the rig, and a blowout preventer stack on "
            "the seabed. Subsea completions send production to the surface through "
            "flowlines and risers. Offshore operations are governed by the same "
            "barrier philosophy as onshore, with stricter requirements because "
            "response to a release is slower."
        ),
        concepts=(
            ("marine_riser", ("marine riser", "riser")),
            ("subsea_wellhead", ("subsea wellhead",)),
            ("blowout_preventer_stack", ("blowout preventer stack", "bop stack")),
            ("floating_rig", ("floating rig", "semisubmersible")),
        ),
        facts=(
            ("marine_riser", "well", "connects", 0.6),
            ("blowout_preventer_stack", "gas_kick", "prevents", 0.8),
        ),
        procedures=(
            "Run the riser and BOP stack before drilling ahead.",
            "Monitor the well and the station-keeping system continuously.",
        ),
    ),
    CorpusEntry(
        key="facilities_and_transport",
        discipline="production",
        title="Surface facilities and transport",
        definition=(
            "Surface facilities separate the produced stream into oil, gas, and "
            "water, treat each to specification, and move them to market. A "
            "separator works by gravity and retention time; a dehydrator removes "
            "water from gas; a stabiliser removes light ends from oil. Pipelines, "
            "tankers, and trucks then transport the products. Facilities capacity "
            "and pressure set the limit on field production."
        ),
        concepts=(
            ("separator", ("separator", "three-phase separator")),
            ("dehydration", ("dehydration", "dehydrator")),
            ("pipeline", ("pipeline",)),
            ("flow_assurance", ("flow assurance",)),
        ),
        facts=(
            ("separator", "production", "processes", 0.7),
            ("facilities_capacity", "production_rate", "limits", 0.7),
            ("flow_assurance", "pipeline", "protects", 0.6),
        ),
        procedures=(
            "Size separation and treatment for the forecast stream.",
            "Monitor for hydrates, wax, and scale in the export line.",
        ),
    ),
    CorpusEntry(
        key="project_management",
        discipline="economics",
        title="Project management",
        definition=(
            "Petroleum projects are delivered through a staged gate process: "
            "identify, assess, select, define, execute, and operate. Each stage "
            "ends at a decision gate where the project is either advanced or "
            "revised. Cost and schedule are estimated with a range that narrows as "
            "the project matures. Risk is managed by a register, and the schedule "
            "is protected by identifying the critical path."
        ),
        concepts=(
            ("stage_gate", ("stage gate", "decision gate")),
            ("cost_estimate", ("cost estimate", "estimate class")),
            ("schedule", ("schedule", "critical path")),
            ("risk_register", ("risk register",)),
        ),
        facts=(
            ("stage_gate", "decision", "structures", 0.6),
            ("cost_estimate", "net_present_value", "affects", 0.6),
            ("risk_register", "risk", "manages", 0.6),
        ),
        procedures=(
            "Define the scope and estimate cost and schedule with ranges.",
            "Pass each gate, then execute and hand over to operations.",
        ),
    ),
    CorpusEntry(
        key="professional_ethics",
        discipline="hse",
        title="Professional ethics",
        definition=(
            "Petroleum engineers hold public safety, the environment, and the "
            "integrity of technical work above commercial convenience. Ethics "
            "requires reporting reserves honestly, disclosing conflicts of "
            "interest, protecting confidential data, and refusing to sign work "
            "that is not sound. A professional society code makes these duties "
            "explicit, and the duty to the public overrides the duty to an "
            "employer."
        ),
        concepts=(
            ("conflict_of_interest", ("conflict of interest",)),
            ("data_ethics", ("data ethics", "confidentiality")),
            ("public_safety", ("public safety",)),
            ("transparency", ("transparency",)),
        ),
        facts=(
            ("public_safety", "decision", "overrides", 0.7),
            ("transparency", "data", "requires", 0.6),
            ("conflict_of_interest", "decision", "threatens", 0.6),
        ),
        procedures=(
            "Disclose conflicts and protect confidential data.",
            "Report results honestly and refuse unsound work.",
        ),
        standards=("Ethics follow the SPE Code of Professional Conduct.",),
    ),
)


def entries() -> tuple[CorpusEntry, ...]:
    """Every corpus entry, grouped by discipline."""
    return _ENTRIES


def entry_keys() -> tuple[str, ...]:
    return tuple(entry.key for entry in _ENTRIES)


def entries_for(discipline: str) -> tuple[CorpusEntry, ...]:
    return tuple(entry for entry in _ENTRIES if entry.discipline == discipline)


def corpus_disciplines() -> tuple[str, ...]:
    """The disciplines the corpus covers, in first-seen order."""
    seen: list[str] = []
    for entry in _ENTRIES:
        if entry.discipline not in seen:
            seen.append(entry.discipline)
    return tuple(seen)


def _add_concept(ontology: Ontology, label: str, aliases: tuple[str, ...]) -> Concept:
    concept = Concept(label, frozenset(aliases))
    ontology.add_concept(concept)
    return concept


def _concept_for(ontology: Ontology, label: str, aliases: tuple[str, ...]) -> Concept:
    existing = ontology.find_concepts(label)
    return existing[0] if existing else _add_concept(ontology, label, aliases)


def _add_relation(
    ontology: Ontology, source: Concept, target: Concept, relation_type: str, confidence: float
) -> None:
    if any(
        relation.source_id == source.id
        and relation.target_id == target.id
        and relation.relation_type == relation_type
        for relation in ontology.relations.values()
    ):
        return
    ontology.add_relation(SemanticRelation(source.id, target.id, relation_type, confidence))


def seed_petroleum_corpus(ontology: Ontology) -> list[Concept]:
    """Plant the corpus concepts and relations, reusing existing nodes.

    Idempotent, and shares nodes with the domain frame and the curriculum, so the
    three seed layers form one graph rather than three disconnected ones.
    """
    concepts: dict[str, Concept] = {}
    for entry in _ENTRIES:
        for label, aliases in entry.concepts:
            if label not in concepts:
                concepts[label] = _concept_for(ontology, label, aliases)
        for source, target, _relation, _confidence in entry.facts:
            for label in (source, target):
                if label not in concepts:
                    concepts[label] = _concept_for(ontology, label, ())
    for entry in _ENTRIES:
        for source, target, relation_type, confidence in entry.facts:
            _add_relation(ontology, concepts[source], concepts[target], relation_type, confidence)
    return list(concepts.values())


def corpus_coverage(ontology: Ontology) -> dict[str, dict[str, int]]:
    """Per-discipline counts of corpus concepts and relations present."""
    coverage: dict[str, dict[str, int]] = {}
    for discipline in corpus_disciplines():
        discipline_entries = entries_for(discipline)
        concept_labels = {label for entry in discipline_entries for label, _ in entry.concepts}
        concept_hits = sum(1 for label in concept_labels if ontology.find_concepts(label))
        expected_concepts = len(concept_labels)
        relation_hits = 0
        expected_relations = 0
        for entry in discipline_entries:
            for source, target, relation_type, _ in entry.facts:
                expected_relations += 1
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
        coverage[discipline] = {
            "concepts": concept_hits,
            "relations": relation_hits,
            "expected_concepts": expected_concepts,
            "expected_relations": expected_relations,
        }
    return coverage


def corpus_coverage_score(ontology: Ontology) -> float:
    """Fraction of the whole corpus's concepts and relations present."""
    expected = 0
    present = 0
    for counts in corpus_coverage(ontology).values():
        expected += counts["expected_concepts"] + counts["expected_relations"]
        present += counts["concepts"] + counts["relations"]
    if expected == 0:
        return 1.0
    return present / expected


def corpus_documents() -> list["Document"]:
    """Turn each corpus entry into a citable document."""
    from core.documents import Document

    documents: list[Document] = []
    for entry in _ENTRIES:
        lines = [entry.definition]
        for label, aliases in entry.concepts:
            aliases_text = f" (also known as {', '.join(aliases)})" if aliases else ""
            lines.append(f"Concept {label}{aliases_text}.")
        for source, target, relation, _confidence in entry.facts:
            lines.append(f"{source} {relation} {target}.")
        for name, expression in entry.formulas:
            lines.append(f"Formula {name}: {expression}.")
        for step in entry.procedures:
            lines.append(f"Procedure: {step}")
        for standard in entry.standards:
            lines.append(f"Standard: {standard}")
        for term, arabic in entry.glossary:
            lines.append(f"Term {term} ({arabic}).")
        documents.append(
            Document(
                doc_id=f"corpus:{entry.key}",
                title=f"{entry.title} ({entry.discipline})",
                text="\n".join(lines),
                source="petroleum_corpus",
                metadata={"kind": "corpus", "key": entry.key, "discipline": entry.discipline},
            )
        )
    return documents


def corpus_sentences() -> list[str]:
    """Training prose generated from the whole corpus."""
    sentences: list[str] = []
    for entry in _ENTRIES:
        sentences.append(entry.definition)
        sentences.append(f"{entry.title} belongs to {entry.discipline} engineering.")
        for label, aliases in entry.concepts:
            sentences.append(f"{label} is a concept in {entry.title}.")
            if aliases:
                sentences.append(f"{label} is also known as {aliases[0]}.")
        for source, target, relation, _confidence in entry.facts:
            sentences.append(f"{source} {relation} {target}.")
        for name, expression in entry.formulas:
            sentences.append(f"The formula for {name} is {expression}.")
        for step in entry.procedures:
            sentences.append(step)
        for standard in entry.standards:
            sentences.append(standard)
    return sentences


__all__ = [
    "CorpusEntry",
    "corpus_coverage",
    "corpus_coverage_score",
    "corpus_disciplines",
    "corpus_documents",
    "corpus_sentences",
    "entries",
    "entries_for",
    "entry_keys",
    "seed_petroleum_corpus",
]
