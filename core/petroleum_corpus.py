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
    # ------------------------------------------------------------------ geoscience
    CorpusEntry(
        key="petroleum_system",
        discipline="geoscience",
        title="The petroleum system",
        definition=(
            "A petroleum system links a source rock that generates hydrocarbons to "
            "the accumulation that traps them. Organic matter buried and heated "
            "matures into oil and then gas along a geothermal gradient. The "
            "generated fluids migrate along carrier beds and faults until a trap "
            "and a sealing cap rock stop them. The system is only complete when "
            "all the elements are present and the timing places migration after "
            "trap formation; missing any one element leaves a dry prospect."
        ),
        concepts=(
            ("burial_history", ("burial history", "subsidence")),
            ("hydrocarbon_charge", ("hydrocarbon charge", "charge")),
            ("play_fairway", ("play fairway",)),
            ("prospect", ("prospect", "lead")),
        ),
        facts=(
            ("burial_history", "thermal_maturation", "controls", 0.7),
            ("geothermal_gradient", "thermal_maturation", "drives", 0.7),
            ("hydrocarbon_charge", "trap", "fills", 0.7),
            ("play_fairway", "prospect", "contains", 0.6),
            ("sedimentary_basin", "play_fairway", "hosts", 0.6),
        ),
        procedures=(
            "Map source, reservoir, seal, and trap elements across the basin.",
            "Rank prospects by charge risk before drilling.",
        ),
        glossary=(("petroleum system", "النظام النفطي"), ("play fairway", "نطاق اللعب")),
    ),
    CorpusEntry(
        key="seismic_interpretation",
        discipline="geoscience",
        title="Seismic interpretation",
        definition=(
            "Seismic interpretation turns reflection data into a structural and "
            "stratigraphic model of the subsurface. Interpreters pick horizons that "
            "follow continuous reflections and faults that offset them, then map "
            "closure to define a trap. Seismic attributes such as amplitude and "
            "inversion products highlight lithology and fluid changes, so an "
            "amplitude anomaly can indicate hydrocarbon. Every interpretation "
            "carries uncertainty that is carried into the volumetric estimate."
        ),
        concepts=(
            ("seismic_attribute", ("seismic attribute",)),
            ("horizon", ("horizon",)),
            ("fault_interpretation", ("fault interpretation",)),
            ("amplitude_anomaly", ("amplitude anomaly", "bright spot")),
            ("inversion_model", ("seismic inversion", "inversion")),
        ),
        facts=(
            ("seismic_section", "horizon", "reveals", 0.6),
            ("fault_interpretation", "trap", "defines", 0.6),
            ("amplitude_anomaly", "hydrocarbon", "indicates", 0.5),
            ("inversion_model", "seismic_attribute", "uses", 0.6),
            ("seismic_attribute", "seismic_section", "derived_from", 0.6),
        ),
        procedures=(
            "Pick horizons and faults on the migrated volume.",
            "Map depth closure and integrate attributes with well control.",
        ),
    ),
    # ------------------------------------------------------------------ facilities
    CorpusEntry(
        key="surface_facilities",
        discipline="facilities",
        title="Surface production facilities",
        definition=(
            "Surface facilities separate, treat, and move what the wells produce. "
            "A separator drops pressure so oil, gas, and water split into phases; "
            "the gas is dehydrated and sweetened, the oil is stabilised, and the "
            "water is treated before disposal or reinjection. Pumps and compressors "
            "move the streams, and pipelines or tankers export them. Facilities "
            "design follows the production profile, so early, plateau, and decline "
            "rates all set equipment sizes."
        ),
        concepts=(
            ("separator", ("separator", "three-phase separator")),
            ("pipeline", ("pipeline", "flowline")),
            ("gas_processing", ("gas processing", "gas plant")),
            ("pump", ("pump", "pumping system")),
            ("storage_tank", ("storage tank", "tank farm")),
        ),
        facts=(
            ("separator", "production", "processes", 0.7),
            ("pipeline", "hydrocarbon", "transports", 0.7),
            ("gas_processing", "hydrocarbon", "treats", 0.6),
            ("pump", "pipeline", "pressurizes", 0.6),
            ("storage_tank", "hydrocarbon", "stores", 0.6),
        ),
        procedures=(
            "Size separators and pumps from the forecast production profile.",
            "Specify gas treating to meet export and sales specifications.",
        ),
        glossary=(("separator", "فاصل"), ("pipeline", "خط أنابيب")),
    ),
    CorpusEntry(
        key="gas_processing_and_lng",
        discipline="facilities",
        title="Gas processing and LNG",
        definition=(
            "Gas processing removes the impurities and heavy hydrocarbons that "
            "would otherwise corrode pipelines or fail sales specifications. "
            "Dehydration removes water, sweetening removes hydrogen sulphide and "
            "carbon dioxide, and a dew-point unit strips condensate. Where a market "
            "is far away, the gas is liquefied to shrink its volume for shipping as "
            "LNG. Each step is a trade between recovery, energy use, and product "
            "value."
        ),
        concepts=(
            ("dehydration", ("dehydration", "glycol dehydration")),
            ("sweetening", ("sweetening", "amine treating")),
            ("nGL_recovery", ("NGL recovery", "natural gas liquids")),
            ("liquefaction", ("liquefaction", "LNG train")),
        ),
        facts=(
            ("dehydration", "hydrate", "prevents", 0.7),
            ("sweetening", "gas", "purifies", 0.7),
            ("nGL_recovery", "gas", "strips", 0.6),
            ("liquefaction", "gas", "condenses", 0.6),
        ),
        procedures=(
            "Dehydrate and sweeten the gas to pipeline specification.",
            "Recover NGL and, for distant markets, liquefy the residue.",
        ),
    ),
    # ------------------------------------------------------------- process_safety
    CorpusEntry(
        key="process_safety_management",
        discipline="process_safety",
        title="Process safety management",
        definition=(
            "Process safety management prevents the loss of containment that turns "
            "a hydrocarbon release into a fire, explosion, or toxic event. It "
            "identifies hazards with a HAZOP or similar study, then layers "
            "protection so that no single failure causes harm: the design, the "
            "safety instrumented system, alarms, and emergency response. Barriers "
            "are monitored so their health is known, and a safety case documents "
            "the argument that the risk is tolerable. A process safety event is "
            "investigated to find the failed barrier, not just the failed person."
        ),
        concepts=(
            ("process_safety_event", ("process safety event", "loss of containment")),
            ("layers_of_protection", ("layers of protection", "protection layers")),
            ("hazard_analysis", ("hazard analysis", "HAZOP")),
            ("safety_case", ("safety case",)),
            ("barrier_management", ("barrier management", "barrier health")),
        ),
        facts=(
            ("hazard_analysis", "process_safety_event", "prevents", 0.7),
            ("layers_of_protection", "process_safety_event", "mitigates", 0.7),
            ("barrier_management", "layers_of_protection", "maintains", 0.6),
            ("safety_case", "hazard_analysis", "documents", 0.6),
            ("process_safety_event", "well_integrity", "threatens", 0.6),
        ),
        procedures=(
            "Run a HAZOP and record every layer of protection.",
            "Track barrier health and investigate each loss of containment.",
        ),
        standards=("Process safety follows API RP 754 for event reporting.",),
    ),
    # ------------------------------------------------------------ unconventional
    CorpusEntry(
        key="unconventional_resources",
        discipline="unconventional",
        title="Unconventional resources",
        definition=(
            "Unconventional reservoirs hold hydrocarbons in rock so tight that the "
            "well must create permeability. Shale and tight-gas plays are developed "
            "with long horizontal laterals placed in a sweet spot and completed "
            "with many hydraulic-fracture stages. Proppant keeps the fractures "
            "conductive, and the stimulated rock volume, not the wellbore alone, "
            "drains the oil or gas. Because every stage is expensive, economics "
            "hinge on landing-zone quality and fracture design."
        ),
        concepts=(
            ("shale_play", ("shale play", "shale")),
            ("horizontal_lateral", ("horizontal lateral", "lateral")),
            ("multi_stage_fracturing", ("multi-stage fracturing", "multi-stage frac")),
            ("tight_gas", ("tight gas", "tight reservoir")),
            ("sweet_spot", ("sweet spot", "landing zone")),
        ),
        facts=(
            ("shale_play", "reservoir", "hosts", 0.6),
            ("horizontal_lateral", "shale_play", "drains", 0.7),
            ("multi_stage_fracturing", "horizontal_lateral", "stimulates", 0.7),
            ("sweet_spot", "horizontal_lateral", "guides", 0.6),
            ("tight_gas", "shale_play", "produced_from", 0.6),
        ),
        procedures=(
            "Identify the sweet spot from logs, core, and seismic.",
            "Drill a long lateral and complete it in many fracture stages.",
        ),
        glossary=(("unconventional", "غير تقليدي"), ("sweet spot", "المنطقة المثلى")),
    ),
    # ---------------------------------------------- advanced reservoir engineering
    CorpusEntry(
        key="reservoir_simulation_modelling",
        discipline="reservoir",
        title="Reservoir simulation and modelling",
        definition=(
            "A reservoir simulator solves the flow equations on a grid to predict "
            "how a field will perform under a development plan. A compositional "
            "model tracks individual components through an equation of state when "
            "the fluids are volatile; a black-oil model is enough for simpler "
            "cases. The model is calibrated by history matching, adjusting "
            "properties until the simulated pressures and rates match measured "
            "history. Because a fine grid is too costly, upscaling coarsens it "
            "while preserving the flow behaviour."
        ),
        concepts=(
            ("history_matching", ("history matching",)),
            ("grid_discretization", ("grid discretization", "gridding")),
            ("upscaling", ("upscaling",)),
            ("equation_of_state", ("equation of state",)),
            ("compositional_model", ("compositional model",)),
            ("simulator", ("reservoir simulator", "simulator")),
        ),
        facts=(
            ("compositional_model", "equation_of_state", "uses", 0.7),
            ("history_matching", "simulator", "calibrates", 0.7),
            ("upscaling", "grid_discretization", "precedes", 0.6),
            ("simulator", "reservoir_simulation", "is_a", 0.7),
        ),
        procedures=(
            "Build the static model and initialise fluids.",
            "History-match, then run forecasts for each development plan.",
        ),
    ),
    CorpusEntry(
        key="advanced_eor_methods",
        discipline="eor",
        title="Advanced enhanced oil recovery",
        definition=(
            "Advanced EOR methods attack the two things waterflooding cannot fix: "
            "an unfavourable mobility ratio and oil trapped by capillary forces. "
            "Polymer flooding thickens the water to improve the mobility ratio and "
            "sweep; surfactant flooding lowers interfacial tension to mobilise "
            "residual oil; miscible gas mixes with the oil and displaces it. "
            "Conformance control seals thief zones so the injected fluid reaches "
            "the oil. Each method is screened against the reservoir before a pilot."
        ),
        concepts=(
            ("polymer_flooding", ("polymer flooding",)),
            ("surfactant_flooding", ("surfactant flooding",)),
            ("miscible_gas_injection", ("miscible gas injection",)),
            ("thermal_recovery", ("thermal recovery", "steam flooding")),
            ("mobility_ratio", ("mobility ratio",)),
            ("residual_oil_saturation", ("residual oil saturation",)),
            ("conformance_control", ("conformance control",)),
        ),
        facts=(
            ("polymer_flooding", "mobility_ratio", "improves", 0.7),
            ("surfactant_flooding", "residual_oil_saturation", "reduces", 0.7),
            ("miscible_gas_injection", "oil", "mixes_with", 0.6),
            ("thermal_recovery", "heavy_oil", "recovers", 0.6),
            ("conformance_control", "mobility_ratio", "manages", 0.6),
        ),
        formulas=(
            ("Mobility ratio", "M = (k_w / mu_w) / (k_o / mu_o)"),
            ("Capillary number", "Nc = v * mu / sigma"),
        ),
        procedures=(
            "Screen the method against fluid and reservoir properties.",
            "Run a single-well or pattern pilot before field-wide expansion.",
        ),
    ),
    CorpusEntry(
        key="waterflood_management",
        discipline="reservoir",
        title="Waterflood management",
        definition=(
            "A waterflood maintains reservoir pressure and sweeps oil toward "
            "producers by injecting water through a pattern of wells. Its success "
            "is measured by sweep efficiency: how much of the reservoir the water "
            "actually contacts. Injection above fracture pressure can channel "
            "water straight to a producer, so injection pressure and rate are "
            "controlled. Surveillance of rates, pressures, and water cut tells the "
            "engineer when to adjust the pattern or add conformance work."
        ),
        concepts=(
            ("injection_pattern", ("injection pattern", "flood pattern")),
            ("sweep_efficiency", ("sweep efficiency",)),
            ("water_cut", ("water cut",)),
            ("voidage_replacement", ("voidage replacement",)),
        ),
        facts=(
            ("injection_pattern", "sweep_efficiency", "controls", 0.7),
            ("waterflood", "reservoir_pressure", "maintains", 0.7),
            ("water_cut", "sweep_efficiency", "indicates", 0.6),
            ("voidage_replacement", "reservoir_pressure", "balances", 0.6),
        ),
        procedures=(
            "Set the pattern and keep voidage replacement near one.",
            "Track water cut and adjust injection to protect sweep.",
        ),
    ),
    CorpusEntry(
        key="field_development_planning_advanced",
        discipline="economics",
        title="Integrated field development planning",
        definition=(
            "An integrated field development plan turns subsurface, wells, "
            "facilities, and economics into one decision. An integrated asset "
            "model couples reservoir, network, and facilities so a change in one is "
            "seen in the others. Teams generate options, test them through decision "
            "gates, and rank them by value, usually net present value under "
            "uncertainty. The plan is a living document: new data reopens the "
            "choices, so opportunities are identified and re-ranked as the field "
            "matures."
        ),
        concepts=(
            ("integrated_asset_model", ("integrated asset model",)),
            ("production_forecast", ("production forecast",)),
            ("opportunity_identification", ("opportunity identification",)),
            ("value_optimization", ("value optimization", "value optimisation")),
            ("decision_gate", ("decision gate", "stage gate")),
        ),
        facts=(
            ("integrated_asset_model", "production_forecast", "produces", 0.7),
            ("opportunity_identification", "value_optimization", "enables", 0.6),
            ("decision_gate", "value_optimization", "governs", 0.6),
            ("production_forecast", "decision", "informs", 0.6),
            ("value_optimization", "net_present_value", "maximizes", 0.6),
        ),
        procedures=(
            "Couple subsurface, network, and facilities in one model.",
            "Rank options by value and pass them through decision gates.",
        ),
    ),
    CorpusEntry(
        key="geomechanics_and_wellbore_stability",
        discipline="drilling",
        title="Geomechanics and wellbore stability",
        definition=(
            "Geomechanics describes the stresses in the rock and how the well "
            "interacts with them. The in-situ stress state, described by a "
            "magnitude and an orientation, sets the mud weight window between pore "
            "pressure and fracture gradient. Drill outside that window and the "
            "borehole collapses or the formation fractures. The same stresses "
            "control sand production and the geometry of a hydraulic fracture, so "
            "geomechanics links drilling, completion, and production."
        ),
        concepts=(
            ("in_situ_stress", ("in situ stress", "stress state")),
            ("fracture_gradient", ("fracture gradient",)),
            ("wellbore_stability", ("wellbore stability",)),
            ("sand_production", ("sand production",)),
            ("stress_orientation", ("stress orientation",)),
        ),
        facts=(
            ("in_situ_stress", "wellbore_stability", "governs", 0.7),
            ("fracture_gradient", "hydraulic_fracturing", "limits", 0.6),
            ("sand_production", "wellbore_stability", "threatens", 0.6),
            ("stress_orientation", "in_situ_stress", "describes", 0.6),
            ("wellbore_stability", "drilling", "constrains", 0.6),
        ),
        formulas=(
            ("Mud weight window", "Ppore < MW * 0.052 * TVD < FracGrad * TVD"),
        ),
        procedures=(
            "Build a stress model from logs, core, and leak-off tests.",
            "Set mud weight inside the stability window and design perforations.",
        ),
    ),
    CorpusEntry(
        key="flow_assurance",
        discipline="production",
        title="Flow assurance",
        definition=(
            "Flow assurance keeps fluids moving from the reservoir to the export "
            "point. As pressure and temperature fall along the flowline, gas and "
            "water can form hydrates that plug the line, wax and asphaltene can "
            "deposit on the wall, and scale can choke a valve. Insulation, chemical "
            "inhibitors, and pigging manage these risks, and slugging is controlled "
            "so the receiving facilities are not overwhelmed. Flow assurance is "
            "designed in early because a blockage offshore is very costly to clear."
        ),
        concepts=(
            ("hydrate", ("hydrate", "gas hydrate")),
            ("wax_deposition", ("wax deposition", "wax")),
            ("asphaltene", ("asphaltene",)),
            ("scale_deposition", ("scale deposition", "scale")),
            ("slugging", ("slugging", "slug flow")),
            ("thermal_insulation", ("thermal insulation",)),
        ),
        facts=(
            ("hydrate", "flow_restriction", "causes", 0.7),
            ("wax_deposition", "flow_restriction", "causes", 0.7),
            ("asphaltene", "flow_restriction", "causes", 0.6),
            ("scale_deposition", "flow_restriction", "causes", 0.6),
            ("thermal_insulation", "hydrate", "prevents", 0.6),
            ("slugging", "flow_restriction", "causes", 0.5),
        ),
        procedures=(
            "Model pressure and temperature along the flowline.",
            "Prevent hydrates and wax with insulation, chemicals, and pigging.",
        ),
    ),
    CorpusEntry(
        key="multiphase_flow",
        discipline="production",
        title="Multiphase flow",
        definition=(
            "Multiphase flow is the simultaneous movement of oil, gas, and water in "
            "a pipe or wellbore. The flow regime, from bubble to slug to annular, "
            "depends on rates, inclination, and fluid properties, and it sets the "
            "pressure gradient and liquid holdup. Because the phases travel at "
            "different velocities, a slip velocity develops between them. Engineers "
            "read the regime from a flow pattern map and use it to size lines and "
            "design artificial lift."
        ),
        concepts=(
            ("flow_regime", ("flow regime",)),
            ("slip_velocity", ("slip velocity",)),
            ("pressure_gradient", ("pressure gradient",)),
            ("holdup", ("liquid holdup", "holdup")),
            ("flow_pattern_map", ("flow pattern map",)),
        ),
        facts=(
            ("flow_regime", "pressure_gradient", "determines", 0.6),
            ("holdup", "slip_velocity", "caused_by", 0.6),
            ("flow_pattern_map", "flow_regime", "predicts", 0.6),
            ("multiphase_flow", "pressure_gradient", "has", 0.6),
            ("gas", "liquid", "flow_together_as", 0.5),
        ),
        procedures=(
            "Determine the flow regime from rates and inclination.",
            "Use the regime to size the line and select lift.",
        ),
    ),
    CorpusEntry(
        key="subsea_and_deepwater",
        discipline="production",
        title="Subsea and deepwater production",
        definition=(
            "In deepwater the wellhead sits on the seabed and the produced fluids "
            "travel to a host through flowlines and a riser. A subsea tree controls "
            "the well, an umbilical supplies power and chemicals, and a manifold "
            "gathers several wells. The host is a floating production unit or a "
            "platform, and it processes and exports the production. Subsea systems "
            "are chosen when a surface platform is impractical, at the cost of "
            "intervention being far harder and more expensive."
        ),
        concepts=(
            ("subsea_well", ("subsea well",)),
            ("subsea_tree", ("subsea tree", "christmas tree")),
            ("production_riser", ("production riser",)),
            ("umbilical", ("umbilical",)),
            ("floating_production", ("floating production", "FPSO")),
            ("subsea_manifold", ("subsea manifold",)),
        ),
        facts=(
            ("subsea_well", "subsea_tree", "controlled_by", 0.7),
            ("subsea_tree", "subsea_manifold", "connects_to", 0.6),
            ("production_riser", "floating_production", "links_to", 0.7),
            ("umbilical", "subsea_tree", "controls", 0.6),
            ("floating_production", "hydrocarbon", "processes", 0.6),
        ),
        procedures=(
            "Gather wells at a manifold and flow them to the host.",
            "Design the riser and umbilical for the water depth.",
        ),
    ),
    CorpusEntry(
        key="data_analytics_and_ml",
        discipline="production",
        title="Petroleum data analytics and machine learning",
        definition=(
            "Data analytics turns the large volume of production, drilling, and "
            "sensor data into decisions. A machine-learning model learns a mapping "
            "from engineered features to an outcome, trained on historical data and "
            "checked on a validation split so it is not merely memorising. Decline "
            "curve analysis forecasts production from history, and predictive "
            "maintenance watches pump vibration to service equipment before it "
            "fails. The model supports the engineer; it never replaces the physics "
            "or the judgement behind the decision."
        ),
        concepts=(
            ("feature_engineering", ("feature engineering",)),
            ("machine_learning_model", ("machine learning model", "ML model")),
            ("training_data", ("training data",)),
            ("validation_split", ("validation split", "holdout")),
            ("predictive_maintenance", ("predictive maintenance",)),
            ("decline_curve_analysis", ("decline curve analysis",)),
        ),
        facts=(
            ("feature_engineering", "machine_learning_model", "feeds", 0.7),
            ("training_data", "machine_learning_model", "trains", 0.7),
            ("validation_split", "machine_learning_model", "evaluates", 0.6),
            ("decline_curve_analysis", "production", "forecasts", 0.6),
            ("predictive_maintenance", "pump", "protects", 0.6),
        ),
        procedures=(
            "Engineer features and split the data into train and validation.",
            "Train, validate, and monitor the model before trusting its output.",
        ),
    ),
    CorpusEntry(
        key="carbon_capture_and_storage",
        discipline="hse",
        title="Carbon capture, utilisation and storage",
        definition=(
            "Carbon capture and storage keeps carbon dioxide out of the atmosphere "
            "by injecting it into a deep geological storage complex. The CO2 is "
            "captured from a plant, compressed, and injected into a porous rock "
            "sealed by an intact caprock. A plume migrates within the storage "
            "complex and is tracked by monitoring wells, while leakage risk is "
            "managed by verifying caprock integrity. The same CO2 can be used to "
            "recover more oil before it is stored permanently."
        ),
        concepts=(
            ("co2_capture", ("CO2 capture", "carbon capture")),
            ("co2_plume", ("CO2 plume",)),
            ("storage_complex", ("storage complex", "storage site")),
            ("monitoring_well", ("monitoring well",)),
            ("leakage_risk", ("leakage risk",)),
            ("caprock_integrity", ("caprock integrity",)),
        ),
        facts=(
            ("co2_capture", "storage_complex", "feeds", 0.7),
            ("co2_plume", "storage_complex", "migrates_within", 0.6),
            ("monitoring_well", "leakage_risk", "detects", 0.6),
            ("caprock_integrity", "leakage_risk", "prevents", 0.7),
            ("co2_plume", "enhanced_oil_recovery", "supports", 0.5),
        ),
        procedures=(
            "Characterise the storage complex and prove caprock integrity.",
            "Inject, monitor the plume, and verify no leakage.",
        ),
        standards=("Storage follows the ISO 27916 and EU CCS directives.",),
    ),
    CorpusEntry(
        key="well_decommissioning",
        discipline="cementing",
        title="Well plugging and decommissioning",
        definition=(
            "When a well reaches the end of its life it is plugged and abandoned so "
            "it can never leak. Cement plugs are placed across each permeable zone "
            "and across the surface, and each plug is verified to form a permanent "
            "barrier. The barriers must isolate every flow path, including the "
            "annulus, and they are tested before the well is signed over. Offshore, "
            "decommissioning ends with the removal or towing away of the platform, "
            "under a regulatory plan."
        ),
        concepts=(
            ("plug_and_abandonment", ("plug and abandonment", "P&A")),
            ("abandonment_barrier", ("abandonment barrier",)),
            ("cement_plug", ("cement plug",)),
            ("platform_removal", ("platform removal",)),
            ("permanent_barrier", ("permanent barrier",)),
        ),
        facts=(
            ("plug_and_abandonment", "abandonment_barrier", "requires", 0.7),
            ("cement_plug", "abandonment_barrier", "forms", 0.7),
            ("abandonment_barrier", "well_integrity", "ensures", 0.7),
            ("platform_removal", "offshore", "concludes", 0.6),
            ("permanent_barrier", "abandonment_barrier", "is_a", 0.7),
        ),
        procedures=(
            "Set and verify a cement plug across every flow path.",
            "Test the barriers, then remove or tow the platform.",
        ),
        standards=("Plugging follows NORSOK D-010 and local regulations.",),
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
    return ontology.add_concept_once(label, frozenset(aliases))


def _concept_for(ontology: Ontology, label: str, aliases: tuple[str, ...]) -> Concept:
    existing = ontology.find_concepts(label)
    return existing[0] if existing else _add_concept(ontology, label, aliases)


def _add_relation(
    ontology: Ontology, source: Concept, target: Concept, relation_type: str, confidence: float
) -> None:
    ontology.add_relation_once(source.id, target.id, relation_type, confidence)


def _seed_glossary_aliases(ontology: Ontology) -> int:
    """Attach each corpus glossary's Arabic term to the concept it names.

    The corpus already records the Arabic word for a term (``("porosity",
    "مسامية")``), but the concept carried only the English label, so an Arabic
    question could never reach it -- the symbolic layer resolved almost none of
    the corpus's own glossary. Each Arabic term is added as an alias of the
    concept the English term names, so the ontology and the corpus agree on what
    a concept is called in either language. Idempotent; returns how many aliases
    were added (an already-present alias is not counted).
    """
    added = 0
    for entry in _ENTRIES:
        for term, arabic in entry.glossary:
            if not arabic.strip():
                continue
            for concept in ontology.find_concepts(term):
                if arabic in concept.aliases:
                    continue
                ontology.add_concept_alias(concept.id, arabic)
                added += 1
    return added


def seed_petroleum_corpus(ontology: Ontology) -> list[Concept]:
    """Plant the corpus concepts and relations, reusing existing nodes.

    Idempotent, and shares nodes with the domain frame and the curriculum, so the
    three seed layers form one graph rather than three disconnected ones. The
    corpus's own Arabic glossary is folded into the concepts as aliases, so the
    Arabic word for a term reaches the same node as its English label.
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
    _seed_glossary_aliases(ontology)
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


def corpus_documents() -> list[Document]:
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
