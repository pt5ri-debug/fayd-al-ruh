"""Curated Arabic aliases for the petroleum ontology (Phase 49).

The project is Arabic-first, but only 39 of its 495 seeded concepts carried an
Arabic alias -- so an Arabic question reached almost none of the graph, and the
reasoning layers (plan, decision, belief, analogy, connections) never fired for
the language the user actually writes. The clearest symptom was the Arabic
analogical question "هل المسامية مثل النفاذية؟": it reached the belief layer
instead of the analogy, because neither the analogy cue nor the concepts were
recognised.

This module is the honest, bounded fix: a curated English-label -> Arabic-term
table, applied to the ontology through ``Ontology.add_concept_alias`` (which keeps
each concept's id, so every relation survives). It is **presentation-free and
additive** -- it never renames a label, never invents a concept, and never touches
a relation. A label the ontology does not hold is *reported* rather than created,
and an alias that is already present is a no-op, so the pass is idempotent and can
run on every controller start.

The Arabic terms are the standard petroleum-engineering vocabulary (the words an
Arabic-speaking engineer writes), kept as secondary aliases exactly as the
project's convention states.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.concept import Concept
from core.ontology import Ontology
from core.tokenize import normalize_phrase

#: The curated table: an English concept **label** (as seeded, underscores and
#: all) to the Arabic term an engineer would write. Only the label is keyed --
#: never an internal ``CON-…`` id, which is regenerated each run -- so the table
#: survives a rebuild. One concept may have more than one Arabic form; the common
#: synonym is kept beside the primary term.
ARABIC_ALIASES: tuple[tuple[str, str], ...] = (
    # --- rock and petrophysics ------------------------------------------------
    ("porosity", "مسامية"),
    ("effective_porosity", "مسامية فعالة"),
    ("primary_porosity", "مسامية أولية"),
    ("permeability", "نفاذية"),
    ("absolute_permeability", "نفاذية مطلقة"),
    ("relative_permeability", "نفاذية نسبية"),
    ("saturation", "تشبع"),
    ("water_saturation", "تشبع مائي"),
    ("hydrocarbon_saturation", "تشبع هيدروكربوني"),
    ("residual_oil_saturation", "تشبع النفط المتبقي"),
    ("wettability", "قابلية البلل"),
    ("capillary_pressure", "الضغط الشعري"),
    ("pore_throat", "عنق المسام"),
    ("pore_pressure", "ضغط المسام"),
    ("grain_size", "حجم الحبيبات"),
    ("grain_sorting", "فرز الحبيبات"),
    ("grains", "حبيبات"),
    ("matrix", "المصفوفة"),
    ("bulk_volume", "الحجم الكلي"),
    ("core_sample", "عينة لب"),
    ("petrophysical_properties", "الخصائص البتروفيزيائية"),
    ("transition_zone", "المنطقة الانتقالية"),
    ("free_water_level", "مستوى الماء الحر"),
    # --- reservoir ------------------------------------------------------------
    ("reservoir", "مكمن"),
    ("reservoir_rock", "صخر المكمن"),
    ("cap_rock", "صخر الغطاء"),
    ("source_rock", "صخر المصدر"),
    ("net_pay", "السمك المنتج"),
    ("net_to_gross", "نسبة الصافي إلى الإجمالي"),
    ("recovery_factor", "معامل الاستخلاص"),
    ("original_oil_in_place", "النفط الأصلي في المكان"),
    ("reserve", "احتياطي"),
    ("drive_mechanism", "آلية الدفع"),
    ("water_drive", "الدفع المائي"),
    ("gas_cap_drive", "الدفع بقبة غازية"),
    ("solution_gas_drive", "الدفع بالغاز المذاب"),
    ("gravity_drainage", "الصرف بالجاذبية"),
    ("water_influx", "التدفق المائي الداخل"),
    ("water_cut", "نسبة الماء"),
    ("mobility_ratio", "نسبة الحركة"),
    ("sweep_efficiency", "كفاءة الإزاحة"),
    ("voidage_replacement", "تعويض الإنتاج"),
    ("material_balance", "موازنة المواد"),
    ("reservoir_pressure", "ضغط المكمن"),
    ("formation_pressure", "ضغط الطبقة"),
    ("bottom_hole_pressure", "ضغط قاع البئر"),
    ("wellhead_pressure", "ضغط رأس البئر"),
    ("hydrostatic_pressure", "الضغط الهيدروستاتيكي"),
    ("fluid_pressure", "ضغط المائع"),
    ("bubble_point", "نقطة الفقاعة"),
    ("dew_point", "نقطة الندى"),
    ("phase_behavior", "سلوك الأطوار"),
    ("equation_of_state", "معادلة الحالة"),
    ("formation_volume_factor", "معامل حجم الطبقة"),
    ("gas_oil_ratio", "نسبة الغاز إلى النفط"),
    ("viscosity", "لزوجة"),
    ("fluid_viscosity", "لزوجة المائع"),
    ("density", "كثافة"),
    ("compressibility", "انضغاطية"),
    ("interfacial_tension", "التوتر السطحي البيني"),
    ("skin_factor", "معامل الجلد"),
    ("productivity_index", "مؤشر الإنتاجية"),
    ("inflow_performance", "أداء التدفق الداخل"),
    ("nodal_analysis", "التحليل العقدي"),
    ("pressure_buildup", "بناء الضغط"),
    ("buildup_test", "اختبار بناء الضغط"),
    ("drawdown_test", "اختبار الهبوط"),
    ("pressure_transient_test", "اختبار الضغط العابر"),
    ("reservoir_simulation", "محاكاة المكمن"),
    ("reservoir_model", "نموذج المكمن"),
    ("simulator", "المحاكي"),
    ("history_matching", "مطابقة التاريخ"),
    ("upscaling", "الترقية الحجمية"),
    ("grid", "الشبكة"),
    ("numerical_model", "نموذج عددي"),
    # --- fluids and flow ------------------------------------------------------
    ("hydrocarbon", "هيدروكربون"),
    ("heavy_oil", "نفط ثقيل"),
    ("water", "ماء"),
    ("flow", "تدفق"),
    ("multiphase_flow", "التدفق متعدد الأطوار"),
    ("laminar_flow", "التدفق الصفحي"),
    ("turbulent_flow", "التدفق المضطرب"),
    ("radial_flow", "التدفق الشعاعي"),
    ("flow_regime", "نظام التدفق"),
    ("flow_assurance", "ضمان التدفق"),
    ("flow_restriction", "تقييد التدفق"),
    ("slugging", "التدفق القطيعي"),
    ("holdup", "المحتوى السائل"),
    ("darcy_law", "قانون دارسي"),
    # --- drilling -------------------------------------------------------------
    ("drilling", "حفر"),
    ("drilling_fluid", "سائل الحفر"),
    ("mud_weight", "وزن الطين"),
    ("mud_engineering", "هندسة الطين"),
    ("drill_bit", "لقمة الحفر"),
    ("drill_string", "عمود الحفر"),
    ("drilling_rig", "جهاز الحفر"),
    ("rate_of_penetration", "معدل الاختراق"),
    ("weight_on_bit", "الوزن على اللقمة"),
    ("casing", "تغليف"),
    ("well_casing", "تغليف البئر"),
    ("surface_casing", "التغليف السطحي"),
    ("intermediate_casing", "التغليف الوسيط"),
    ("production_casing", "تغليف الإنتاج"),
    ("cement", "أسمنت"),
    ("cement_slurry", "مزيج الأسمنت"),
    ("cement_sheath", "غلاف الأسمنت"),
    ("well_casing_cementing", "تسميت تغليف البئر"),
    ("blowout_preventer", "مانع الانفجار"),
    ("gas_kick", "تدفق غازي مفاجئ"),
    ("kill_mud", "طين الإخماد"),
    ("directional_well", "بئر اتجاهي"),
    ("horizontal_well", "بئر أفقي"),
    ("well_trajectory", "مسار البئر"),
    ("measured_depth", "العمق المقاس"),
    ("true_vertical_depth", "العمق الرأسي الحقيقي"),
    ("inclination", "الميل"),
    ("azimuth", "السمت"),
    ("wellbore_stability", "استقرار جدار البئر"),
    ("formation_damage", "تلف الطبقة"),
    # --- completion and production --------------------------------------------
    ("production", "إنتاج"),
    ("production_rate", "معدل الإنتاج"),
    ("production_decline", "انحدار الإنتاج"),
    ("decline_curve", "منحنى الانحدار"),
    ("artificial_lift", "الرفع الصناعي"),
    ("gas_lift", "الرفع بالغاز"),
    ("sucker_rod_pump", "مضخة القضبان الماصة"),
    ("electrical_submersible_pump", "المضخة الكهربائية الغاطسة"),
    ("esp", "المضخة الكهربائية الغاطسة"),
    ("pump", "مضخة"),
    ("pump_efficiency", "كفاءة المضخة"),
    ("completion", "إكمال البئر"),
    ("perforating", "تثقيب"),
    ("packer", "الحزمة العازلة"),
    ("wellhead_equipment", "معدات رأس البئر"),
    ("christmas_tree", "شجرة رأس البئر"),
    ("separator", "فاصل"),
    ("pipeline", "خط أنابيب"),
    ("storage_tank", "خزان التخزين"),
    ("sand_control", "التحكم في الرمل"),
    ("sand_production", "إنتاج الرمل"),
    ("scale_deposition", "ترسيب القشور"),
    ("wax_deposition", "ترسيب الشمع"),
    ("hydrate", "الهيدرات"),
    ("asphaltene", "الأسفلتين"),
    ("gas_processing", "معالجة الغاز"),
    ("sweetening", "تحلية الغاز"),
    ("dehydration", "إزالة الماء"),
    # --- stimulation ----------------------------------------------------------
    ("well_stimulation", "تحفيز البئر"),
    ("acid_stimulation", "التحفيز الحمضي"),
    ("acid_stimulation", "التحميض"),
    ("matrix_acidizing", "التحميض المصفوفي"),
    ("hydraulic_fracturing", "التكسير الهيدروليكي"),
    ("fracturing_fluid", "سائل التكسير"),
    ("proppant", "المادة الداعمة"),
    ("fracture", "كسر"),
    ("fracture_conductivity", "موصلية الكسر"),
    ("refracturing", "إعادة التكسير"),
    ("stimulated_rock_volume", "حجم الصخر المحفز"),
    # --- enhanced recovery ----------------------------------------------------
    ("enhanced_oil_recovery", "الاستخلاص المعزز للنفط"),
    ("thermal_recovery", "الاستخلاص الحراري"),
    ("steam_flooding", "الإغراق بالبخار"),
    ("polymer_flooding", "الإغراق بالبوليمر"),
    ("chemical_flooding", "الإغراق الكيميائي"),
    ("miscible_gas_injection", "حقن الغاز الممتزج"),
    ("co2_capture", "احتجاز ثاني أكسيد الكربون"),
    # --- geology --------------------------------------------------------------
    ("rock", "صخر"),
    ("sedimentary_rock", "صخر رسوبي"),
    ("igneous_rock", "صخر ناري"),
    ("metamorphic_rock", "صخر متحول"),
    ("sandstone", "حجر رملي"),
    ("limestone", "حجر جيري"),
    ("shale_play", "الطبقة السجيلية"),
    ("strata", "الطبقات"),
    ("formation", "الطبقة"),
    ("lithology", "الليثولوجيا"),
    ("facies", "السحنة"),
    ("horizon", "الأفق"),
    ("crust", "القشرة"),
    ("sedimentary_basin", "حوض رسوبي"),
    ("trap", "مصيدة"),
    ("structural_trap", "مصيدة تركيبية"),
    ("stratigraphic_trap", "مصيدة طبقية"),
    ("fault", "صدع"),
    ("fold", "طية"),
    ("migration", "الهجرة"),
    ("primary_migration", "الهجرة الأولية"),
    ("secondary_migration", "الهجرة الثانوية"),
    ("organic_matter", "المادة العضوية"),
    ("thermal_maturation", "النضوج الحراري"),
    ("geologic_time", "الزمن الجيولوجي"),
    ("play_fairway", "ممر الاستكشاف"),
    ("prospect", "الاحتمال الاستكشافي"),
    ("seismic_survey", "المسح الزلزالي"),
    ("seismic_wave", "الموجة الزلزالية"),
    ("seismic_reflection", "الانعكاس الزلزالي"),
    ("seismic_section", "المقطع الزلزالي"),
    ("seismic_attribute", "السمة الزلزالية"),
    ("rock_cycle", "دورة الصخور"),
    # --- well logs ------------------------------------------------------------
    ("well_log", "سجل البئر"),
    ("log_interpretation", "تفسير السجل"),
    ("log_calibration", "معايرة السجل"),
    ("log_qc", "ضبط جودة السجل"),
    ("density_log", "سجل الكثافة"),
    ("neutron_log", "سجل النيوترون"),
    ("sonic_log", "سجل الصوت"),
    ("gamma_ray_log", "سجل أشعة جاما"),
    ("resistivity_log", "سجل المقاومة"),
    ("electrical_log", "السجل الكهربائي"),
    ("porosity_log", "سجل المسامية"),
    ("cement_bond_log", "سجل ترابط الأسمنت"),
    ("shale_volume", "حجم السجيل"),
    # --- facilities and safety ------------------------------------------------
    ("offshore", "بحري"),
    ("offshore_platform", "منصة بحرية"),
    ("floating_production", "الإنتاج العائم"),
    ("well_integrity", "سلامة البئر"),
    ("well_barrier", "حاجز البئر"),
    ("primary_barrier", "الحاجز الأولي"),
    ("secondary_barrier", "الحاجز الثانوي"),
    ("plug_and_abandonment", "السد والتخلي"),
    ("risk", "خطر"),
    ("risk_assessment", "تقييم المخاطر"),
    ("risk_management", "إدارة المخاطر"),
    ("hazard_analysis", "تحليل الأخطار"),
    ("process_safety_event", "حدث سلامة عملياتي"),
    ("emergency_response", "الاستجابة للطوارئ"),
    ("personal_protective_equipment", "معدات الوقاية الشخصية"),
    ("safety_valve", "صمام الأمان"),
    ("pressure_barrier", "حاجز الضغط"),
    ("leakage_risk", "خطر التسرب"),
    # --- the remaining seeded concepts (Phase 51) -----------------------------
    # The Phase 51 measurement found 258 seeded labels still carried no Arabic
    # alias, so the reasoning layers had no Arabic term to render them with and
    # printed the raw identifier. These are the established Arabic terms for the
    # remaining concepts, grouped by the domain they belong to. Every entry keys
    # the concept's own label (never an id), so the table survives a rebuild.
    # Rock, geology, and geophysics
    ("rocks", "الصخور"),
    ("mineral", "المعدن"),
    ("deposition", "الترسيب"),
    ("burial_history", "تاريخ الدفن"),
    ("cross_section", "المقطع العرضي"),
    ("structural_accumulation", "المصيدة التركيبية"),
    ("fault_interpretation", "تفسير الصدوع"),
    ("reservoir_boundary", "حدود المكمن"),
    ("reservoir_connectivity", "اتصال المكمن"),
    ("natural_fracture", "الكسر الطبيعي"),
    ("caprock_integrity", "سلامة صخرة الغطاء"),
    ("hydrocarbon_charge", "شحن الهيدروكربون"),
    ("amplitude_anomaly", "شذوذ السعة"),
    ("wave_velocity", "سرعة الموجة"),
    ("waves", "الموجات"),
    ("gravity_survey", "المسح الجاذبي"),
    ("geomagnetism", "المغناطيسية الأرضية"),
    ("geothermal_gradient", "التدرج الحراري الأرضي"),
    # Rock mechanics
    ("in_situ_stress", "الإجهاد في الموقع"),
    ("effective_stress", "الإجهاد الفعّال"),
    ("tectonic_stress", "الإجهاد التكتوني"),
    ("stress", "الإجهاد"),
    ("stress_orientation", "اتجاه الإجهاد"),
    ("strain", "الانفعال"),
    ("rock_strength", "مقاومة الصخر"),
    ("rock_failure", "فشل الصخر"),
    ("compressive_strength", "مقاومة الانضغاط"),
    ("tensile_strength", "مقاومة الشد"),
    ("young_s_modulus", "معامل يونغ"),
    ("bending", "الانحناء"),
    ("torsion", "الالتواء"),
    # Reservoir and fluids
    ("reservoir_fluid", "مائع المكمن"),
    ("reservoir_study", "دراسة المكمن"),
    ("reservoir_transmissibility", "انتقالية المكمن"),
    ("compositional_model", "النموذج التركيبي"),
    ("formation_factor", "معامل التكوين"),
    ("archie_equation", "معادلة آرتشي"),
    ("natural_water_drive", "الدفع المائي الطبيعي"),
    ("waterflood", "الإغراق المائي"),
    ("water_flooding", "الإغراق المائي"),
    ("injection_pattern", "نمط الحقن"),
    ("well_count", "عدد الآبار"),
    ("monitoring_well", "بئر المراقبة"),
    ("conformance_control", "ضبط المطابقة"),
    ("co2_plume", "سحابة ثاني أكسيد الكربون"),
    ("miscible_gas", "الغاز الممتزج"),
    ("tight_gas", "الغاز المحكم"),
    ("nGL_recovery", "استخلاص سوائل الغاز الطبيعي"),
    ("decline_curve_analysis", "تحليل منحنى الانحدار"),
    ("production_forecast", "توقع الإنتاج"),
    ("well_flow", "تدفق البئر"),
    ("wellbore_storage", "خزن جوف البئر"),
    ("pressure_derivative", "مشتقة الضغط"),
    ("drawdown", "الهبوط"),
    ("fluid", "المائع"),
    ("liquid", "السائل"),
    ("solution", "المحلول"),
    ("flow_control", "التحكم في التدفق"),
    ("flow_pattern_map", "خريطة أنماط التدفق"),
    ("slip_velocity", "سرعة الانزلاق"),
    ("reynolds_number", "رقم رينولدز"),
    # Drilling and completions
    ("drillship", "سفينة الحفر"),
    ("jack_up_rig", "الحفار ذاتي الرفع"),
    ("floating_rig", "الحفار العائم"),
    ("marine_riser", "الأنبوب الصاعد البحري"),
    ("production_riser", "أنبوب الإنتاج الصاعد"),
    ("horizontal_lateral", "الفرع الأفقي"),
    ("drillers_method", "طريقة الحفار"),
    ("blowout_preventer_stack", "منظومة مانع الانفجار"),
    ("kill_mud_weight", "وزن طين الإخماد"),
    ("weighted_kill_mud", "طين الإخماد الموزون"),
    ("wait_and_weight", "الانتظار والترجيح"),
    ("shut_in", "الإغلاق"),
    ("shut_in_pressure", "ضغط الإغلاق"),
    ("pressure_kick", "اندفاع الضغط"),
    ("annulus_pressure", "ضغط الحلقة"),
    ("burst_load", "حمل الانفجار"),
    ("cased_hole_completion", "إكمال البئر المبطن"),
    ("open_hole_completion", "إكمال البئر المكشوف"),
    ("cement_plug", "سدادة أسمنتية"),
    ("cement_additives", "إضافات الأسمنت"),
    ("cement_slurry_design", "تصميم مزيج الأسمنت"),
    ("light_cement", "الأسمنت الخفيف"),
    ("lightweight_cement", "الأسمنت خفيف الوزن"),
    ("retarder", "المؤخِّر"),
    ("setting_time", "زمن التصلب"),
    ("bond_quality", "جودة الترابط"),
    ("integrity_test", "اختبار السلامة"),
    ("pressure_test", "اختبار الضغط"),
    # Stimulation and sand control
    ("acid_recipe", "وصفة التحميض"),
    ("breakdown_pressure", "ضغط الانهيار"),
    ("fracture_pressure", "ضغط الكسر"),
    ("fracture_gradient", "تدرج كسر الطبقة"),
    ("fracture_geometry", "هندسة الكسر"),
    ("multi_stage_fracturing", "التكسير متعدد المراحل"),
    ("cluster_spacing", "تباعد العناقيد"),
    ("conductivity", "الموصلية"),
    ("gravel", "الحصى"),
    ("gravel_pack", "حزمة الحصى"),
    ("screen", "المصفاة"),
    ("acidity", "الحموضة"),
    # Enhanced recovery
    ("alkaline_flooding", "الإغراق القلوي"),
    ("surfactant_flooding", "الإغراق بالمُخفضات السطحية"),
    ("chemical_eor", "الاستخلاص الكيميائي المعزز"),
    ("thermal_eor", "الاستخلاص الحراري المعزز"),
    ("chemical_injection", "الحقن الكيميائي"),
    # Artificial lift and production
    ("lift_design", "تصميم الرفع"),
    ("continuous_gas_lift", "الرفع الغازي المستمر"),
    ("gas_lift_valve", "صمام الرفع الغازي"),
    ("plunger_lift", "الرفع بالمكبس"),
    ("progressing_cavity_pump", "المضخة التجويفية"),
    ("pump_station", "محطة الضخ"),
    ("production_valve", "صمام الإنتاج"),
    ("production_metering", "قياس الإنتاج"),
    ("production_treatment", "معالجة الإنتاج"),
    ("well_equipment", "معدات البئر"),
    ("facilities_capacity", "سعة المنشآت"),
    ("integrated_asset_model", "نموذج الأصل المتكامل"),
    ("storage_complex", "مجمع الخزن"),
    # Subsea
    ("subsea_bop", "مانع انفجار قاعي"),
    ("subsea_christmas_tree", "شجرة عيد الميلاد القاعية"),
    ("subsea_tree", "الشجرة القاعية"),
    ("subsea_wellhead", "رأس البئر القاعي"),
    ("subsea_well", "البئر القاعي"),
    ("subsea_flowline", "خط التدفق القاعي"),
    ("subsea_manifold", "المشعب القاعي"),
    ("umbilical", "الكابل الممتد"),
    # Well logs and interpretation
    ("log_interpretation_workflow", "سير عمل تفسير المجسات"),
    ("cutoff", "حد القطع"),
    # Facilities, safety, and environment
    ("abandonment_barrier", "حاجز التخلي"),
    ("permanent_barrier", "الحاجز الدائم"),
    ("barrier_management", "إدارة الحواجز"),
    ("bow_tie", "تحليل القوس والربطة"),
    ("layers_of_protection", "طبقات الحماية"),
    ("safety_case", "ملف السلامة"),
    ("safety_factor", "معامل الأمان"),
    ("safety_first", "السلامة أولًا"),
    ("safety_instrumented_system", "نظام السلامة الآلي"),
    ("ignition_source", "مصدر الإشعال"),
    ("industrial_accidents", "الحوادث الصناعية"),
    ("occupational_health", "الصحة المهنية"),
    ("public_safety", "السلامة العامة"),
    ("pipeline_integrity", "سلامة الأنابيب"),
    ("risk_register", "سجل المخاطر"),
    ("predictive_maintenance", "الصيانة التنبؤية"),
    ("inspection", "الفحص"),
    ("failure", "الفشل"),
    ("contamination", "التلوث"),
    ("emissions", "الانبعاثات"),
    ("environment", "البيئة"),
    ("groundwater_protection", "حماية المياه الجوفية"),
    ("waste_management", "إدارة النفايات"),
    ("platform_removal", "إزالة المنصة"),
    # Petroleum economics and project management
    ("cost", "التكلفة"),
    ("cost_estimate", "تقدير التكلفة"),
    ("budget", "الميزانية"),
    ("cash_flow", "التدفق النقدي"),
    ("cash_flow_analysis", "تحليل التدفق النقدي"),
    ("discount_rate", "معدل الخصم"),
    ("internal_rate_of_return", "معدل العائد الداخلي"),
    ("payback_period", "فترة الاسترداد"),
    ("oil_price", "سعر النفط"),
    ("revenue", "الإيراد"),
    ("royalty", "الإتاوة"),
    ("contracting", "التعاقد"),
    ("value_optimization", "تحسين القيمة"),
    ("opportunity_identification", "تحديد الفرص"),
    ("screening_study", "دراسة الفرز"),
    ("development_concept", "مفهوم التطوير"),
    ("project_scope", "نطاق المشروع"),
    ("project_life_cycle", "دورة حياة المشروع"),
    ("stage_gate", "البوابة المرحلية"),
    ("decision_gate", "بوابة القرار"),
    ("stakeholders", "أصحاب المصلحة"),
    ("schedule", "الجدول الزمني"),
    # Data, computing, and mathematics
    ("data", "البيانات"),
    ("database", "قاعدة البيانات"),
    ("data_structure", "بنية البيانات"),
    ("data_ethics", "أخلاقيات البيانات"),
    ("algorithm", "الخوارزمية"),
    ("computer_algorithm", "الخوارزمية الحاسوبية"),
    ("program", "البرنامج"),
    ("programming_language", "لغة البرمجة"),
    ("engineering_software", "البرمجيات الهندسية"),
    ("machine_learning_model", "نموذج تعلّم الآلة"),
    ("feature_engineering", "هندسة السمات"),
    ("training_data", "بيانات التدريب"),
    ("validation_split", "تقسيم التحقق"),
    ("regression", "الانحدار"),
    ("mathematical_model", "النموذج الرياضي"),
    ("numerical_computing", "الحوسبة العددية"),
    ("numerical_analysis", "التحليل العددي"),
    ("numerical_approximation", "التقريب العددي"),
    ("grid_discretization", "تقطيع الشبكة"),
    ("differential_equations", "المعادلات التفاضلية"),
    ("linear_algebra", "الجبر الخطي"),
    ("calculus", "التفاضل والتكامل"),
    ("vectors", "المتجهات"),
    ("equation", "المعادلة"),
    ("iteration", "التكرار"),
    ("convergence", "التقارب"),
    ("estimation", "التقدير"),
    ("inversion_model", "نموذج الانعكاس"),
    ("sensitivity_analysis", "تحليل الحساسية"),
    ("correlation", "الارتباط"),
    ("distribution", "التوزيع"),
    ("probability", "الاحتمال"),
    ("variance", "التباين"),
    ("mean", "المتوسط"),
    ("series", "المتسلسلة"),
    ("result", "النتيجة"),
    ("error", "الخطأ"),
    ("measurement", "القياس"),
    ("instrument", "الجهاز"),
    ("sample", "العينة"),
    ("experiment", "التجربة"),
    ("chemical_analysis", "التحليل الكيميائي"),
    ("integration", "التكامل"),
    ("stability", "الاستقرار"),
    # Physics and chemistry
    ("atom", "ذرة"),
    ("molecule", "الجزيء"),
    ("chemical_bond", "الرابطة الكيميائية"),
    ("chemical_reaction", "التفاعل الكيميائي"),
    ("energy", "الطاقة"),
    ("kinetic_energy", "الطاقة الحركية"),
    ("heat", "الحرارة"),
    ("work", "الشغل"),
    ("force", "القوة"),
    ("light", "الضوء"),
    ("electricity", "الكهرباء"),
    ("magnetism", "المغناطيسية"),
    ("electrical_conductivity", "الموصلية الكهربائية"),
    ("pressure", "الضغط"),
    ("pressure_distribution", "توزيع الضغط"),
    ("pressure_gradient", "تدرج الضغط"),
    ("operating_pressure", "ضغط التشغيل"),
    ("temperature", "درجة الحرارة"),
    ("enthalpy", "الإنثالبي"),
    ("entropy", "الإنتروبيا"),
    ("first_law", "القانون الأول"),
    ("second_law", "القانون الثاني"),
    ("law", "القانون"),
    ("darcy_s_law", "قانون دارسي"),
    ("bernoulli", "برنولي"),
    ("mechanics", "الميكانيكا"),
    ("engineering_design", "التصميم الهندسي"),
    ("thermal_equilibrium", "الاتزان الحراري"),
    ("thermal_expansion", "التمدد الحراري"),
    ("thermal_insulation", "العزل الحراري"),
    ("liquefaction", "الإسالة"),
    # Professional practice and ethics
    ("professional_responsibility", "المسؤولية المهنية"),
    ("social_responsibility", "المسؤولية الاجتماعية"),
    ("scientific_integrity", "النزاهة العلمية"),
    ("transparency", "الشفافية"),
    ("conflict_of_interest", "تضارب المصالح"),
    ("quality_assurance", "ضمان الجودة"),
    ("as_low_as_reasonably_practicable", "أقل ما يمكن تحقيقه عمليًا"),
    ("decision", "القرار"),
)

#: Arabic script range, for the honest coverage report. Used instead of a full
#: Unicode script property so the module stays dependency-free.
_ARABIC_RANGE = ("\u0600", "\u06ff")


@dataclass(frozen=True)
class GlossaryReport:
    """What one application of the glossary did."""

    added: tuple[str, ...]
    """Labels that gained an Arabic alias in this call."""

    already_present: tuple[str, ...]
    """Labels that already carried an Arabic alias (or the one being added)."""

    unknown: tuple[str, ...]
    """Table labels the ontology does not hold -- reported, never invented."""

    @property
    def total(self) -> int:
        return len(self.added) + len(self.already_present)

    def describe(self) -> str:
        return (
            f"{len(self.added)} Arabic alias(es) added, "
            f"{len(self.already_present)} already present, "
            f"{len(self.unknown)} unknown label(s) skipped."
        )


def has_arabic(text: str) -> bool:
    """Whether ``text`` contains any Arabic-script character."""
    return any(_ARABIC_RANGE[0] <= char <= _ARABIC_RANGE[1] for char in text)


def _concept_for_label(ontology: Ontology, label: str) -> Concept | None:
    """The concept whose **label** is exactly ``label``, or ``None``.

    A label is the table's key, and it must resolve to the concept it names --
    never to a *different* concept that happens to share an alias. ``acidizing``
    is an alias of both ``well_stimulation`` and ``acid_stimulation``, so a lookup
    by that word returned the wrong concept and the Arabic term was silently lost;
    matching the label exactly (case-insensitively) is the honest resolution.
    """
    wanted = label.casefold()
    for concept in ontology.concepts.values():
        if concept.label.casefold() == wanted:
            return concept
    return None


def apply_arabic_aliases(ontology: Ontology, table: tuple[tuple[str, str], ...] = ARABIC_ALIASES) -> GlossaryReport:
    """Add the curated Arabic aliases to ``ontology`` in place, idempotently.

    Each entry is resolved by its **label** (never an id, and never a shared
    alias), so the pass works on a freshly seeded graph and on a restored one and
    always targets the concept it names. A label the ontology does not hold is
    collected in ``unknown`` and skipped -- nothing is invented. A term already
    present is a no-op, so calling this twice changes nothing; a concept may carry
    more than one Arabic term, so a synonym is added beside the primary one. The
    concept's id is preserved, so every relation that references it stays intact.
    """
    added: list[str] = []
    already: list[str] = []
    unknown: list[str] = []
    for label, arabic in table:
        term = arabic.strip()
        if not term:
            continue
        concept = _concept_for_label(ontology, label)
        if concept is None:
            unknown.append(label)
            continue
        if term in concept.aliases:
            already.append(label)
            continue
        ontology.add_concept_alias(concept.id, term)
        added.append(label)
    return GlossaryReport(tuple(added), tuple(already), tuple(unknown))


def missing_arabic_labels(ontology: Ontology) -> tuple[str, ...]:
    """The seeded concept labels that still carry no Arabic alias, sorted.

    Honest by construction: it reports the remaining gap rather than hiding it, so
    a caller (or a test) can see exactly how far the glossary reaches.
    """
    labels = [
        concept.label
        for concept in ontology.concepts.values()
        if not has_arabic(concept.label)
        and not any(has_arabic(alias) for alias in concept.aliases)
    ]
    return tuple(sorted(labels))


def arabic_alias_coverage(ontology: Ontology) -> tuple[int, int]:
    """``(concepts with an Arabic alias, total concepts)`` -- the coverage figure."""
    total = len(ontology.concepts)
    covered = sum(
        1
        for concept in ontology.concepts.values()
        if has_arabic(concept.label) or any(has_arabic(alias) for alias in concept.aliases)
    )
    return covered, total


def normalize_arabic_term(term: str) -> str:
    """The project's normalised form of an Arabic term, for a caller that matches."""
    return normalize_phrase(term)


__all__ = [
    "ARABIC_ALIASES",
    "GlossaryReport",
    "apply_arabic_aliases",
    "arabic_alias_coverage",
    "has_arabic",
    "missing_arabic_labels",
    "normalize_arabic_term",
]
