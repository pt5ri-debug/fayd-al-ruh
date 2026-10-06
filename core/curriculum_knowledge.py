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
    ("النموذج_الرياضي", ("mathematical model",)),
    ("النموذج_العددي", ("numerical model",)),
    ("الطاقة", ("energy",)),
    ("القوة", ("force",)),
    ("الشغل", ("work",)),
    ("الحرارة", ("heat",)),
    ("درجة_الحرارة", ("temperature",)),
    ("المائع", ("fluid",)),
    ("التدفق", ("flow",)),
    ("التجربة", ("experiment",)),
    ("البيانات", ("data",)),
    ("القياس", ("measurement",)),
    ("الجهاز", ("instrument", "device")),
    ("الخطأ", ("error", "uncertainty")),
    ("التقدير", ("estimation",)),
    ("المخاطرة", ("risk",)),
    ("القرار", ("decision",)),
    ("التكلفة", ("cost",)),
    ("القانون", ("law", "principle")),
    ("المعادلة", ("equation",)),
    ("الخوارزمية", ("algorithm",)),
    ("البرنامج", ("program", "software")),
    ("المحاكاة", ("simulation",)),
    ("الحل", ("result",)),
    ("المعدن", ("mineral",)),
    ("الصخر_الرسوبي", ("sedimentary rock",)),
    ("الطبقة_الحاملة", ("reservoir rock", "carrier bed")),
    ("الحوض_الرسوبي", ("sedimentary basin", "basin")),
    ("الهجرة", ("migration",)),
    ("الفخ", ("trap",)),
    ("السائل", ("liquid",)),
    ("الغاز", ("gas",)),
    ("النفط", ("oil", "crude")),
    ("التلوث", ("contamination", "pollution")),
    ("البيئة", ("environment",)),
)


_COURSES: tuple[CourseKnowledge, ...] = (
    CourseKnowledge(
        key="math",
        label="الرياضيات",
        category="أساسيات",
        concepts=(
            ("التفاضل", ("calculus", "differentiation")),
            ("التكامل", ("integration", "integral")),
            ("المعادلات_التفاضلية", ("differential equations",)),
            ("الجبر_الخطي", ("linear algebra",)),
            ("المصفوفات", ("matrix", "matrices")),
            ("المتجهات", ("vectors",)),
            ("المتسلسلات", ("series",)),
        ),
        relations=(
            ("التفاضل", "المعادلات_التفاضلية", "used_in", 0.7),
            ("التكامل", "المعادلات_التفاضلية", "used_in", 0.7),
            ("المصفوفات", "الجبر_الخطي", "part_of", 0.7),
            ("المتجهات", "الجبر_الخطي", "part_of", 0.7),
            ("المعادلات_التفاضلية", "النموذج_الرياضي", "builds", 0.6),
        ),
    ),
    CourseKnowledge(
        key="physics",
        label="الفيزياء",
        category="أساسيات",
        concepts=(
            ("الميكانيكا", ("mechanics",)),
            ("الكهرباء", ("electricity",)),
            ("المغناطيسية", ("magnetism",)),
            ("الموجات", ("waves",)),
            ("الضوء", ("light", "optics")),
            ("الطاقة_الحركية", ("kinetic energy",)),
        ),
        relations=(
            ("القوة", "الميكانيكا", "part_of", 0.7),
            ("الميكانيكا", "الطاقة", "conserves", 0.6),
            ("الكهرباء", "المغناطيسية", "related_to", 0.7),
            ("الموجات", "الطاقة", "transports", 0.6),
            ("الضوء", "الموجات", "is_a", 0.7),
            ("القوة", "الطاقة_الحركية", "produces", 0.6),
        ),
    ),
    CourseKnowledge(
        key="chemistry",
        label="الكيمياء",
        category="أساسيات",
        concepts=(
            ("الذرة", ("atom",)),
            ("الجزيء", ("molecule",)),
            ("الرابطة_الكيميائية", ("chemical bond",)),
            ("التفاعل_الكيميائي", ("chemical reaction",)),
            ("الحموضة", ("acidity", "pH")),
            ("المحلول", ("solution",)),
            ("التحليل_الكيميائي", ("chemical analysis",)),
        ),
        relations=(
            ("الذرة", "الجزيء", "forms", 0.7),
            ("الجزيء", "الرابطة_الكيميائية", "held_by", 0.7),
            ("التفاعل_الكيميائي", "الحموضة", "affects", 0.6),
            ("المحلول", "التفاعل_الكيميائي", "hosts", 0.6),
            ("التحليل_الكيميائي", "المحلول", "measures", 0.6),
        ),
    ),
    CourseKnowledge(
        key="statistics",
        label="الإحصاء_وتحليل_البيانات",
        category="أساسيات",
        concepts=(
            ("الاحتمال", ("probability",)),
            ("التوزيع_الإحصائي", ("distribution",)),
            ("المتوسط", ("mean", "average")),
            ("التباين", ("variance",)),
            ("الانحدار", ("regression",)),
            ("الارتباط", ("correlation",)),
            ("عينة_البيانات", ("sample", "dataset")),
        ),
        relations=(
            ("الاحتمال", "التوزيع_الإحصائي", "describes", 0.7),
            ("التوزيع_الإحصائي", "المتوسط", "has", 0.6),
            ("التوزيع_الإحصائي", "التباين", "has", 0.6),
            ("الانحدار", "الارتباط", "measures", 0.6),
            ("عينة_البيانات", "الانحدار", "trains", 0.6),
            ("عينة_البيانات", "البيانات", "is_a", 0.7),
        ),
    ),
    CourseKnowledge(
        key="computing",
        label="الحوسبة_والبرمجة",
        category="أساسيات",
        concepts=(
            ("الخوارزمية_الحاسوبية", ("computer algorithm",)),
            ("لغة_البرمجة", ("programming language",)),
            ("هيكل_البيانات", ("data structure",)),
            ("قاعدة_البيانات", ("database",)),
            ("الحساب_العددي", ("numerical computing",)),
            ("البرمجيات_الهندسية", ("engineering software",)),
        ),
        relations=(
            ("الخوارزمية_الحاسوبية", "البرنامج", "implemented_by", 0.7),
            ("لغة_البرمجة", "البرنامج", "writes", 0.6),
            ("هيكل_البيانات", "الخوارزمية_الحاسوبية", "supports", 0.6),
            ("قاعدة_البيانات", "البيانات", "stores", 0.7),
            ("البرمجيات_الهندسية", "المحاكاة", "runs", 0.6),
            ("الحساب_العددي", "النموذج_العددي", "computes", 0.6),
        ),
    ),
    CourseKnowledge(
        key="numerical_methods",
        label="الطرق_العددية",
        category="أساسيات",
        concepts=(
            ("التقريب_العددي", ("numerical approximation",)),
            ("التحليل_العددي", ("numerical analysis",)),
            ("التكرار_العددي", ("iteration",)),
            ("الشبكة_الحسابية", ("grid", "mesh")),
            ("استقرار_الحل", ("stability",)),
            ("الاقتران_العددي", ("convergence",)),
        ),
        relations=(
            ("التقريب_العددي", "النموذج_العددي", "builds", 0.6),
            ("التحليل_العددي", "الخطأ", "bounds", 0.6),
            ("التكرار_العددي", "الاقتران_العددي", "seeks", 0.6),
            ("الشبكة_الحسابية", "المعادلات_التفاضلية", "discretizes", 0.6),
            ("استقرار_الحل", "الحل", "qualifies", 0.6),
        ),
    ),
    CourseKnowledge(
        key="geology",
        label="الجيولوجيا_العامة",
        category="علوم_الأرض",
        concepts=(
            ("القشرة_الأرضية", ("crust",)),
            ("الصخور", ("rocks",)),
            ("الصخر_الناري", ("igneous rock",)),
            ("الصخر_المتحول", ("metamorphic rock",)),
            ("الدورة_الجيولوجية", ("rock cycle",)),
            ("الزمن_الجيولوجي", ("geologic time",)),
            ("الطبقات", ("strata", "layers")),
        ),
        relations=(
            ("الصخر_الناري", "الصخور", "is_a", 0.7),
            ("الصخر_المتحول", "الصخور", "is_a", 0.7),
            ("الصخر_الرسوبي", "الصخور", "is_a", 0.7),
            ("الصخور", "الدورة_الجيولوجية", "undergo", 0.6),
            ("الطبقات", "الزمن_الجيولوجي", "dated_by", 0.6),
            ("القشرة_الأرضية", "الصخور", "composed_of", 0.7),
        ),
    ),
    CourseKnowledge(
        key="structural_geology",
        label="الجيولوجيا_البنيوية",
        category="علوم_الأرض",
        concepts=(
            ("الطية", ("fold",)),
            ("الصدع", ("fault",)),
            ("الفالق", ("fracture",)),
            ("الإجهاد_التكتوني", ("tectonic stress",)),
            ("التصاعد_البنائي", ("structural trap",)),
            ("المقطع_الجيولوجي", ("cross-section",)),
        ),
        relations=(
            ("الإجهاد_التكتوني", "الطية", "forms", 0.6),
            ("الإجهاد_التكتوني", "الصدع", "forms", 0.6),
            ("الصدع", "الفالق", "creates", 0.6),
            ("التصاعد_البنائي", "الفخ", "is_a", 0.7),
            ("المقطع_الجيولوجي", "الطبقات", "shows", 0.6),
        ),
    ),
    CourseKnowledge(
        key="sedimentology",
        label="علم_الرسوبيات",
        category="علوم_الأرض",
        concepts=(
            ("الترسيب", ("deposition",)),
            ("الحبيبات", ("grains", "clasts")),
            ("التصنيف_الحجمي", ("grain sorting",)),
            ("السحنة", ("facies",)),
            ("الحجر_الرملي", ("sandstone",)),
            ("الحجر_الجيري", ("limestone",)),
            ("المسامية_الأولية", ("primary porosity",)),
        ),
        relations=(
            ("الترسيب", "الصخر_الرسوبي", "forms", 0.7),
            ("الحبيبات", "التصنيف_الحجمي", "sorted_by", 0.6),
            ("الحجر_الرملي", "الصخر_الرسوبي", "is_a", 0.7),
            ("الحجر_الجيري", "الصخر_الرسوبي", "is_a", 0.7),
            ("السحنة", "الترسيب", "records", 0.6),
            ("المسامية_الأولية", "الترسيب", "results_from", 0.6),
        ),
    ),
    CourseKnowledge(
        key="petroleum_geology",
        label="جيولوجيا_البترول",
        category="علوم_الأرض",
        concepts=(
            ("الصخر_المصدر", ("source rock",)),
            ("المادة_العضوية", ("organic matter",)),
            ("النضوج_الحراري", ("thermal maturation",)),
            ("الهجرة_الأولية", ("primary migration",)),
            ("الهجرة_الثانوية", ("secondary migration",)),
            ("الفخ_البنائي", ("structural trap",)),
            ("الفخ_الطبقي", ("stratigraphic trap",)),
            ("الغطاء_الصخري", ("cap rock", "seal")),
        ),
        relations=(
            ("الصخر_المصدر", "المادة_العضوية", "contains", 0.7),
            ("المادة_العضوية", "النضوج_الحراري", "transformed_by", 0.7),
            ("النضوج_الحراري", "النفط", "generates", 0.7),
            ("النضوج_الحراري", "الغاز", "generates", 0.7),
            ("النفط", "الهجرة_الأولية", "undergoes", 0.6),
            ("الهجرة_الأولية", "الهجرة_الثانوية", "followed_by", 0.6),
            ("الهجرة_الثانوية", "الفخ", "accumulates_in", 0.7),
            ("الغطاء_الصخري", "الفخ", "seals", 0.7),
            ("الفخ_البنائي", "الفخ", "is_a", 0.7),
            ("الفخ_الطبقي", "الفخ", "is_a", 0.7),
        ),
    ),
    CourseKnowledge(
        key="geophysics",
        label="الجيوفيزياء",
        category="علوم_الأرض",
        concepts=(
            ("الموجة_الزلزالية", ("seismic wave",)),
            ("المسح_الزلزالي", ("seismic survey",)),
            ("الانعكاس_الزلزالي", ("seismic reflection",)),
            ("سرعة_الموجة", ("wave velocity",)),
            ("الرصد_الجذبي", ("gravity survey",)),
            ("المغناطيسية_الأرضية", ("geomagnetism",)),
            ("المقطع_الزلزالي", ("seismic section",)),
        ),
        relations=(
            ("المسح_الزلزالي", "الموجة_الزلزالية", "generates", 0.7),
            ("الانعكاس_الزلزالي", "الموجة_الزلزالية", "is_a", 0.6),
            ("الموجة_الزلزالية", "سرعة_الموجة", "has", 0.6),
            ("المقطع_الزلزالي", "الانعكاس_الزلزالي", "built_from", 0.6),
            ("الرصد_الجذبي", "الصخور", "images", 0.5),
            ("المغناطيسية_الأرضية", "الصخور", "images", 0.5),
            ("المسح_الزلزالي", "الفخ", "locates", 0.6),
        ),
    ),
    CourseKnowledge(
        key="fluid_mechanics",
        label="ميكانيكا_الموائع",
        category="هندسة_أساسية",
        concepts=(
            ("اللزوجة", ("viscosity",)),
            ("الكثافة", ("density",)),
            ("ضغط_المائع", ("fluid pressure",)),
            ("قانون_برنولي", ("Bernoulli",)),
            ("دارسي_للتدفق", ("Darcy's law",)),
            ("عدد_رينولدز", ("Reynolds number",)),
            ("التدفق_الصفحي", ("laminar flow",)),
            ("التدفق_المضطرب", ("turbulent flow",)),
        ),
        relations=(
            ("اللزوجة", "التدفق", "resists", 0.7),
            ("الكثافة", "ضغط_المائع", "affects", 0.6),
            ("قانون_برنولي", "ضغط_المائع", "relates", 0.6),
            ("دارسي_للتدفق", "التدفق", "governs", 0.7),
            ("عدد_رينولدز", "التدفق_المضطرب", "predicts", 0.6),
            ("التدفق_الصفحي", "التدفق_المضطرب", "transition_to", 0.6),
            ("دارسي_للتدفق", "النفاذية", "depends_on", 0.6),
        ),
    ),
    CourseKnowledge(
        key="thermodynamics",
        label="الديناميكا_الحرارية",
        category="هندسة_أساسية",
        concepts=(
            ("القانون_الأول_للديناميكا", ("first law",)),
            ("القانون_الثاني_للديناميكا", ("second law",)),
            ("الإنتروبيا", ("entropy",)),
            ("المحتوى_الحراري", ("enthalpy",)),
            ("الحالة_الحركية", ("phase behavior",)),
            ("التوازن_الحراري", ("thermal equilibrium",)),
            ("معادلة_الحالة", ("equation of state",)),
        ),
        relations=(
            ("القانون_الأول_للديناميكا", "الطاقة", "conserves", 0.7),
            ("القانون_الثاني_للديناميكا", "الإنتروبيا", "introduces", 0.7),
            ("المحتوى_الحراري", "الطاقة", "measures", 0.6),
            ("الحالة_الحركية", "معادلة_الحالة", "described_by", 0.6),
            ("التوازن_الحراري", "درجة_الحرارة", "equalizes", 0.6),
            ("معادلة_الحالة", "النفط", "applies_to", 0.5),
        ),
    ),
    CourseKnowledge(
        key="mechanics_of_materials",
        label="مقاومة_المواد",
        category="هندسة_أساسية",
        concepts=(
            ("الإجهاد", ("stress",)),
            ("الانفعال", ("strain",)),
            ("معامل_يونغ", ("Young's modulus",)),
            ("الانحناء", ("bending",)),
            ("الالتواء", ("torsion",)),
            ("الانهيار", ("failure",)),
            ("عامل_الأمان", ("safety factor",)),
        ),
        relations=(
            ("الإجهاد", "الانفعال", "causes", 0.7),
            ("معامل_يونغ", "الإجهاد", "relates", 0.7),
            ("الانحناء", "الإجهاد", "induces", 0.6),
            ("الالتواء", "الإجهاد", "induces", 0.6),
            ("الانهيار", "الإجهاد", "results_from", 0.6),
            ("عامل_الأمان", "الانهيار", "prevents", 0.6),
            ("عامل_الأمان", "التصميم_الهندسي", "guards", 0.6),
        ),
    ),
    CourseKnowledge(
        key="rock_mechanics",
        label="ميكانيكا_الصخور",
        category="هندسة_أساسية",
        concepts=(
            ("الإجهاد_الأرضي", ("in-situ stress", "overburden")),
            ("الضغط_المسامي", ("pore pressure",)),
            ("إجهاد_الشد", ("tensile strength",)),
            ("إجهاد_الضغط", ("compressive strength",)),
            ("التشقق_الطبيعي", ("natural fracture",)),
            ("الانهيار_الصخري", ("rock failure",)),
            ("ضغط_التشقق", ("fracture pressure",)),
        ),
        relations=(
            ("الإجهاد_الأرضي", "ضغط_المسامي", "opposed_by", 0.6),
            ("ضغط_التشقق", "الإجهاد_الأرضي", "depends_on", 0.6),
            ("التشقق_الطبيعي", "إجهاد_الشد", "opens_under", 0.6),
            ("الانهيار_الصخري", "إجهاد_الضغط", "governed_by", 0.6),
            ("الضغط_المسامي", "الانهيار_الصخري", "triggers", 0.5),
            ("إجهاد_الضغط", "عامل_الأمان", "checked_by", 0.5),
        ),
    ),
    CourseKnowledge(
        key="petrophysics",
        label="البتروفيزياء",
        category="تخصصات_البترول",
        concepts=(
            ("مسامية_فعالة", ("effective porosity",)),
            ("نفاذية_مطلقة", ("absolute permeability",)),
            ("التشبع_المائي", ("water saturation",)),
            ("تشبع_الهيدروكربون", ("hydrocarbon saturation",)),
            ("عامل_التكوين", ("formation factor",)),
            ("التوصيلية_الكهربائية", ("electrical conductivity",)),
            ("الانضغاطية", ("compressibility",)),
            ("الخصائص_البتروفيزيائية", ("petrophysical properties",)),
        ),
        relations=(
            ("مسامية_فعالة", "المسامية", "is_a", 0.7),
            ("نفاذية_مطلقة", "النفاذية", "is_a", 0.7),
            ("التشبع_المائي", "الماء", "measures", 0.7),
            ("تشبع_الهيدروكربون", "الهيدروكربون", "measures", 0.7),
            ("عامل_التكوين", "مسامية_فعالة", "depends_on", 0.6),
            ("التوصيلية_الكهربائية", "التشبع_المائي", "indicates", 0.6),
            ("الانضغاطية", "المسامية", "affects", 0.5),
            ("الخصائص_البتروفيزيائية", "الطبقة_الحاملة", "describe", 0.7),
        ),
    ),
    CourseKnowledge(
        key="reservoir_fluids",
        label="خصائص_سوائل_المكمن",
        category="تخصصات_البترول",
        concepts=(
            ("سائل_المكمن", ("reservoir fluid",)),
            ("نسبة_الغاز_للنفط", ("gas-oil ratio", "GOR")),
            ("عامل_حجم_التكوين", ("formation volume factor",)),
            ("ضغط_الفقاعة", ("bubble point",)),
            ("ضغط_الندى", ("dew point",)),
            ("اللزوجة_الفعالة", ("fluid viscosity",)),
            ("التمدد_الحراري", ("thermal expansion",)),
        ),
        relations=(
            ("سائل_المكمن", "النفط", "includes", 0.7),
            ("سائل_المكمن", "الغاز", "includes", 0.7),
            ("سائل_المكمن", "الماء", "includes", 0.6),
            ("نسبة_الغاز_للنفط", "سائل_المكمن", "characterizes", 0.6),
            ("عامل_حجم_التكوين", "سائل_المكمن", "characterizes", 0.6),
            ("ضغط_الفقاعة", "الغاز", "releases", 0.6),
            ("ضغط_الندى", "السائل", "condenses", 0.5),
            ("اللزوجة_الفعالة", "اللزوجة", "is_a", 0.6),
        ),
    ),
    CourseKnowledge(
        key="reservoir_engineering",
        label="هندسة_المكامن",
        category="تخصصات_البترول",
        concepts=(
            ("المعادلة_المواد", ("material balance",)),
            ("مؤشر_الانتاجية", ("productivity index",)),
            ("آلية_الدفع", ("drive mechanism",)),
            ("الدفع_بالغاز_المذاب", ("solution gas drive",)),
            ("الدفع_المائي_الطبيعي", ("natural water drive",)),
            ("توزيع_الضغط", ("pressure distribution",)),
            ("توصيل_المكمن", ("reservoir connectivity",)),
            ("دراسة_جدوى_المكمن", ("reservoir study",)),
        ),
        relations=(
            ("المعادلة_المواد", "الاحتياطي", "estimates", 0.7),
            ("مؤشر_الانتاجية", "معدل_الانتاج", "predicts", 0.6),
            ("آلية_الدفع", "ضغط_المكمن", "maintains", 0.7),
            ("الدفع_بالغاز_المذاب", "آلية_الدفع", "is_a", 0.6),
            ("الدفع_المائي_الطبيعي", "آلية_الدفع", "is_a", 0.6),
            ("توصيل_المكمن", "محاكاة_المكمن", "modeled_by", 0.6),
            ("توزيع_الضغط", "ضغط_المكمن", "maps", 0.6),
            ("دراسة_جدوى_المكمن", "الاحتياطي", "verifies", 0.6),
        ),
    ),
    CourseKnowledge(
        key="drilling_engineering",
        label="هندسة_الحفر",
        category="تخصصات_البترول",
        concepts=(
            ("منظومة_الحفر", ("drilling rig",)),
            ("عمود_الحفر", ("drill string",)),
            ("الوزن_على_الرأس", ("weight on bit",)),
            ("معدل_الحفر", ("rate of penetration",)),
            ("غلاف_البئر", ("well casing",)),
            ("ضغط_التكوين", ("formation pressure", "pore pressure")),
            ("هندسة_الطين", ("mud engineering",)),
            ("هندسة_المسار", ("well trajectory",)),
        ),
        relations=(
            ("منظومة_الحفر", "البئر", "drills", 0.7),
            ("عمود_الحفر", "رأس_الحفر", "connects", 0.6),
            ("الوزن_على_الرأس", "معدل_الحفر", "affects", 0.6),
            ("معدل_الحفر", "الحفر", "measures", 0.6),
            ("غلاف_البئر", "بطانة_البئر", "is_a", 0.6),
            ("هندسة_الطين", "سائل_الحفر", "designs", 0.7),
            ("سائل_الحفر", "ضغط_التكوين", "balances", 0.7),
            ("هندسة_المسار", "البئر_الموجه", "plans", 0.6),
            ("معدل_الحفر", "التكلفة", "affects", 0.5),
        ),
    ),
    CourseKnowledge(
        key="production_engineering",
        label="هندسة_الإنتاج",
        category="تخصصات_البترول",
        concepts=(
            ("تدفق_البئر", ("well flow",)),
            ("معادلة_التدفق_الداخلي", ("inflow performance", "IPR")),
            ("ضغط_قاع_البئر", ("bottom-hole pressure",)),
            ("ضغط_رأس_البئر", ("wellhead pressure",)),
            ("خط_الانتاج", ("flowline",)),
            ("التحكم_في_التدفق", ("flow control",)),
            ("معالجة_الإنتاج", ("production treatment",)),
        ),
        relations=(
            ("تدفق_البئر", "التدفق", "is_a", 0.7),
            ("معادلة_التدفق_الداخلي", "تدفق_البئر", "predicts", 0.7),
            ("ضغط_قاع_البئر", "تدفق_البئر", "drives", 0.7),
            ("ضغط_رأس_البئر", "ضغط_قاع_البئر", "depends_on", 0.6),
            ("خط_الانتاج", "التدفق", "carries", 0.6),
            ("التحكم_في_التدفق", "معدل_الانتاج", "regulates", 0.6),
            ("معالجة_الإنتاج", "الغاز", "separates", 0.6),
            ("معالجة_الإنتاج", "الماء", "separates", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_logging",
        label="تسجيلات_الآبار",
        category="تخصصات_البترول",
        concepts=(
            ("السجل_الجذبي", ("gamma ray log",)),
            ("سجل_المقاومة", ("resistivity log",)),
            ("سجل_الكثافة", ("density log",)),
            ("سجل_الصوت", ("sonic log",)),
            ("سجل_المسامية", ("porosity log",)),
            ("تفسير_السجل", ("log interpretation",)),
            ("معايرة_السجل", ("log calibration",)),
        ),
        relations=(
            ("السجل_الجذبي", "السجل", "is_a", 0.7),
            ("سجل_المقاومة", "السجل", "is_a", 0.7),
            ("سجل_الكثافة", "السجل", "is_a", 0.7),
            ("سجل_الصوت", "السجل", "is_a", 0.7),
            ("سجل_المسامية", "المسامية", "measures", 0.7),
            ("سجل_المقاومة", "التشبع_المائي", "estimates", 0.7),
            ("تفسير_السجل", "الطبقة_الحاملة", "identifies", 0.6),
            ("معايرة_السجل", "تفسير_السجل", "improves", 0.6),
            ("سجل_الكثافة", "المسامية", "measures", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_completions",
        label="إكمال_الآبار",
        category="تخصصات_البترول",
        concepts=(
            ("إكمال_بطانة", ("cased-hole completion",)),
            ("إكمال_مكشوف", ("open-hole completion",)),
            ("التثقيب_المتفجر", ("perforation",)),
            ("المصفاة", ("screen",)),
            ("التغليف_بالحصى", ("gravel pack",)),
            ("صمام_الإنتاج", ("production valve",)),
            ("تجهيز_البئر", ("well equipment",)),
        ),
        relations=(
            ("إكمال_بطانة", "الاكمال", "is_a", 0.7),
            ("إكمال_مكشوف", "الاكمال", "is_a", 0.7),
            ("التثقيب_المتفجر", "التثقيب", "is_a", 0.7),
            ("التثقيب_المتفجر", "إكمال_بطانة", "enables", 0.6),
            ("المصفاة", "الرمل_المصقول", "controls", 0.7),
            ("التغليف_بالحصى", "الرمل_المصقول", "controls", 0.6),
            ("صمام_الإنتاج", "تدفق_البئر", "regulates", 0.6),
            ("تجهيز_البئر", "الاكمال", "supports", 0.6),
        ),
    ),
    CourseKnowledge(
        key="cementing",
        label="إسمنت_الآبار",
        category="تخصصات_البترول",
        concepts=(
            ("ملاط_الاسمنت", ("cement slurry",)),
            ("زمن_التصلب", ("setting time",)),
            ("الإضافات_الاسمنتية", ("cement additives",)),
            ("السدادة_الاسمنتية", ("cement plug",)),
            ("جودة_الربط", ("bond quality",)),
            ("الاسمنت_الخفيف_الوزن", ("lightweight cement",)),
            ("الاسمنت_المتأخر", ("retarder",)),
        ),
        relations=(
            ("ملاط_الاسمنت", "الاسمنت", "is_a", 0.7),
            ("ملاط_الاسمنت", "زمن_التصلب", "has", 0.6),
            ("الإضافات_الاسمنتية", "زمن_التصلب", "control", 0.6),
            ("السدادة_الاسمنتية", "الاسمنت", "is_a", 0.6),
            ("جودة_الربط", "الاسمنت", "verifies", 0.6),
            ("الاسمنت_الخفيف_الوزن", "الاسمنت_الخفيف", "is_a", 0.6),
            ("الاسمنت_المتأخر", "الإضافات_الاسمنتية", "is_a", 0.6),
            ("ملاط_الاسمنت", "بطانة_البئر", "fills", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_stimulation",
        label="تحفيز_الآبار",
        category="تخصصات_البترول",
        concepts=(
            ("التحمض_المصفوفي", ("matrix acidizing",)),
            ("التشقيق_الهيدروليكي_الضخم", ("hydraulic fracturing",)),
            ("المائع_الحامل", ("fracturing fluid",)),
            ("الدعامة", ("proppant",)),
            ("الضغط_الانهياري", ("breakdown pressure",)),
            ("الموصلية_المتضررة", ("formation damage",)),
        ),
        relations=(
            ("التحمض_المصفوفي", "التحمض", "is_a", 0.7),
            ("التشقيق_الهيدروليكي_الضخم", "التشقيق_الهيدروليكي", "is_a", 0.7),
            ("المائع_الحامل", "التشقيق_الهيدروليكي", "used_in", 0.6),
            ("المائع_الحامل", "ضغط_التشقق", "requires", 0.6),
            ("الدعامة", "المائع_الحامل", "carried_by", 0.6),
            ("الضغط_الانهياري", "التشقيق_الهيدروليكي", "initiates", 0.6),
            ("التحمض_المصفوفي", "الموصلية_المتضررة", "removes", 0.6),
        ),
    ),
    CourseKnowledge(
        key="artificial_lift",
        label="الرفع_الصناعي",
        category="تخصصات_البترول",
        concepts=(
            ("الرفع_بالغاز_المستمر", ("gas lift",)),
            ("المضخة_الغاطسة_الكهربائية", ("electrical submersible pump", "ESP")),
            ("المضخة_ذات_القضبان", ("sucker rod pump",)),
            ("الرفع_بالنفخ", ("plunger lift",)),
            ("ضغط_التشغيل", ("operating pressure",)),
            ("كفاءة_المضخة", ("pump efficiency",)),
            ("التصميم_الهيدروليكي", ("lift design",)),
        ),
        relations=(
            ("الرفع_بالغاز_المستمر", "الرفع_بالغاز", "is_a", 0.7),
            ("المضخة_الغاطسة_الكهربائية", "المضخة_الغاطسة", "is_a", 0.7),
            ("المضخة_ذات_القضبان", "المضخة", "is_a", 0.7),
            ("الرفع_بالنفخ", "الرفع_الصناعي", "is_a", 0.6),
            ("ضغط_التشغيل", "الرفع_الصناعي", "drives", 0.6),
            ("كفاءة_المضخة", "المضخة", "measures", 0.6),
            ("التصميم_الهيدروليكي", "الرفع_الصناعي", "optimizes", 0.6),
            ("الرفع_الصناعي", "معدل_الانتاج", "increases", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_testing",
        label="اختبار_الآبار",
        category="تخصصات_البترول",
        concepts=(
            ("اختبار_الضغط_البناء", ("buildup test",)),
            ("اختبار_السقوط", ("drawdown test",)),
            ("مشتقة_الضغط", ("pressure derivative",)),
            ("نموذج_الخزان", ("reservoir model",)),
            ("عامل_الجلد", ("skin factor",)),
            ("توصيلية_المكمن", ("reservoir transmissibility",)),
            ("حدود_المكمن", ("reservoir boundary",)),
        ),
        relations=(
            ("اختبار_الضغط_البناء", "اختبار_الضغط", "is_a", 0.7),
            ("اختبار_السقوط", "اختبار_الضغط", "is_a", 0.7),
            ("مشتقة_الضغط", "اختبار_الضغط_البناء", "analyzed_from", 0.6),
            ("نموذج_الخزان", "اختبار_الضغط", "interpreted_with", 0.6),
            ("عامل_الجلد", "توصيلية_المكمن", "affects", 0.6),
            ("حدود_المكمن", "نموذج_الخزان", "detected_by", 0.6),
            ("توصيلية_المكمن", "النفاذية", "depends_on", 0.6),
        ),
    ),
    CourseKnowledge(
        key="eor",
        label="الاستخلاص_المعزز",
        category="تخصصات_البترول",
        concepts=(
            ("الإزاحة_الكيميائية", ("chemical flooding",)),
            ("الحقن_بالبوليمر", ("polymer flooding",)),
            ("الحقن_بالقلويات", ("alkaline flooding",)),
            ("الحقن_بالغاز_المختلط", ("miscible gas injection",)),
            ("الحقن_الحراري", ("thermal recovery",)),
            ("الإزاحة_بالماء", ("water flooding",)),
            ("كفاءة_الإزاحة", ("sweep efficiency",)),
            ("الشد_السطحي", ("interfacial tension",)),
        ),
        relations=(
            ("الإزاحة_الكيميائية", "الاستخلاص_المعزز", "is_a", 0.7),
            ("الحقن_بالبوليمر", "الإزاحة_الكيميائية", "is_a", 0.6),
            ("الحقن_بالقلويات", "الإزاحة_الكيميائية", "is_a", 0.6),
            ("الحقن_بالغاز_المختلط", "الاستخلاص_المعزز", "is_a", 0.6),
            ("الحقن_الحراري", "الاستخلاص_المعزز", "is_a", 0.6),
            ("الإزاحة_بالماء", "الدفع_المائي", "is_a", 0.6),
            ("كفاءة_الإزاحة", "الشد_السطحي", "depends_on", 0.6),
            ("الحقن_بالبوليمر", "كفاءة_الإزاحة", "improves", 0.6),
            ("الحقن_الحراري", "البخار", "uses", 0.6),
        ),
    ),
    CourseKnowledge(
        key="well_control",
        label="التحكم_في_الآبار",
        category="تخصصات_البترول",
        concepts=(
            ("مانع_الانفجار", ("blowout preventer", "BOP")),
            ("رشح_الضغط", ("pressure kick",)),
            ("طين_القتل_الموزون", ("weighted kill mud",)),
            ("طريقة_الانتظار_والوزن", ("wait and weight",)),
            ("ضغط_الاغلاق", ("shut-in pressure",)),
            ("حاجز_البئر", ("well barrier",)),
            ("اختبار_الضغط_البئري", ("pressure test",)),
        ),
        relations=(
            ("مانع_الانفجار", "رشح_الغاز", "controls", 0.7),
            ("رشح_الضغط", "رشح_الغاز", "is_a", 0.6),
            ("طين_القتل_الموزون", "طين_القتل", "is_a", 0.7),
            ("طريقة_الانتظار_والوزن", "طين_القتل_الموزون", "uses", 0.6),
            ("ضغط_الاغلاق", "الاغلاق", "measures", 0.6),
            ("حاجز_البئر", "حاجز_الضغط", "is_a", 0.6),
            ("اختبار_الضغط_البئري", "حاجز_البئر", "verifies", 0.6),
            ("مانع_الانفجار", "صمام_الامان", "complements", 0.5),
        ),
    ),
    CourseKnowledge(
        key="facilities",
        label="المنشآت_والنقل_والتخزين",
        category="تخصصات_البترول",
        concepts=(
            ("معدات_رأس_البئر", ("wellhead equipment",)),
            ("الفاصل", ("separator",)),
            ("معالجة_الغاز", ("gas processing",)),
            ("خط_الأنابيب", ("pipeline",)),
            ("محطة_الضخ", ("pump station",)),
            ("خزان_التخزين", ("storage tank",)),
            ("قياس_الإنتاج", ("production metering",)),
            ("سلامة_الأنابيب", ("pipeline integrity",)),
        ),
        relations=(
            ("معدات_رأس_البئر", "ضغط_رأس_البئر", "handles", 0.6),
            ("الفاصل", "النفط", "separates", 0.6),
            ("الفاصل", "الغاز", "separates", 0.6),
            ("الفاصل", "الماء", "separates", 0.6),
            ("معالجة_الغاز", "الغاز", "treats", 0.6),
            ("خط_الأنابيب", "النفط", "transports", 0.7),
            ("محطة_الضخ", "خط_الأنابيب", "drives", 0.6),
            ("خزان_التخزين", "النفط", "stores", 0.6),
            ("قياس_الإنتاج", "معدل_الانتاج", "measures", 0.6),
            ("سلامة_الأنابيب", "التفتيش", "requires", 0.6),
        ),
    ),
    CourseKnowledge(
        key="offshore",
        label="الحفر_البحري",
        category="تخصصات_البترول",
        concepts=(
            ("المنصة_البحرية", ("offshore platform",)),
            ("منصة_الحفر_المتحركة", ("jack-up rig",)),
            ("السفينة_الحفارة", ("drillship",)),
            ("رأس_البئر_البحري", ("subsea wellhead",)),
            ("شجرة_عيد_الميلاد", ("subsea christmas tree",)),
            ("خط_الإنتاج_البحري", ("subsea flowline",)),
            ("موانع_الانفجار_البحرية", ("subsea BOP",)),
        ),
        relations=(
            ("المنصة_البحرية", "منظومة_الحفر", "is_a", 0.6),
            ("منصة_الحفر_المتحركة", "المنصة_البحرية", "is_a", 0.6),
            ("السفينة_الحفارة", "المنصة_البحرية", "is_a", 0.6),
            ("رأس_البئر_البحري", "معدات_رأس_البئر", "is_a", 0.6),
            ("شجرة_عيد_الميلاد", "معدات_رأس_البئر", "is_a", 0.6),
            ("خط_الإنتاج_البحري", "خط_الأنابيب", "is_a", 0.6),
            ("موانع_الانفجار_البحرية", "مانع_الانفجار", "is_a", 0.6),
            ("المنصة_البحرية", "الماء", "stands_in", 0.5),
        ),
    ),
    CourseKnowledge(
        key="petroleum_economics",
        label="اقتصاديات_البترول",
        category="تخصصات_البترول",
        concepts=(
            ("صافي_القيمة_الحالية_للمشروع", ("net present value", "NPV")),
            ("معدل_الخصم", ("discount rate",)),
            ("معدل_العائد_الداخلي", ("internal rate of return", "IRR")),
            ("فترة_الاسترداد", ("payback period",)),
            ("تحليل_التدفق_النقدي", ("cash flow analysis",)),
            ("سعر_النفط", ("oil price",)),
            ("ضريبة_الإنتاج", ("royalty", "production tax")),
            ("تحليل_الحساسية", ("sensitivity analysis",)),
        ),
        relations=(
            ("صافي_القيمة_الحالية_للمشروع", "صافي_القيمة_الحالية", "is_a", 0.7),
            ("معدل_الخصم", "صافي_القيمة_الحالية_للمشروع", "discounts", 0.7),
            ("معدل_العائد_الداخلي", "التكلفة", "compares_to", 0.6),
            ("فترة_الاسترداد", "التكلفة", "recovers", 0.6),
            ("تحليل_التدفق_النقدي", "الايراد", "projects", 0.6),
            ("سعر_النفط", "الايراد", "determines", 0.7),
            ("ضريبة_الإنتاج", "التكلفة", "adds_to", 0.6),
            ("تحليل_الحساسية", "المخاطرة", "quantifies", 0.6),
            ("صافي_القيمة_الحالية_للمشروع", "القرار", "informs", 0.6),
        ),
    ),
    CourseKnowledge(
        key="project_management",
        label="إدارة_المشاريع",
        category="تخصصات_البترول",
        concepts=(
            ("دورة_حياة_المشروع", ("project life cycle",)),
            ("نطاق_المشروع", ("project scope",)),
            ("الجدول_الزمني", ("schedule",)),
            ("الميزانية", ("budget",)),
            ("إدارة_المخاطر", ("risk management",)),
            ("التعاقد", ("contracting",)),
            ("ضمان_الجودة", ("quality assurance",)),
            ("أصحاب_المصلحة", ("stakeholders",)),
        ),
        relations=(
            ("دورة_حياة_المشروع", "نطاق_المشروع", "defines", 0.6),
            ("الجدول_الزمني", "الميزانية", "balanced_with", 0.6),
            ("إدارة_المخاطر", "المخاطرة", "manages", 0.7),
            ("التعاقد", "التكلفة", "sets", 0.6),
            ("ضمان_الجودة", "القرار", "supports", 0.5),
            ("أصحاب_المصلحة", "القرار", "influence", 0.6),
            ("الميزانية", "التكلفة", "allocates", 0.6),
        ),
    ),
    CourseKnowledge(
        key="hse",
        label="السلامة_والصحة_والبيئة",
        category="تخصصات_البترول",
        concepts=(
            ("تقييم_المخاطر", ("risk assessment",)),
            ("معدات_الوقاية_الشخصية", ("personal protective equipment", "PPE")),
            ("الحوادث_الصناعية", ("industrial accidents",)),
            ("الانبعاثات_الغازية", ("emissions",)),
            ("إدارة_النفايات", ("waste management",)),
            ("الاستجابة_للطوارئ", ("emergency response",)),
            ("حماية_المياه_الجوفية", ("groundwater protection",)),
            ("الصحة_المهنية", ("occupational health",)),
        ),
        relations=(
            ("تقييم_المخاطر", "المخاطرة", "assesses", 0.7),
            ("معدات_الوقاية_الشخصية", "الحوادث_الصناعية", "reduces", 0.6),
            ("الحوادث_الصناعية", "سلامة_البئر", "threaten", 0.6),
            ("الانبعاثات_الغازية", "البيئة", "pollutes", 0.7),
            ("إدارة_النفايات", "التلوث", "prevents", 0.6),
            ("الاستجابة_للطوارئ", "رشح_الغاز", "mitigates", 0.6),
            ("حماية_المياه_الجوفية", "الماء", "protects", 0.6),
            ("الصحة_المهنية", "الحوادث_الصناعية", "prevents", 0.6),
        ),
    ),
    CourseKnowledge(
        key="professional_ethics",
        label="الأخلاقيات_والمهنية",
        category="تخصصات_البترول",
        concepts=(
            ("المسؤولية_المهنية", ("professional responsibility",)),
            ("النزاهة_العلمية", ("scientific integrity",)),
            ("السلامة_أولاً", ("safety first",)),
            ("الشفافية", ("transparency",)),
            ("تعارض_المصالح", ("conflict of interest",)),
            ("أخلاقيات_البيانات", ("data ethics",)),
            ("المسؤولية_الاجتماعية", ("social responsibility",)),
        ),
        relations=(
            ("المسؤولية_المهنية", "القرار", "guides", 0.7),
            ("النزاهة_العلمية", "البيانات", "protects", 0.6),
            ("السلامة_أولاً", "سلامة_البئر", "prioritizes", 0.7),
            ("الشفافية", "البيانات", "requires", 0.6),
            ("تعارض_المصالح", "القرار", "threatens", 0.6),
            ("أخلاقيات_البيانات", "البيانات", "governs", 0.6),
            ("المسؤولية_الاجتماعية", "البيئة", "protects", 0.6),
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
    by ``core.domain_knowledge`` (e.g. ``المكمن`` or ``المسامية``) is reused rather
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
