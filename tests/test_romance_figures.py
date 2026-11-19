"""The four ready-made figures (woman, girl, man, boy) and imported models.

The Body & shape studio now opens on one of four complete human bodies chosen from
the character being edited, instead of a single generic figure. These tests pin:

* the four slots and their built-in parameters (each a real body at that age/sex,
  one connected skin);
* the character -> slot rule (gender and age, with the documented fallback);
* importing a ready-made model (OBJ/STL/PLY) into a slot, showing it, and
  clearing it back to the built-in;
* that a slot with an imported model renders that model, and one without renders
  the built-in;
* the service wiring and the CLI.
"""

from __future__ import annotations

import pytest
from core.romance.body import default_parameters
from core.romance.figures import (
    ADULT_AGE,
    DEFAULT_SLOT,
    FIGURE_SLOTS,
    SLOT_BOY,
    SLOT_GIRL,
    SLOT_MAN,
    SLOT_WOMAN,
    FigureLibrary,
    slot_for_character,
    slot_parameters,
    slot_parameters_for_character,
    slot_title,
    slot_titles,
)
from core.scene3d_figures import FIGURE_PRESETS, build_figure_mesh


class _Record:
    """A minimal stand-in for a character record (gender + age + name)."""

    def __init__(self, name: str = "", gender: str = "", age: int | None = None) -> None:
        self.name = name
        self.gender = gender
        self.fictional_age = age


class _Bundle:
    def __init__(self, record: _Record) -> None:
        self.character = record


def _write_tetra_obj(path) -> None:
    """A tiny valid OBJ (a tetrahedron) that the reader accepts."""
    path.write_text(
        "o tetra\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 2 0\n"
        "v 0 0 3\n"
        "f 1 2 3\n"
        "f 1 2 4\n"
        "f 1 3 4\n"
        "f 2 3 4\n",
        encoding="utf-8",
    )


# -- the four slots -------------------------------------------------------


def test_there_are_four_slots_in_a_fixed_order() -> None:
    assert FIGURE_SLOTS == (SLOT_WOMAN, SLOT_GIRL, SLOT_MAN, SLOT_BOY)


def test_every_slot_has_a_built_in_figure_and_a_bilingual_title() -> None:
    for slot in FIGURE_SLOTS:
        assert slot in FIGURE_PRESETS
        assert slot_title(slot, "en")
        assert slot_title(slot, "ar")
    assert slot_title("unknown") == "unknown"
    assert dict(slot_titles("ar"))[SLOT_GIRL] == "بنت"


def test_the_figures_are_real_bodies_at_their_age_and_sex() -> None:
    woman = slot_parameters(SLOT_WOMAN)
    man = slot_parameters(SLOT_MAN)
    girl = slot_parameters(SLOT_GIRL)
    boy = slot_parameters(SLOT_BOY)
    # Adults are their gender's build; children are shorter with a bigger head.
    assert woman.build == "feminine" and woman.bust > 0.0
    assert man.build == "masculine" and man.muscle > 0.0
    assert girl.height < woman.height and girl.head_scale > 1.0
    assert boy.height < man.height and boy.head_scale > 1.0
    # A girl is not given an adult bust, and neither child is a "neutral" blob.
    assert girl.bust == 0.0
    assert girl.build == "feminine" and boy.build == "masculine"


def test_every_figure_is_one_connected_shell_not_a_pile_of_parts() -> None:
    for slot in FIGURE_SLOTS:
        params = slot_parameters(slot).with_changes(subdivisions=0)
        mesh = build_figure_mesh(params)
        assert len(mesh.vertices) > 0
        # The body is one connected skin: every vertex of the largest shell is
        # reachable from its first vertex, and that shell is the whole body. The
        # eyes and the two rows of teeth are small closed sub-surfaces inside it,
        # as in any character asset -- never a second body part.
        adjacency: dict[int, set[int]] = {}
        for face in mesh.faces:
            for index in face:
                adjacency.setdefault(index, set()).update(face)
        seen = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        assert len(seen) > 12_000, f"{slot}: the body shell is only {len(seen)} vertices"
        assert len(seen) > 0.97 * len(mesh.vertices), f"{slot} is not one body shell"
        for index in range(len(mesh.vertices)):
            if index not in seen:
                # A detail vertex must belong to a tiny closed surface.
                component = _component_of(adjacency, index)
                assert len(component) < 200, f"{slot}: a {len(component)}-vertex extra shell"


def _component_of(adjacency: dict[int, set[int]], start: int) -> set[int]:
    """The connected component of one vertex, by shared faces."""
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


# -- the character -> slot rule -------------------------------------------


def test_slot_follows_the_characters_gender_and_age() -> None:
    assert slot_for_character(_Record("Amina", "woman", 30)) == SLOT_WOMAN
    assert slot_for_character(_Record("Karim", "man", 40)) == SLOT_MAN
    assert slot_for_character(_Record("Sara", "girl", 10)) == SLOT_GIRL
    assert slot_for_character(_Record("Ali", "boy", 12)) == SLOT_BOY


def test_a_minor_is_never_drawn_with_an_adult_body() -> None:
    # A character recorded under the adult age gets the younger figure of its sex.
    assert slot_for_character(_Record("Mona", "woman", ADULT_AGE - 1)) == SLOT_GIRL
    assert slot_for_character(_Record("Omar", "man", ADULT_AGE - 1)) == SLOT_BOY
    assert slot_for_character(_Record("Mona", "woman", ADULT_AGE)) == SLOT_WOMAN


def test_an_unrecorded_gender_falls_back_to_the_documented_default() -> None:
    assert slot_for_character(_Record("Nour", "", None)) == DEFAULT_SLOT
    assert slot_for_character(_Record("Nour", "non-binary", 25)) == DEFAULT_SLOT
    assert slot_for_character(None) == DEFAULT_SLOT
    # An unrecorded age is shown as an adult (the program cannot invent an age).
    assert slot_for_character(_Record("Amina", "woman", None)) == SLOT_WOMAN


def test_arabic_gender_words_are_understood() -> None:
    assert slot_for_character(_Record("أمينة", "امرأة", 30)) == SLOT_WOMAN
    assert slot_for_character(_Record("كريم", "رجل", 30)) == SLOT_MAN
    assert slot_for_character(_Record("سارة", "بنت", 9)) == SLOT_GIRL


def test_slot_parameters_for_character_names_the_figure_after_the_character() -> None:
    slot, params = slot_parameters_for_character(_Record("Amina", "woman", 30))
    assert slot == SLOT_WOMAN
    assert params.name == "Amina"
    assert params.build == "feminine"


def test_default_parameters_opens_on_the_four_figures() -> None:
    # A blank character opens on the default figure; a woman on the woman's body.
    assert default_parameters().build == slot_parameters(DEFAULT_SLOT).build
    assert default_parameters(_Record("Karim", "man", 30)).build == "masculine"
    # A named preset still wins, so the Templates section can put any body on.
    assert default_parameters(_Record("Karim", "man", 30), preset="slim").build == "masculine"


# -- the library: importing a model ---------------------------------------


def test_a_slot_with_nothing_imported_shows_the_built_in(tmp_path) -> None:
    library = FigureLibrary(tmp_path)
    assert library.has_imported(SLOT_WOMAN) is False
    assert library.imported(SLOT_WOMAN) is None
    assert "built-in" in library.summary(SLOT_WOMAN)
    assert all(not imported for _slot, _title, imported in library.slots())


def test_importing_a_model_stores_it_and_records_where_it_came_from(tmp_path) -> None:
    source = tmp_path / "daz_figure.obj"
    _write_tetra_obj(source)
    library = FigureLibrary(tmp_path / "data")
    entry = library.import_model(SLOT_WOMAN, source, origin="https://www.daz3d.com/x")
    assert entry.slot == SLOT_WOMAN
    assert entry.filename == "daz_figure.obj"
    assert entry.source == "https://www.daz3d.com/x"
    assert library.has_imported(SLOT_WOMAN) is True
    assert library.imported(SLOT_WOMAN) is not None
    assert "imported" in library.summary(SLOT_WOMAN)
    assert library.slots()[0] == (SLOT_WOMAN, "Woman", True)


def test_importing_refuses_a_file_this_build_cannot_read(tmp_path) -> None:
    bad = tmp_path / "not_a_model.obj"
    bad.write_text("this is not a mesh at all\n", encoding="utf-8")
    library = FigureLibrary(tmp_path / "data")
    with pytest.raises(Exception):
        library.import_model(SLOT_MAN, bad)
    # Nothing was stored, so the slot still shows its built-in figure.
    assert library.has_imported(SLOT_MAN) is False


def test_importing_refuses_an_unknown_slot(tmp_path) -> None:
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    library = FigureLibrary(tmp_path / "data")
    with pytest.raises(ValueError):
        library.import_model("dragon", source)


def test_importing_refuses_a_missing_file(tmp_path) -> None:
    library = FigureLibrary(tmp_path / "data")
    with pytest.raises(ValueError):
        library.import_model(SLOT_BOY, tmp_path / "does_not_exist.obj")


def test_clearing_a_slot_returns_it_to_the_built_in(tmp_path) -> None:
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    library = FigureLibrary(tmp_path / "data")
    library.import_model(SLOT_GIRL, source)
    assert library.clear(SLOT_GIRL) is True
    assert library.has_imported(SLOT_GIRL) is False
    # A second clear is an honest no-op.
    assert library.clear(SLOT_GIRL) is False


def test_a_slot_whose_file_went_missing_falls_back_to_the_built_in(tmp_path) -> None:
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    root = tmp_path / "data"
    library = FigureLibrary(root)
    library.import_model(SLOT_WOMAN, source)
    # The user deletes the imported file: the slot must not report a broken model.
    (root / SLOT_WOMAN / "figure.obj").unlink()
    assert library.has_imported(SLOT_WOMAN) is False
    assert library.imported(SLOT_WOMAN) is None


def test_a_corrupt_record_reads_as_nothing_imported(tmp_path) -> None:
    root = tmp_path / "data"
    folder = root / SLOT_MAN
    folder.mkdir(parents=True)
    (folder / "figure.json").write_text("{ not json", encoding="utf-8")
    library = FigureLibrary(root)
    assert library.imported(SLOT_MAN) is None


# -- the library: the mesh and the render ---------------------------------


def test_a_slot_with_no_import_shows_the_built_in_mesh(tmp_path) -> None:
    library = FigureLibrary(tmp_path)
    mesh = library.mesh(SLOT_MAN)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_a_slot_with_an_import_shows_the_imported_mesh(tmp_path) -> None:
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    library = FigureLibrary(tmp_path / "data")
    library.import_model(SLOT_WOMAN, source)
    mesh = library.mesh(SLOT_WOMAN)
    # The tetrahedron normalised to the slot's height has exactly four vertices.
    assert len(mesh.vertices) == 4
    assert len(mesh.faces) == 4


def test_a_render_is_a_real_png_for_every_slot(tmp_path) -> None:
    library = FigureLibrary(tmp_path)
    for slot in FIGURE_SLOTS:
        data = library.render(slot, width=120, height=180)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_render_of_an_imported_slot_uses_the_imported_mesh(tmp_path) -> None:
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    library = FigureLibrary(tmp_path / "data")
    library.import_model(SLOT_WOMAN, source)
    # The scene a slot builds holds the imported mesh (four vertices) and no hair,
    # because a ready-made model carries its own.
    scene = library.scene(SLOT_WOMAN)
    assert len(scene.objects) == 1
    assert len(scene.objects[0].mesh.vertices) == 4


# -- the service wiring ---------------------------------------------------


def test_the_service_resolves_slots_and_imports_models(tmp_path) -> None:
    from gui.romance import RomanceStudioService

    service = RomanceStudioService(root=tmp_path)
    assert service.figure_slot_titles()[0][0] == SLOT_WOMAN
    assert service.character_figure_slot("") == DEFAULT_SLOT
    assert service.figure_slot_parameters(SLOT_BOY).build == "masculine"
    assert "built-in" in service.figure_slot_summary(SLOT_GIRL)
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    entry = service.import_figure_model(SLOT_GIRL, str(source), origin="a gift")
    assert entry.filename == "figure.obj"
    assert service.figure_slot_has_import(SLOT_GIRL) is True
    assert service.clear_figure_slot(SLOT_GIRL) is True
    assert service.figure_slot_has_import(SLOT_GIRL) is False


def test_the_service_renders_a_slot(tmp_path) -> None:
    from gui.romance import RomanceStudioService

    service = RomanceStudioService(root=tmp_path)
    data = service.render_figure_slot(SLOT_MAN, width=100, height=150)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


# -- the CLI --------------------------------------------------------------


def _run_cli(argv: list[str]) -> tuple[int, str]:
    import io
    from contextlib import redirect_stdout

    from cli.romance import main

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(argv)
    return code, buffer.getvalue()


def test_the_cli_lists_the_four_figures(tmp_path) -> None:
    code, out = _run_cli(["body", "--data-root", str(tmp_path), "--figures"])
    assert code == 0
    for title in ("Woman", "Girl", "Man", "Boy"):
        assert title in out


def test_the_cli_imports_and_clears_a_figure(tmp_path) -> None:
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    root = str(tmp_path / "data")
    code, out = _run_cli(
        ["body", "--data-root", root, "--figure", "girl",
         "--import-figure", str(source), "--figure-source", "a gift"]
    )
    assert code == 0
    assert "Imported into the girl figure" in out
    code, out = _run_cli(["body", "--data-root", root, "--figures"])
    assert code == 0
    assert "girl" in out and "imported" in out.lower()
    code, out = _run_cli(
        ["body", "--data-root", root, "--figure", "girl", "--clear-figure"]
    )
    assert code == 0
    assert "Removed the imported model" in out


def test_the_cli_import_needs_a_slot_or_a_character(tmp_path) -> None:
    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    code, _out = _run_cli(
        ["body", "--data-root", str(tmp_path), "--import-figure", str(source)]
    )
    assert code != 0


def test_the_cli_figure_flag_selects_the_body(tmp_path) -> None:
    code, out = _run_cli(
        ["body", "--data-root", str(tmp_path), "--figure", "boy", "--summary"]
    )
    assert code == 0
    assert out.strip()


# -- the window -----------------------------------------------------------


def _display_available() -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
    except Exception:
        return False
    return True


def _wait_for(root, predicate, *, timeout: float = 60.0) -> bool:
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        root.update()
        if predicate():
            return True
        _time.sleep(0.01)
    return predicate()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_window_opens_on_the_characters_own_figure(tmp_path) -> None:
    import gc
    import tkinter as tk

    from core.romance.model import ROLE_COMPANION
    from gui.app import DARK_THEME
    from gui.romance import RomanceStudioService
    from gui.romance_body import BodyStudioDialog

    service = RomanceStudioService(root=tmp_path)
    character = service.create_character(name="Karim", role=ROLE_COMPANION, gender="man")
    root = tk.Tk()
    root.withdraw()
    dialog = None
    try:
        dialog = BodyStudioDialog(
            root, service, character.id, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        # A man opens on the man's figure (not a generic one).
        assert dialog._figure_slot == SLOT_MAN
        assert dialog._params.build == "masculine"
        # The four figure buttons exist and one is marked active.
        assert set(dialog._figure_buttons) == set(FIGURE_SLOTS)
        assert _wait_for(root, lambda: dialog._photo is not None)
        # Switching to another figure changes the body at once.
        dialog._show_figure_slot(SLOT_BOY)
        root.update()
        assert dialog._figure_slot == SLOT_BOY
        assert dialog._params.build == "masculine"
        assert dialog._params.height < 1.5
    finally:
        if dialog is not None:
            dialog._close()
        root.destroy()
        gc.collect()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_window_imports_a_model_into_a_figure(tmp_path) -> None:
    import gc
    import tkinter as tk

    from core.romance.model import ROLE_COMPANION
    from gui.app import DARK_THEME
    from gui.romance import RomanceStudioService
    from gui.romance_body import BodyStudioDialog

    source = tmp_path / "figure.obj"
    _write_tetra_obj(source)
    service = RomanceStudioService(root=tmp_path / "data")
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    root = tk.Tk()
    root.withdraw()
    dialog = None
    try:
        dialog = BodyStudioDialog(
            root, service, character.id, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        assert dialog._figure_slot == SLOT_WOMAN
        # Import through the service (the window's own handler opens a file dialog,
        # which cannot run headless), then load the figure as the editable model.
        service.import_figure_model(SLOT_WOMAN, str(source), origin="a gift")
        dialog._edit_figure_slot()
        assert _wait_for(root, lambda: dialog._mesh is not None)
        assert len(dialog._mesh.vertices) == 4  # the imported tetrahedron
        # Clearing the slot returns the figure to its built-in body.
        dialog._clear_figure_slot()
        root.update()
        assert service.figure_slot_has_import(SLOT_WOMAN) is False
    finally:
        if dialog is not None:
            dialog._close()
        root.destroy()
        gc.collect()
