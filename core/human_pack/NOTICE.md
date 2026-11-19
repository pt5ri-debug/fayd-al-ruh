The human body in the Body & shape studio is built from the MakeHuman base mesh
and a curated set of MakeHuman morph targets. Both are released under **CC0 1.0
Universal** (public domain dedication).

Upstream project: MakeHuman — https://github.com/makehumancommunity/makehuman
Copyright: originally Manuel Bastioni; transferred in 2020 to the MakeHuman
community (Data Collection AB / Joel Palmius). The asset release is documented in
the upstream `LICENSE.md`, section *C. The license for the bundled assets*.

What this project ships, and where it comes from
------------------------------------------------

`core/human_pack/human_pack.bin.z` is a zlib-compressed binary assembled by
`tools/build_human_pack.py` from the following upstream files:

* `makehuman/data/3dobjs/base.obj` — the base body mesh. This project keeps the
  `body` group plus the two eye groups and the two teeth rows, and the joint
  marker groups used to place the rig.
* `makehuman/data/targets/**` — 57 curated morph targets (each an `-incr` /
  `-decr` pair), covering the measurements, torso, hips, pelvis, head, face and
  limbs that the studio's dials drive.

Nothing else is bundled, and no upstream texture, clothing, hair, pose or
expression asset is shipped.

CC0 1.0 Universal summary
-------------------------

To the extent possible under law, the authors have waived all copyright and
related or neighbouring rights to these assets. You may copy, modify, distribute
and use them, even for commercial purposes, all without asking permission.

Full text: https://creativecommons.org/publicdomain/zero/1.0/legalcode

Rebuilding the pack
-------------------

The pack is built once, offline, from a local MakeHuman checkout and committed:

    python tools/build_human_pack.py --makehuman /path/to/makehuman

The program itself never downloads the mesh or the targets: it reads the
committed pack with `zlib` + `struct` only, so it works fully offline.
