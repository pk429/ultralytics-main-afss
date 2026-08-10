from __future__ import annotations

import base64
import mimetypes
import os
import random
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps, JpegImagePlugin

# ========= 修改这里 =========
BASE_URL = os.getenv("IMAGE_API_BASE_URL", "http://47.89.248.63/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-c08e618a035fb7d0135c7d58f40422d33c98cedc663d20433602bb4c92047cf0")
MODEL = os.getenv("IMAGE_EDIT_MODEL", "gpt-image-2")

# 可以填单张背景图，也可以填一个背景图目录。不需要 YOLO label txt。
INPUT_SOURCE = "/mnt/sda1/xzm/datasets/cargoship_visible_obb_dataset/20260723/images/"
OUTPUT_DIR = "/mnt/sda1/xzm/datasets/ship_transfer/20260804_cargoship_20260723_mix/"

# 每张背景图生成几张不同随机样本。
OUTPUTS_PER_IMAGE = 2

# 每张输出图中合成几组过驳船组；每组固定为 货船 - 接驳船 - 货船。
TRANSSHIPMENT_GROUPS_PER_IMAGE = 4

# 输出命名方式:
# "suffix" 生成 image_transshipment_001.jpg
# "original" 仅适合 OUTPUTS_PER_IMAGE=1，会覆盖/复用原文件名
OUTPUT_NAME_MODE = "suffix"
OUTPUT_NAME_SUFFIX = "_transshipment"
OVERWRITE_EXISTING = False

# None 或 "original" 表示保持输入原图尺寸；也可以填固定尺寸，例如 "1920x1080"。
IMAGE_SIZE = None
ENFORCE_OUTPUT_SIZE = True

MAX_INPUT_IMAGES = None
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


BASE_PROMPT = """
Edit the provided UAV / drone river surveillance background into one realistic synthetic
dataset image of inland ship-to-ship transshipment or anchorage transshipment.

Preserve the original drone top-down or oblique-aerial viewpoint, river surface, shoreline,
riverbanks, roads, bridges, vegetation, buildings, vehicles, existing boats and ships, haze,
lighting direction, shadows, color temperature, sharpness, noise, compression artifacts, and
surveillance-video texture. Do not change the background geometry.

The input image may already contain boats, ships, barges, workboats, docks, wake trails, or
moored vessels. These original vessels are part of the background and must remain unchanged:
do not erase, replace, repaint, resize, move, deform, cover, merge with, or convert any
existing vessel into a new transshipment group. Add new transshipment groups only in empty
water areas that do not overlap original boats or ships. Leave a visible clean-water buffer
between every newly generated group and every original vessel, original wake, docked boat, or
moored ship. The synthetic groups must not contaminate real data by touching or partially
covering any original vessel pixels.

Add only new realistic vessels and their local water-contact effects on empty river water.
Never place any new vessel on land, road, bank, bridge, vegetation, building roof, vehicle,
sky, or on top of an original vessel. Keep all added vessels physically sitting in the water,
with correct perspective, scale, occlusion, soft shadow, hull-water contact, reflection,
small ripples, and weak wake.

The target behavior is transshipment: the generated target must contain exactly four
separate working three-vessel groups. Every group must follow the order cargo ship -
transfer boat - cargo ship. The two side vessels are both cargo vessels, and their scale
relationship should be randomized across groups: some groups may have two large cargo
vessels, while other groups may have one large bulk carrier / container ship on one side and
one smaller cargo barge / lighter cargo vessel on the other side. The transfer boat remains
the middle work platform. In each group, all three vessels must be parallel alongside each
other, tightly moored or fendered together, with the transfer boat sandwiched between the two
cargo vessels or tightly alongside both hulls. Use only narrow water gaps or realistic hull
contact inside each group; do not scatter the vessels. The gaps inside a group must be
physically plausible: narrow, even, and protected by fenders or mooring contact. Do not let
hulls overlap, intersect, crush into each other, bend, fuse, or leave an implausibly large
empty gap. Each group must look stopped or moving very slowly in an anchored working posture.
If a crane is generated, make it a real maritime
crane or excavator-like loading crane on the transfer boat or crane barge, with a boom
reaching toward one of the open cargo holds or container decks.

The middle transfer boat must be a realistic inland transshipment vessel, not a random small
boat. It should look like a low steel lightering barge, crane barge, deck-work boat, or
small cargo-transfer vessel with a flat working deck, side fenders, mooring ropes, low
freeboard, compact wheelhouse, crane pedestal, excavator-like loading arm, conveyor, grab
bucket, deck winch, or loading equipment when scale allows. It must not look like a speed
boat, yacht, rescue boat, passenger ferry, abstract object, broken hull fragment, or
unrelated decorative boat.

Include the real-world side-lightering visual pattern shown by typical aerial anchorage
operations: a large cargo ship with a random realistic industrial hull color, several
rectangular open cargo holds with hatch covers slid or folded aside, exactly one low middle
transfer/lightering boat pressed between two cargo vessels, and a blue/orange/yellow
floating crane or deck crane platform on that middle boat. The crane boom may be blue,
orange, yellow, gray, or weathered steel and should angle diagonally across the open cargo
holds. This pattern should look like practical cargo handling at anchor, not ordinary vessel
parking.

Use the reference-image visual logic as engineering guidance, not as an exact copy. The
important cues are: a large bulk carrier with repeated rectangular open holds and hatch
covers slid or folded aside; a second cargo vessel on the other side; exactly one lower middle
transfer/lightering/crane boat between the two cargo vessels; a blue, yellow, orange, or
gray floating-crane platform mounted on the middle boat; long crane booms angled over open
holds; grab buckets, hooks, spreaders, hoppers, conveyor bridges, or chutes oriented between
vessels; taut mooring lines; side fenders; and small localized swirl/ripple patterns in the
working gaps. Do not add extra tugboats or extra workboats as part of a group. Do not copy
sample-image watermarks, captions, logos, visible website text, or low horizon composition.
Adapt these cues to the input UAV / drone perspective.

The vessels may be different inland freight types and random realistic colors: red-brown,
dark blue, black, dark green, olive, gray, white cabin blocks, rusted steel, faded cyan,
weathered orange, worn industrial paint, and mixed weathered hulls. Do not force all cargo
ships to be red or black.
Cargo may include exposed black coal, yellow sand, gray gravel, mixed aggregate, earthwork
soil, open bulk material, or stacked shipping containers. Except for container-transfer
scenes, cargo vessels must be in open-hatch loading/unloading state: cargo holds should be
open, visible, and not sealed by full hatch covers. Bulk cargo surfaces must look real from a
UAV view: matte, granular or compacted, with low mounds, loader marks, edge buildup, subtle
clumps, dust, or uneven tones. Container cargo must appear as realistic
rectangular ISO container stacks or rows on deck: red, blue, green, gray, white, or weathered
containers with correct perspective, small shadows, lane spacing, and scale. Do not make
cargo look like flat paint, water reflection, or a tarp unless it is explicitly a partial
cover.

Keep cargo logic consistent inside every group. A container-transfer group should involve
container vessels or general-cargo/container-feeder vessels on both sides, using a spreader
or container-capable crane. A sand/gravel/coal/soil bulk-transfer group should involve bulk
cargo vessels on both sides, using a grab bucket, conveyor, hopper, chute, excavator-like
handler, or bulk crane. Do not mix incompatible cargo operations inside one group: no
container ship transferring to a sand barge, no coal barge transferring to a container deck,
no container spreader over loose sand, and no bulk grab dumping into a container stack.

Most importantly, every added group must show visible operational evidence of cargo
transfer, not merely three parked vessels. Include plausible work traces such as a crane boom
rotated toward a cargo hold, a raised or lowered grab bucket, a suspended container or bulk
grab load, uneven cargo surface where material has been scooped, partial empty areas in a
hold, scattered coal/sand/gravel on deck edges, ropes/fenders under load, small localized
water disturbance between hulls, and cargo-placement gaps. For container cargo, do not fill
the deck completely; leave missing slots, uneven rows, a partially moved container, or a
container being lifted. For sand, coal, gravel, or soil, show mound height variation,
scooped pits, ridges, fresh heap slopes, and loader/grab marks.

Strictly avoid text, watermarks, arrows, detection boxes, labels, segmentation masks,
cartoon, illustration, CGI, unrealistic toy-like boats, dramatic disaster scenes, fire,
smoke, people close-ups, excessive commercial-photography styling, and any global
background rewrite.
"""


TRANSSHIPMENT_SCENARIOS = [
    {
        "name": "large_ship_lightering_barge_crane",
        "prompt": """
Use this style for the four transshipment groups: each group is a realistic three-vessel
lightering operation in fixed order cargo ship - transfer boat - cargo ship. Randomize the
two side cargo vessels: some groups can use two large cargo ships, and some groups can use
one large bulk carrier plus one smaller cargo barge or lighter cargo vessel. The transfer
boat must be in the middle, tightly parallel between the two cargo vessels, like a working
lightering boat physically servicing both sides. Add a crane or excavator boom on some
transfer boats if scale allows, with the boom reaching toward an open cargo hold. Use
low-speed or anchored water contact: minimal wake, small ripples, tight mooring spacing,
fenders or ropes if visible, and no collision damage.
""",
    },
    {
        "name": "reference_style_side_lightering",
        "prompt": """
Use this reference-style operation for the four transshipment groups: each group should
resemble a real aerial anchorage lightering scene. Put one larger cargo ship with a random
realistic hull color and rectangular open holds with hatch covers slid aside on one side, a
low lightering barge or transfer boat tightly alongside it, and a second cargo vessel/barge
tightly alongside the transfer boat. The second cargo vessel may also be large, or it can be
a smaller sand barge, coal barge, container feeder, or lighter cargo vessel. Non-container
cargo vessels must show open holds in active loading/unloading state. The middle vessel
should carry a visible blue, orange, yellow, gray, or weathered steel crane pedestal, grab
crane, or conveyor-like loading arm, with a boom angled diagonally over the open cargo holds.
Keep all three hulls parallel and physically close, with fenders, mooring ropes, narrow
water gaps, and weak stationary ripples if scale allows.
""",
    },
    {
        "name": "two_barges_with_transfer_boat_between",
        "prompt": """
Use this style for the four transshipment groups: each group has two different bulk cargo
barges with a smaller transfer boat parallel between them. The order must be cargo barge -
transfer boat - cargo barge, with all three hulls tightly side-by-side and nearly parallel.
The smaller boat should be visibly used for transshipment, not simply passing through. Add a
small crane, deck crane, or loading arm to some groups if scale allows. Every group must be
stationary or very slow, with weak ripples and narrow working gaps.
""",
    },
    {
        "name": "anchorage_coal_transfer",
        "prompt": """
Use this style for the four transshipment groups: each group is an anchorage coal-transfer
three-vessel group with one coal-loaded cargo barge, one smaller transfer boat in the middle,
and a second cargo barge or cargo ship on the other side. All three vessels in each group
must be parallel and tightly alongside. Exposed matte black coal should be visible in at
least some cargo holds. Add crane/boom or loading equipment on some middle transfer boats if
scale allows. Keep it realistic for inland river surveillance, not an ocean-port close-up.
""",
    },
    {
        "name": "sand_gravel_lightering",
        "prompt": """
Use this style for the four transshipment groups: each group is a sand or gravel
transshipment three-vessel group with one yellow-sand or gray-gravel cargo barge, one smaller
transfer boat in the middle, and a second cargo vessel tightly parallel on the other side.
The cargo should be visible as matte sand/gravel with subtle mound slopes, loader marks, and
irregular edges. Use different hull colors and realistic weathering, but keep every
three-vessel group aligned and tightly side-by-side.
""",
    },
    {
        "name": "floating_crane_between_cargo_vessels",
        "prompt": """
Use this style for the four transshipment groups: each group is a floating-crane operation in
fixed order cargo vessel - crane/transfer boat - cargo vessel. The middle vessel should be
smaller and may carry a crane or loading boom angled toward one of the cargo holds. The two
cargo vessels can differ in size, color, and cargo type. The spatial relationship in every
group must clearly read as transfer work: all three are parallel, tightly moored, slow or
anchored, with plausible crane reach.
""",
    },
    {
        "name": "small_river_transfer_cluster",
        "prompt": """
Use this style for the four transshipment groups: each group is a compact inland-river
three-vessel transshipment group using two medium cargo barges and one smaller transfer boat
between them. Optionally include a small deck crane or excavator-like boom on some middle
transfer boats. The groups should be modest in scale, naturally integrated into the water
surface, and suitable for a far or mid-distance UAV monitoring frame. The hulls in every
group must remain parallel and tightly alongside each other.
""",
    },
    {
        "name": "container_barge_parallel_transfer",
        "prompt": """
Use this style for the four transshipment groups: each group is a realistic container-transfer
three-vessel group with one larger container cargo vessel, one smaller transfer/crane boat in
the middle, and a second cargo vessel tightly parallel on the other side. Add rows or low
stacks of colored shipping containers on at least some cargo vessel decks, and crane booms or
loading arms on some middle boats reaching toward container decks if scale allows. The three
hulls in every group must be side-by-side, almost parallel, close enough to look moored for
cargo transfer, with subtle fenders, ropes, narrow water gaps, and weak stationary ripples.
""",
    },
    {
        "name": "general_cargo_same_type_lightering",
        "prompt": """
Use this style for the four transshipment groups: each group is a same-business lightering
operation. Both side cargo vessels must carry compatible cargo types: either both are
open-hold bulk vessels with coal/sand/gravel/soil/aggregate, or both are general-cargo /
container-capable vessels with compatible deck cargo. Do not pair a container vessel with a
sand/coal/gravel barge in the same group. The transfer boat in the middle must carry matching
equipment for that cargo type, and all three hulls must remain parallel, close-alongside, and
slow or anchored.
""",
    },
]


VESSEL_VARIANTS = [
    """
Use a weathered rust-red open-hold bulk barge with black rubber fenders, scuffed side paint,
a small white stern cabin, and exposed dark bulk cargo.
""",
    """
Use a dark blue steel cargo vessel with faded paint, low side coamings, worn deck edges,
and a compact aft wheelhouse.
""",
    """
Use a charcoal-black or gray low-freeboard barge with a long rectangular hold, dull steel
highlights, rust streaks near the waterline, and utilitarian inland-river proportions.
""",
    """
Use a dark green or olive aggregate barge with dusty deck edges, gray gravel or yellow sand
inside the hold, and small workboat details.
""",
    """
Use a brown-red cargo barge with sectional hatch rails, open cargo holds, hatch covers slid
aside or folded to the ends, faded paint, and visible loading-wear marks.
""",
    """
Use a smaller blue-red workboat or transfer boat with a tiny cabin, side fenders, deck
equipment, and realistic scale relative to the larger cargo vessels.
""",
    """
Use a container cargo barge or small container vessel with low stacks of red, blue, green,
gray, and white shipping containers, weathered deck paint, compact bridge cabin, and visible
container rows aligned with the vessel length.
""",
    """
Use a crane barge or transfer boat with a compact orange, yellow, or blue deck crane,
stabilizing equipment, side fenders, mooring ropes, and a boom angled toward a nearby cargo
hold or container deck.
""",
]


TRANSFER_BOAT_REALISM_RULES = [
    """
Use the reference-style side-lightering feature: a blue floating crane platform or blue deck
crane mounted on the middle transfer boat, with a long blue boom angled diagonally toward the
large cargo ship's rectangular open hold. The transfer boat should be pressed along the cargo
ship side, not separated in open water.
""",
    """
The middle transfer boat can be a low dark, red-brown, blue-gray, green, or weathered steel
lightering barge with an open empty hold or working deck, tied alongside a larger cargo ship
with any realistic industrial hull color. Add black rubber fenders and a thin water gap to
show realistic side contact.
""",
    """
The middle transfer boat should be a low steel lightering barge with a rectangular working
deck, black rubber side fenders, mooring ropes, small bollards, and a compact cabin. It is
usually the lowest or narrowest vessel in the group, but it must still be large enough to
plausibly carry transfer equipment.
""",
    """
The middle vessel can be a floating crane barge: flat deck, low freeboard, side fenders,
stabilizing legs or deck equipment, and one compact crane pedestal with a boom reaching
toward a cargo hold. Keep the crane size plausible and integrated with the deck.
""",
    """
The middle vessel can be a small cargo-transfer workboat with a deck crane, conveyor, grab
bucket, hopper, or loading arm. Its deck equipment should point toward one of the larger
cargo vessels, clearly supporting cargo transfer rather than normal navigation.
""",
    """
The transfer boat must have realistic mooring contact: narrow water gaps, aligned hull sides,
rubber fenders, ropes, or small contact shadows between vessels. It should not float loosely
far away from the cargo vessels.
""",
    """
Avoid unrealistic transfer-boat shapes: no speedboat hulls, leisure boats, passenger
launches, rescue craft, oversized cabin blocks, toy-like colors, random floating platforms,
or malformed geometry. Keep it industrial, weathered, and consistent with inland river
workboat proportions.
""",
]


CRANE_STRUCTURE_VARIATION_RULES = [
    """
Use a floating crane barge with a lattice-boom crane: triangular truss boom, vertical hoist
cables, grab bucket or hook hanging below, counterweight block on the rear of the crane
house, and a low blue/yellow/orange crane pedestal fixed to the middle transfer boat.
""",
    """
Use a pedestal slewing crane on the transfer boat: compact rotating tower, box boom or
lattice boom, small operator cabin, base ring, hoist cables, and a boom swung left or right
toward an open cargo hold.
""",
    """
Use an excavator-like material handler on the transfer boat: tracked or pedestal base,
articulated boom and stick, clamshell grab bucket, hydraulic cylinders, and the bucket
lowered into coal/sand/gravel rather than floating unrealistically.
""",
    """
Use a belt-conveyor or chute transfer setup where suitable: conveyor bridge from the middle
boat to an open hold, small hopper on the transfer boat, angled chute or boom support frame,
and scattered bulk material around the hopper.
""",
    """
Use a compact shipboard crane / derrick style: mast-like tower, angled jib, stay cables,
small hook or spreader, and deck winches. Keep the tower and boom physically attached to the
middle transfer boat's deck.
""",
    """
Across the four groups, vary crane/tower construction and pose: different boom length,
boom color, base height, left/right rotation, raised/lowered angle, grab bucket vs hook vs
container spreader, and whether the equipment serves the left or right cargo vessel.
""",
]


OPERATION_EVIDENCE_RULES = [
    """
Show crane-motion evidence in static form: the middle transfer boat's crane boom should be
rotated left or right toward one cargo vessel, raised or lowered at a realistic angle, with a
grab bucket, hook, spreader, or small suspended cargo load if scale allows. Across the four
groups, vary boom direction and height so the operation looks active rather than parked.
""",
    """
For container-transfer groups, do not fill the container deck completely. Leave visible
empty slots, uneven container rows, small gaps between stacks, one partially isolated
container, or a container/spreader being lifted by the crane. Containers must stay aligned
with the deck perspective and remain small enough for UAV scale.
""",
    """
For sand, gravel, coal, or soil groups, make cargo surfaces visibly disturbed by handling:
uneven mounds, scooped depressions, grab-bucket bite marks, fresh ridges, sloped heap edges,
small spilled material near hatch coamings, and different cargo height between the two cargo
vessels.
""",
    """
Add subtle work-contact traces only where realistic: tight fender contact, taut mooring
ropes, small dark contact shadows between hulls, local ripples trapped in the narrow water
gap, and weak disturbed water near the crane/transfer boat. Avoid large wakes because the
operation is anchored or very slow.
""",
    """
Make the transfer action readable from UAV view: cargo-handling equipment should point from
the middle transfer boat toward one of the cargo holds or container decks, and at least one
cargo vessel should show a partially loaded or partially emptied area instead of a perfectly
uniform full load.
""",
]


REFERENCE_VISUAL_STYLE_RULES = [
    """
Use a bulk-carrier lightering layout like the reference samples: one large cargo ship with
several rectangular open holds and hatch covers slid or folded aside, a second cargo vessel/barge
on the opposite side, and exactly one lower middle transfer/crane boat positioned tightly but
not deformed between the two cargo vessels. Keep the three ships parallel and nearly
touching, with practical mooring spacing.
""",
    """
Add a floating-crane or crane-barge signature when scale allows: a blue, yellow, orange, or
gray crane base on a low working platform, a long lattice boom or box boom angled diagonally
over a hatch, vertical hoist cables, and a grab bucket / hook / spreader suspended above a
hold or deck. The crane should look mounted to the transfer boat, not floating in the air.
""",
    """
Use realistic hatch and cargo states from the references: some cargo holds open and dark,
some hatch covers slid or folded aside at the ends or edges, some holds visibly fuller than
another, and cargo surfaces with fresh handling marks. Non-container cargo holds must remain
open and visible; do not close them with full hatch covers.
""",
    """
For bulk-transfer groups, include a visible transfer path: a crane boom reaching from the
middle vessel to a cargo hold, a conveyor bridge/chute between vessels, a small hopper on
the transfer boat, a grab bucket above coal/sand/gravel, or spilled material near the hold
edge. The equipment orientation must explain the behavior.
""",
    """
For container or general-cargo groups, use partial container stacks and handling gaps: rows
of colored containers should be incomplete, with an empty slot or a single container under a
spreader/boom. Avoid a fully packed container deck that hides the transfer behavior.
""",
    """
Add realistic marine work details only at UAV-visible scale: black rubber fenders between
hulls, mooring ropes crossing the narrow gap, subtle prop wash or circular ripples, and small
contact shadows where the three vessels sit close. Do not add an extra tug or service boat
inside the group; the group must remain exactly three vessels.
""",
]


CARGO_COMPATIBILITY_RULES = [
    """
Use same-cargo logic inside each group. If one side is carrying coal, the other side should
also be a coal/bulk vessel and the middle equipment should be grab bucket, conveyor, hopper,
or bulk crane. Do not pair coal with containers in the same group.
""",
    """
For sand, gravel, aggregate, or soil operations, both side vessels should be bulk barges or
bulk carriers with open holds and compatible loose material. Use a grab bucket, excavator
material handler, conveyor, chute, or hopper; do not use a container spreader.
""",
    """
For container-transfer operations, both side vessels should be container-capable or
general-cargo vessels with container rows, partial stacks, empty slots, or deck lanes. Use a
spreader/hook/container crane; do not generate exposed sand, coal, or gravel as the transfer
target in that same group.
""",
    """
For general-cargo or hatch-cover vessels, keep the transfer target compatible: packaged
cargo, pallets, containers, or hatch/deck cargo should transfer to another general-cargo
deck, not into a loose sand/coal/gravel hold.
""",
    """
Across the four groups, cargo types can vary from group to group, but each individual group
must be internally consistent. It is fine to have one coal group, one sand group, one
container group, and one gravel group in the same image, as long as no single three-vessel
group mixes incompatible cargo businesses.
""",
]


PLACEMENT_RULES = [
    """
Place four separate transshipment groups in open water areas with enough room for all hulls.
Align all groups with the dominant river direction and the existing camera perspective. Each
working group must contain exactly three vessels in cargo ship - transfer boat - cargo ship
order. Inside each group, the three hulls must be parallel side-by-side, not crossing,
scattered, bow-to-bow, or randomly angled. Keep all groups away from roads, bridges, tree
crowns, banks, buildings, and any pre-existing boats or ships unless there is enough empty
water separation to avoid covering or altering them. Do not place a new group so close to an
existing vessel that their wakes, shadows, hulls, cranes, or cargo visually merge.
""",
    """
Use a near-middle-far composition across the four groups: one or two groups can be larger
and clearer, other groups should be medium or farther away, smaller, softer, and lower
contrast. In every group, the side cargo vessels can be asymmetric: one large cargo ship and
one smaller cargo barge is realistic and encouraged. The middle transfer boat should remain
the work platform between them. Match local haze, compression, and drone-video softness.
""",
    """
Place the three vessels inside each group side-by-side and nearly parallel, optionally
slightly staggered along their length, with the smaller transfer boat between the two cargo
vessels. Separate the four groups from each other with visible water gaps so they remain
four labelable samples. Also keep visible water gaps from any original vessel already in the
background. Inside each three-vessel group, keep the two side gaps consistent and realistic:
close enough for loading work, but not squeezed, warped, or overlapping. Add only weak
ripples and small disturbed water around the new hulls, because the operation is anchored or
slow-moving.
""",
    """
If the background has a clear current or existing water texture, make all four groups follow
the waterway direction and keep wakes subtle. The added objects should not erase or flatten
the river texture except at hull contact areas. Do not crowd all four groups into one
unreadable mass.
""",
]


INTENSITY_RULES = [
    """
Use a light transshipment scene for each group: exactly two cargo vessels and one small
transfer boat close together, with subtle cargo-transfer equipment. The three vessels in
each group must still be parallel and closely alongside each other. Keep the overall edit
natural even though four groups are present.
""",
    """
Use a moderate transshipment scene for each group: two clear cargo vessels, one smaller
middle transfer barge or workboat, visible cargo, and crane/boom or loading equipment on at
least some groups. The three hulls in each group should be parallel side-by-side with narrow
working gaps.
""",
    """
Use a stronger but realistic scene: four compact three-vessel operations, each with two main
barges or cargo ships and a transfer boat between them. Some groups can use one large ship
and one smaller cargo barge to match real lightering operations. Add visible floating cranes
or deck cranes on some groups if scale allows. Keep every group parallel, moored-looking,
physically plausible, and separated enough to remain readable.
""",
]


def guess_mime_type(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def parse_image_size(image_size: str) -> tuple[int, int]:
    try:
        width_text, height_text = image_size.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise ValueError(f"IMAGE_SIZE 格式错误，应类似 1920x1080: {image_size}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"IMAGE_SIZE 必须是正整数: {image_size}")

    return width, height


def resolve_output_size(image_path: Path) -> tuple[int, int]:
    if IMAGE_SIZE is None or str(IMAGE_SIZE).lower() == "original":
        with Image.open(image_path) as image:
            return ImageOps.exif_transpose(image).size

    return parse_image_size(str(IMAGE_SIZE))


def format_image_size(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def save_b64_image(b64_string: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(b64_string))


def download_image(image_url: str, output_path: Path) -> None:
    response = requests.get(image_url, timeout=300)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def save_image_with_source_quality(
    image: Image.Image,
    output_path: Path,
    source_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    save_kwargs = {}

    if suffix in {".jpg", ".jpeg"}:
        save_kwargs.update({"quality": 95, "subsampling": 0})

    try:
        with Image.open(source_path) as source_image:
            exif = source_image.info.get("exif")
            if exif:
                save_kwargs["exif"] = exif

            if suffix in {".jpg", ".jpeg"} and source_image.format == "JPEG":
                qtables = getattr(source_image, "quantization", None)
                if qtables:
                    save_kwargs["qtables"] = qtables
                    save_kwargs.pop("quality", None)
                save_kwargs["subsampling"] = JpegImagePlugin.get_sampling(source_image)
    except Exception:
        pass

    image.convert("RGB").save(output_path, **save_kwargs)


def enforce_output_size(output_path: Path, source_path: Path, target_size: tuple[int, int]) -> None:
    if not ENFORCE_OUTPUT_SIZE:
        return

    with Image.open(output_path) as image:
        image = ImageOps.exif_transpose(image)
        original_size = image.size

        if original_size == target_size:
            print(f"输出尺寸已符合要求: {original_size[0]}x{original_size[1]}")
            return

        resized = image.convert("RGB").resize(target_size, LANCZOS)
        save_image_with_source_quality(resized, output_path, source_path)

    print(
        "已强制调整输出尺寸:",
        f"{original_size[0]}x{original_size[1]} -> {target_size[0]}x{target_size[1]}",
    )


def collect_input_images(input_source: str) -> list[Path]:
    source = Path(input_source)

    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = [
            path for path in sorted(source.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        raise FileNotFoundError(f"找不到输入图片或目录: {input_source}")

    if not images:
        raise FileNotFoundError(f"没有可处理的图片: {input_source}")

    if MAX_INPUT_IMAGES is not None:
        images = images[:MAX_INPUT_IMAGES]

    return images


def make_output_path(output_dir: Path, image_path: Path, variant_index: int) -> Path:
    if OUTPUT_NAME_MODE == "original":
        if OUTPUTS_PER_IMAGE != 1:
            raise ValueError('OUTPUT_NAME_MODE="original" 时 OUTPUTS_PER_IMAGE 必须为 1')
        return output_dir / image_path.name

    if OUTPUT_NAME_MODE != "suffix":
        raise ValueError('OUTPUT_NAME_MODE 只能是 "suffix" 或 "original"')

    if OUTPUTS_PER_IMAGE == 1:
        return output_dir / f"{image_path.stem}{OUTPUT_NAME_SUFFIX}{image_path.suffix}"

    return output_dir / f"{image_path.stem}{OUTPUT_NAME_SUFFIX}_{variant_index:03d}{image_path.suffix}"


def build_prompt() -> tuple[str, str]:
    scenario = random.choice(TRANSSHIPMENT_SCENARIOS)
    placement_rule = random.choice(PLACEMENT_RULES).strip()
    intensity_rule = random.choice(INTENSITY_RULES).strip()
    transfer_boat_rules = random.sample(
        TRANSFER_BOAT_REALISM_RULES,
        k=min(3, len(TRANSFER_BOAT_REALISM_RULES)),
    )
    transfer_boat_text = "\n\n".join(
        f"{index}. {rule.strip()}" for index, rule in enumerate(transfer_boat_rules, start=1)
    )
    crane_structure_rules = random.sample(
        CRANE_STRUCTURE_VARIATION_RULES,
        k=min(4, len(CRANE_STRUCTURE_VARIATION_RULES)),
    )
    crane_structure_text = "\n\n".join(
        f"{index}. {rule.strip()}" for index, rule in enumerate(crane_structure_rules, start=1)
    )
    operation_rules = random.sample(
        OPERATION_EVIDENCE_RULES,
        k=min(4, len(OPERATION_EVIDENCE_RULES)),
    )
    operation_text = "\n\n".join(f"{index}. {rule.strip()}" for index, rule in enumerate(operation_rules, start=1))
    reference_style_rules = random.sample(
        REFERENCE_VISUAL_STYLE_RULES,
        k=min(4, len(REFERENCE_VISUAL_STYLE_RULES)),
    )
    reference_style_text = "\n\n".join(
        f"{index}. {rule.strip()}" for index, rule in enumerate(reference_style_rules, start=1)
    )
    cargo_compatibility_rules = random.sample(
        CARGO_COMPATIBILITY_RULES,
        k=min(3, len(CARGO_COMPATIBILITY_RULES)),
    )
    cargo_compatibility_text = "\n\n".join(
        f"{index}. {rule.strip()}" for index, rule in enumerate(cargo_compatibility_rules, start=1)
    )
    vessel_variants = VESSEL_VARIANTS.copy()
    random.shuffle(vessel_variants)
    vessel_text = "\n\n".join(
        f"{index}. {variant.strip()}" for index, variant in enumerate(vessel_variants[:4], start=1)
    )

    prompt = f"""
{BASE_PROMPT.strip()}

Specific transshipment scenario:
{scenario["name"]}

Scenario instruction:
{scenario["prompt"].strip()}

Placement instruction:
{placement_rule}

Scene intensity:
{intensity_rule}

Required transfer boat realism:
{transfer_boat_text}

Required crane / transfer-equipment structure diversity:
{crane_structure_text}

Required visible operation evidence:
{operation_text}

Reference-style visual requirements:
{reference_style_text}

Cargo compatibility requirements:
{cargo_compatibility_text}

Use realistic vessel diversity from these appearance cues:
{vessel_text}

Final quality checks:
- Output exactly one realistic UAV river image, not a collage and not multiple panels.
- The generated target must contain exactly {TRANSSHIPMENT_GROUPS_PER_IMAGE} separate
  ship-to-ship transfer groups. Do not generate fewer or more groups.
- Each group must read as ship-to-ship transfer or anchorage transshipment, not ordinary
  passing traffic.
- Every group must be one three-vessel unit in this order: cargo ship, transfer boat, cargo
  ship. That means the final image should contain {TRANSSHIPMENT_GROUPS_PER_IMAGE * 3}
  generated vessels total, organized as {TRANSSHIPMENT_GROUPS_PER_IMAGE} labelable groups.
- Never create a two-vessel transfer group. A valid group is not cargo ship + transfer boat,
  and not cargo ship + crane barge only. It must have two cargo vessels plus exactly one
  middle transfer/crane boat.
- Randomize the scale relationship of the two side cargo vessels across the four groups:
  some groups can have two large cargo vessels, and some groups can have one large cargo
  ship plus one smaller cargo barge / lighter cargo vessel. Both side vessels must still be
  cargo vessels, not random small boats.
- Do not add extra tugboats, extra workboats, or fourth/fifth vessels inside a target group.
  A tug-like or workboat-like hull is allowed only if it is the single middle transfer boat
  of that three-vessel group.
- In each group, the transfer boat must be the middle vessel between the two cargo vessels or
  tightly touching both sides.
- Every transfer boat must be an industrial lightering / crane / cargo-transfer workboat with
  plausible deck equipment, fenders, mooring relationship, and realistic inland-river scale.
  Do not generate random small boats, speedboats, yachts, ferry-like boats, malformed
  objects, or decorative platforms.
- Include a clear relationship inside each group: parallel close alongside placement,
  low-speed or anchored posture, transfer boat/workboat, and crane/boom/loading equipment
  when scale allows.
- Every group must show at least one clear cargo-transfer behavior trace. Examples include a
  crane boom rotated toward a hold, raised/lowered grab bucket, suspended container or cargo
  load, partially emptied hold, missing container slots, disturbed sand/coal/gravel mounds,
  deck spillage, taut mooring lines, or local ripples in the working gap.
- In each group, the viewer should be able to infer the transfer path: which middle transfer
  boat/crane is serving which cargo hold or container deck. Equipment must point toward the
  target hold/deck and the cargo state should support that action.
- Each group must be cargo-compatible. Do not create unrealistic mixed-business transfer:
  no container vessel transferring to a sand/coal/gravel barge, no bulk grab unloading into
  a container deck, no container spreader over loose bulk cargo, and no sand/gravel/coal
  material appearing on a container-only vessel.
- Cargo type may vary between groups, but within one three-vessel group the two cargo vessels
  and the middle equipment must match the same operation type: container/general cargo with
  container-capable equipment, or bulk cargo with grab/conveyor/hopper/excavator equipment.
- Use reference-style side-lightering geometry where appropriate: large bulk carrier with
  repeated rectangular hatches, a second cargo vessel on the other side, exactly one low
  middle lightering/crane boat between them, and diagonal boom/hoist cables over the cargo
  hold.
- Hatch covers and cargo holds should vary naturally: some open dark holds, some hatch
  covers slid or folded aside, some loaded holds, some partially emptied holds. Avoid
  identical repeated hatch patterns across all generated vessels. Non-container cargo holds
  must remain visibly open.
- Except for container-transfer groups, both side cargo vessels must be in open-hatch
  loading/unloading state. Their cargo holds should be visibly open with coal, sand, gravel,
  soil, aggregate, empty hold bottoms, or disturbed material visible. Do not generate fully
  sealed or fully closed hatch-cover cargo vessels for bulk-cargo transfer.
- Do not make container vessels fully packed edge-to-edge. Container cargo must have
  realistic gaps, partial stacks, missing rows, or one container being handled so the
  transfer behavior is visible.
- Do not make sand, coal, gravel, or soil loads flat and uniform. Bulk cargo must show
  realistic handling traces such as ridges, scooped pits, uneven heap height, grab marks,
  material spill, and fresh mound slopes.
- Crane and loading equipment should not all face the same direction. Across the four groups,
  vary crane boom angle, left/right rotation, raised/lowered pose, and target hold/deck while
  keeping the geometry physically plausible.
- Crane/tower structures on the middle transfer boats should be diverse and realistic:
  floating crane barge, pedestal slewing crane, lattice boom, box boom, excavator-like
  material handler, grab bucket crane, conveyor/chute system, or derrick-style jib. The
  equipment must be physically mounted on the middle boat.
- At least some groups should include the reference-style feature: a large cargo ship with
  random realistic hull color and rectangular open cargo holds, a low lightering barge
  pressed along its side, and a blue/orange/yellow/gray floating crane or deck crane with a
  diagonal boom reaching over the hold.
- All three vessels inside every group must be nearly parallel, with aligned hull directions
  and tight working spacing. Avoid perpendicular crossing, random angles, separated ships,
  normal passing traffic, or boats that only happen to be nearby.
- Inside each group, the two side cargo vessels and the middle transfer boat must keep
  realistic side-by-side spacing: narrow working gaps, fenders, mooring contact, and aligned
  hull sides. Do not overlap hulls, fuse vessels together, squeeze/deform the middle boat, or
  leave a wide unrealistic empty gap.
- Keep the four groups spatially separated with visible water between groups. Do not merge
  them into one crowded mass, and do not overlap one group with another.
- Use different ship sizes and colors naturally. The two side cargo vessels can both be
  large, or one can be large while the other is a smaller cargo barge; the middle transfer
  boat is the work platform between them.
- Randomize vessel hull colors across the four groups. Use plausible weathered industrial
  colors such as red-brown, black, dark blue, blue-gray, dark green, olive, gray, white cabin
  blocks, faded cyan, rusted steel, and weathered orange. Do not make every cargo ship the
  same color.
- Cargo can be coal, sand, gravel, soil, mixed aggregate, open bulk material, or shipping
  containers, but it must stay inside open cargo holds or on valid container decks and match
  the UAV scale.
- If containers are used, render them as realistic small rectangular container rows or low
  stacks with varied colors, correct deck alignment, tiny shadows, and muted video-frame
  clarity. Do not make them oversized blocks or toy-like graphics.
- Preserve the original image resolution, framing, background, lighting, noise, compression,
  and monitoring-video texture.
- Add vessels only on water. Do not change shorelines, roads, trees, bridges, buildings,
  existing vehicles, existing text, existing boats, existing ships, existing wakes, docks,
  or other background objects.
- Existing boats or ships in the input image must remain intact and visible. Do not erase,
  cover, replace, redraw, recolor, resize, move, or transform them. New four transshipment
  groups must be added in empty water areas with clear separation from original vessels.
- Do not let any generated hull, crane boom, shadow, wake, cargo, rope, fender, or ripple
  touch or cover a pre-existing boat/ship/wake/dock. Keep a clear clean-water buffer so the
  synthetic sample does not contaminate real vessel pixels.
- Do not add any watermark, caption, red box, arrow, label, UI overlay, or synthetic marker.
""".strip()

    return prompt, scenario["name"]


def request_image_edit(image_path: Path, prompt: str, target_size: tuple[int, int]) -> requests.Response:
    if not API_KEY:
        raise RuntimeError("未设置 OPENAI_API_KEY 环境变量")

    url = f"{BASE_URL.rstrip('/')}/images/edits"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    mime_type = guess_mime_type(image_path)

    with open(image_path, "rb") as image_file:
        files = {
            "image": (image_path.name, image_file, mime_type),
        }
        data = {
            "model": MODEL,
            "prompt": prompt,
            "n": 1,
            "size": format_image_size(target_size),
        }
        return requests.post(url, headers=headers, files=files, data=data, timeout=300)


def edit_image(image_path: Path, output_path: Path, prompt: str) -> bool:
    target_size = resolve_output_size(image_path)

    for attempt in range(1, MAX_RETRIES + 2):
        response = request_image_edit(image_path, prompt, target_size)
        print("HTTP 状态码:", response.status_code)

        if response.status_code == 200:
            break

        print("请求失败，返回内容如下:")
        print(response.text)

        if attempt <= MAX_RETRIES:
            print(f"{RETRY_SLEEP_SECONDS} 秒后重试，第 {attempt + 1} 次请求...")
            time.sleep(RETRY_SLEEP_SECONDS)
    else:
        return False

    try:
        result = response.json()
    except Exception:
        print("返回不是合法 JSON，原始返回如下:")
        print(response.text)
        return False

    if not result.get("data"):
        print("返回中没有 data 字段，完整返回如下:")
        print(result)
        return False

    item = result["data"][0]

    if "b64_json" in item:
        save_b64_image(item["b64_json"], output_path)
    elif "url" in item:
        download_image(item["url"], output_path)
    else:
        print("返回中既没有 b64_json，也没有 url。完整返回如下:")
        print(result)
        return False

    enforce_output_size(output_path, image_path, target_size)
    return True


def main() -> None:
    input_images = collect_input_images(INPUT_SOURCE)
    output_dir = Path(OUTPUT_DIR)

    print("请求地址:", f"{BASE_URL.rstrip('/')}/images/edits")
    print(f"输入图片数量: {len(input_images)}")
    print(f"每张图片输出数量: {OUTPUTS_PER_IMAGE}")
    print(f"每张输出图过驳船组数量: {TRANSSHIPMENT_GROUPS_PER_IMAGE}")
    print(f"输出目录: {output_dir}")
    print("输入 label: 不需要")

    total = 0
    succeeded = 0

    for image_path in input_images:
        for variant_index in range(1, OUTPUTS_PER_IMAGE + 1):
            total += 1
            output_path = make_output_path(output_dir, image_path, variant_index)

            if output_path.exists() and not OVERWRITE_EXISTING:
                enforce_output_size(output_path, image_path, resolve_output_size(image_path))
                succeeded += 1
                print(f"\n[{total}] 输出已存在，跳过: {output_path}")
                continue

            prompt, scenario_name = build_prompt()

            print(f"\n[{total}] 开始编辑: {image_path} -> {output_path}")
            print(f"随机场景: {scenario_name}")

            if edit_image(image_path, output_path, prompt):
                succeeded += 1
                print(f"生成成功，输出图片已保存到: {output_path}")
            else:
                print(f"生成失败: {image_path}")

    print(f"\n完成: {succeeded}/{total} 张生成成功")


if __name__ == "__main__":
    main()
