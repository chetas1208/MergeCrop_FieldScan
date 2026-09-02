from __future__ import annotations

from cropmerge.pipeline.schemas import SemanticClass

# Priority for overlap resolution (higher wins)
CLASS_PRIORITY: dict[SemanticClass, int] = {
    SemanticClass.INFRASTRUCTURE: 70,
    SemanticClass.WATER: 60,
    SemanticClass.ROAD_PATH: 55,
    SemanticClass.TREE_VEGETATION: 50,
    SemanticClass.BARE_SOIL: 40,
    SemanticClass.CROP: 35,
    SemanticClass.FIELD: 20,
    SemanticClass.UNKNOWN: 0,
}

CLASS_COLORS_BGR: dict[SemanticClass, tuple[int, int, int]] = {
    SemanticClass.FIELD: (80, 140, 60),
    SemanticClass.CROP: (40, 180, 40),
    SemanticClass.BARE_SOIL: (40, 90, 160),
    SemanticClass.ROAD_PATH: (90, 90, 90),
    SemanticClass.TREE_VEGETATION: (20, 90, 20),
    SemanticClass.WATER: (200, 120, 30),
    SemanticClass.INFRASTRUCTURE: (0, 0, 200),
    SemanticClass.UNKNOWN: (40, 40, 40),
}

PROMPT_TO_CLASS = {
    "field": SemanticClass.FIELD,
    "crop": SemanticClass.CROP,
    "bare_soil": SemanticClass.BARE_SOIL,
    "road": SemanticClass.ROAD_PATH,
    "trees": SemanticClass.TREE_VEGETATION,
    "water": SemanticClass.WATER,
    "infrastructure": SemanticClass.INFRASTRUCTURE,
}
