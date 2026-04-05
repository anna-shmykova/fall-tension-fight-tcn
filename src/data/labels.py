from __future__ import annotations


DEFAULT_POSITIVE_EVENT_ID = 4
KNOWN_LABEL_MODES = {
    "binary": [DEFAULT_POSITIVE_EVENT_ID],
    "binary_tension": [DEFAULT_POSITIVE_EVENT_ID],
    "binary_event4": [DEFAULT_POSITIVE_EVENT_ID],
}


def _normalize_positive_event_ids(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return [int(item) for item in items]


def resolve_label_cfg(cfg=None):
    cfg = dict(cfg or {})
    mode = str(cfg.get("mode", cfg.get("label_mode", cfg.get("task", "binary_tension")))).lower()

    positive_event_ids = _normalize_positive_event_ids(cfg.get("positive_event_ids"))
    if positive_event_ids is None and cfg.get("positive_event_id") is not None:
        positive_event_ids = [int(cfg.get("positive_event_id"))]

    any_nonzero = bool(cfg.get("any_nonzero", False))
    if positive_event_ids is None:
        if mode in {"binary_any", "binary_any_nonzero", "any_nonzero"}:
            any_nonzero = True
            positive_event_ids = []
        elif mode in KNOWN_LABEL_MODES:
            positive_event_ids = list(KNOWN_LABEL_MODES[mode])
        else:
            raise ValueError(
                f"Unsupported label mode: {mode}. "
                "Pass labels.positive_event_id(s) explicitly for custom mappings."
            )

    return {
        "mode": mode,
        "positive_event_ids": positive_event_ids,
        "any_nonzero": any_nonzero,
    }


def events_to_label(frame, cfg=None):
    resolved_cfg = resolve_label_cfg(cfg)
    events = [int(event) for event in (frame.get("group_events", []) or []) if event is not None]

    if resolved_cfg["any_nonzero"]:
        return int(any(event != 0 for event in events))

    positive_events = set(int(event_id) for event_id in resolved_cfg["positive_event_ids"])
    return int(any(event in positive_events for event in events))
