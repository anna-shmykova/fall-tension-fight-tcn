
def events_to_label(frame, cfg=None):
    # adjust key to your json
    events = frame.get("group_events", [])  

    if 4 in events:
        #return 2
        return 1
    #if 3 in events:
        #return 1
    return 0