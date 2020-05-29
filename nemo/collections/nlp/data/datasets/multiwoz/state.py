"""
convlab2
"""
__all__ = ['default_state']


def default_state():
    state = dict(user_action=[], system_action=[], belief_state={}, request_state={}, terminated=False, history=[])
    state['belief_state'] = {
        "police": {"book": {"booked": []}, "semi": {}},
        "hotel": {
            "book": {"booked": [], "people": "", "day": "", "stay": ""},
            "semi": {"name": "", "area": "", "parking": "", "pricerange": "", "stars": "", "internet": "", "type": ""},
        },
        "attraction": {"book": {"booked": []}, "semi": {"type": "", "name": "", "area": ""}},
        "restaurant": {
            "book": {"booked": [], "people": "", "day": "", "time": ""},
            "semi": {"food": "", "pricerange": "", "name": "", "area": "",},
        },
        "hospital": {"book": {"booked": []}, "semi": {"department": ""}},
        "taxi": {"book": {"booked": []}, "semi": {"leaveAt": "", "destination": "", "departure": "", "arriveBy": ""}},
        "train": {
            "book": {"booked": [], "people": ""},
            "semi": {"leaveAt": "", "destination": "", "day": "", "arriveBy": "", "departure": ""},
        },
    }
    return state
