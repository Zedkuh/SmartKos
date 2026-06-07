"""
koswatt_agent.py
----------------
KosWatt AI core. Import in app.py: from koswatt_agent import core_koswatt_agent
"""

import numpy as np
import skfuzzy as fuzz
from collections import deque
import warnings
warnings.filterwarnings('ignore')

STANDBY_WATTS = 10
LAMP_WATTS    = 20
TV_WATTS      = 80
AC_HOT_WATTS  = 400
AC_COOL_WATTS = 250
AC_ECO_WATTS  = 150

watt_range        = np.arange(0, 551, 1)
occupancy_range   = np.arange(0, 1.01, 0.01)
temperature_range = np.arange(20, 41, 1)
tod_range         = np.arange(0, 1.01, 0.01)
waste_range       = np.arange(0, 101, 1)

watt_low    = fuzz.trimf(watt_range, [0,   0,   100])
watt_medium = fuzz.trimf(watt_range, [100, 275, 400])
watt_high   = fuzz.trimf(watt_range, [300, 550, 550])
occ_empty    = fuzz.trimf(occupancy_range, [0, 0, 0.5])
occ_occupied = fuzz.trimf(occupancy_range, [0.5, 1, 1])
temp_cool = fuzz.trimf(temperature_range, [20, 20, 30])
temp_hot  = fuzz.trimf(temperature_range, [28, 40, 40])
tod_day   = fuzz.trimf(tod_range, [0, 0, 0.5])
tod_night = fuzz.trimf(tod_range, [0.5, 1, 1])
waste_low    = fuzz.trimf(waste_range, [0,   0,   40])
waste_medium = fuzz.trimf(waste_range, [25,  50,  75])
waste_high   = fuzz.trimf(waste_range, [60, 100, 100])

_ZERO = np.zeros_like(waste_range, dtype=float)


def compute_waste_score(watt_val, occupancy_val, temperature_val, time_of_day_val,
                        tv_on=False, ac_on=False, lamp_on=False):
    """
    Device-aware Mamdani fuzzy inference.

    TV (rigid):  empty + on => HIGH, no tolerance.
    AC (adaptive): empty + hot => MEDIUM; empty + cool => HIGH.
    Lamp (adaptive): empty + day => HIGH; empty + night => MEDIUM.
    """
    mu_watt_low    = fuzz.interp_membership(watt_range, watt_low,    watt_val)
    mu_watt_medium = fuzz.interp_membership(watt_range, watt_medium, watt_val)
    mu_watt_high   = fuzz.interp_membership(watt_range, watt_high,   watt_val)
    mu_occ_empty    = fuzz.interp_membership(occupancy_range, occ_empty,    occupancy_val)
    mu_occ_occupied = fuzz.interp_membership(occupancy_range, occ_occupied, occupancy_val)
    mu_temp_cool = fuzz.interp_membership(temperature_range, temp_cool, temperature_val)
    mu_temp_hot  = fuzz.interp_membership(temperature_range, temp_hot,  temperature_val)
    mu_tod_day   = fuzz.interp_membership(tod_range, tod_day,   time_of_day_val)
    mu_tod_night = fuzz.interp_membership(tod_range, tod_night, time_of_day_val)

    # Generic: occupied overuse => HIGH; occupied normal => LOW; standby => LOW
    clip_occ_high   = np.fmin(np.fmin(mu_occ_occupied, mu_watt_high),   waste_high)
    clip_occ_low    = np.fmin(np.fmin(mu_occ_occupied,
                              np.fmax(mu_watt_low, mu_watt_medium)),     waste_low)
    clip_standby    = np.fmin(np.fmin(mu_occ_empty, mu_watt_low),       waste_low)

    # TV rigid: empty + tv_on => HIGH (activation = mu_occ_empty, gated by tv_on)
    tv_gate  = mu_occ_empty if tv_on else 0.0
    clip_tv  = np.fmin(tv_gate, waste_high)

    # AC adaptive
    clip_ac_med  = np.fmin(np.fmin(mu_occ_empty, mu_temp_hot),  waste_medium) if ac_on else _ZERO
    clip_ac_high = np.fmin(np.fmin(mu_occ_empty, mu_temp_cool), waste_high)   if ac_on else _ZERO

    # Lamp adaptive
    clip_lamp_high = np.fmin(np.fmin(mu_occ_empty, mu_tod_day),   waste_high)   if lamp_on else _ZERO
    clip_lamp_med  = np.fmin(np.fmin(mu_occ_empty, mu_tod_night), waste_medium) if lamp_on else _ZERO

    # Aggregate all HIGH contributors
    agg_high = np.fmax(clip_occ_high,
               np.fmax(clip_tv,
               np.fmax(clip_ac_high,
                       clip_lamp_high)))

    # Aggregate all MEDIUM contributors
    agg_medium = np.fmax(clip_ac_med, clip_lamp_med)

    # Aggregate all LOW contributors
    agg_low = np.fmax(clip_occ_low, clip_standby)

    # Final aggregate + defuzzify
    aggregated = np.fmax(agg_low, np.fmax(agg_medium, agg_high))
    if aggregated.max() == 0:
        return 0.0
    return float(np.clip(fuzz.defuzz(waste_range, aggregated, 'centroid'), 0, 100))


class StripsPlanner:
    def __init__(self):
        self.operators = [
            {
                'name'        : 'TURN_OFF_TV',
                'precondition': lambda s: s.get('tv_on', False),
                'effect'      : lambda s: {**s, 'tv_on': False}
            },
            {
                'name'        : 'TURN_OFF_LAMP',
                'precondition': lambda s: s.get('lamp_on', False),
                'effect'      : lambda s: {**s, 'lamp_on': False}
            },
            {
                'name'        : 'TURN_OFF_AC',
                'precondition': lambda s: s.get('ac_on', False) and not s.get('ac_eco', False),
                'effect'      : lambda s: {**s, 'ac_on': False, 'ac_eco': False}
            },
            {
                'name'        : 'SET_AC_TO_ECO',
                'precondition': lambda s: s.get('ac_on', False) and not s.get('ac_eco', False),
                'effect'      : lambda s: {**s, 'ac_eco': True}
            },
        ]

    def _build_goal_state(self, current_state, temperature, time_of_day):
        goal = dict(current_state)
        goal['tv_on'] = False
        if goal.get('ac_on', False):
            if temperature >= 32:
                goal['ac_eco'] = True
            else:
                goal['ac_on'] = False
                goal['ac_eco'] = False
        if goal.get('lamp_on', False) and time_of_day == 0:
            goal['lamp_on'] = False
        return goal

    def _satisfies(self, state, goal):
        return all(state.get(k) == v for k, v in goal.items())

    def _key(self, state):
        return tuple(sorted(state.items()))

    def plan(self, start_state, temperature, time_of_day):
        goal = self._build_goal_state(start_state, temperature, time_of_day)
        if self._satisfies(start_state, goal):
            return []
        queue   = deque([(start_state, [])])
        visited = {self._key(start_state)}
        while queue:
            state, actions = queue.popleft()
            for op in self.operators:
                if op['precondition'](state):
                    next_s   = op['effect'](state)
                    next_key = self._key(next_s)
                    if next_key not in visited:
                        new_actions = actions + [op['name']]
                        if self._satisfies(next_s, goal):
                            return new_actions
                        visited.add(next_key)
                        queue.append((next_s, new_actions))
        return []


def calculate_watt(status_ac, status_lamp, status_tv, temperature):
    total = STANDBY_WATTS
    if status_lamp: total += LAMP_WATTS
    if status_tv:   total += TV_WATTS
    if status_ac:
        total += AC_HOT_WATTS if temperature > 30 else AC_COOL_WATTS
    return total


def calculate_post_action_watt(final_state, temperature):
    total = STANDBY_WATTS
    if final_state.get('lamp_on'): total += LAMP_WATTS
    if final_state.get('tv_on'):   total += TV_WATTS
    if final_state.get('ac_on'):
        total += AC_ECO_WATTS if final_state.get('ac_eco') else \
                 (AC_HOT_WATTS if temperature > 30 else AC_COOL_WATTS)
    return total


def apply_actions_to_state(start_state, actions):
    planner = StripsPlanner()
    op_map  = {op['name']: op for op in planner.operators}
    state   = dict(start_state)
    for name in actions:
        if name in op_map:
            state = op_map[name]['effect'](state)
    return state


def core_koswatt_agent(occupancy, temperature, time_of_day,
                       status_ac, status_lamp, status_tv):
    initial_watt = calculate_watt(status_ac, status_lamp, status_tv, temperature)
    fuzzy_score  = compute_waste_score(
        watt_val        = initial_watt,
        occupancy_val   = float(occupancy),
        temperature_val = float(temperature),
        time_of_day_val = float(time_of_day),
        tv_on           = bool(status_tv),
        ac_on           = bool(status_ac),
        lamp_on         = bool(status_lamp)
    )

    if fuzzy_score > 65:
        waste_category = 'HIGH WASTE'
    elif fuzzy_score > 35:
        waste_category = 'MEDIUM WASTE'
    else:
        waste_category = 'LOW WASTE'

    start_state = {
        'ac_on'  : bool(status_ac),
        'lamp_on': bool(status_lamp),
        'tv_on'  : bool(status_tv),
        'ac_eco' : False
    }

    action_sequence = []
    if waste_category == 'HIGH WASTE':
        planner         = StripsPlanner()
        action_sequence = planner.plan(start_state, temperature, time_of_day)

    final_state = apply_actions_to_state(start_state, action_sequence)
    final_watt  = calculate_post_action_watt(final_state, temperature)

    return {
        'fuzzy_score'    : round(fuzzy_score, 2),
        'waste_category' : waste_category,
        'action_sequence': action_sequence,
        'initial_watt'   : initial_watt,
        'final_state'    : final_state,
        'final_watt'     : final_watt
    }
