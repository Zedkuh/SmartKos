"""
koswatt_agent.py
----------------
KosWatt AI core.
Import in app.py: from koswatt_agent import core_koswatt_agent

Architecture
------------
1. Fuzzy inference engine (Mamdani): evaluates energy waste on a 0-100 scale
   using occupancy, temperature, time of day, and per-device context. Rules
   are device-gated -- no rule fires for a device that is off.

2. STRIPS planner (BFS forward search): generates the minimal ordered action
   sequence to reach an energy-efficient goal state. Activated only when the
   fuzzy score exceeds the HIGH WASTE threshold (> 65).

3. Ethical guardrails: embedded in both subsystems.
     TV is classified as a RIGID device -- no tolerance in an empty room.
     AC is classified as ADAPTIVE -- thermal context determines tolerance.
     Lamp is classified as ADAPTIVE -- time of day determines tolerance.
   The STRIPS goal builder and the fuzzy rules both reference AC_ECO_THRESHOLD
   (28 C), ensuring consistency between the two subsystems.

Waste category thresholds
-------------------------
> 65  : HIGH WASTE   -- aligned with the centroid of a fully-activated
                        waste_high rule (membership rises from 60, peaks at 100).
> 35  : MEDIUM WASTE -- aligned with the crossover mid-point between
                        waste_low (zero at 40) and waste_medium (rising from 25).
<= 35 : LOW WASTE

Device wattage assumptions
--------------------------
Based on a 1HP split-type inverter AC, LED lamp, and mid-size LED television
in a single boarding house room. These are configurable constants and should
be calibrated to actual device specs when deployed with real sensors and
smart plugs.

Fixes applied over the initial version
---------------------------------------
- watt_val removed from compute_waste_score signature. Waste classification
  is driven entirely by device context and occupancy; the raw wattage figure
  is used only for display and savings estimation, not inference.
- Waste category thresholds (35, 65) documented with rationale derived from
  the fuzzy membership function crossover points.
- status_ac_eco added to core_koswatt_agent: the AC initial state can now
  represent a device already in ECO mode, closing a model completeness gap.
- get_medium_recommendations now covers both the AC ECO case (R3) and the
  Lamp nighttime case (R6), matching the coverage of the reasoning log.
- Fuzzy rules are fully device-gated (no rule fires for a device that is off).
- AC ethical threshold unified at AC_ECO_THRESHOLD = 28 C across both the
  fuzzy engine and STRIPS goal builder.
- STRIPS_OPERATORS moved to module level.
- apply_actions_to_state is only called when action_sequence is non-empty.
- Input validation added: occupancy, temperature, and time_of_day are clamped
  and coerced before entering the inference pipeline.
- get_reasoning() produces human-readable decision trace for each agent
  decision, exposed in the UI reasoning panel.
"""

import numpy as np
import skfuzzy as fuzz
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# ── Device wattage constants ──────────────────────────────────────────────────
STANDBY_WATTS  = 10     # Baseline idle draw (router, smart meter, etc.)
LAMP_WATTS     = 15     # LED lamp
TV_WATTS       = 100    # Mid-size LED television
AC_HOT_WATTS   = 450    # 1HP split AC, high-load mode (ambient > 30 C)
AC_COOL_WATTS  = 280    # 1HP split AC, mild weather mode (ambient <= 30 C)
AC_ECO_WATTS   = 150    # 1HP split AC, sleep or ECO mode

# Ethical guardrail temperature threshold (Celsius).
# At or above this value, AC in an empty room is permitted to remain in ECO
# mode. Below this value, continued AC operation has no thermal justification
# and the STRIPS planner will schedule a full shutoff.
# Aligned with the onset of temp_hot membership to ensure consistency between
# the fuzzy inference rules and the STRIPS goal builder.
AC_ECO_THRESHOLD = 28


# ── Fuzzy universe ranges ─────────────────────────────────────────────────────
occupancy_range   = np.arange(0, 1.01, 0.01)
temperature_range = np.arange(20, 41, 1)
tod_range         = np.arange(0, 1.01, 0.01)
waste_range       = np.arange(0, 101, 1)


# ── Membership functions ──────────────────────────────────────────────────────
occ_empty    = fuzz.trimf(occupancy_range, [0,   0,   0.5])
occ_occupied = fuzz.trimf(occupancy_range, [0.5, 1,   1  ])

# temp_hot onset is aligned with AC_ECO_THRESHOLD.
# The 28-30 C overlap zone is intentional -- it represents the ambiguous
# transitional state where neither fully cool nor fully hot has membership.
temp_cool = fuzz.trimf(temperature_range, [20, 20, 30])
temp_hot  = fuzz.trimf(temperature_range, [28, 40, 40])

tod_day   = fuzz.trimf(tod_range, [0,   0,   0.5])
tod_night = fuzz.trimf(tod_range, [0.5, 1,   1  ])

waste_low    = fuzz.trimf(waste_range, [0,  0,   40 ])
waste_medium = fuzz.trimf(waste_range, [25, 50,  75 ])
waste_high   = fuzz.trimf(waste_range, [60, 100, 100])

_ZERO = np.zeros_like(waste_range, dtype=float)


# ── STRIPS operators (module level) ──────────────────────────────────────────
# Defined here so both StripsPlanner and apply_actions_to_state reference
# the same list without requiring a planner instance to read operators.
STRIPS_OPERATORS = [
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
        # Handles the case where AC was already in ECO mode at the start of the
        # planning cycle and the thermal context does not justify even ECO operation
        # (ambient < AC_ECO_THRESHOLD). TURN_OFF_AC cannot apply in this state
        # because its precondition requires not ac_eco.
        'name'        : 'TURN_OFF_AC_ECO',
        'precondition': lambda s: s.get('ac_on', False) and s.get('ac_eco', False),
        'effect'      : lambda s: {**s, 'ac_on': False, 'ac_eco': False}
    },
    {
        'name'        : 'SET_AC_TO_ECO',
        'precondition': lambda s: s.get('ac_on', False) and not s.get('ac_eco', False),
        'effect'      : lambda s: {**s, 'ac_eco': True}
    },
]


# ── Fuzzy inference ───────────────────────────────────────────────────────────

def compute_waste_score(occupancy_val, temperature_val, time_of_day_val,
                        tv_on=False, ac_on=False, lamp_on=False):
    """
    Mamdani fuzzy inference engine for energy waste classification.

    Fuzzy rules
    -----------
    R1  Occupied room => waste LOW.
        Occupants have a right to use their devices. No interference with an
        occupied room regardless of wattage.
    R2  Empty + TV on => waste HIGH.
        TV is a rigid device. No contextual tolerance in an empty room.
    R3  Empty + AC on + hot ambient (>= 28 C) => waste MEDIUM.
        Thermal justification: preserving partial cooling for returning
        occupants is a valid ethical reason to withhold full shutoff.
    R4  Empty + AC on + cool ambient (< 28 C) => waste HIGH.
        No thermal justification. Cooling an empty room when it is not hot
        is classified as unambiguous waste.
    R5  Empty + Lamp on + daytime => waste HIGH.
        Natural light is available. Artificial lighting is unnecessary.
    R6  Empty + Lamp on + nighttime => waste MEDIUM.
        Safety and deterrence lighting has partial justification at night.

    All device rules are gated by actual on/off status. Waste classification
    is driven by device context and occupancy, not raw wattage.
    """
    # ── SHORT-CIRCUIT GUARD ───────────────────────────────────────────────────
    # Force absolute zero if all managed devices are completely turned off.
    # This prevents centroid defuzzification of R1 from returning 13.3 W.
    if not (tv_on or ac_on or lamp_on):
        return 0.0
    # ──────────────────────────────────────────────────────────────────────────

    mu_occ_empty    = fuzz.interp_membership(occupancy_range, occ_empty,    occupancy_val)
    mu_occ_occupied = fuzz.interp_membership(occupancy_range, occ_occupied, occupancy_val)
    mu_temp_cool    = fuzz.interp_membership(temperature_range, temp_cool,  temperature_val)
    mu_temp_hot     = fuzz.interp_membership(temperature_range, temp_hot,   temperature_val)
    mu_tod_day      = fuzz.interp_membership(tod_range, tod_day,            time_of_day_val)
    mu_tod_night    = fuzz.interp_membership(tod_range, tod_night,          time_of_day_val)

    # R1: Occupied => LOW
    clip_occupied_low = np.fmin(mu_occ_occupied, waste_low)

    # R2: TV -- rigid device
    tv_gate      = mu_occ_empty if tv_on else 0.0
    clip_tv_high = np.fmin(tv_gate, waste_high)

    # R3/R4: AC -- adaptive device (rules gated by ac_on)
    clip_ac_medium = np.fmin(np.fmin(mu_occ_empty, mu_temp_hot),  waste_medium) if ac_on else _ZERO
    clip_ac_high   = np.fmin(np.fmin(mu_occ_empty, mu_temp_cool), waste_high)   if ac_on else _ZERO

    # R5/R6: Lamp -- adaptive device (rules gated by lamp_on)
    clip_lamp_high   = np.fmin(np.fmin(mu_occ_empty, mu_tod_day),   waste_high)   if lamp_on else _ZERO
    clip_lamp_medium = np.fmin(np.fmin(mu_occ_empty, mu_tod_night), waste_medium) if lamp_on else _ZERO

    agg_low    = clip_occupied_low
    agg_medium = np.fmax(clip_ac_medium, clip_lamp_medium)
    agg_high   = np.fmax(clip_tv_high, np.fmax(clip_ac_high, clip_lamp_high))

    aggregated = np.fmax(agg_low, np.fmax(agg_medium, agg_high))
    if aggregated.max() == 0:
        return 0.0
    return float(np.clip(fuzz.defuzz(waste_range, aggregated, 'centroid'), 0, 100))


# ── STRIPS planner ────────────────────────────────────────────────────────────

class StripsPlanner:
    """
    STRIPS-based energy intervention planner.

    State space : boolean device flags {ac_on, lamp_on, tv_on, ac_eco}.
    Goal state  : energy-efficient configuration derived from ethical guardrails.
    Search      : BFS forward chaining, guaranteeing the shortest action sequence.

    Goal construction respects the same ethical guardrails as the fuzzy engine:
      TV is always off in an empty room (rigid device).
      AC: ECO mode if ambient >= AC_ECO_THRESHOLD, OFF otherwise (adaptive).
      Lamp: OFF during daytime, tolerated at nighttime (adaptive).
    """

    def __init__(self):
        self.operators = STRIPS_OPERATORS

    def _build_goal_state(self, current_state, temperature, time_of_day):
        goal = dict(current_state)
        # TV: rigid -- always off in empty room
        goal['tv_on'] = False
        # AC: adaptive -- ECO if at or above threshold, OFF otherwise
        if goal.get('ac_on', False):
            if temperature >= AC_ECO_THRESHOLD:
                goal['ac_eco'] = True
            else:
                goal['ac_on']  = False
                goal['ac_eco'] = False
        # Lamp: adaptive -- off during daytime, tolerated at night
        if goal.get('lamp_on', False) and time_of_day < 0.5:
            goal['lamp_on'] = False
        return goal

    def _satisfies(self, state, goal):
        return all(state.get(k) == v for k, v in goal.items())

    def _state_key(self, state):
        return tuple(sorted(state.items()))

    def plan(self, start_state, temperature, time_of_day):
        """
        BFS forward search from start_state toward goal.
        Returns an ordered list of action name strings.
        Returns [] if start_state already satisfies the goal.
        """
        goal = self._build_goal_state(start_state, temperature, time_of_day)
        if self._satisfies(start_state, goal):
            return []
        queue   = deque([(start_state, [])])
        visited = {self._state_key(start_state)}
        while queue:
            state, actions = queue.popleft()
            for op in self.operators:
                if op['precondition'](state):
                    next_state = op['effect'](state)
                    next_key   = self._state_key(next_state)
                    if next_key not in visited:
                        new_actions = actions + [op['name']]
                        if self._satisfies(next_state, goal):
                            return new_actions
                        visited.add(next_key)
                        queue.append((next_state, new_actions))
        return []


# ── Watt calculation helpers ──────────────────────────────────────────────────

def calculate_watt(status_ac, status_lamp, status_tv, temperature, status_ac_eco=False):
    """Estimate total power draw in watts given current device states."""
    total = STANDBY_WATTS
    if status_lamp: total += LAMP_WATTS
    if status_tv:   total += TV_WATTS
    if status_ac:
        if status_ac_eco:
            total += AC_ECO_WATTS
        else:
            total += AC_HOT_WATTS if temperature > 30 else AC_COOL_WATTS
    return total


def calculate_post_action_watt(final_state, temperature):
    """Estimate power draw after STRIPS actions have been applied."""
    total = STANDBY_WATTS
    if final_state.get('lamp_on'): total += LAMP_WATTS
    if final_state.get('tv_on'):   total += TV_WATTS
    if final_state.get('ac_on'):
        total += (AC_ECO_WATTS if final_state.get('ac_eco')
                  else (AC_HOT_WATTS if temperature > 30 else AC_COOL_WATTS))
    return total


def apply_actions_to_state(start_state, actions):
    """
    Apply an ordered sequence of STRIPS action names to start_state and
    return the resulting device state dict. References STRIPS_OPERATORS
    directly -- no StripsPlanner instance required.
    """
    op_map = {op['name']: op for op in STRIPS_OPERATORS}
    state  = dict(start_state)
    for name in actions:
        if name in op_map:
            state = op_map[name]['effect'](state)
    return state


# ── Advisory and reasoning helpers ───────────────────────────────────────────

def get_medium_recommendations(start_state, temperature, time_of_day):
    """
    Returns advisory (non-binding) action suggestions for MEDIUM WASTE cases.
    These represent what the STRIPS planner would propose if unrestricted.
    The agent does not execute them autonomously -- they are surfaced to the
    user interface as recommendations only.
    """
    recs = []
    if start_state.get('ac_on') and not start_state.get('ac_eco'):
        recs.append('SUGGEST_AC_ECO')
    if start_state.get('lamp_on') and time_of_day >= 0.5:
        recs.append('SUGGEST_LAMP_OFF')
    return recs


def get_reasoning(waste_category, start_state, action_sequence, temperature,
                  time_of_day, medium_recommendations=None):
    """
    Returns an ordered list of human-readable strings explaining the
    agent's decision. Each string represents one reasoning step or observation.
    Intended for display in the Agent Reasoning panel of the UI.
    """
    lines = []

    if waste_category == 'LOW WASTE':
        active = [k for k in ('ac_on', 'lamp_on', 'tv_on') if start_state.get(k)]
        if not active:
            lines.append(
                "No active devices detected. Only standby draw is present. "
                "No evaluation required."
            )
        else:
            lines.append(
                "Power draw is within acceptable range for the current occupancy "
                "and environmental context. Active devices are operating within "
                "ethical parameters. No intervention required."
            )

    elif waste_category == 'MEDIUM WASTE':
        if start_state.get('ac_on') and temperature >= AC_ECO_THRESHOLD:
            if start_state.get('ac_eco'):
                lines.append(
                    f"AC is active in ECO mode in an empty room at {int(temperature)} C. "
                    f"Ambient temperature at or above {AC_ECO_THRESHOLD} C provides thermal "
                    "justification for continued operation. ECO mode is already the "
                    "energy-reduced state; no further reduction is applicable. "
                    "No action required."
                )
            else:
                lines.append(
                    f"AC is active in an empty room at {int(temperature)} C. Ambient temperature "
                    f"at or above {AC_ECO_THRESHOLD} C provides partial thermal justification "
                    "for continued operation. Shutting the AC off risks occupant discomfort on "
                    "return. Autonomous action withheld under the adaptive device guardrail."
                )
        if start_state.get('lamp_on') and time_of_day >= 0.5:
            lines.append(
                "Lamp is active in an empty room at night. Safety and deterrence "
                "lighting carries partial justification during nighttime hours. "
                "Autonomous shutoff withheld under the adaptive device guardrail."
            )
        if medium_recommendations:
            lines.append(
                "Non-binding advisory recommendations are available to reduce draw "
                "without triggering autonomous action. No STRIPS plan has been executed."
            )

    else:  # HIGH WASTE
        if not action_sequence:
            lines.append(
                "Waste score exceeds the HIGH threshold but device states are already "
                "at the energy-optimal configuration. No further actions are required."
            )
        for action in action_sequence:
            if action == 'TURN_OFF_TV':
                lines.append(
                    "TV is active in an empty room. No contextual justification exists "
                    "for an entertainment device when occupancy is zero. "
                    "TV scheduled for shutoff."
                )
            elif action == 'TURN_OFF_AC':
                lines.append(
                    f"AC is active in an empty room at {int(temperature)} C. Temperature is "
                    f"below the thermal threshold ({AC_ECO_THRESHOLD} C). No justification "
                    "for continued cooling. AC scheduled for shutoff."
                )
            elif action == 'TURN_OFF_AC_ECO':
                lines.append(
                    f"AC is in ECO mode in an empty room at {int(temperature)} C. Temperature "
                    f"is below the thermal threshold ({AC_ECO_THRESHOLD} C). ECO mode does not "
                    "provide sufficient justification for continued operation at this ambient "
                    "temperature. AC scheduled for shutoff."
                )
            elif action == 'SET_AC_TO_ECO':
                lines.append(
                    f"AC is active in an empty room at {int(temperature)} C. Temperature at "
                    f"or above {AC_ECO_THRESHOLD} C provides partial thermal justification. "
                    "Full shutoff is withheld under the adaptive guardrail. Switching to "
                    "ECO mode preserves comfort margin while reducing draw."
                )
            elif action == 'TURN_OFF_LAMP':
                lines.append(
                    "Lamp is active in an empty room during daytime. Natural light "
                    "renders artificial lighting unnecessary. Lamp scheduled for shutoff."
                )

    return lines


# ── Main agent entry point ────────────────────────────────────────────────────

def core_koswatt_agent(occupancy, temperature, time_of_day,
                       status_ac, status_lamp, status_tv,
                       status_ac_eco=False):
    """
    Main KosWatt agent entry point.

    Parameters
    ----------
    occupancy     : 0 (empty) or 1 (occupied); clamped to [0.0, 1.0]
    temperature   : ambient temperature in Celsius; clamped to [20.0, 40.0]
    time_of_day   : 0 (day) or 1 (night); clamped to [0.0, 1.0]
    status_ac     : bool -- AC currently switched on
    status_lamp   : bool -- Lamp currently switched on
    status_tv     : bool -- TV currently switched on
    status_ac_eco : bool -- AC is already in ECO mode (default False)

    Returns
    -------
    dict with keys:
      fuzzy_score             float  waste score 0-100
      waste_category          str    'LOW WASTE' | 'MEDIUM WASTE' | 'HIGH WASTE'
      action_sequence         list   ordered STRIPS action names (autonomous)
      initial_watt            int    estimated draw before AI intervention
      final_state             dict   device states after actions applied
      final_watt              int    estimated draw after AI intervention
      medium_recommendations  list   advisory action names (MEDIUM WASTE only)
      reasoning               list   human-readable decision explanation strings
    """
    # Input validation and type coercion
    occupancy   = float(np.clip(float(occupancy),   0.0, 1.0))
    temperature = float(np.clip(float(temperature), 20.0, 40.0))
    time_of_day = float(np.clip(float(time_of_day), 0.0, 1.0))
    status_ac     = bool(status_ac)
    status_lamp   = bool(status_lamp)
    status_tv     = bool(status_tv)
    # If AC is off, ECO state is meaningless; force it to False.
    status_ac_eco = bool(status_ac_eco) if status_ac else False

    # Step 1: Estimate current power draw
    initial_watt = calculate_watt(status_ac, status_lamp, status_tv, temperature,
                                  status_ac_eco=status_ac_eco)

    # Step 2: Fuzzy inference -- score the waste level
    fuzzy_score = compute_waste_score(
        occupancy_val   = occupancy,
        temperature_val = temperature,
        time_of_day_val = time_of_day,
        tv_on           = status_tv,
        ac_on           = status_ac,
        lamp_on         = status_lamp
    )

    # Step 3: Classify waste category
    # Thresholds are aligned with the fuzzy membership function crossover points.
    # waste_high has full membership at 100 and rises from 60; its centroid for
    # a fully-activated high rule lands above 65, making > 65 the natural HIGH
    # boundary. waste_medium is centred at 50 with full membership between 25
    # and 75; the LOW/MEDIUM boundary at 35 sits at the crossover between
    # waste_low (zero at 40) and waste_medium (rising from 25), which is the
    # mid-point of that overlap region.
    if fuzzy_score > 65:
        waste_category = 'HIGH WASTE'
    elif fuzzy_score > 35:
        waste_category = 'MEDIUM WASTE'
    else:
        waste_category = 'LOW WASTE'

    # Step 4: Build initial device state dict
    start_state = {
        'ac_on'  : status_ac,
        'lamp_on': status_lamp,
        'tv_on'  : status_tv,
        'ac_eco' : status_ac_eco
    }

    # Step 5: STRIPS planning -- autonomous action, HIGH WASTE only
    action_sequence = []
    if waste_category == 'HIGH WASTE':
        planner         = StripsPlanner()
        action_sequence = planner.plan(start_state, temperature, time_of_day)

    # Step 6: Advisory recommendations -- non-binding, MEDIUM WASTE only
    medium_recommendations = []
    if waste_category == 'MEDIUM WASTE':
        medium_recommendations = get_medium_recommendations(
            start_state, temperature, time_of_day
        )

    # Step 7: Apply actions and compute final power state.
    # Only runs when actions exist, avoiding redundant computation on
    # LOW and MEDIUM paths where no actions are taken.
    if action_sequence:
        final_state = apply_actions_to_state(start_state, action_sequence)
        final_watt  = calculate_post_action_watt(final_state, temperature)
    else:
        final_state = dict(start_state)
        final_watt  = initial_watt

    # Step 8: Generate human-readable reasoning trace
    reasoning = get_reasoning(
        waste_category, start_state, action_sequence,
        temperature, time_of_day, medium_recommendations
    )

    return {
        'fuzzy_score'           : round(fuzzy_score, 2),
        'waste_category'        : waste_category,
        'action_sequence'       : action_sequence,
        'initial_watt'          : initial_watt,
        'final_state'           : final_state,
        'final_watt'            : final_watt,
        'medium_recommendations': medium_recommendations,
        'reasoning'             : reasoning,
    }
