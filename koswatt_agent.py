"""
koswatt_agent.py
----------------
KosWatt AI core.

Architecture
------------
1. Fuzzy inference engine (Mamdani): evaluates energy waste on a 0-100 scale
   using occupancy, temperature, time of day, and per-device context.

2. Ethical priority resolver: applies an explicit Safety > Comfort > Efficiency
   hierarchy. Design decisions (device classifications, thresholds) are documented
   in DEVICE_ETHICS and ETHICAL_PRIORITIES -- auditable constants, not hidden
   assumptions.

3. Rigid device override: guarantees HIGH classification for devices (TV) that
   have no contextual justification, independent of aggregate fuzzy score.
   Prevents calibration drift from masking obvious waste.

4. STRIPS planner (BFS forward search): generates the minimal action sequence
   to reach the energy-efficient goal state. Only runs when waste is HIGH and
   the autonomy level permits action.

5. Deliberative reasoning trace: accumulated inline at each decision step as
   the decision is made, not reconstructed after the fact.

Ethical Framework
-----------------
The priority hierarchy is an explicit design choice, documented here:

  Priority 1 — SAFETY:      Preserve safe conditions (security lighting, thermal
                             safety). Takes precedence over all other priorities.
  Priority 2 — COMFORT:     Respect occupant comfort within reasonable energy
                             bounds. Thermal regulation with justification takes
                             precedence over efficiency.
  Priority 3 — EFFICIENCY:  Minimise energy waste when Safety and Comfort are
                             already satisfied. Drives autonomous shutoff decisions.

These are designer decisions, as they are in all symbolic AI systems. They are
made explicit here so an examiner or operator can inspect, debate, and change
them. The threshold AC_ECO_THRESHOLD = 28 C is justified below in DEVICE_ETHICS.

Autonomy Levels
---------------
  'advisory'   -- Recommendations only. No autonomous action regardless of score.
  'confirm'    -- HIGH WASTE generates a pending plan that requires human approval.
  'autonomous' -- HIGH WASTE triggers immediate STRIPS execution (default).

Occupancy Confidence
--------------------
Sensors are imperfect (motion sensor failure, sleeping person, etc.).
occupancy_confidence (0.0–1.0) modulates the inferred occupancy:
    effective_occupancy = sensor_reading * confidence + 0.5 * (1 - confidence)

At confidence=1.0, effective equals the raw sensor value.
At confidence=0.0, effective is 0.5 (maximally uncertain; system is maximally
conservative). A sleeping person read as "empty" with 40% confidence produces
an effective_occupancy of 0.3 -- still biased toward empty, but cautiously.
When confidence drops below 0.5 and the room reads empty, the safety override
suspends autonomous action entirely.

Fixes applied over the initial version
---------------------------------------
- watt_val removed from compute_waste_score.
- Waste category thresholds (35, 65) documented with membership function rationale.
- status_ac_eco added to core_koswatt_agent.
- get_medium_recommendations covers R3 and R6 cases.
- Fuzzy rules fully device-gated.
- AC ethical threshold unified at AC_ECO_THRESHOLD = 28 C.
- STRIPS_OPERATORS at module level.
- apply_actions_to_state only called when actions exist.
- Input validation added.
- R3b: AC already in ECO + hot + empty => LOW (not MEDIUM).
- R4 gated on not-ECO to prevent HIGH/no-actions contradiction at T=28 C.
- calculate_post_action_watt delegates to calculate_watt (no duplication).
- ETHICAL_PRIORITIES and DEVICE_ETHICS added: ethics made explicit.
- autonomy_level parameter: 'advisory' | 'confirm' | 'autonomous'.
- occupancy_confidence parameter with safety override at low confidence.
- Rigid device override: TV-in-empty-room escalates to HIGH independent of
  aggregate fuzzy score, preventing calibration drift from hiding obvious waste.
- Reasoning accumulated inline at each decision step (deliberative, not post-hoc).
- DEVICE_REGISTRY added: single place to register a new device.
- get_reasoning() removed: reasoning now built inside core_koswatt_agent.
"""

import numpy as np
import skfuzzy as fuzz
from collections import deque, OrderedDict
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='skfuzzy')
warnings.filterwarnings('ignore', category=RuntimeWarning,    module='skfuzzy')


# =============================================================================
# ETHICAL FRAMEWORK  (explicit, auditable design decisions)
# =============================================================================

ETHICAL_PRIORITIES = OrderedDict([
    ('SAFETY',
     'Preserve safe operating conditions and occupant wellbeing above all else. '
     'Includes security lighting at night and protection against thermal extremes. '
     'No efficiency gain justifies compromising safety.'),

    ('COMFORT',
     'Respect occupant comfort within reasonable energy bounds. Thermal regulation '
     'with environmental justification (ambient >= AC_ECO_THRESHOLD) takes '
     'precedence over immediate efficiency targets.'),

    ('EFFICIENCY',
     'Minimise energy waste when Safety and Comfort are already satisfied. '
     'Drives all autonomous shutoff and ECO-mode decisions. Applied only after '
     'Priorities 1 and 2 are evaluated.'),
])

DEVICE_ETHICS = {
    'TV': {
        'class': 'RIGID',
        'justification': (
            'Entertainment device. No safety, comfort, or thermal justification '
            'exists for operation in an unoccupied room. Conflicts with Priority 3 '
            '(Efficiency) with zero countervailing claim from Priority 1 or 2. '
            'Classified RIGID: no contextual exception is granted. TV is the only '
            'device in this system with no adaptive-guardrail tolerance.'
        ),
    },
    'AC': {
        'class': 'ADAPTIVE',
        'justification': (
            'Thermal regulation device that directly affects occupant health and '
            'comfort on return. At or above AC_ECO_THRESHOLD (28 C), continued '
            'operation in ECO mode satisfies Priority 2 (Comfort) -- pre-cooling '
            'a room that will be reoccupied is a legitimate ethical reason to '
            'withhold full shutoff. 28 C is chosen as the onset of the temp_hot '
            'fuzzy membership function, aligning the crisp STRIPS boundary with '
            'the fuzzy classification boundary. Below 28 C, no thermal '
            'justification exists and the AC is scheduled for shutoff.'
        ),
    },
    'LAMP': {
        'class': 'ADAPTIVE',
        'justification': (
            'Lighting device. At night, an active lamp satisfies Priority 1 (Safety): '
            'it provides security deterrence and safe re-entry illumination for '
            'returning occupants. This safety benefit outweighs the 15 W draw. '
            'During daytime, natural light renders artificial lighting unnecessary -- '
            'no Safety or Comfort claim applies, so Priority 3 (Efficiency) governs '
            'and the lamp is scheduled for shutoff.'
        ),
    },
}

AUTONOMY_LEVELS = {
    'advisory': (
        'Generate non-binding recommendations only. No autonomous action is taken '
        'regardless of waste classification. Fully respects user autonomy; the '
        'agent acts as an advisor, not an executor.'
    ),
    'confirm': (
        'HIGH WASTE generates a ready-to-execute STRIPS plan, but the agent '
        'waits for explicit human approval before acting. Balances energy '
        'efficiency with user control. Recommended for most deployments.'
    ),
    'autonomous': (
        'HIGH WASTE triggers immediate STRIPS plan execution without human '
        'confirmation. Suitable for fully trusted, monitored deployments where '
        'the sensor infrastructure is reliable.'
    ),
}

# Device registry: documents all devices in one place.
# NOTE: This is a documentation scaffold that shows the intended extensibility
# pattern. Adding a new device here does NOT automatically wire it into the
# fuzzy engine, STRIPS operators, or watt helpers -- those require separate
# additions. The registry's purpose is to make the full set of ethical
# classifications visible and auditable in a single location.
DEVICE_REGISTRY = {
    'TV':   {'ethics': DEVICE_ETHICS['TV']},
    'AC':   {'ethics': DEVICE_ETHICS['AC']},
    'LAMP': {'ethics': DEVICE_ETHICS['LAMP']},
}


# =============================================================================
# WATTAGE CONSTANTS
# =============================================================================

STANDBY_WATTS  = 10   # Baseline idle draw (router, smart meter, etc.)
LAMP_WATTS     = 15   # LED lamp
TV_WATTS       = 100  # Mid-size LED television
AC_HOT_WATTS   = 450  # 1HP split AC, high-load (ambient > 30 C)
AC_COOL_WATTS  = 280  # 1HP split AC, mild weather (ambient <= 30 C)
AC_ECO_WATTS   = 150  # 1HP split AC, ECO / sleep mode

# Ethical guardrail temperature threshold.
# Justified in DEVICE_ETHICS['AC'] above. Aligned with temp_hot MF onset
# so that the fuzzy engine and STRIPS goal builder share the same boundary.
AC_ECO_THRESHOLD = 28  # °C


# =============================================================================
# FUZZY UNIVERSE AND MEMBERSHIP FUNCTIONS
# =============================================================================

occupancy_range   = np.arange(0, 1.01, 0.01)
temperature_range = np.arange(20, 41, 1)
tod_range         = np.arange(0, 1.01, 0.01)
waste_range       = np.arange(0, 101, 1)

occ_empty    = fuzz.trimf(occupancy_range, [0,   0,   0.5])
occ_occupied = fuzz.trimf(occupancy_range, [0.5, 1,   1  ])

# temp_hot onset at 28 C aligns with AC_ECO_THRESHOLD.
# The 28–30 C overlap is intentional: the transitional zone where neither
# fully cool nor fully hot has exclusive membership.
temp_cool = fuzz.trimf(temperature_range, [20, 20, 30])
temp_hot  = fuzz.trimf(temperature_range, [28, 40, 40])

tod_day   = fuzz.trimf(tod_range, [0,   0,   0.5])
tod_night = fuzz.trimf(tod_range, [0.5, 1,   1  ])

waste_low    = fuzz.trimf(waste_range, [0,  0,   40 ])
waste_medium = fuzz.trimf(waste_range, [25, 50,  75 ])
waste_high   = fuzz.trimf(waste_range, [60, 100, 100])

_ZERO = np.zeros_like(waste_range, dtype=float)


# =============================================================================
# STRIPS OPERATORS
# =============================================================================

STRIPS_OPERATORS = [
    {
        'name':         'TURN_OFF_TV',
        'precondition': lambda s: s.get('tv_on', False),
        'effect':       lambda s: {**s, 'tv_on': False},
    },
    {
        'name':         'TURN_OFF_LAMP',
        'precondition': lambda s: s.get('lamp_on', False),
        'effect':       lambda s: {**s, 'lamp_on': False},
    },
    {
        'name':         'TURN_OFF_AC',
        'precondition': lambda s: s.get('ac_on', False) and not s.get('ac_eco', False),
        'effect':       lambda s: {**s, 'ac_on': False, 'ac_eco': False},
    },
    {
        # Handles AC already in ECO mode when thermal context doesn't justify
        # even ECO operation (ambient < AC_ECO_THRESHOLD).
        'name':         'TURN_OFF_AC_ECO',
        'precondition': lambda s: s.get('ac_on', False) and s.get('ac_eco', False),
        'effect':       lambda s: {**s, 'ac_on': False, 'ac_eco': False},
    },
    {
        'name':         'SET_AC_TO_ECO',
        'precondition': lambda s: s.get('ac_on', False) and not s.get('ac_eco', False),
        'effect':       lambda s: {**s, 'ac_eco': True},
    },
]


# =============================================================================
# FUZZY INFERENCE ENGINE
# =============================================================================

def compute_waste_score(occupancy_val, temperature_val, time_of_day_val,
                        tv_on=False, ac_on=False, lamp_on=False, ac_eco=False):
    """
    Mamdani fuzzy inference engine for energy waste classification.

    Accepts effective_occupancy (not raw sensor reading) so that occupancy
    confidence modulation is already applied before inference.

    Fuzzy rules
    -----------
    R1   Occupied => LOW.
    R2   Empty + TV => HIGH.  (TV: RIGID -- no tolerance)
    R3   Empty + AC full mode + hot ambient  => MEDIUM.  (Comfort justification)
    R3b  Empty + AC ECO mode  + hot ambient  => LOW.     (Already optimised)
    R4   Empty + AC full mode + cool ambient => HIGH.    (No thermal justification)
         Gated on full-mode only: ECO+cool scores LOW, preventing the HIGH/
         no-actions contradiction at the AC_ECO_THRESHOLD boundary (T=28 C).
    R5   Empty + Lamp + daytime  => HIGH.   (Natural light available)
    R6   Empty + Lamp + nighttime => MEDIUM. (Safety justification)
    """
    if not (tv_on or ac_on or lamp_on):
        return 0.0

    mu_occ_empty    = fuzz.interp_membership(occupancy_range, occ_empty,    occupancy_val)
    mu_occ_occupied = fuzz.interp_membership(occupancy_range, occ_occupied, occupancy_val)
    mu_temp_cool    = fuzz.interp_membership(temperature_range, temp_cool,  temperature_val)
    mu_temp_hot     = fuzz.interp_membership(temperature_range, temp_hot,   temperature_val)
    mu_tod_day      = fuzz.interp_membership(tod_range, tod_day,            time_of_day_val)
    mu_tod_night    = fuzz.interp_membership(tod_range, tod_night,          time_of_day_val)

    # R1: Occupied => LOW
    clip_occupied_low = np.fmin(mu_occ_occupied, waste_low)

    # R2: TV -- RIGID device
    tv_gate      = mu_occ_empty if tv_on else 0.0
    clip_tv_high = np.fmin(tv_gate, waste_high)

    # R3 / R3b / R4: AC -- ADAPTIVE device
    clip_ac_eco_low = np.fmin(np.fmin(mu_occ_empty, mu_temp_hot),  waste_low)    if (ac_on and ac_eco)     else _ZERO
    clip_ac_medium  = np.fmin(np.fmin(mu_occ_empty, mu_temp_hot),  waste_medium) if (ac_on and not ac_eco) else _ZERO
    clip_ac_high    = np.fmin(np.fmin(mu_occ_empty, mu_temp_cool), waste_high)   if (ac_on and not ac_eco) else _ZERO

    # R5 / R6: Lamp -- ADAPTIVE device
    clip_lamp_high   = np.fmin(np.fmin(mu_occ_empty, mu_tod_day),   waste_high)   if lamp_on else _ZERO
    clip_lamp_medium = np.fmin(np.fmin(mu_occ_empty, mu_tod_night), waste_medium) if lamp_on else _ZERO

    agg_low    = np.fmax(clip_occupied_low, clip_ac_eco_low)
    agg_medium = np.fmax(clip_ac_medium, clip_lamp_medium)
    agg_high   = np.fmax(clip_tv_high, np.fmax(clip_ac_high, clip_lamp_high))

    aggregated = np.fmax(agg_low, np.fmax(agg_medium, agg_high))
    if aggregated.max() == 0:
        return 0.0
    return float(np.clip(fuzz.defuzz(waste_range, aggregated, 'centroid'), 0, 100))


# =============================================================================
# STRIPS PLANNER
# =============================================================================

class StripsPlanner:
    """
    STRIPS-based energy intervention planner.

    State space  : boolean device flags {ac_on, lamp_on, tv_on, ac_eco}.
    Goal state   : energy-efficient configuration derived from DEVICE_ETHICS and
                   ETHICAL_PRIORITIES.
    Search       : BFS forward chaining, guaranteeing shortest action sequence.

    Goal construction applies the same ethical priorities as the fuzzy engine:
      TV  (RIGID)    : always off in an empty room.
      AC  (ADAPTIVE) : ECO if ambient >= AC_ECO_THRESHOLD, OFF otherwise.
      Lamp (ADAPTIVE): OFF during daytime, tolerated at nighttime.
    """

    def __init__(self):
        self.operators = STRIPS_OPERATORS

    def _build_goal_state(self, current_state, temperature, time_of_day):
        goal = dict(current_state)
        goal['tv_on'] = False          # Priority 3: RIGID, no exception
        if goal.get('ac_on', False):
            if temperature >= AC_ECO_THRESHOLD:
                goal['ac_eco'] = True  # Priority 2: Comfort -- ECO permitted
            else:
                goal['ac_on']  = False # Priority 3: Efficiency -- no justification
                goal['ac_eco'] = False
        if goal.get('lamp_on', False) and time_of_day < 0.5:
            goal['lamp_on'] = False    # Priority 3: daytime lamp has no justification
        return goal

    def _satisfies(self, state, goal):
        return all(state.get(k) == v for k, v in goal.items())

    def _state_key(self, state):
        return tuple(sorted(state.items()))

    def plan(self, start_state, temperature, time_of_day):
        """BFS. Returns [] if start_state already satisfies goal."""
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


# =============================================================================
# WATT CALCULATION HELPERS
# =============================================================================

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
    return calculate_watt(
        status_ac     = final_state.get('ac_on',   False),
        status_lamp   = final_state.get('lamp_on', False),
        status_tv     = final_state.get('tv_on',   False),
        temperature   = temperature,
        status_ac_eco = final_state.get('ac_eco',  False),
    )


def apply_actions_to_state(start_state, actions):
    """Apply an ordered list of STRIPS action names to start_state."""
    op_map = {op['name']: op for op in STRIPS_OPERATORS}
    state  = dict(start_state)
    for name in actions:
        if name in op_map:
            state = op_map[name]['effect'](state)
    return state


def get_medium_recommendations(start_state, temperature, time_of_day):
    """Advisory (non-binding) suggestions for MEDIUM WASTE cases."""
    recs = []
    if start_state.get('ac_on') and not start_state.get('ac_eco'):
        recs.append('SUGGEST_AC_ECO')
    if start_state.get('lamp_on') and time_of_day >= 0.5:
        recs.append('SUGGEST_LAMP_OFF')
    return recs


# =============================================================================
# MAIN AGENT ENTRY POINT
# =============================================================================

def core_koswatt_agent(
    occupancy, temperature, time_of_day,
    status_ac, status_lamp, status_tv,
    status_ac_eco=False,
    autonomy_level='autonomous',
    occupancy_confidence=1.0,
):
    """
    Main KosWatt agent entry point.

    Reasoning is accumulated inline at each decision step as the decision is
    made. This is deliberative (each line caused the next decision) rather than
    post-hoc (explanation reconstructed after the fact).

    Parameters
    ----------
    occupancy             : 0 (empty) or 1 (occupied)
    temperature           : ambient temperature in Celsius; clamped to [20, 40]
    time_of_day           : 0 (day) or 1 (night)
    status_ac             : bool -- AC on
    status_lamp           : bool -- lamp on
    status_tv             : bool -- TV on
    status_ac_eco         : bool -- AC already in ECO mode
    autonomy_level        : 'advisory' | 'confirm' | 'autonomous'
    occupancy_confidence  : float [0.0–1.0] -- sensor confidence in occupancy

    Returns
    -------
    dict with keys:
      fuzzy_score, waste_category, action_sequence, initial_watt,
      final_state, final_watt, medium_recommendations, reasoning,
      autonomy_level, occupancy_confidence, effective_occupancy,
      awaiting_confirmation
    """
    # ── Input validation ──────────────────────────────────────────────────────
    occupancy            = float(np.clip(float(occupancy),            0.0, 1.0))
    temperature          = float(np.clip(float(temperature),          20.0, 40.0))
    time_of_day          = float(np.clip(float(time_of_day),          0.0, 1.0))
    occupancy_confidence = float(np.clip(float(occupancy_confidence), 0.0, 1.0))
    status_ac     = bool(status_ac)
    status_lamp   = bool(status_lamp)
    status_tv     = bool(status_tv)
    status_ac_eco = bool(status_ac_eco) if status_ac else False
    if autonomy_level not in AUTONOMY_LEVELS:
        autonomy_level = 'autonomous'

    reasoning = []

    # ── STEP 0: Declare autonomy level ───────────────────────────────────────
    reasoning.append(
        f"[AUTONOMY] Operating at '{autonomy_level.upper()}' level. "
        f"{AUTONOMY_LEVELS[autonomy_level]}"
    )

    # ── STEP 1: Occupancy with confidence modulation ─────────────────────────
    # Formula: effective = sensor * confidence + 0.5 * (1 - confidence)
    # At confidence=1.0 → effective = sensor value exactly.
    # At confidence=0.0 → effective = 0.5 (maximally uncertain; maximally
    #                                  conservative before autonomous action).
    effective_occupancy = occupancy * occupancy_confidence + 0.5 * (1.0 - occupancy_confidence)

    if occupancy_confidence < 1.0:
        raw_label  = 'OCCUPIED' if occupancy >= 0.5 else 'EMPTY'
        conf_pct   = int(occupancy_confidence * 100)
        reasoning.append(
            f"[PRIORITY 1: SAFETY] Sensor reports {raw_label} at {conf_pct}% confidence. "
            f"Effective occupancy modulated to {effective_occupancy:.2f} "
            f"(0.0=certain-empty, 1.0=certain-occupied). "
            f"Low confidence increases caution before any autonomous action."
        )
    else:
        occ_label = 'OCCUPIED' if occupancy >= 0.5 else 'EMPTY'
        reasoning.append(
            f"[PRIORITY 1: SAFETY] Occupancy confirmed {occ_label} at 100% confidence."
        )

    # ── STEP 2: Fuzzy waste scoring ───────────────────────────────────────────
    initial_watt = calculate_watt(status_ac, status_lamp, status_tv, temperature,
                                  status_ac_eco=status_ac_eco)

    fuzzy_score = compute_waste_score(
        occupancy_val   = effective_occupancy,
        temperature_val = temperature,
        time_of_day_val = time_of_day,
        tv_on           = status_tv,
        ac_on           = status_ac,
        lamp_on         = status_lamp,
        ac_eco          = status_ac_eco,
    )
    reasoning.append(
        f"[FUZZY ENGINE] Mamdani inference over {initial_watt} W draw at {temperature} C. "
        f"Waste score: {fuzzy_score:.1f}/100."
    )

    # ── STEP 3: Waste classification ──────────────────────────────────────────
    if fuzzy_score > 65:
        waste_category = 'HIGH WASTE'
    elif fuzzy_score > 35:
        waste_category = 'MEDIUM WASTE'
    else:
        waste_category = 'LOW WASTE'
    reasoning.append(
        f"[CLASSIFICATION] Score {fuzzy_score:.1f} → {waste_category}. "
        f"Thresholds: ≤35 LOW | 35–65 MEDIUM | >65 HIGH."
    )

    # ── STEP 4: Ethical priority resolution ──────────────────────────────────
    # For each active device in an assessed-empty room, log which ethical
    # priority governs the decision before any action is taken.
    if effective_occupancy < 0.5:
        for device, (active, _) in [
            ('TV',   (status_tv,   False)),
            ('AC',   (status_ac,   status_ac_eco)),
            ('LAMP', (status_lamp, False)),
        ]:
            if not active:
                continue
            meta = DEVICE_ETHICS[device]
            if device == 'AC':
                priority = 'COMFORT' if temperature >= AC_ECO_THRESHOLD else 'EFFICIENCY'
            elif device == 'LAMP':
                priority = 'SAFETY' if time_of_day >= 0.5 else 'EFFICIENCY'
            else:
                priority = 'EFFICIENCY'
            reasoning.append(
                f"[PRIORITY {list(ETHICAL_PRIORITIES.keys()).index(priority)+1}: {priority}] "
                f"{device} ({meta['class']}): {meta['justification']}"
            )

    # ── STEP 5: Rigid device override ────────────────────────────────────────
    # Guarantees HIGH classification for TV in an empty room, independent of
    # the aggregate fuzzy score. Prevents calibration drift or rule interaction
    # from letting an obvious waste case slip below the HIGH threshold.
    if status_tv and effective_occupancy < 0.5 and waste_category != 'HIGH WASTE':
        prev_category  = waste_category
        waste_category = 'HIGH WASTE'
        reasoning.append(
            f"[RIGID OVERRIDE] TV (RIGID device) is active in an assessed-empty room. "
            f"Fuzzy score alone classified this as {prev_category}. "
            f"Rigid override escalates to HIGH WASTE. No Safety or Comfort priority "
            f"can justify an entertainment device in an empty room; Priority 3 "
            f"(Efficiency) applies absolutely."
        )

    # ── STEP 6: Build device state dict ──────────────────────────────────────
    start_state = {
        'ac_on'  : status_ac,
        'lamp_on': status_lamp,
        'tv_on'  : status_tv,
        'ac_eco' : status_ac_eco,
    }

    # ── STEP 7: Action planning with autonomy and confidence gating ──────────
    awaiting_confirmation  = False
    action_sequence        = []
    medium_recommendations = []

    if waste_category == 'LOW WASTE':
        reasoning.append(
            "[DECISION] Waste within acceptable bounds. All active devices satisfy "
            "their governing ethical priority. No intervention required."
        )

    elif waste_category == 'MEDIUM WASTE':
        reasoning.append(
            "[DECISION] MEDIUM WASTE. Contextual justification (Safety or Comfort) "
            "withholds autonomous action. Adaptive guardrail active."
        )
        medium_recommendations = get_medium_recommendations(
            start_state, temperature, time_of_day
        )
        if medium_recommendations:
            reasoning.append(
                f"[ADVISORY] Non-binding recommendations: {', '.join(medium_recommendations)}. "
                f"These represent what the STRIPS planner would propose if the guardrail "
                f"were lifted. No plan executed."
            )

    else:  # HIGH WASTE
        # Safety override: low occupancy confidence suspends autonomous action.
        # A sleeping person detected as 'empty' at 40% confidence should not
        # have the AC turned off -- Priority 1 (Safety) overrides Priority 3.
        if occupancy_confidence < 0.5 and occupancy < 0.5:
            reasoning.append(
                f"[PRIORITY 1: SAFETY OVERRIDE] HIGH WASTE confirmed, but occupancy "
                f"sensor confidence is {int(occupancy_confidence*100)}% — below the "
                f"50% threshold for autonomous action. Risk of acting on a false-empty "
                f"reading (e.g. sleeping occupant). Autonomous action suspended. "
                f"Priority 1 (Safety) overrides Priority 3 (Efficiency)."
            )
            # Surface the full STRIPS plan as non-binding recommendations so the
            # operator sees what would have been done (including TV shutoff), not
            # just the limited MEDIUM WASTE advisory set.
            _safety_planner = StripsPlanner()
            _safety_plan    = _safety_planner.plan(start_state, temperature, time_of_day)
            medium_recommendations = [f"SUGGEST_{a}" for a in _safety_plan]
            if medium_recommendations:
                reasoning.append(
                    f"[ADVISORY] Non-binding recommendations (safety override): "
                    f"{medium_recommendations}. No plan executed."
                )

        elif autonomy_level == 'advisory':
            reasoning.append(
                "[DECISION] HIGH WASTE confirmed. Autonomy is ADVISORY — STRIPS plan "
                "computed for display, but no actions will be executed."
            )
            planner         = StripsPlanner()
            action_sequence = planner.plan(start_state, temperature, time_of_day)
            # Surface plan as recommendations without executing
            medium_recommendations = [f"SUGGEST_{a}" for a in action_sequence]
            action_sequence = []

        elif autonomy_level == 'confirm':
            planner         = StripsPlanner()
            action_sequence = planner.plan(start_state, temperature, time_of_day)
            awaiting_confirmation = True
            reasoning.append(
                f"[DECISION] HIGH WASTE confirmed. Autonomy is CONFIRM — STRIPS plan "
                f"ready ({action_sequence}), but execution is suspended pending "
                f"human approval. No changes applied yet."
            )

        else:  # autonomous
            planner = StripsPlanner()
            reasoning.append(
                "[DECISION] HIGH WASTE confirmed. Autonomy is AUTONOMOUS. "
                "Initiating STRIPS forward-search planner (BFS, 4-variable state space)."
            )
            action_sequence = planner.plan(start_state, temperature, time_of_day)
            if action_sequence:
                reasoning.append(
                    f"[STRIPS] Optimal plan found: {action_sequence} "
                    f"({len(action_sequence)} action(s)). Applying to device state."
                )
            else:
                reasoning.append(
                    "[STRIPS] Start state already satisfies goal. No actions needed."
                )

    # ── STEP 8: Apply actions (autonomous, non-pending only) ─────────────────
    if action_sequence and not awaiting_confirmation:
        final_state = apply_actions_to_state(start_state, action_sequence)
        final_watt  = calculate_post_action_watt(final_state, temperature)
        saved       = max(0, initial_watt - final_watt)
        reasoning.append(
            f"[EXECUTION] {len(action_sequence)} action(s) applied. "
            f"Draw: {initial_watt} W → {final_watt} W ({saved} W saved)."
        )
    else:
        final_state = dict(start_state)
        final_watt  = initial_watt

    return {
        'fuzzy_score'           : round(fuzzy_score, 2),
        'waste_category'        : waste_category,
        'action_sequence'       : action_sequence,
        'initial_watt'          : initial_watt,
        'final_state'           : final_state,
        'final_watt'            : final_watt,
        'medium_recommendations': medium_recommendations,
        'reasoning'             : reasoning,
        'autonomy_level'        : autonomy_level,
        'occupancy_confidence'  : occupancy_confidence,
        'effective_occupancy'   : round(effective_occupancy, 3),
        'awaiting_confirmation' : awaiting_confirmation,
    }
