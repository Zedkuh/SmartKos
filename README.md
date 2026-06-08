# KosWatt

**Smart Energy Agentic AI -- Boarding House Energy Monitor**

KosWatt is a symbolic AI system that monitors room energy consumption and autonomously intervenes when device usage is ethically unjustified. It combines a Mamdani fuzzy inference engine for waste classification with a STRIPS forward-search planner for action selection, governed by an explicit three-priority ethical framework.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Ethical Framework](#ethical-framework)
- [Fuzzy Inference Rules](#fuzzy-inference-rules)
- [STRIPS Planner](#strips-planner)
- [Autonomy Levels](#autonomy-levels)
- [Occupancy Confidence](#occupancy-confidence)
- [Device Classification](#device-classification)
- [Wattage Model](#wattage-model)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Simulation Controls](#simulation-controls)
- [Design Decisions and Limitations](#design-decisions-and-limitations)

---

## Overview

KosWatt simulates an AI agent managing three devices in a single boarding house room: an air conditioning unit, a lamp, and a television. At each cycle the agent:

1. Reads occupancy, ambient temperature, and time of day.
2. Applies occupancy confidence modulation to account for sensor imperfection.
3. Runs Mamdani fuzzy inference to score energy waste on a 0 to 100 scale.
4. Resolves a three-priority ethical hierarchy to determine whether intervention is justified.
5. If waste is HIGH and the autonomy level permits, invokes a STRIPS BFS planner to generate the minimal corrective action sequence.
6. Produces a deliberative reasoning trace, accumulated inline at each decision step.

The system is intentionally symbolic: no machine learning or training is involved. All ethical design decisions are explicit, auditable constants rather than learned weights or hidden heuristics.

---

## Architecture

```
Sensor Inputs
    occupancy, temperature, time_of_day, device states
        |
        v
Occupancy Confidence Modulation
    effective_occupancy = sensor * confidence + 0.5 * (1 - confidence)
        |
        v
Mamdani Fuzzy Inference Engine
    7 rules over occupancy, temperature, time of day, and device context
    Output: waste score 0 to 100
        |
        v
Ethical Priority Resolution
    Priority 1: Safety  >  Priority 2: Comfort  >  Priority 3: Efficiency
    Per-device classification logged before any action is taken
        |
        v
Rigid Device Override
    TV in empty room -> guaranteed HIGH WASTE regardless of fuzzy score
        |
        v
Autonomy Gate
    advisory  -> recommendations only, no execution
    confirm   -> plan generated, execution suspended pending approval
    autonomous -> STRIPS planner executes immediately
        |
        v
STRIPS BFS Planner  (HIGH WASTE + autonomous only)
    State space: {ac_on, ac_eco, lamp_on, tv_on}  (2^4 = 16 states)
    Guarantees shortest action sequence to goal state
        |
        v
Deliberative Reasoning Trace
    Each line generated at the point the decision is made
```

---

## Ethical Framework

The priority hierarchy is defined as explicit, auditable constants in `koswatt_agent.py` under `ETHICAL_PRIORITIES`. Every automated decision references one of the three priorities by name.

| Priority | Name       | Scope |
|----------|------------|-------|
| 1        | SAFETY     | Preserve safe conditions. Security lighting at night. Protection against thermal extremes. No efficiency gain overrides this. |
| 2        | COMFORT    | Respect occupant comfort within reasonable energy bounds. Thermal regulation with environmental justification takes precedence over efficiency. |
| 3        | EFFICIENCY | Minimise energy waste when Safety and Comfort are already satisfied. Drives all autonomous shutoff and ECO-mode decisions. |

---

## Fuzzy Inference Rules

The Mamdani engine uses trimf (triangular) membership functions over the following universes:

- **Occupancy**: 0.0 (empty) to 1.0 (occupied)
- **Temperature**: 20 C to 40 C
- **Time of day**: 0.0 (day) to 1.0 (night)
- **Waste output**: 0 to 100

| Rule | Condition | Output | Ethical Basis |
|------|-----------|--------|---------------|
| R1  | Occupied | LOW | Priority 2: Comfort -- occupants have a right to their devices |
| R2  | Empty + TV on | HIGH | Priority 3: Efficiency -- TV is RIGID, no contextual exception |
| R3  | Empty + AC (full mode) + hot ambient (>= 28 C) | MEDIUM | Priority 2: Comfort -- pre-cooling withholds full shutoff |
| R3b | Empty + AC (ECO mode) + hot ambient (>= 28 C) | LOW | Already at the energy-optimal state, no further action |
| R4  | Empty + AC (full mode) + cool ambient (< 28 C) | HIGH | Priority 3: Efficiency -- no thermal justification |
| R5  | Empty + Lamp + daytime | HIGH | Priority 3: Efficiency -- natural light available |
| R6  | Empty + Lamp + nighttime | MEDIUM | Priority 1: Safety -- security and re-entry lighting |

Rules R3b and R4 are gated on ECO mode state. This prevents the HIGH/no-actions contradiction that occurs at exactly T = 28 C, where `temp_cool` membership is 0.2 but the STRIPS goal builder already considers ECO acceptable (threshold >= 28 C).

Waste classification thresholds:

| Score | Category |
|-------|----------|
| 0 to 35 | LOW WASTE |
| 35 to 65 | MEDIUM WASTE |
| > 65 | HIGH WASTE |

---

## STRIPS Planner

The planner uses BFS forward search over the 4-variable boolean device state space. It is only invoked for HIGH WASTE cases when the autonomy level permits execution.

**Operators:**

| Operator | Precondition | Effect |
|----------|-------------|--------|
| TURN_OFF_TV | tv_on | tv_on = False |
| TURN_OFF_LAMP | lamp_on | lamp_on = False |
| TURN_OFF_AC | ac_on and not ac_eco | ac_on = False, ac_eco = False |
| TURN_OFF_AC_ECO | ac_on and ac_eco | ac_on = False, ac_eco = False |
| SET_AC_TO_ECO | ac_on and not ac_eco | ac_eco = True |

**Goal construction** applies the same ethical priorities as the fuzzy engine:

- TV: always off in an empty room (RIGID, Priority 3)
- AC: ECO mode if ambient temperature >= 28 C (Priority 2), off otherwise (Priority 3)
- Lamp: off during daytime (Priority 3), tolerated at night (Priority 1)

BFS guarantees the shortest valid action sequence. The state space is small enough (maximum 16 states) that BFS is appropriate; the architecture is designed to scale to larger device sets.

---

## Autonomy Levels

| Level | Behaviour |
|-------|-----------|
| `autonomous` | HIGH WASTE triggers immediate STRIPS plan execution without human confirmation. Default. |
| `confirm` | HIGH WASTE generates a ready-to-execute plan, shown with projected watt savings. Execution is suspended until the operator approves. |
| `advisory` | The agent generates non-binding recommendations only. No device state is changed regardless of waste classification. |

Autonomy level is configurable in the sidebar at runtime.

---

## Occupancy Confidence

Real motion sensors are imperfect. A sleeping occupant or a stationary person may register as an empty room. The `occupancy_confidence` parameter (0.0 to 1.0) modulates the raw sensor reading before it enters the fuzzy engine:

```
effective_occupancy = sensor_reading * confidence + 0.5 * (1 - confidence)
```

At 100% confidence, the effective value equals the raw reading.
At 0% confidence, the effective value is 0.5 (maximally uncertain).

An additional hard safety override applies when confidence drops below 50% and the room reads empty: **autonomous action is suspended entirely**, regardless of waste category. Priority 1 (Safety) prevents the system from acting on a potentially false-empty reading.

---

## Device Classification

Defined in `DEVICE_ETHICS` in `koswatt_agent.py`.

| Device | Class | Justification |
|--------|-------|---------------|
| TV | RIGID | Entertainment device. No safety, comfort, or thermal justification exists for operation in an unoccupied room. No contextual exception is granted. |
| AC | ADAPTIVE | Thermal regulation device. At or above 28 C, ECO-mode operation satisfies Priority 2 (Comfort) for returning occupants. Below 28 C, no thermal justification exists. |
| Lamp | ADAPTIVE | At night, satisfies Priority 1 (Safety) via security deterrence and re-entry illumination. Daytime operation has no Safety or Comfort justification. |

RIGID devices are also subject to the rigid device override: if the fuzzy engine's aggregate score falls below HIGH for any calibration reason, the override guarantees HIGH classification for a RIGID device in an empty room.

---

## Wattage Model

| Device | State | Draw |
|--------|-------|------|
| Standby (always) | -- | 10 W |
| Lamp | On | 15 W |
| TV | On | 100 W |
| AC | On, ambient > 30 C | 450 W |
| AC | On, ambient <= 30 C | 280 W |
| AC | ECO mode | 150 W |

The watt draw is used for display and savings estimation only. It does not influence the fuzzy waste classification, which is driven entirely by device context and occupancy.

---

## Installation

**Requirements:** Python 3.9 or later.

```bash
pip install streamlit plotly numpy scikit-fuzzy
```

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard opens in the default browser at `http://localhost:8501`.

---

## Project Structure

```
.
+-- app.py               Streamlit dashboard (UI, session state, rendering)
+-- koswatt_agent.py     AI core (fuzzy engine, STRIPS planner, ethical framework)
+-- README.md            This file
```

**`koswatt_agent.py` exports:**

| Symbol | Type | Description |
|--------|------|-------------|
| `core_koswatt_agent` | function | Main agent entry point |
| `ETHICAL_PRIORITIES` | OrderedDict | Priority hierarchy (Safety, Comfort, Efficiency) |
| `DEVICE_ETHICS` | dict | Per-device ethical classification and justification |
| `AUTONOMY_LEVELS` | dict | Descriptions for each autonomy level |
| `DEVICE_REGISTRY` | dict | Documentation scaffold listing all registered devices |
| `STRIPS_OPERATORS` | list | BFS operator definitions |
| `compute_waste_score` | function | Mamdani fuzzy inference engine |
| `StripsPlanner` | class | BFS forward-search planner |
| `calculate_watt` | function | Power draw estimation |
| `AC_ECO_THRESHOLD` | int | Ethical temperature threshold (28 C) |

---

## Simulation Controls

The sidebar exposes all simulation parameters at runtime.

**Room State**

| Control | Values | Description |
|---------|--------|-------------|
| Occupancy | Room Empty / Room Occupied | Binary occupancy sensor reading |
| Room Temperature | 20 C to 40 C | Ambient temperature in Celsius |
| Time of Day | Daytime / Nighttime | Used by lamp adaptive rule (R5/R6) |

**Device Switch State**

| Control | Description |
|---------|-------------|
| AC Unit | Toggle AC on/off |
| AC in ECO mode | Toggle ECO mode (only available when AC is on) |
| Lamp | Toggle lamp on/off |
| TV | Toggle TV on/off |

**Agent Settings**

| Control | Values | Description |
|---------|--------|-------------|
| Autonomy Level | Autonomous / Confirm First / Advisory Only | Controls whether the agent acts, waits, or only advises |
| Sensor Confidence | 0% to 100% | Visible when room reads empty. Modulates effective occupancy. Below 50%, the safety override suspends autonomous action. |

**Session Stats**

Displayed in the sidebar across the current browser session:

- Total readings logged
- High waste event count
- Medium waste event count
- Average watts saved per high waste action (projected for non-autonomous modes)

---

## Design Decisions and Limitations

**Ethical rules are designer decisions, not learned values.**
All thresholds and device classifications are explicit constants with documented justifications. This is true of all symbolic AI systems. The advantage is full auditability; the tradeoff is that the ethics do not adapt to new contexts without manual update.

**Why STRIPS over a direct rule lookup?**
With 4 boolean variables the state space is small (16 states). BFS is trivially fast here. STRIPS is used because it demonstrates the principle of automated plan generation and is extensible to a larger device set without algorithmic changes, only new operator definitions.

**Reasoning is deliberative, not post-hoc.**
The reasoning trace is accumulated at each decision step as the decision is made. Each line in the trace is the direct output of the logic that produced the next decision. This contrasts with post-hoc explanation, where a rationale is reconstructed after all decisions are complete.

**Occupancy is binary in the sensor input but continuous in inference.**
The UI presents a binary room state toggle for simplicity. The confidence slider introduces continuous uncertainty into the fuzzy engine's effective occupancy value, allowing the system to model imperfect sensing without requiring a probabilistic sensor model.

**No machine learning.**
KosWatt uses fuzzy logic and classical planning. There is no training, no gradient descent, and no learned parameters. It is a symbolic AI system in the tradition of knowledge-based expert systems, extended with an explicit ethical priority hierarchy and an autonomy control layer.
