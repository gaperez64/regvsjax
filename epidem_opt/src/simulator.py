from datetime import date, timedelta
from pathlib import Path

from epidem_opt.src import epistep
from epidem_opt.src.epidata import EpiData, JaxFriendlyEpiData


def check_pop_conservation(old, new):
    (S, E, I, R, V, day) = old
    (Snew, Enew, Inew, Rnew, Vnew, daynew) = new
    tot_pop_before = S.sum() + E.sum() + I.sum() + R.sum() + V.sum()
    tot_pop_after = Snew.sum() + Enew.sum() + Inew.sum() + Rnew.sum() + Vnew.sum()
    # assert tot_pop_before == tot_pop_after, f"Difference in population: {tot_pop_before-tot_pop_after}"
    # TODO: check that this matches within 20 units.


def simulate_trajectories(epi_data: EpiData, state_date: date, start_state,
                          end_date: date, drop_before: date=date(year=2000, month=1, day=1)):
    """
        Function that simulates the epidemic and returns the trajectories.

        This is used for the "burn-in" step.
    """
    state = start_state
    trajectories = []
    cur_date = state_date
    idx = 1
    print(f"Start date {cur_date}")
    while cur_date <= end_date:
        (S, E, inf, R, V, day) = state

        # STEP 1: reset flu cycle
        if (cur_date.month, cur_date.day) == epi_data.peak_date:
            print(f"Reseting flu cycle {cur_date} (day {idx}:{day})")
            day = 0
            state = (S, E, inf, R, V, day)

        # STEP 2: add disease
        if (cur_date.month, cur_date.day) == epi_data.seed_date:
            print(f"Seeding infections {cur_date} (day {idx}:{day})")
            state = epistep.seedInfs(epi_data, *state)

        # STEP 3: apply step
        # (newS, newE, newInf, newR, newV, day + 1,
        #             ambCost, noMedCost, hospCost, vaxCost,
        #             ambQaly, noMedQaly, hospQaly, lifeyrsLost)
        ext_state = epistep.step(epi_data, *state)

        # sanity check
        check_pop_conservation(state, ext_state[0:6])

        # state = (newS, newE, newInf, newR, newV, day)
        state = ext_state[0:6]

        # TODO: call m.switchProgram("prog name") after an appropriate number of days

        # STEP 4: perform vaccination
        if (cur_date.month, cur_date.day) == epi_data.vacc_date:
            print(f"Vaccinating {cur_date} (day {idx}:{day})")
            (*state, extra_vax_cost) = epistep.vaccinate(epi_data, epi_data.vacc_rates, *state)
            (*rest, vc, aq, nmq, hq, ll) = ext_state
            ext_state = (*rest, extra_vax_cost + vc, aq, nmq, hq, ll)

        # STEP 5: register current values
        if cur_date >= drop_before:
            assert len(ext_state) == 14  # Sanity check
            trajectories.append(ext_state)

        # STEP 6: apply aging
        if (cur_date.month, cur_date.day) == epi_data.birthday:
            print(f"Aging population {cur_date} (day {idx}:{day})")
            state = epistep.age(epi_data, *state)

        cur_date = cur_date + timedelta(days=1)
        idx += 1
    print(f"End date {cur_date} (day {idx}:{day})")
    return trajectories


def simulate_cost(vacc_rates, epi_data, state, start_date, end_date):
    """
        Function that simulates the epidemic and computes all the costs, including the QALY cost.

        "state" is the initial state.
        This function is used for the gradient descent step. Do NOT add any state-modifying behaviour or I/O.
    """
    cur_date = start_date
    total_cost = 0
    while cur_date <= end_date:

        # STEP 1: reset flu cycle
        if (cur_date.month, cur_date.day) == epi_data.peak_date:
            (S, E, Inf, R, V, day) = state
            day = 0
            state = (S, E, Inf, R, V, day)

        # STEP 2: add disease
        if (cur_date.month, cur_date.day) == epi_data.seed_date:
            state = epistep.seedInfs(epi_data, *state)

        # STEP 3: apply step
        (*state,
         amb_cost, nomed_cost, hosp_cost, vax_cost,
         amb_qaly, nomed_qaly, hosp_qaly, lifeyrs_lost) = epistep.step(epi_data, *state)

        # STEP 4: perform vaccination
        if (cur_date.month, cur_date.day) == epi_data.vacc_date:
            (*state, extra_vax_cost) = epistep.vaccinate(epi_data, vacc_rates, *state)
            vax_cost += extra_vax_cost

        # STEP 5: register current values
        total_cost += ((amb_cost.sum() +
                        nomed_cost.sum() +
                        hosp_cost.sum() +
                        vax_cost.sum()) +
                       (amb_qaly.sum() +
                        nomed_qaly.sum() +
                        hosp_qaly.sum() +
                        lifeyrs_lost.sum()) * 35000)  # TODO: is this constant the QALY constant?

        # STEP 6: apply aging
        if (cur_date.month, cur_date.day) == epi_data.birthday:
            state = epistep.age(epi_data, *state)

        cur_date = cur_date + timedelta(days=1)
    return total_cost

