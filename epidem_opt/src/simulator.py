from datetime import date, timedelta
from typing import Callable


from epidem_opt.src import epistep
from epidem_opt.src.epidata import EpiData, JaxFriendlyEpiData

import jax.numpy as jnp


def _get_max_compartment_difference_(actual, expected):
    """
        This is the infinity norm of the difference between the two specified vectors.
        Each vector is expected to be a compartment of 100 age groups.
    """
    expected = jnp.asarray(expected)
    diff = jnp.max(jnp.abs(actual - expected))
    return diff


def _check_pop_conservation_(old, new):
    """
        Verify that the total pop before and after the iteration differs by less than 10.000.
    """
    (S, E, I, R, V, day) = old
    (Snew, Enew, Inew, Rnew, Vnew, daynew) = new
    tot_pop_before = S.sum() + E.sum() + I.sum() + R.sum() + V.sum()
    tot_pop_after = Snew.sum() + Enew.sum() + Inew.sum() + Rnew.sum() + Vnew.sum()
    assert abs(tot_pop_before - tot_pop_after) <= 10000, f"Difference in population: {tot_pop_before-tot_pop_after}"
    # Max difference: 5 * 100 * 20 = 10000


def date_to_ordinal_set(periodic_date: tuple[int, int], begin_date: date, end_date: date):
    """
        Return a set that contains all dates between begin_date and end_date, inclusive, that has the same
        day and month als the specified tuple (month, day).

        Each date in the set is specified as an ordinal: datetime.toordinal().
    """
    # month, day
    ordinal_set: set[int] = set()
    cur_date_temp = begin_date
    while cur_date_temp <= end_date:
        if (cur_date_temp.month, cur_date_temp.day) == periodic_date:
            ordinal_set.add(cur_date_temp.toordinal())
        cur_date_temp = cur_date_temp + timedelta(days=1)
    return ordinal_set


def simulate_trajectories(epi_data: EpiData,
                          begin_date: date,
                          start_state,
                          end_date: date,
                          drop_before: date = date(year=2000, month=1, day=1),
                          enforce_invariant: bool = False):
    """
        Wrapper that accepts datetime objects.
    """

    begin_date_int = begin_date.toordinal()
    end_date_int = end_date.toordinal()
    drop_before_int = drop_before.toordinal()

    vacc_dates = date_to_ordinal_set(epi_data.vacc_date, begin_date, end_date)
    peak_dates = date_to_ordinal_set(epi_data.peak_date, begin_date, end_date)
    seed_dates = date_to_ordinal_set(epi_data.seed_date, begin_date, end_date)
    birth_dates = date_to_ordinal_set(epi_data.birthday, begin_date, end_date)

    return _internal_sim_(epi_data=epi_data, begin_date=begin_date_int, end_date=end_date_int,
                          vacc_dates=vacc_dates, birth_dates=birth_dates, seed_dates=seed_dates, peak_dates=peak_dates,
                          start_state=start_state, drop_before=drop_before_int,
                          enforce_invariant=enforce_invariant)


def _internal_sim_(epi_data: EpiData,
                   begin_date: int,
                   start_state,
                   end_date: int,
                   vacc_dates: set[int],
                   birth_dates: set[int],
                   seed_dates: set[int],
                   peak_dates: set[int],
                   drop_before: int,
                   enforce_invariant: bool
                   ):
    """
        Function that simulates the epidemic and returns the trajectories.

        This is used for the "burn-in" step.

        Simulator loop that only accepts integers as dates.
    """

    epi_state = start_state
    trajectories = []
    cur_date = begin_date
    # idx = 1
    print(f"Start date {cur_date}")
    while cur_date <= end_date:
        (S, E, inf, R, V, day) = epi_state

        # STEP 1: reset flu cycle
        if cur_date in peak_dates:
            print(f"Reseting flu cycle {cur_date} (day: {day})")
            day = 0
            epi_state = (S, E, inf, R, V, day)

        # STEP 2: add disease
        if cur_date in seed_dates:
            print(f"Seeding infections {cur_date} (day: {day})")
            epi_state = epistep.seedInfs(epi_data, *epi_state)

        # STEP 3: apply step
        # (newS, newE, newInf, newR, newV, day + 1,
        #             ambCost, noMedCost, hospCost, vaxCost,
        #             ambQaly, noMedQaly, hospQaly, lifeyrsLost)
        epi_state_old = epi_state
        ext_state = epistep.step(epi_data, *epi_state)

        # sanity check
        if enforce_invariant:
            _check_pop_conservation_(old=epi_state_old[0:6], new=ext_state[0:6])

        # state = (newS, newE, newInf, newR, newV, day)
        epi_state = ext_state[0:6]

        # TODO: call m.switchProgram("prog name") after an appropriate number of days

        # STEP 4: perform vaccination
        if cur_date in vacc_dates:
            print(f"Vaccinating {cur_date} (day: {day})")
            (*epi_state, extra_vax_cost) = epistep.vaccinate(epi_data, epi_data.vacc_rates, *epi_state)
            (*rest, vc, aq, nmq, hq, ll) = ext_state
            ext_state = (*rest, extra_vax_cost + vc, aq, nmq, hq, ll)

        # STEP 5: register current values
        if drop_before <= cur_date:
            assert len(ext_state) == 14  # Sanity check
            trajectories.append(ext_state)

        # STEP 6: apply aging
        if cur_date in birth_dates:
            print(f"Aging population {cur_date} (day: {day})")
            epi_state = epistep.age(epi_data, *epi_state)

        cur_date += 1
        # idx += 1
    print(f"End date {cur_date} (day: {day})")
    return trajectories


def simulate_cost(vacc_rates,
                  epi_data: JaxFriendlyEpiData,
                  epi_state,
                  start_date: int,
                  end_date: int,
                  vacc_dates: Callable[[int], bool],
                  birth_dates: Callable[[int], bool],
                  seed_dates: Callable[[int], bool],
                  peak_dates: Callable[[int], bool]):
    """
        Function that simulates the epidemic and computes all the costs, including the QALY cost.

        "state" is the initial state.
        This function is used for the gradient descent step. Do NOT add any state-modifying behaviour or I/O.
    """
    cur_date = start_date
    total_cost = 0
    while cur_date <= end_date:

        # STEP 1: reset flu cycle
        # if (cur_date.month, cur_date.day) == epi_data.peak_date:
        # if cur_date in peak_dates:
        if peak_dates(cur_date):
            # jax.debug.print("peak date")
            # print("peak date")
            (S, E, Inf, R, V, day) = epi_state
            day = 0
            epi_state = (S, E, Inf, R, V, day)

        # STEP 2: add disease
        # if (cur_date.month, cur_date.day) == epi_data.seed_date:
        # if cur_date in seed_dates:
        if seed_dates(cur_date):
            # jax.debug.print("seed date")
            # print("seed date")
            epi_state = epistep.seedInfs(epi_data, *epi_state)

        # STEP 3: apply step
        (*epi_state,
         amb_cost, nomed_cost, hosp_cost, vax_cost,
         amb_qaly, nomed_qaly, hosp_qaly, lifeyrs_lost) = epistep.step(epi_data, *epi_state)

        # STEP 4: perform vaccination
        # if (cur_date.month, cur_date.day) == epi_data.vacc_date:
        # if cur_date in vacc_dates:
        if vacc_dates(cur_date):
            # jax.debug.print("vacc date")
            # print("vacc date")
            (*epi_state, extra_vax_cost) = epistep.vaccinate(epi_data, vacc_rates, *epi_state)
            vax_cost += extra_vax_cost

        # STEP 5: register current values
        # print("BEFORE", cur_date, total_cost, type(total_cost))
        # diff_cost = int(
        diff_cost = (
            ((amb_cost.sum() +
                        nomed_cost.sum() +
                        hosp_cost.sum() +
                        vax_cost.sum()) +
                       (amb_qaly.sum() +
                        nomed_qaly.sum() +
                        hosp_qaly.sum() +
                        lifeyrs_lost.sum()) * 35000)
        )

        # if diff_cost >  13049654503:
            # print("\t econ", (amb_cost.sum() +
            #             nomed_cost.sum() +
            #             hosp_cost.sum() +
            #             vax_cost.sum()))
            # print("\t\t", amb_cost.sum())
            # print("\t\t", nomed_cost.sum())
            # print("\t\t", hosp_cost.sum())
            # print("\t\t", vax_cost.sum())
            # print("\t qaly", (amb_qaly.sum() +
            #             nomed_qaly.sum() +
            #             hosp_qaly.sum() +
            #             lifeyrs_lost.sum()) * 35000)
            # print("\t\t", amb_qaly.sum())
            # print("\t\t", nomed_qaly.sum())
            # print("\t\t", hosp_qaly.sum())
            # print("\t\t", lifeyrs_lost.sum())

        # print("DIFFERENCE (date, additional cost, data type)", cur_date, diff_cost, type(diff_cost))
        total_cost += diff_cost  # TODO: is this constant the QALY constant?
        # print("AFTER", cur_date, total_cost, type(total_cost))
        # jax.debug.print("currrent cost {x}", x=total_cost)
        # print("--")

        # STEP 6: apply aging
        # if (cur_date.month, cur_date.day) == epi_data.birthday:
        # if cur_date in birth_dates:
        if birth_dates(cur_date):
            # jax.debug.print("birth date")
            # print("birth date")
            epi_state = epistep.age(epi_data, *epi_state)

        # cur_date = cur_date + timedelta(days=1)
        cur_date = cur_date + 1
    return total_cost
