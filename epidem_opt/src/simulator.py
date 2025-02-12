from datetime import date, timedelta

from epidem_opt.src import epistep



def simulate(m, endDate, drop_before=date(year=2000, month=1, day=1)):
    # TODO: make start state an ndarray
    state = m.start_state()
    trajectories = []
    cur_date = m.start_date
    idx = 1
    print(f"Start date {cur_date}")
    while cur_date <= endDate:
        (S, E, inf, R, V, day) = state

        # STEP 1: reset flu cycle
        if (cur_date.month, cur_date.day) == m.peak_date:
            print(f"Reseting flu cycle {cur_date} (day {idx}:{day})")
            day = 0
            state = (S, E, inf, R, V, day)

        # STEP 2: add disease
        if (cur_date.month, cur_date.day) == m.seed_date:
            print(f"Seeding infections {cur_date} (day {idx}:{day})")
            state = epistep.seedInfs(m, *state)

        # STEP 3: apply step
        # (newS, newE, newInf, newR, newV, day + 1,
        #             ambCost, noMedCost, hospCost, vaxCost,
        #             ambQaly, noMedQaly, hospQaly, lifeyrsLost)
        ext_state = epistep.step(m, *state)

        # state = (newS, newE, newInf, newR, newV, day)
        state = ext_state[0:6]

        # TODO: call m.switchProgram("prog name") after an appropriate number of days

        # STEP 4: perform vaccination
        if (cur_date.month, cur_date.day) == m.vacc_date:
            print(f"Vaccinating {cur_date} (day {idx}:{day})")
            (*state, extra_vax_cost) = epistep.vaccinate(m, m.vacc_rates, *state)
            (*rest, vc, aq, nmq, hq, ll) = ext_state
            ext_state = (*rest, extra_vax_cost + vc, aq, nmq, hq, ll)

        # STEP 5: register current values
        if cur_date >= drop_before:
            assert len(ext_state) == 14  # Sanity check
            trajectories.append(ext_state)

        # STEP 6: apply aging
        if (cur_date.month, cur_date.day) == m.birthday:
            print(f"Aging population {cur_date} (day {idx}:{day})")
            state = epistep.age(m, *state)

        cur_date = cur_date + timedelta(days=1)
        idx += 1
    print(f"End date {cur_date} (day {idx}:{day})")
    return trajectories
