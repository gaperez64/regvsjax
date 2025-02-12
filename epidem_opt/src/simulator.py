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

        if (cur_date.month, cur_date.day) == m.peak_date:
            print(f"Reseting flu cycle {cur_date} (day {idx}:{day})")
            day = 0
            state = (S, E, inf, R, V, day)

        if (cur_date.month, cur_date.day) == m.seed_date:
            print(f"Seeding infections {cur_date} (day {idx}:{day})")
            state = epistep.seedInfs(m, *state)

        # (newS, newE, newInf, newR, newV, day + 1,
        #             ambCost, noMedCost, hospCost, vaxCost,
        #             ambQaly, noMedQaly, hospQaly, lifeyrsLost)
        extState = epistep.step(m, *state)

        # state = (newS, newE, newInf, newR, newV, day)
        state = extState[0:6]

        # TODO: call m.switchProgram("prog name") after an appropriate number of days

        if (cur_date.month, cur_date.day) == m.vacc_date:
            print(f"Vaccinating {cur_date} (day {idx}:{day})")
            # vaxdState = (newS, newE, newInf, newR, newV, day, vaxCost)
            vaxd_state = epistep.vaccinate(m, m.vacc_rates, *state)
            state = vaxd_state[0:6]
            # if cur_date >= drop_before:
            (*rest, vc, aq, nmq, hq, ll) = extState
            extState = (*rest, vaxd_state[-1] + vc, aq, nmq, hq, ll)
                # extState = update_vax_cost(extState, vaxd_state[-1])

        if cur_date >= drop_before:
            trajectories.append(extState)

        if (cur_date.month, cur_date.day) == m.birthday:
            print(f"Aging population {cur_date} (day {idx}:{day})")
            state = epistep.age(m, *state)

        cur_date = cur_date + timedelta(days=1)
        idx += 1
    print(f"End date {cur_date} (day {idx}:{day})")
    return trajectories
