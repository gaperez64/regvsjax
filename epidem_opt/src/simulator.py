from datetime import date, timedelta

from epidem_opt.src import epistep


def updateVaxCost(t, vaxCost):
    # The last five entries are
    # vaxCosts, ambulatoryCosts,
    # noMedCosts, hospCosts, lifeyrsLost
    (*rest, vc, aq, nmq, hq, ll) = t
    return (*rest, vaxCost + vc, aq, nmq, hq, ll)



def simulate(m, endDate, dropBefore=date(year=2000, month=1, day=1)):
    # TODO: run this, and then refactor step by step
    #   - replace state with extState
    #   - maybe find a more elegant solution, where stuff is put in a dict or object.
    # TODO: make start state an ndarray
    state = m.startState()
    trajectories = []
    curDate = m.startDate
    idx = 1
    print(f"Start date {curDate}")
    while curDate <= endDate:
        (S, E, Inf, R, V, day) = state

        if (curDate.month, curDate.day) == m.peakDate:
            print(f"Reseting flu cycle {curDate} (day {idx}:{day})")
            day = 0
            state = (S, E, Inf, R, V, day)

        if (curDate.month, curDate.day) == m.seedDate:
            print(f"Seeding infections {curDate} (day {idx}:{day})")
            state = epistep.seedInfs(m, *state)

        # (newS, newE, newInf, newR, newV, day + 1,
        #             ambCost, noMedCost, hospCost, vaxCost,
        #             ambQaly, noMedQaly, hospQaly, lifeyrsLost)
        extState = epistep.step(m, *state)

        # state = (newS, newE, newInf, newR, newV, day)
        state = extState[0:6]

        # TODO: call m.switchProgram("prog name") after an appropriate number of days

        if (curDate.month, curDate.day) == m.vaccDate:
            print(f"Vaccinating {curDate} (day {idx}:{day})")
            # vaxdState = (newS, newE, newInf, newR, newV, day, vaxCost)
            vaxdState = epistep.vaccinate(m, m.vaccRates, *state)
            state = vaxdState[0:6]
            if curDate >= dropBefore:
                extState = updateVaxCost(extState, vaxdState[-1])

        if curDate >= dropBefore:
            trajectories.append(extState)

        if (curDate.month, curDate.day) == m.birthday:
            print(f"Aging population {curDate} (day {idx}:{day})")
            state = epistep.age(m, *state)

        curDate = curDate + timedelta(days=1)
        idx += 1
    print(f"End date {curDate} (day {idx}:{day})")
    return trajectories
