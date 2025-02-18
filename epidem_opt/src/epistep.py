import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


@partial(jax.jit, static_argnums=0)
def vaccinate(epi_data, vacc_rates, S, E, Inf, R, V, day):
    """
        This is NOT something that's easy to make a method of the model class because
        the vaccination rates and costs could change after switching vaccination
        strategy! To optimize via JAX compilation, this version of the function is
        externalized.
    """
    # vaccination = element-wise product with vaccRates
    S2V = S * vacc_rates
    E2V = E * vacc_rates
    I2V = Inf * vacc_rates
    R2V = R * vacc_rates
    # updates
    new_s = S - S2V
    new_e = E - E2V
    new_inf = Inf - I2V
    new_r = R - R2V
    new_v = V + S2V + E2V + I2V + R2V

    # cost
    vax_cost = (new_v - V) * epi_data.vacc_costs
    return new_s, new_e, new_inf, new_r, new_v, day, vax_cost


@partial(jax.jit, static_argnums=0)
def seedInfs(epi_data, S, E, Inf, R, V, day):
    newS = S.at[epi_data.seed_ages].add(-epi_data.seed_inf)
    newInf = Inf.at[epi_data.seed_ages].add(epi_data.seed_inf)
    return newS, E, newInf, R, V, day


@partial(jax.jit, static_argnums=0)
def step(epi_data, S, E, Inf, R, V, day):
    """
    We simulate one step of (forward) Newton integration
    with h = 1 (day). For efficiency, the method is JIT compiled
    NOTE: if any object attributes change after the first call,
    this will result in incorrect results as we assume config
    to be static
    """
    z = 1 + epi_data.delta * jnp.sin(2 * np.pi * (day / 365))
    beta = epi_data.contact * epi_data.q
    force = z * jnp.dot(beta, Inf)

    # daily transitions
    S2E = S * force  # element-wise product
    E2I = E * epi_data.sigma
    I2R = Inf * epi_data.gamma
    R2S = R * epi_data.omega_imm
    V2S = V * epi_data.omega_vacc

    # daily mortality = element-wise product with mortality
    # rates
    S2D = S * epi_data.dailyMort
    E2D = E * epi_data.dailyMort
    I2D = Inf * epi_data.dailyMort
    R2D = R * epi_data.dailyMort
    V2D = V * epi_data.dailyMort

    # new values for all components
    newS = S - S2E + R2S + V2S - S2D
    newE = E + S2E - E2I - E2D
    newInf = Inf + E2I - I2R - I2D
    newR = R + I2R - R2S - R2D
    newV = V - V2S - V2D

    # breakdown of new infections
    confirmedInf = E2I * epi_data.influenzaRate
    noMedCare = (confirmedInf / epi_data.no_med_care) - confirmedInf
    hospd = confirmedInf * epi_data.hospRate
    fatal = confirmedInf * epi_data.caseFatalityRate

    # costs
    ambCost = (confirmedInf - hospd) * epi_data.ambulatory_costs
    noMedCost = noMedCare * epi_data.nomedCosts
    hospCost = hospd * (epi_data.hospCosts + epi_data.hospAmbCosts)
    vaxCost = 0

    # qalys
    ambQaly = (confirmedInf - hospd) * epi_data.ambulatory_qalys
    noMedQaly = noMedCare * epi_data.nomed_qalys
    hospQaly = hospd * epi_data.hosp_qalys
    lifeyrsLost = fatal * epi_data.disc_life_ex

    return (newS, newE, newInf, newR, newV, day + 1,
            ambCost, noMedCost, hospCost, vaxCost,
            ambQaly, noMedQaly, hospQaly, lifeyrsLost)


@partial(jax.jit, static_argnums=0)
def age(config, S, E, Inf, R, V, day):
    totPop = config.tot_pop
    newS = jnp.roll(S, 1).at[0].set(0)
    newE = jnp.roll(E, 1).at[0].set(0)
    newInf = jnp.roll(Inf, 1).at[0].set(0)
    newR = jnp.roll(R, 1).at[0].set(0)
    newV = jnp.roll(V, 1).at[0].set(0)
    # reincarnate dead people
    curPop = jnp.asarray([newS.sum(),
                          newE.sum(),
                          newInf.sum(),
                          newR.sum(),
                          newV.sum()]).sum()
    newS = newS.at[0].set(totPop - curPop)
    return newS, newE, newInf, newR, newV, day
