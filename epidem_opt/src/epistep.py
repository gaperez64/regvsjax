import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


@partial(jax.jit, static_argnums=0)
def vaccinate(config, vaccRates, S, E, Inf, R, V, day):
    """
        This is NOT something that's easy to make a method of the model class because
        the vaccination rates and costs could change after switching vaccination
        strategy! To optimize via JAX compilation, this version of the function is
        externalized.
    """
    # vaccination = element-wise product with vaccRates
    S2V = S * vaccRates
    E2V = E * vaccRates
    I2V = Inf * vaccRates
    R2V = R * vaccRates
    # updates
    newS = S - S2V
    newE = E - E2V
    newInf = Inf - I2V
    newR = R - R2V
    newV = V + S2V + E2V + I2V + R2V

    # cost
    vaxCost = (newV - V) * config.vacc_costs
    return (newS, newE, newInf, newR, newV, day, vaxCost)


@partial(jax.jit, static_argnums=0)
def seedInfs(config, S, E, Inf, R, V, day):
    newS = S.at[config.seed_ages].add(-config.seed_inf)
    newInf = Inf.at[config.seed_ages].add(config.seed_inf)
    return newS, E, newInf, R, V, day


@partial(jax.jit, static_argnums=0)
def step(config, S, E, Inf, R, V, day):
    """
    We simulate one step of (forward) Newton integration
    with h = 1 (day). For efficiency, the method is JIT compiled
    NOTE: if any object attributes change after the first call,
    this will result in incorrect results as we assume config
    to be static
    """
    z = 1 + config.delta * jnp.sin(2 * np.pi * (day / 365))
    beta = config.contact * config.q
    force = z * jnp.dot(beta, Inf)

    # daily transitions
    S2E = S * force  # element-wise product
    E2I = E * config.sigma
    I2R = Inf * config.gamma
    R2S = R * config.omega_imm
    V2S = V * config.omega_vacc

    # daily mortality = element-wise product with mortality
    # rates
    S2D = S * config.dailyMort
    E2D = E * config.dailyMort
    I2D = Inf * config.dailyMort
    R2D = R * config.dailyMort
    V2D = V * config.dailyMort

    # new values for all components
    newS = S - S2E + R2S + V2S - S2D
    newE = E + S2E - E2I - E2D
    newInf = Inf + E2I - I2R - I2D
    newR = R + I2R - R2S - R2D
    newV = V - V2S - V2D

    # breakdown of new infections
    confirmedInf = E2I * config.influenzaRate
    noMedCare = (confirmedInf / config.no_med_care) - confirmedInf
    hospd = confirmedInf * config.hospRate
    fatal = confirmedInf * config.caseFatalityRate

    # costs
    ambCost = (confirmedInf - hospd) * config.ambulatory_costs
    noMedCost = noMedCare * config.nomedCosts
    hospCost = hospd * (config.hospCosts + config.hospAmbCosts)
    vaxCost = 0

    # qalys
    ambQaly = (confirmedInf - hospd) * config.ambulatory_qalys
    noMedQaly = noMedCare * config.nomed_qalys
    hospQaly = hospd * config.hosp_qalys
    lifeyrsLost = fatal * config.disc_life_ex

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
