
using Distributions
import Distributions: quantile, rand, cdf, logccdf, invlogccdf
import Base: rand, push!, isless

export TransitionDistribution, WrappedDistribution, TransitionExponential
export TransitionWeibull, TransitionGamma, TransitionLogLogistic
export rand, test, hazard_integral, implicit_hazard_integral, cdf
export parameters, quantile
export EmpiricalDistribution, push!, build!
export NelsonAalenDistribution, multiple_measures

# These are the distributions of stochastic processes in absolute time.
# So it's exponential, starting at some enabling time, or
# Weibull, but with an enabling time. They can be sampled at any
# time with the assumption that they have not fired up until
# some time called "now."
abstract TransitionDistribution

type WrappedDistribution <: TransitionDistribution
    # Relative to the enabling time.
    relative_distribution::Distributions.ContinuousUnivariateDistribution
    enabling_time::Float64
    WrappedDistribution(d, e)=new(d, e)
end

# Given a distribution F
# P(x<=t | x>t0)=P(x<=t && x>t0) / P(x>t0)
function rand(distribution::WrappedDistribution, now::Float64,
        rng::MersenneTwister)
    quantile(distribution, now, rand(rng))
end

# Given a distribution F
# P(x<=t | x>t0)=P(x<=t && x>t0) / P(x>t0)
# U is a uniform variable between 0<U<=1
function quantile(distribution::WrappedDistribution, t0::Float64,
        U::Float64)
    te=distribution.enabling_time
    te+quantile(distribution.relative_distribution,
        U+(1-U)*cdf(distribution.relative_distribution, t0-te))
end

# The current time of the system is "now".
# The cdf is being evaluated for a future time, "when".
function cdf(dist::WrappedDistribution, when::Float64, now::Float64)
    t0te=cdf(dist.relative_distribution, now-dist.enabling_time)
    tte=cdf(dist.relative_distribution, when-dist.enabling_time)
    (tte-t0te)/(1-t0te)
end

# Given a hazard, the integral of that hazard over a time.
# int_{t0}^{t1} hazard(s, te) ds
function hazard_integral(dist::WrappedDistribution, t1, t2)
    # logccdf is log(1-cdf(d, x))
    rel=dist.relative_distribution
    te=dist.enabling_time
    logccdf(rel, t1-te)-logccdf(rel, t2-te)
end

# xa = int_{t0}^{t} hazard(s, te) ds. Solve for t.
function implicit_hazard_integral(dist::WrappedDistribution, xa, t0)
    rel=dist.relative_distribution
    te=dist.enabling_time
    t=te+invlogccdf(rel, -xa+logccdf(rel, t0-te))
    @assert(t>=t0)
    t
end


######### Exponential
type TransitionExponential <: TransitionDistribution
    relative_distribution::Distributions.Exponential # relative time
    enabling_time::Float64
end

function TransitionExponential(rate::Real, enabling_time::Real)
    dist=Distributions.Exponential(1.0/rate)
    TransitionExponential(dist, enabling_time)
end

parameters(d::TransitionExponential)=[1.0/scale(d.relative_distribution),
        d.enabling_time]

function rand(distribution::TransitionExponential, now::Float64, rng)
    # We store the distribution for this call. Doing the inverse with
    # a log() is very slow compared to the Ziggurat method, which should
    # be available here.
    now+quantile(distribution.relative_distribution,
            rand(rng))
end

function hazard_integral(dist::TransitionExponential, start, finish)
    @assert(finish>=start)
    (finish-start)/scale(dist.relative_distribution)
end

function cdf(dist::TransitionExponential, when, now)
    1-exp(-hazard_integral(dist, now, when))
end

function implicit_hazard_integral(dist::TransitionDistribution,
        cumulative_hazard, current_time)
    @assert(cumulative_hazard>=0)
    current_time+cumulative_hazard*scale(dist.relative_distribution)
end

function test(TransitionExponential)
    rng=MersenneTwister()
    rate=2.0
    dist=TransitionExponential(rate, 0.0)
    ed=EmpiricalDistribution()
    for i in 1:100000
        push!(ed, rand(dist, 0.0, rng))
    end
    lambda_estimator=1/Base.mean(ed.samples)
    too_low=(rate<lambda_estimator*(1-1.96/sqrt(length(ed))))
    too_high=(rate>lambda_estimator*(1+1.96/sqrt(length(ed))))
    @debug("TransitionExponential low ", too_low, " high ", too_high)
end

########### Weibull
# F(T)=1-exp(-((T-Te)/lambda)^k)
type TransitionWeibull <: TransitionDistribution
    parameters::Array{Float64,1}
end
function TransitionWeibull(lambda, k, enabling_time)
    TransitionWeibull([lambda, k, enabling_time])
end
parameters(tw::TransitionWeibull)=tw.parameters

function rand(dist::TransitionWeibull, now::Float64, rng::MersenneTwister)
    (λ, k, tₑ)=dist.parameters
    d=now-tₑ
    value=0
    U=Base.rand(rng)
    if d>0
        value=λ*(-log(1-U)+(d/λ)^k)^(1/k)-d
    else
        value=-d+λ*(-log(1-U))^(1/k)
    end
    now+value
end

function hazard_integral(dist::TransitionWeibull, last, now)
    (λ, k, tₑ)=dist.parameters
    ((now-tₑ)/λ)^k - ((last-tₑ)/λ)^k
end

function cdf(dist::TransitionWeibull, when, now)
    1-exp(-hazard_integral(dist, now, when))
end

function implicit_hazard_integral(dist::TransitionWeibull,
        cumulative_hazard, now)
    (λ, k, tₑ)=dist.parameters
    tₑ + λ*(cumulative_hazard + ((now-tₑ)/λ)^k)^(1.0/k)
end

function test(dist::TransitionWeibull)
    rng=MersenneTwister()
    (λ, k, tₑ)=dist.parameters
    ed=EmpiricalDistribution()
    for i in 1:10000
        push!(ed, rand(dist, 0.0, rng))
    end
    expected_mean=λ*gamma(1+1/k)
    actual_mean=mean(ed)
    @debug("mean expected ", expected_mean, " actual ", actual_mean,
        " diff ", abs(expected_mean-actual_mean))

    expected_variance=λ^2*(gamma(1+2/k)-gamma(1+1/k)^2)
    obs_var=variance(ed)
    @debug("variance expected ", expected_variance, " actual ", obs_var,
        " diff ", abs(expected_variance-obs_var))

    min_value=min(ed)
    mink=min_value^k
    total=0.0
    for i in 1:length(ed)
        total+=ed.samples[i]^k
    end
    λ_estimator=(total/length(ed)-mink)^(1/k)
    @debug("λ expected ", λ, " actual ", λ_estimator,
        " diff ", abs(λ-λ_estimator))

    numerator=0.0
    denominator=0.0
    logsum=0.0
    for s in ed.samples
        numerator+=s^k*log(s) - mink*log(min_value)
        denominator+=s^k - mink
        logsum+=log(s)
    end
    k_est_inv=numerator/denominator - logsum/length(ed)
    k_est=1.0/k_est_inv
    @debug("k expected ", k, " actual ", k_est,
        " diff ", abs(k-k_est))
end

#################################
# α - shape parameter
# β - inverse scale parameter, also called rate parameter
#
# pdf=(β^α/Γ(α))x^(α-1) e^(-βx)
function TransitionGamma(α::Float64, β::Float64, te::Float64)
    # The supplied version uses θ=1/β.
    relative=Distributions.Gamma(α, 1/β)
    WrappedDistribution(relative, te)
end

##################################
# F(t)=1/(1 + ((t-te)/α)^(-β))
type LogLogistic <: Distributions.ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
end

function rand(d::LogLogistic, rng::MersenneTwister)
    quantile(d, rand(rng))
end

function quantile(d::LogLogistic, U::Float64)
    d.alpha*(U/(1-U))^(1/d.beta)
end

function cdf(d::LogLogistic, t::Real)
    1/(1+(t/d.alpha)^(-d.beta))
end

# Survival
function ccdf(d::LogLogistic, t::Real)
    1/(1+(t/d.alpha)^d.beta)
end

function logccdf(d::LogLogistic, t::Real)
    -log( 1 + (t/d.alpha)^d.beta )
end

# needed to change this from the original version
# due to some sort of updated definition of invlogccdf
# caused errors, but fixed by sending Real->Float64
# see: http://julia.readthedocs.org/en/latest/manual/methods/
function invlogccdf(d::LogLogistic, lp::Float64)
    d.alpha*(1-exp(-lp)^(1/d.beta))
end

function TransitionLogLogistic(a::Float64, b::Float64, t::Float64)
    WrappedDistribution(LogLogistic(a,b), t)
end

#################################
type EmpiricalDistribution
    samples::Array{Float64,1}
    built::Bool
    EmpiricalDistribution()=new(Array(Float64,0), false)
end

function cdf(ed::EmpiricalDistribution, which::Int)
    (ed.samples[which], which/length(ed.samples))
end

function build!(ed::EmpiricalDistribution)
    if !ed.built
        sort!(ed.samples)
        ed.built=true
    end
end

function push!(ed::EmpiricalDistribution, value)
    push!(ed.samples, value)
end

length(ed::EmpiricalDistribution)=length(ed.samples)

function mean(ed::EmpiricalDistribution)
    Base.mean(ed.samples)
end

function min(ed::EmpiricalDistribution)
    Base.minimum(ed.samples)
end

function variance(ed::EmpiricalDistribution)
    m=Base.mean(ed.samples)
    total=0.0
    for d in ed.samples
        total+=(m+d)^2
    end
    total/length(ed.samples)
end


function kolmogorov_smirnov_statistic(ed::EmpiricalDistribution, other)
    build!(ed)
    sup_diff=0.0
    n=length(ed.samples)
    for i in 1:n
        t=ed.samples[i]
        F_empirical=i/n
        sup_diff=max(sup_diff, abs(F_empirical-cdf(other, t)))
    end
    c_alpha=1.36 # 0.05 confidence interval
    (sup_diff, sup_diff > c_alpha*sqrt(2/n))
end


#############################################
type NelsonAalenEntry
    when::Float64
    hazard_sum::Float64
end

# H(t)=\int \lambda=\sum_{t_i<=t} (d/n)
type NelsonAalenDistribution
    integrated_hazard::Array{NelsonAalenEntry,1}
    NelsonAalenDistribution(cnt::Int)=new(Array(NelsonAalenEntry, cnt))
end

function cdf(dist::NelsonAalenDistribution, when::Float64)
    entry_idx=findfirst(x->(x.when>when), dist.integrated_hazard)
    if entry_idx==0
        entry_idx=length(dist.hazard_sum)+1
    end
    1-exp(-dist.integrated_hazard[entry_idx-1].hazard_sum)
end


function cdf(dist::NelsonAalenDistribution, bypoint::Int)
    entry=dist.integrated_hazard[bypoint]
    (entry.when, 1-exp(-entry.hazard_sum))
end


function kolmogorov_smirnov_statistic(ed::NelsonAalenDistribution, other)
    sup_diff=0.0
    n=length(ed.integrated_hazard)
    for i in 1:n
        t=ed.integrated_hazard[i].when
        F_empirical=ed.integrated_hazard[i].hazard_sum
        sup_diff=max(sup_diff, abs(F_empirical-cdf(other, t)))
    end
    c_alpha=1.36 # 0.05 confidence interval
    (sup_diff, sup_diff > c_alpha*sqrt(2/n))
end


type NelsonAalenSortEntry
    time::Float64
    fired::Int
end

function isless(a::NelsonAalenSortEntry, b::NelsonAalenSortEntry)
    a.time<b.time
end

function NelsonAalenDistribution(fired::Array{Float64,1}, not_fired::Array{Float64,1})
end


# Given measurements of several outcomes, construct Nelson-Aalen distributions
function multiple_measures(eds::Array{EmpiricalDistribution,1})
    dist_len=Int64[length(x) for x in eds]
    nad=Array(NelsonAalenDistribution, length(eds))
    for nad_idx=1:length(eds)
        # +1 for the entry with integrated hazard of zero
        nad[nad_idx]=NelsonAalenDistribution(dist_len[nad_idx]+1)
    end
    for ed in eds
        build!(ed)
    end
    total=sum(dist_len)
    fired=Array(NelsonAalenSortEntry, total)
    fired_idx=1
    for add_idx=1:length(eds)
        for s in eds[add_idx].samples
            fired[fired_idx]=NelsonAalenSortEntry(s, add_idx)
            fired_idx+=1
        end
    end
    sort!(fired)
    dist_idx=ones(Int, length(eds))
    for start_zero=1:length(eds)
        nad[start_zero].integrated_hazard[dist_idx[start_zero]]=NelsonAalenEntry(0.0, 0.0)
        dist_idx[start_zero]+=1
    end
    for entry_idx=1:length(fired)
        at_risk=total-entry_idx+1
        who=fired[entry_idx].fired
        when=fired[entry_idx].time
        previous=nad[who].integrated_hazard[dist_idx[who]-1].hazard_sum
        nad[who].integrated_hazard[dist_idx[who]]=NelsonAalenEntry(when, previous+1.0/at_risk)
        dist_idx[who]+=1
    end
    for check_idx=1:length(eds)
        assert(dist_idx[check_idx]==2+dist_len[check_idx])
    end
    nad
end

