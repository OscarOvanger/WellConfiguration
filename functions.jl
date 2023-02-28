using Plots
using Random
using GaussianRandomFields
using Interpolations
using Optim
using Distributions
using Roots
using StatsBase
using PlotlyJS
using LinearAlgebra
using LineSearches
using CSV
using DataFrames
using DataStructures
using MvNormalCDF
using LaTeXStrings

function prior_sample(prior_mu,prior_sigma,m)
    samps = rand(MvNormal(prior_mu,prior_sigma),m)
    return transpose(samps)
end

function theta_to_params(theta)
    params = zeros(size(theta))
    for i = 1:size(params,1)
        if i == 1
            params[i,1] = exp(theta[i,1])
            params[i,2] = exp(theta[i,2])
        else
            params[i,1] = exp(theta[i,1]) + params[i-1,1]
            params[i,2] = exp(theta[i,2]) + params[i-1,2]
        end
    end
    return params
end

function log_prior_pdf(thetas,prior_mu,prior_sigma,m)
    p_theta = 0
    for i in 1:m
        p_theta += logpdf(MvNormal(prior_mu,prior_sigma),thetas[i,:])
    end
    return p_theta
end

function reject_sampler(proposal_distr,log_target_pdf,N,lim)
    count = 0
    sample = zeros(N,length(rand(proposal_distr)))
    while count < N
        proposal = rand(proposal_distr)
        rand_nr = log(rand()*(lim))
        if rand_nr < log_target_pdf(proposal)
            count += 1
            sample[count,:] .= proposal
        end
    end
    return sample
end

function simulate_pc(param_MAP,grf_params,m,N)
    x = range(0,100,length=101)
    cova= CovarianceFunction(1, Matern(grf_params[1], grf_params[2],σ=grf_params[3]))
    grf = GaussianRandomField(cova, CirculantEmbedding(), x)
    configu_MAP = []
    roots_MAP = []
    for i in 1:N
        realiz = zeros(m,101)
        f = x -> 1
        surfs = Array{Function}(undef, m)
        for j in 1:m
            realiz[j,:] = mu_func(x,param_MAP,j) .+ GaussianRandomFields.sample(grf)
            interp = LinearInterpolation(x,realiz[j,:],extrapolation_bc=Line())
            surf = x -> interp(x)
            surfs[j] = surf
        end
        res = well_data(surfs)
        if [res[2]] != []
            append!(configu_MAP,[Int.(res[1])])
            append!(roots_MAP,[res[2]])
        end
    end
    new_configu_MAP = configu_MAP[findall(x->x!=[0.0],configu_MAP)]
    p_c_MAP = config_probs(new_configu_MAP,m+1)
    return p_c_MAP
end

function p_c_d_alg(configs,m,d,prior_mu,prior_sigma,grf_params,p_c_df,h,N)
    p_cd = zeros(length(configs)) #p(c|d)
    j = 1
    for c in configs
        #p_hat = zeros(m) #log p(c_j|d_i)
        if p_c_df.true_count[findall(x->x==string(c),p_c_df.configurations)][1] == 0
            p_cd[j] = -Inf
        else
            #println("surface nr: ",i)
            #println("prior_mu: ", prior_mu_curr)
            post_mu = posterior_mu(d,c,prior_mu,prior_sigma,grf_params,m) #find Laplace-approximation mean
            #println("post_mu: ",post_mu)
            post_sigma = post_sigma_approx(d,c,prior_mu,prior_sigma,grf_params,post_mu,m,h) #find Laplace-approximation covariance
            #println("post_sigma: ",post_sigma)
            post_sigma = (post_sigma .+ transpose(post_sigma)) ./ 2
            #println("post_sigma: ", post_sigma)
            param_MAP = transpose(reshape(post_mu,2,m))
            prior = log_prior_pdf(param_MAP,prior_mu,prior_sigma,m) #log prior pdf
            likeli = log_likeli(d,c,grf_params,param_MAP,m) #log prior likelihood equality
            post_approx = logpdf(MvNormal(post_mu,post_sigma),post_mu) #log posterior approximation pdf(LA)
            p_c_map_df = simulate_pc(param_MAP,grf_params,m,N)
            p_c_map = 0
            if length(findall(x->x==string(Int.(c)),p_c_map_df.configurations)) == 0
                p_c_map = 0
            else
                p_c_map = p_c_map_df.true_count[findall(x->x==string(Int.(c)),p_c_map_df.configurations)][1]
            end
            p_cd[j] = prior + likeli + log(p_c_map) - post_approx #logp(c_j|d_i) evaluated at sample s 
            #println("p_hat[i]: ",p_hat[i])
        end
        j += 1
        #println("p_cd[j]: ",p_cd[j])
    end
    #println("p_cd: " , p_cd)
    p_cd = exp.(p_cd) ./ sum(exp.(p_cd)) #normalize and convert from log-scale. 
    return p_cd
end

function p_c_d_alg_with_theta(configs,m,d,prior_mu,prior_sigma,grf_params,p_c_df,h,N)
    p_cd = zeros(length(configs)) #p(c|d)
    theta_MAPS = []
    Sigma_MAPS = []
    j = 1
    for c in configs
        #p_hat = zeros(m) #log p(c_j|d_i)
        if p_c_df.true_count[findall(x->x==string(c),p_c_df.configurations)][1] == 0
            p_cd[j] = -Inf
            append!(theta_MAPS,[prior_mu])
            append!(Sigma_MAPS,[prior_sigma])
        else
            #println("surface nr: ",i)
            #println("prior_mu: ", prior_mu_curr)
            post_mu = posterior_mu(d,c,prior_mu,prior_sigma,grf_params,m) #find Laplace-approximation mean
            append!(theta_MAPS,[post_mu])
            #println("post_mu: ",post_mu)
            post_sigma = post_sigma_approx(d,c,prior_mu,prior_sigma,grf_params,post_mu,m,h) #find Laplace-approximation covariance
            #println("post_sigma: ",post_sigma)
            post_sigma = (post_sigma .+ transpose(post_sigma)) ./ 2
            append!(Sigma_MAPS,[post_sigma])
            #println("post_sigma: ", post_sigma)
            param_MAP = transpose(reshape(post_mu,2,m))
            prior = log_prior_pdf(param_MAP,prior_mu,prior_sigma,m) #log prior pdf
            likeli = log_likeli(d,c,grf_params,param_MAP,m) #log prior likelihood equality
            post_approx = logpdf(MvNormal(post_mu,post_sigma),post_mu) #log posterior approximation pdf(LA)
            p_c_map_df = simulate_pc(param_MAP,grf_params,m,N)
            p_c_map = 0
            if length(findall(x->x==string(Int.(c)),p_c_map_df.configurations)) == 0
                p_c_map = 0
            else
                p_c_map = p_c_map_df.true_count[findall(x->x==string(Int.(c)),p_c_map_df.configurations)][1]
            end
            p_cd[j] = prior + likeli + log(p_c_map) - post_approx #logp(c_j|d_i) evaluated at sample s 
            #println("p_hat[i]: ",p_hat[i])
        end
        j += 1
        #println("p_cd[j]: ",p_cd[j])
    end
    #println("p_cd: " , p_cd)
    p_cd = exp.(p_cd) ./ sum(exp.(p_cd)) #normalize and convert from log-scale. 
    return p_cd,theta_MAPS,Sigma_MAPS
end

function posterior_mu(d,c,prior_mu,prior_sigma,grf_params,m)
    post = theta -> -log_prior_pdf(transpose(reshape(theta,2,m)),prior_mu,prior_sigma,m) - log_likeli(d,c,grf_params,transpose(reshape(theta,2,m)),m)
    means = prior_mu
    init = zeros(m*2)
    for i = 1:m
        init[2*i-1:2*i] = means 
    end
    if log_likeli(d,c,grf_params,transpose(reshape(init,2,m)),m) == -Inf
        suces = false
        iter = 1
        while (suces == false) && (iter < 1000)
            init_try = prior_sample(prior_mu,prior_sigma,m)
            if log_likeli(d,c,grf_params,init_try,m) != -Inf
                suces = true
                init = reshape(transpose(init_try),m*2)
            end
            iter += 1
        end
    end
    res = Optim.optimize(post,init,Optim.Options(iterations=1000,g_tol=1e-10))
    #println(res)
    param_MAP = Optim.minimizer(res)
    return param_MAP
end

function posterior_mu2(d,c,prior_mu,prior_sigma,grf_params,m)
    post = theta -> -log_prior_pdf(theta,prior_mu,prior_sigma,m) - log_likeli(d,c,grf_params,theta) 
    lower_bound = [0,0]
    upper_bound = [10.0,10.0]
    guess = init_guess(d,c,prior_mu,prior_sigma,grf_params,k)
    res = Optim.optimize(post,guess,BFGS(linesearch=LineSearches.BackTracking()))
    param_MAP = Optim.minimizer(res)
    return param_MAP
end

function init_guess(d,c,prior_mu,prior_sigma,grf_params,k)
    N = 100
    log_aggs = range(0,3.0,length=N)
    log_progs = range(0,3.0,length=N)
    posteriori = zeros(N,N)
    i=1
    for log_agg in log_aggs
        j=1
        for log_prog in log_progs
            sample = [log_prog,log_agg]
            posteriori[i,j] = logpdf(MvNormal(prior_mu,prior_sigma),sample) + log_likeli_e(d,c,grf_params,sample,k) + log_likeli_ie(d,c,grf_params,sample,k)
            j += 1
        end
        i += 1
    end
    param_MAP = [log_progs[argmax(posteriori)[2]],log_aggs[argmax(posteriori)[1]]]
    return param_MAP
end

function posterior_sigma(d,c,prior_mu,prior_sigma,grf_params,param_MAP,k)
    post_Sigma = -prior_sigma + log_likeli_e_hess(d,c,grf_params,param_MAP,k)
    post_Sigma[1,2] = post_Sigma[2,1]
    return post_Sigma
end

function post_sigma_approx(d,c,prior_mu,prior_sigma,grf_params,param_MAP,m,h)
    f = x -> log_likeli(d,c,grf_params,transpose(reshape(x,2,m)),m) + log_prior_pdf(transpose(reshape(x,2,m)),prior_mu,prior_sigma,m)
    #h = 1e-2 # step size
    approx_hess = hessian(f,param_MAP,h)
    #println(approx_hess)
    sigma = -inv(approx_hess)
    #println(sigma)
    return sigma
end

function post_sigma_approx_ie(d,c,prior_mu,prior_sigma,grf_params,param_MAP,k)
    h = 0.01
    x = param_MAP[1]
    y = param_MAP[2]
    f = (x,y) -> log_likeli_ie(d,c,grf_params,[x,y],k) + log_likeli_e(d,c,grf_params,[x,y],k) + logpdf(MvNormal(prior_mu,prior_sigma),[x,y])
    approx_hess = hessi(x,y,h,f)
    #println("approx_hess: ",approx_hess)
    approx_hess = -inv(approx_hess)
    return approx_hess
end

function log_likeli_ie(d,c,grf_params,theta,k)
    ls = lowers(d,c,k)
    us = uppers(d,c,k)
    low_sum = 0
    for i in 1:length(ls)
        mu = mu_func(ls[i],theta)
        low_sum += log(1-cdf(Normal(mu,grf_params[3]),well_func(ls[i])))
    end
    up_sum = 0
    for i in 1:length(us)
        mu = mu_func(us[i],theta)
        up_sum += logcdf(Normal(mu,grf_params[3]),well_func(us[i]))
    end
    log_pdf = low_sum + up_sum
end

function log_likeli(d,c,grf_params,theta,m)
    ec = equality_configs(c) # find layers that hit well
    #println("theta: ",theta)
    rel_points = unique(ec)
    log_p = 0 
    start = Int(minimum(rel_points)-1)
    if start == 0
        start = 1
    end
    stop = Int(maximum(rel_points)+1)
    if stop > m
        stop = m
    end
    #println("start: ", start)
    #println("stop: ", stop)
    for k = start:stop
        #println("k: ", k)
        ls = lowers(d,c,k)
        #println("ls: ", ls)
        us = uppers(d,c,k)
        #println("us: ", us)
        da = filter(x->x!=0,d)[findall(x->x==k,ec)] #find data value of equality constraints
        #println("da: ", da)
        if length(da) == 0
            #println("inequality only")
            mu_ineq,sigma_ineq = grf_moments_ineq(vcat(ls,us),c,grf_params,theta,k)
            sigma_ineq = (sigma_ineq + transpose(sigma_ineq)) ./ 2
            #println("mu_ineq: ",mu_ineq)
            #println("sigma_ineq: ",sigma_ineq)
            lows = vcat(well_func.(ls),-Inf .* ones(length(us)))
            ups = vcat(Inf .* ones(length(ls)),well_func.(us))
            #println("lows: ",lows)
            #println("ups: ",ups)
            lows = convert(Array{Float64,1}, lows)
            ups = convert(Array{Float64,1}, ups)
            #println(mvnormcdf(MvNormal(mu_ineq,sigma_ineq),lows,ups)[1])
            if size(sigma_ineq) == (1,1)
                #println("one dimensional")
                log_p += log(cdf(Normal(mu_ineq[1],sqrt(reshape(sigma_ineq,1)[1])),ups[1])-cdf(Normal(mu_ineq[1],sqrt(reshape(sigma_ineq,1)[1])),lows[1]))
            else
                #println("more than one dimensional")
                log_p += log(mvnormcdf(MvNormal(mu_ineq,sigma_ineq),lows,ups)[1])
            end
            #println("log_p: ",log_p)
        else
           #println("inequality and equality")
            mu,sigma = grf_moments(d,c,grf_params,theta,k) #mean and covariance of grf
            #println("mu: ", mu)
            #println("sigma: ", sigma)
            log_p += logpdf(MvNormal(mu,sigma),da) #log pdf of likelihood
            #println("log_p: ",log_p)
            mu_ineq,sigma_ineq = grf_moments_cond(d,vcat(ls,us),c,grf_params,theta,k)
            sigma_ineq = (sigma_ineq + transpose(sigma_ineq)) ./ 2
            #println("mu_ineq: ",mu_ineq)
            #println("sigma_ineq: ",sigma_ineq)
            lows = vcat(well_func.(ls),-Inf .* ones(length(us)))
            ups = vcat(Inf .* ones(length(ls)),well_func.(us))
            lows = convert(Array{Float64,1}, lows)
            ups = convert(Array{Float64,1}, ups)
            #println("lows: ",lows)
            #println("ups: ",ups)
            #println(mvnormcdf(MvNormal(mu_ineq,sigma_ineq),lows,ups)[1])
            if size(sigma_ineq) == (1,1)
                #println("one dimensional")
                log_p += log(cdf(Normal(mu_ineq[1],sqrt(reshape(sigma_ineq,1)[1])),ups[1])-cdf(Normal(mu_ineq[1],sqrt(reshape(sigma_ineq,1)[1]),lows[1])))
            else
                #println("more than one dimensional")
                log_p += log(mvnormcdf(MvNormal(mu_ineq,sigma_ineq),lows,ups)[1])
            end
            #println("log_p: ",log_p)
        end
    end
    return log_p
end

function init_optim(d,c,prior_mu,prior_sigma,grf_params,theta,m)
    ec = equality_configs(c) # find layers that hit well
    ec_unique = unique(ec)
    log_theta_MAP = zeros(length(ec_unique),2)
    i = 1
    for equal_surf in ec_unique
        prior_mu_curr = log.(exp.(prior_mu) .* equal_surf)
        log_theta_MAP[i,:] = init_guess(d,c,prior_mu_curr,prior_sigma,grf_params,equal_surf)
        i += 1
    end

    

end

function log_likeli_e_hess(d,c,grf_params,theta,k)
    ec = equality_configs(c)
    data_x = findall(x->x!=0,d)[findall(x->x==k,ec)]
    cova= CovarianceFunction(1, Matern(grf_params[1], grf_params[2],σ=grf_params[3]))
    sigma = apply(cova,data_x).*grf_params[3]
    d_mu = zeros(length(data_x),2)
    dd_mu = zeros(length(data_x),2,2)
    i = 1
    for da in data_x
        d_mu[i,:] = [exp(theta[1]) * exp(-(da - exp(theta[1])) / 100) / 100 , exp(theta[2])]
        dd_mu[i,:,:] = [[(exp(theta[1]) * exp(-(da - exp(theta[1])) / 100) / 100) * (1 + exp(theta[1])/100) , 0] [0 , exp(theta[2])]]
        i += 1
    end
    dd_mu_flat = reshape(dd_mu,(length(data_x),4))
    first = reshape(transpose(dd_mu_flat)*(sigma\(d[data_x]-mu_func(data_x,theta))),(2,2))
    tot = transpose(d_mu)*(sigma\d_mu) .- first
    return tot
end

function hessi(x, y, h, f)
    H = zeros(2,2)
    #println("f(x,y): ",f(x,y))
    H[1,1] = (f(x+h,y) - 2*f(x,y) + f(x-h,y)) / (h^2)
    H[1,2] = (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)) / (4*h^2)
    H[2,1] = H[1,2] # off-diagonal element is symmetric with H[1,2]
    H[2,2] = (f(x,y+h) - 2*f(x,y) + f(x,y-h)) / (h^2)
    #println("H: ",H)
    return H
end

function hessian(f, x,h)
    n = length(x)
    H = zeros(n, n)
    for i = 1:n
        xi_plus = copy(x)
        #println("xi_plus: ", xi_plus)
        xi_minus = copy(x)
        #println("xi_minus: ", xi_minus)
        xi_plus[i] += h/2
        xi_minus[i] -= h/2
        for j = 1:n
            xj_plus = copy(xi_plus)
            xj_minus = copy(xi_minus)
            xj_plus[j] += h/2
            xj_minus[j] -= h/2
            xj_plus_minus = copy(xi_plus)
            xj_minus_plus = copy(xi_minus)
            xj_plus_minus[j] -= h/2
            xj_minus_plus[j] += h/2
            H[i, j] = (f(xj_plus) - f(xj_plus_minus) - f(xj_minus_plus) + f(xj_minus)) / h^2
        end
    end
    return H
end

function line_segments(d,c)
    data_x = findall(x->x!=0,d)
    line_segs = []
    len = length(data_x)
    for i in 1:len+1
        if i == 1
            append!(line_segs,[collect(20:data_x[1]-1)])
        elseif i == len+1
            append!(line_segs,[collect(data_x[len]+1:80)])
        else
            append!(line_segs,[collect(data_x[i-1]+1:data_x[i]-1)])
        end
    end
    return line_segs
end

function lowers(d,c,k)
    c_index = findall(x->x==k,c)
    line_segs = line_segments(d,c)
    line_points = [line_seg[Int(ceil(length(line_seg)/2))] for line_seg in line_segs]
    ls = []
    for ci in c_index
        ls = vcat(ls,line_points[ci])
    end
    return ls
end

function uppers(d,c,k)
    c_index = findall(x->x>k,c)
    line_segs = line_segments(d,c)
    line_points = [line_seg[Int(ceil(length(line_seg)/2))] for line_seg in line_segs]
    us = []
    for ci in c_index
        us = vcat(us,line_points[ci])
    end
    return us
end

function grf_moments_ineq(xs,c,grf_params,theta,k)
    mu_ineq = mu_func(xs,theta,k)
    cova= CovarianceFunction(1, Matern(grf_params[1], grf_params[2],σ=grf_params[3]))
    sigma_ineq = apply(cova,xs).*grf_params[3]
    return mu_ineq,sigma_ineq
end

function grf_moments_cond(d,xs,c,grf_params,theta,k)
    cova= CovarianceFunction(1, Matern(grf_params[1], grf_params[2],σ=grf_params[3]))
    ec = equality_configs(c)
    data_x = findall(x->x!=0,d)[findall(x->x==k,ec)]
    sigma = apply(cova,vcat(xs,data_x)).*grf_params[3]
    sigma_xs = sigma[1:length(xs),1:length(xs)]
    sigma_xs_data = sigma[1:length(xs),length(xs)+1:end]
    sigma_data = sigma[length(xs)+1:end,length(xs)+1:end]
    mu_ineq = mu_func(xs,theta,k) + sigma_xs_data * (sigma_data\(filter(x->x!=0,d)[findall(x->x==k,ec)]-mu_func(data_x,theta,k)))
    sigma_ineq = sigma_xs - sigma_xs_data * (sigma_data \ transpose(sigma_xs_data))
    return mu_ineq,sigma_ineq
end

function grf_moments(d,c,grf_params,theta,k)
    ec = equality_configs(c)
    data_x = findall(x->x!=0,d)[findall(x->x==k,ec)]
    mu = mu_func(data_x,theta,k)
    cova= CovarianceFunction(1, Matern(grf_params[1], grf_params[2],σ=grf_params[3]))
    sigma = apply(cova,data_x).*grf_params[3]
    return mu,sigma
end

function mu_func(x,theta,k)
    params = theta_to_params(theta)
    alpha = params[k,2]
    phi = params[k,1]
    mean = min.(ones(length(x)) .+ alpha,alpha .+ exp.(-(x .- phi) ./100))
    return mean
end

function well_func(x)
    a = 1e-3
    b = -10e-2
    c = 6
    g = a*x^2+b*x+c
    return g
end

function stable_sum(p)
    l_m = maximum(p)
    res = l_m + log(sum(exp.(p.-l_m)))
    return res
end

function equality_configs(c)
    ec = min.(c[2:end],c[1:end-1])
    return ec
end

function well_data(surfs)
    roots = []
    well = x -> well_func(x)
    for i in 1:length(surfs)
        diff_func = x -> surfs[i](x)-well(x)
        new_roots = find_zeros(diff_func,20,80)
        append!(roots,new_roots)
    end
    sorted_roots = sort(roots)
    configs = zeros(length(roots)+1)
    if length(roots) > 0
        pos = sorted_roots[1]-0.5
        j=1
        while surfs[j](pos) < well(pos)
            j += 1
            if j > length(surfs)
                break
            end
        end
        configs[1] = j
    end
    for i in 1:length(roots)
        if i < length(roots)
            y = (sorted_roots[i+1]-sorted_roots[i])/2 + sorted_roots[i]
        else
            y = sorted_roots[i]+0.5
        end
        j = 1
        while surfs[j](y) < well(y)
            j += 1
            if j > length(surfs)
                break
            end
        end
        configs[i+1] = j
    end 
    return configs,roots
end

function make_sims(prior_mu,prior_sigma,grf_params,nr_realiz,disp)
    #Set grid and create well
    x_well = collect(20:80)
    x = range(0,100,length=101)

    #Create realizations
    # Here we create different realisations from the prior and observe the data given the well
    #Random.seed!(2022)
    #Seed = 2022, 10 000 runs
    realiz1_list = []
    #realiz2_list = []
    #realiz3_list = []
    configu = []
    roots = []
    param_list = []
    for i in 1:nr_realiz
        params = zeros(3,2)
        params = prior_sample(prior_mu,prior_sigma,3)
        append!(param_list,[params])  
        realiz1 = mu_func(x,params,1) .+ GaussianRandomFields.sample(grf)
        #realiz2 = mu_func(x,params,2) .+ GaussianRandomFields.sample(grf)
        #realiz3 = mu_func(x,params,3) .+ GaussianRandomFields.sample(grf)
        append!(realiz1_list,[realiz1])
        #append!(realiz2_list,[realiz2])
        #append!(realiz3_list,[realiz3])
        interp1 = LinearInterpolation(x,realiz1,extrapolation_bc=Line())
        #interp2 = LinearInterpolation(x,realiz2,extrapolation_bc=Line())
        #interp3 = LinearInterpolation(x,realiz3,extrapolation_bc=Line())
        surf1 = x -> interp1(x)
        #surf2 = x -> interp2(x)
        #surf3 = x -> interp3(x)
        surfs = [surf1]
        #surfs = [surf1,surf2,surf3]
        #surfs = [surf1,surf2]
        res = well_data(surfs)
        if [res[2]] != []
            append!(configu,[res[1]])
            append!(roots,[res[2]])
        end
        if disp 
            Plots.plot(x,zeros(length(x)),fillrange=mu_func(x,zeros(3,2),1),fillcolor="gray",color="gray",label="")
            Plots.plot!(x,mu_func(x,zeros(3,2),1),fillrange=realiz1,fillcolor="ivory1",color="ivory1",label="")
            Plots.plot!(x,realiz1, fillrange=realiz2,fillcolor="ivory2",color="ivory2",label = "")
            #Plots.plot!(x,realiz2, fillrange=realiz3,fillcolor="ivory3",color="ivory3",label = "")
            #Plots.plot!(x,realiz3,title=string(Int.(res[1])),color="ivory3",label="")
            Plots.plot!(x_well,well_func.(x_well),color="grey0",label = "",linewidth=2)
            Plots.xlims!((0,100))
            Plots.ylims!((0,7))
            display(Plots.plot!(title=string(Int.(res[1]))))
            Plots.savefig(Plots.plot!(xlabel="x",ylabel="z",title=string(Int.(res[1]))),"sim_nr_$i.png")
        end
    end
    #Discard no observations
    new_roots = roots[findall(x->x!=[],roots)]
    new_configu = configu[findall(x->x!=[0.0],configu)]
    params_list = param_list[findall(x->x!=[0.0],configu)]
    realiz1_list = realiz1_list[findall(x->x!=[0.0],configu)]
    #realiz2_list = realiz2_list[findall(x->x!=[0.0],configu)]
    #realiz3_list = realiz3_list[findall(x->x!=[0.0],configu)]
    #compute well configuration probabilities
    data = zeros(length(new_roots),length(x))
    for i in 1:length(new_roots)
        data[i,trunc.(Int,ceil.(new_roots[i]))] = well_func.(trunc.(Int,ceil.(new_roots[i])))
    end
    new_data = []
    new_configsu = []
    new_params = []
    new_realiz1_list = []
    #new_realiz2_list = []
    #new_realiz3_list = []
    for i in 1:length(new_roots)
        data_x = findall(x->x!=0,data[i,:])
        data_x1 = vcat(0,data_x)
        data_x2 = vcat(data_x,0)
        diff = (data_x2 .- data_x1)[1:end-1]
        config1 = vcat(0,new_configu[i][1:end-1])
        diff2 = (new_configu[i] .- config1)[2:end]
        check = findall(x->x==0,diff2)
        if (sum(findall(x->x==1,diff)) == 0) && (length(check) == 0)
            check2 = findall(x->x in [20,80],data_x)
            if length(check2) == 0
                append!(new_data,[data[i,:]])
                append!(new_configsu,[new_configu[i]])
                append!(new_params,[params_list[i]])
                append!(new_realiz1_list,[realiz1_list[i]])
                #append!(new_realiz2_list,[realiz2_list[i]])
                #append!(new_realiz3_list,[realiz3_list[i]])
            end
        end
    end 

    #Plotting the distribution of configurations
    configu_copy = copy(new_configu)
    elements = unique(new_configsu)
    non_zero_elements = elements[findall(x->x!=[0.0],elements)]
    j=1
    for e in non_zero_elements
        configu_copy[findall(x->x==e,new_configsu)].=string(e)
        j += 1
    end
    count_elements = zeros(length(non_zero_elements))
    for i in 1:length(non_zero_elements)
        count_elements[i] = countmap(configu_copy)[string.(non_zero_elements)[i]]
    end
    display(Plots.bar(count_elements, orientation=:h, yticks=(1:length(non_zero_elements), string.(non_zero_elements)), yflip=true))

    return new_data,new_configsu,new_params,new_realiz1_list
end

function makepaths(m, n)
    paths = [[j] for j in 1:m]
    for t in 1:n
        newpaths = []
        for path in paths
            for neighbor in neighbors(path[end], m)
                newpath = vcat(path, [neighbor])
                if !(newpath in paths)
                    push!(newpaths, newpath)
                end
            end
        end
        append!(paths, newpaths)
    end
    return paths[length.(paths) .== n+1]
end

function neighbors(i, n)
    if n == 1
        return []
    elseif i == 1
        return [2]
    elseif i == n
        return [n-1]
    else
        return [i-1, i+1]
    end
end

function config_probs(configu,m)
    unique_configs = unique(configu)
    int_configu = [Int.(x) for x in configu]
    config_count = counter(int_configu)
    max_len = maximum([length(c) for c in int_configu])
    all_configs = collect(Iterators.flatten([makepaths(m,mi) for mi in collect(1:max_len-1)]))
    df_data = DataFrame(configurations = string.(collect(keys(config_count))),true_count=collect(values(config_count)))
    df_all_configs = DataFrame(configurations = string.(all_configs))
    df_all = outerjoin(df_data, df_all_configs, on = :configurations)
    df_all[!,"true_count"] = coalesce.(df_all[!,"true_count"], 0)
    df_all[!,"true_count"] = df_all[!,"true_count"] ./ sum(df_all[!,"true_count"])
    return df_all
end