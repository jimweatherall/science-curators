#Intended to run on the former long-term stable release Julia 1.0

using Distributed

addprocs(33-length(procs()))

@everywhere using Distributions
@everywhere using DelimitedFiles

@everywhere mutable struct World #Structure to store the true mean and standard deviation
    mu #actual mean
    stdDev #actual stdDev
end

@everywhere mutable struct Player #Structure to store a plyers parameters
    mu0 #mean of player's belief, initialized as prior
    nu #nu parameter for normal-inverse gamma distribution; updates as nu -> nu + n where n is observations
    alpha #alpha parameter for normal-inverse gamma distribution; updates as alpha -> alpha + n/2
    beta #beta parameter for normal-inverse gamma distribution, initialized as prior standard deviation; updates as beta-> beta + 1/2 sum of squared deviations from mean
    bias #player bias parameter
end

@everywhere function posterior(player,x)
    return pdf(Normal(player.mu0,player.beta/(player.alpha-1)),x)
end


#@everywhere function journoPDF(world,h,c,f::Bool,x)
#    if h<1
#        error("h must be >=1")
#    end
#    if !f
#        lSide = quadgk(y->pdf(Normal(h*world.mu,h^2*world.stdDev),y),-Inf,-c)[1]
#        rSide = quadgk(y->pdf(Normal(h*world.mu,h^2*world.stdDev),y),c,Inf)[1]
        #lNorm = lSide[1]/(lSide[1] + rSide[1])
        #rNorm = rSide[1]/(lSide[1] + rSide[1])
#        d(y) = lSide*pdf(truncated(Normal(h*world.mu,h^2*world.stdDev),-Inf,-c),y) + rSide*pdf(truncated(Normal(h*world.mu,h^2*world.stdDev),c,Inf),y)
#        norm = quadgk(d,-Inf,Inf)[1]
#        return d(x)/norm
#    else
#        e(y) = pdf(truncated(Normal(h*world.mu,h^2*world.stdDev),-Inf,-c),y) + pdf(truncated(Normal(h*world.mu,h^2*world.stdDev),c,Inf),y)
#        norm = quadgk(e,-Inf,Inf)[1]
#        return e(x)/norm
#    end
#end


@everywhere function update(world,player,h,c,f::Bool)
    if !f #If fair&balanced is off
        @label restart1
        global draw = rand(Normal(h*world.mu,h*world.stdDev)) #Draw a sample from the hyperbolized world distribution
        while -c < draw && draw < c
            global draw = rand(Normal(h*world.mu,h*world.stdDev))
        end
        if player.bias !=0
            if (draw <= player.mu0 && rand() > cdf(Normal(player.mu0,1/player.bias),draw)) || (draw > player.mu0 && rand() > 1-cdf(Normal(player.mu0,1/player.bias),draw))
                @goto restart1
            end
        end
        player.mu0 = (player.nu*player.mu0 + draw)/(player.nu+1)
        player.nu += 1
        player.alpha += 1/2
        player.beta += player.nu*(draw-player.mu0)^2/(2*(player.nu + 1))
        return player
    else
        global draw1 = rand(Normal(h*world.mu,h*world.stdDev))
        while -c < draw1 && draw1 < c
            global draw1 = rand(Normal(h*world.mu,h*world.stdDev))
        end
        global draw2 = rand(Normal(h*world.mu,h*world.stdDev))
        while -c < draw2 && draw2 < c || draw1*draw2 > 0
            global draw2 = rand(Normal(h*world.mu,h*world.stdDev))
        end
        if player.bias !=0
            check1 = (draw1 <= player.mu0 && rand() > cdf(Normal(player.mu0,1/player.bias),draw1)) || (draw1 > player.mu0 && rand() > 1-cdf(Normal(player.mu0,1/player.bias),draw1))
            check2 = (draw2 <= player.mu0 && rand() > cdf(Normal(player.mu0,1/player.bias),draw2)) || (draw2 > player.mu0 && rand() > 1-cdf(Normal(player.mu0,1/player.bias),draw2))
            if !check1
                player.mu0 = (player.nu*player.mu0 + draw1)/(player.nu+1)
                player.nu += 1
                player.alpha += 1/2
                player.beta += player.nu*(draw1-player.mu0)^2/(2*(player.nu + 1))
            end
            if !check2
                player.mu0 = (player.nu*player.mu0 + draw2)/(player.nu+1)
                player.nu += 1
                player.alpha += 1/2
                player.beta += player.nu*(draw2-player.mu0)^2/(2*(player.nu + 1))
            end
        else
            player.mu0 = (player.nu*player.mu0 + draw1)/(player.nu+1)
            player.nu += 1
            player.alpha += 1/2
            player.beta += player.nu*(draw1-player.mu0)^2/(2*(player.nu + 1))

            player.mu0 = (player.nu*player.mu0 + draw2)/(player.nu+1)
            player.nu += 1
            player.alpha += 1/2
            player.beta += player.nu*(draw2-player.mu0)^2/(2*(player.nu + 1))

            #player.mu0 = (player.nu*player.mu0 + draw1+draw2)/(player.nu+2)
            #player.nu += 2
            #player.alpha += 1
            #player.beta += (draw1-draw2)^2 + player.nu*((draw1+draw2)/2-player.mu0)^2/((player.nu + 2))
        end
        return player
    end
end

@everywhere function runAlot(h,c,f,mu0,s,q,mu,stdDev,runs)
    world = World(mu,stdDev)

    results = Array{Float64}(undef,0,2)

    for i=1:runs
        player = Player(mu0,0,2,1,s)
        while player.nu < q
            player = update(world,player,h,c,f)
        end
        #println(player.mu0," ",player.beta)
        results = vcat(results,[player.mu0 player.beta/(player.alpha-1)])
    end

    msemu = []
    msesigma = []
    for i=1:length(results[:,1])
        msemu = vcat(msemu, (results[i,1] - mu)^2)
        msesigma = vcat(msesigma, (results[i,2]-1)^2)
    end


    #println(mu," ",stdDev," ",h," ",c," ",f," ",mu0," ",s," ",q," ",runs," ",mean(results[:,1])," ",mean(results[:,2]))
    return [mu mu0 s h c f  mean(results[:,1]) mean(results[:,2]) mean(results[:,1] - mu*ones(length(results[:,1]))) mean(results[:,2] - ones(length(results[:,2]))) mean(msemu) mean(msesigma)]

end

results = Array{Any}(undef,0,12)

for mu in [0,.5,1,1.5,2]
    for stdDev in [1]
        for h in [1,1.5,2,2.5,3]
            for c in [0,.25,.5,.75, 1,1.25,1.5]
                for f in [true, false]
                    for mu0 in [-5,-1, 0,1,5]
                        for s in [0,.1,.5,1]
                            for q in [1000]
                                writedlm("journalism results sweep $mu $h $c $f $mu0 $s.csv",runAlot(h,c,f,mu0,s,q,mu,stdDev,1000),',')
                            end
                        end
                    end
                end
            end
        end
    end
end
