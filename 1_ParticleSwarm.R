#####################################################
#
#    *********        *********        *********
#    *       *        *                *       *
#    *       *        *                *       *
#    *       *        *                *       *
#    *********        *********        *       *
#    *                        *        *       *
#    *                        *        *       *
#    *                        *        *       *
#    *                *********        *********
#
######################################################

# author: Shu Yang
# date: Dec 14, 2017
# implement: argmin(x^2 + y^2 + z^2) without constraints

# Reference: Eberhart, R., & Kennedy, J. (1995, October). A new optimizer using particle swarm theory. In Micro Machine and Human Science, 1995. MHS'95., Proceedings of the Sixth International Symposium on (pp. 39-43). IEEE.

# Key concepts in the paper:
# 1. optimization of continous nonlinear functions
# 2. global best solution
#   a. in the paper: gbest
#   b. in the code: GolbalBestSolution
# 3. local best solution
#   a. in the paper: pbest
#   b. in the code: CurrentBestSolution
# 4. single generation contains multiple solutions
#   a. generation (usually in genetic algorithm) = swarm
#   b. solution = particle

library(ggplot2)
library(lubridate)

# The paper says: 'It is also necessary to clamp velocities to some maximum to prevent overflow'
PSO.Threshold.Velocity = function(x, threshold)
{
    if (x < -threshold)
    {
        return(-threshold);
    }
    else if (x > threshold)
    {
        return(threshold);
    }
    else
        {
        return(x);
    }
}

PSO.InitializeVelocity = function(generation)
{
    Velocity = matrix(0, nrow = nrow(generation), ncol = ncol(generation));
    return(Velocity);
}
PSO.InitializeGeneration = function(numSolution)
{
    X = runif(n = numSolution, min = -5, max = 5);
    Y = runif(n = numSolution, min = -5, max = 5);
    Z = runif(n = numSolution, min = -5, max = 5);

    InitialGerantion = data.frame(X, Y, Z);
    return(InitialGerantion);
}

PSO.EvaluateSolution = function(singleSolution)
{
    Result = singleSolution$X ^ 2 + singleSolution$Y ^ 2 + singleSolution$Z ^ 2;
    return(Result);
}

PSO.EvaluateGeneration = function(generation)
{
    Fitness = c();

    for (i in 1: nrow(generation))
    {
        Fitness[i] = PSO.EvaluateSolution(generation[i,]);
    }

    SortedFitness = sort(Fitness, index.return = T);
    return(SortedFitness);
}

PSO.SortGeneration = function(originalGen, sortIndex)
{
    return(originalGen[sortIndex, ]);
}
PSO.BestSolution = function(sortedGen)
{
    return(sortedGen[1, ]);
}
PSO.BestFitness = function(sortedFitness)
{
    return(sortedFitness[1]);
}

PSO.NewGeneration = function(originalGen, velocity, inertiaWeight, globalBestSolution, globalBestFitness)
{
    NumSolution = nrow(originalGen);
    NumVariable = ncol(originalGen);

    SortedFitness = PSO.EvaluateGeneration(originalGen);
    SortedOriginalGen = PSO.SortGeneration(originalGen, SortedFitness$ix);

    CurrentBestSolution = PSO.BestSolution(SortedOriginalGen);
    CurrentBestFitness = PSO.BestFitness(SortedFitness$x);

    # in the paper: C1 = C2 = ACC_CONST = 2.0
    C1 = 2;
    C2 = 2;

    Velocity = c();
    for (i in 1: NumSolution)
    {
        Velocity.Solution = inertiaWeight * velocity[i,] +
                    C1 * runif(n = NumVariable) * (CurrentBestSolution - originalGen[i,]) +
                    C2 * runif(n = NumVariable) * (globalBestSolution - originalGen[i,]);

        # Minimizing (x^2 + y^2 + z^2) is an easy task
        # No significant differences of PSO behaviors can be observed with or wo velocity threshold
        Velocity.Solution = apply(Velocity.Solution, 2, PSO.Threshold.Velocity, 4);

        originalGen[i,] = originalGen[i,] + Velocity.Solution;
        Velocity = rbind(Velocity, Velocity.Solution);
    }

    Result = list();
    if (CurrentBestFitness < globalBestFitness)
    {
        Result$GlobalBestSolution = CurrentBestSolution;
        Result$GlobalBestFitness = CurrentBestFitness;
        Result$OriginalGen = originalGen;
        Result$Velocity = Velocity;
    }
    else
    {
        Result$GlobalBestSolution = globalBestSolution;
        Result$GlobalBestFitness = globalBestFitness;
        Result$OriginalGen = originalGen;
        Result$Velocity = Velocity;
    }
    return(Result);
}

PSO.UnitTest = function()
{
    TrackBestFitness = c();

    InitialGeneration = PSO.InitializeGeneration(numSolution = 50);
    SortedFitness = PSO.EvaluateGeneration(InitialGeneration);
    SortedGeneration = PSO.SortGeneration(InitialGeneration, SortedFitness$ix);

    GlobalBestSolution = PSO.BestSolution(SortedGeneration);
    GlobalBestFitness = PSO.BestFitness(SortedFitness$x);
    TrackBestFitness = c(TrackBestFitness, GlobalBestFitness);

    NumIteration = 50;
    Velocity = PSO.InitializeVelocity(InitialGeneration);
    InertiaWeight = seq(from = 0.1, to = 0.01, length = NumIteration);

    for (i in 1: NumIteration)
    {
        Result = PSO.NewGeneration(InitialGeneration, Velocity, InertiaWeight[i],
                    GlobalBestSolution, GlobalBestFitness);

        GlobalBestSolution = Result$GlobalBestSolution;
        GlobalBestFitness = Result$GlobalBestFitness;
        InitialGeneration = Result$OriginalGen;
        Velocity = Result$Velocity;

        TrackBestFitness = c(TrackBestFitness, GlobalBestFitness);

        print(paste(c('global best solution = ', GlobalBestSolution, 
            'global best fitness = ', GlobalBestFitness), collapse = ' '));
    }

    TrackBestFitness = as.data.frame(cbind(1:length(TrackBestFitness), TrackBestFitness));
    colnames(TrackBestFitness) = c('iteration', 'fitness');

    ResultPlot = ggplot() +
        geom_line(data = TrackBestFitness, aes(x = iteration, y = fitness), col = 'darkblue', size = 2) +
        ggtitle('Best fitness');
    print(ResultPlot);
}