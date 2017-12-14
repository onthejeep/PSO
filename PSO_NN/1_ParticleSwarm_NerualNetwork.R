library(ggplot2)
library(lubridate)


source('1_ActivationFunction.R');

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
    Velocity.Generation = list();
    SingleSolutionStructure = generation[[1]];
    for (i in 1: length(generation))
    {
        Velocity.Solution = list();
        for (j in 1:length(SingleSolutionStructure))
        {
            Velocity.Solution[[j]] =
                matrix(0, nrow = nrow(SingleSolutionStructure[[j]]), ncol = ncol(SingleSolutionStructure[[j]]));
        }
        Velocity.Generation[[i]] = Velocity.Solution;
    }

    return(Velocity.Generation);
}
PSO.InitializeGeneration = function(numSolution = 100, nnStructure)
{
    InitialGeneration = list();
    for (i in 1: numSolution)
    {
        InitialGeneration[[i]] = PSO.InitializeSolution(nnStructure);
    }
    return(InitialGeneration);
}
PSO.InitializeSolution = function(nnStructure)
{
    # nnStructure is a vector
    # example: c(3, 7, 2) = (numInput = 3, hiddenNeuron = 7, numOutput = 2)
    # example: c(3, 7, 4, 2) = (numInput = 3, hidden1 = 7, hidden2 = 4, numOutput = 2)

    InitialSolution = list();
    for (i in 1:(length(nnStructure) - 1))
    {
        NumRow = nnStructure[i];
        NumCol = nnStructure[i + 1];
        InitialSolution[[i]] = matrix(runif(NumRow * NumCol, min = -3, max = 3),
            nrow = NumRow, ncol = NumCol);
    }
    return(InitialSolution);
}

PSO.EvaluateSolution = function(input, singleSolution)
{
    # input is a vector
    # example: input = matrix(data = 1:3, nrow = 1, ncol = 3);

    Result = input;
    NumHiddenLayer = length(singleSolution) - 1;

    # Shu's comments:
    # After testing different activation functions, the logistic function performs best with other parameters under control
    for (i in 1:NumHiddenLayer)
    {
        Result = Result %*% singleSolution[[i]];
        Result = apply(Result, c(1, 2), FUN = Activation.Logistic);
    }
    
    Result = Result %*% singleSolution[[NumHiddenLayer + 1]];
    Result = apply(Result, c(1, 2), FUN = Activation.Logistic);
    
    return(Result);
}

PSO.Prediction2Classification = function(prediction)
{
    Classification = apply(prediction, 1, which.max);
    Classification = matrix(data = Classification, nrow = length(Classification));
    return(Classification);
}

PSO.EvaluateError = function(prediction, output)
{
    # root-mean-square error
    Error = sqrt(sum((prediction - output) ^ 2) / nrow(prediction));

    # 0.5 * (DesiredData ¨C CalculatedData)^2
    # Error = 0.5 * sum((prediction - output) ^ 2);
    return(Error);
}

PSO.EvaluateGeneration = function(generation, input, output)
{
    Fitness = c();

    for (i in 1:length(generation))
    {
        Prediction = PSO.EvaluateSolution(input, generation[[i]]);
        Fitness[i] = PSO.EvaluateError(Prediction, output);
    }

    SortedFitness = sort(Fitness, index.return = T);
    return(SortedFitness);
}

PSO.SortGeneration = function(originalGen, sortIndex)
{
    return(originalGen[sortIndex]);
}
PSO.BestSolution = function(sortedGen)
{
    return(sortedGen[[1]]);
}
PSO.BestFitness = function(sortedFitness)
{
    return(sortedFitness[1]);
}

PSO.NewGeneration = function(originalGen, velocity, inertiaWeight, globalBestSolution, globalBestFitness, input, output)
{
    NumSolution = length(originalGen);

    SortedFitness = PSO.EvaluateGeneration(originalGen, input, output);
    SortedOriginalGen = PSO.SortGeneration(originalGen, SortedFitness$ix);

    CurrentBestSolution = PSO.BestSolution(SortedOriginalGen);
    CurrentBestFitness = PSO.BestFitness(SortedFitness$x);

    C1 = 2;
    C2 = 2;

    Velocity.Generation = list();
    for (i in 1:NumSolution)
    {
        SingleSolution = originalGen[[i]];
        Velocity.Solution = list();
        for (j in 1: length(SingleSolution))
        {
            NumRow = nrow(SingleSolution[[j]]);
            NumCol = ncol(SingleSolution[[j]]);

            Velocity.Solution[[j]] = inertiaWeight * velocity[[i]][[j]] +
                    C1 * matrix(data = runif(n = NumRow * NumCol), nrow = NumRow, ncol = NumCol) * (CurrentBestSolution[[j]] - SingleSolution[[j]]) +
                    C2 * matrix(data = runif(n = NumRow * NumCol), nrow = NumRow, ncol = NumCol) * (globalBestSolution[[j]] - SingleSolution[[j]]);

            Velocity.Solution[[j]] = apply(Velocity.Solution[[j]], c(1, 2), FUN = PSO.Threshold.Velocity, 4);

            originalGen[[i]][[j]] = originalGen[[i]][[j]] + Velocity.Solution[[j]];
        }

        Velocity.Generation[[i]] = Velocity.Solution;
    }

    Result = list();
    if (CurrentBestFitness < globalBestFitness)
    {
        Result$GlobalBestSolution = CurrentBestSolution;
        Result$GlobalBestFitness = CurrentBestFitness;
        Result$OriginalGen = originalGen;
        Result$Velocity = Velocity.Generation;
    }
    else
    {
        Result$GlobalBestSolution = globalBestSolution;
        Result$GlobalBestFitness = globalBestFitness;
        Result$OriginalGen = originalGen;
        Result$Velocity = Velocity.Generation;
    }
    return(Result);
}


PSO.Execute = function(input, output)
{
    # the neural network structure is predefined
    NNStructure = c(ncol(input), 16, 16, 16, ncol(output));

    TrackBestFitness = c();

    # Initialize generation and calcualte initial global best solution
    InitialGeneration = PSO.InitializeGeneration(numSolution = 100, nnStructure = NNStructure);
    SortedFitness = PSO.EvaluateGeneration(InitialGeneration, input, output);
    SortedGeneration = PSO.SortGeneration(InitialGeneration, SortedFitness$ix);

    GlobalBestSolution = PSO.BestSolution(SortedGeneration);
    GlobalBestFitness = PSO.BestFitness(SortedFitness$x);
    TrackBestFitness = c(TrackBestFitness, GlobalBestFitness);

    NumIteration = 300;
    Velocity = PSO.InitializeVelocity(InitialGeneration);
    InertiaWeight = seq(from = 0.2, to = 0.001, length = NumIteration);

    for (i in 1:NumIteration)
    {
        Result = PSO.NewGeneration(InitialGeneration, Velocity, InertiaWeight[i],
            GlobalBestSolution, GlobalBestFitness, input, output);

        GlobalBestSolution = Result$GlobalBestSolution;
        GlobalBestFitness = Result$GlobalBestFitness;
        InitialGeneration = Result$OriginalGen;
        Velocity = Result$Velocity;

        TrackBestFitness = c(TrackBestFitness, GlobalBestFitness);
    }

    TrackBestFitness = as.data.frame(cbind(1:length(TrackBestFitness), TrackBestFitness));
    colnames(TrackBestFitness) = c('Iteration', 'Fitness');

    Result = list();
    Result$TrackBestFitness = TrackBestFitness;
    Result$BestSolution = GlobalBestSolution;
    return(Result);
}

PSO.UnitTest1 = function()
{
    # Input
    Input = matrix(data = c(0.1, 0.1,
    0.2, 0.2,
    0.1, 0.3,
    0.3, 0.1,
    0.7, 0.1,
    0.8, 0.3,
    0.9, 0.2,
    0.6, 0.4,
    0.7, 0.7,
    0.8, 0.9,
    0.6, 0.8,
    0.1, 0.9,
    0.2, 0.7,
    0.3, 0.8), nrow = 14, ncol = 2, byrow = T);
    
    # Output
    Output = matrix(data = c(1, 0, 0, 0,
                    1, 0, 0, 0,
                    1, 0, 0, 0,
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 1, 0, 0,
                    0, 1, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 1, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                    0, 0, 0, 1,
    0, 0, 0, 1), nrow = 14, ncol = 4, byrow = T);

    Result = PSO.Execute(Input, Output);

    P1 = ggplot() +
        geom_line(data = Result$TrackBestFitness, aes(x = Iteration, y = Fitness), col = 'darkblue', size = 2) +
        ggtitle('Best fitness');
    print(P1);

    Prediction = PSO.EvaluateSolution(Input, Result$BestSolution);
    Classification = PSO.Prediction2Classification(Prediction);
    print('---Classification Prediction---');
    print(Classification);
}

PSO.UnitTest2 = function()
{
    # Input
    Input = matrix(data = c(0.2, 0.5,
    1, 0,
    2, 0,
    2, 2,
    1, 2,
    5, 6,
    6, 6,
    6, 7,
    8, 7,
    7, 7,
    7, 6,
    0, 15,
    1, 14,
    2, 13,
    3, 12,
    4, 16,
    5, 15,
    6, 14,
    6, 13,
    7, 15,
    8, 15), nrow = 21, ncol = 2, byrow = T);

    # Output
    Output = matrix(data = c(1, 0, 0,
    1, 0, 0, 
    1, 0, 0, 
    1, 0, 0,
    1, 0, 0, 
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1), nrow = 21, ncol = 3, byrow = T);

    Result = PSO.Execute(Input, Output);

    P1 = ggplot() +
        geom_line(data = Result$TrackBestFitness, aes(x = Iteration, y = Fitness), col = 'darkblue', size = 2) +
        ggtitle('Best fitness');
    print(P1);

    Prediction = PSO.EvaluateSolution(Input, Result$BestSolution);
    Classification = PSO.Prediction2Classification(Prediction);
    print('---Classification Prediction---');
    print(Classification);
}

