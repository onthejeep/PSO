
# output ranges (-inf, inf)
Activation.Identity = function(x)
{
    return(x);
}

# output ranges {0, 1}
Activation.Binary = function(x, threshold = 0.5)
{
    ifelse(x < threshold, return(0), return(1));
}

# output ranges (0, 1)
Activation.Logistic = function(x)
{
    return(1 / (1 + exp(-x)));
}

# output ranges (-1, 1)
Activation.TanH = function(x)
{
    return(tanh(x));
}

# output ranges (-1, 1)
Activation.Softsign = function(x)
{
    return(x / (1 + abs(x)));
}

