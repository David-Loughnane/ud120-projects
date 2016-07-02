def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    for i in range(len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], abs(predictions[i] - net_worths[i])))
    cleaned_data.sort(key=lambda tup: tup[2])
    cleaned_data = cleaned_data[:81]
    
    return cleaned_data

