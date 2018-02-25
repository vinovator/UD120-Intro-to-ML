#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    # len of predictions, ages, net_worths = 90
    
    # determine error as square to get absolute error
    error = (predictions - net_worths)**2
    # type(error) = numpy.ndarray
    
    # Coverting array to list
    # resulting list is a list of list with inner list containing 1 element
    # strip this to create a clean list
    # repeat for error, ages and networths
    error = error.tolist()
    error2 = list()
    for e in error:
        error2.append(e[0])
     
    ages = ages.tolist()
    ages2 = list()
    for a in ages:
        ages2.append(a[0])
        
    net_worths = net_worths.tolist()
    net_worths2 = list()
    for n in net_worths:
        net_worths2.append(n[0])
    
    # Form the cleaned_data list as tuple of age, networth and error
    for item in zip(ages2, net_worths2, error2):
        cleaned_data.append(item)
       
    # Sort by error
    cleaned_data.sort(key=lambda x:x[2])   # sort by errors
    
    # Clean away 10% outliers
    cleaned_data = cleaned_data[:81]
    
    # print(cleaned_data)

    
    return cleaned_data

