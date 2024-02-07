import copy, math  
import numpy as np 

def compute_model(x, w, b):
    """
    Compute the linear regression model.

    Parameters:
    x (numpy.ndarray): Input data.
    w (numpy.ndarray): Weight vector.
    b (float): Bias term.

    Returns:
    numpy.ndarray: The computed regression model.

    """
    reg = np.dot(x, w) + b
    return reg

def compute_cost(x, y, w, b):
    """
    Compute the cost function for linear regression.

    Parameters:
    x (numpy.ndarray): Input features.
    y (numpy.ndarray): Target values.
    w (numpy.ndarray): Weight vector.
    b (float): Bias term.

    Returns:
    float: The cost value.
    """

    total_cost = np.sum(((np.dot(w, x) + b) - y) ** 2) / (2 * len(x)) 
    return total_cost


def compute_gradient(x, y, w, b):
    """
    Compute the gradient of the cost function with respect to the weights and bias.

    Parameters:
    x (numpy.ndarray): Input features.
    y (numpy.ndarray): Target values.
    w (numpy.ndarray): Weight vector.
    b (float): Bias value.

    Returns:
    dj_dw (float): Gradient of the cost function with respect to the weights.
    dj_db (float): Gradient of the cost function with respect to the bias.
    """

    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0

    dj_dw_i = 0.0
    dj_db_i = 0.0

    f_wb = np.zeros(m)
    f_wb = np.dot(x, w) + b
    # print(f"f_wb: {f_wb}")
    for i in range(m):
        dj_dw_i += (f_wb - y[i]) * x[i]
        # print(f"dj_dw_i: {dj_dw_i}")
        dj_db_i += (f_wb - y[i])
        # print(f"dj_db_i: {dj_db_i}")

    dj_dw = dj_dw_i / m
    # print(f"dj_dw: {dj_dw}")
    dj_db = dj_db_i / m
    # print(f"dj_db: {dj_db}")
    return dj_dw, dj_db

def gradient_decent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Perform gradient decent optimization to learn the weights and bias for the linear regression model.

    Parameters:
    x (numpy.ndarray): Input features.
    y (numpy.ndarray): Target values.
    w_in (numpy.ndarray): Initial weight vector.
    b_in (float): Initial bias value.
    alpha (float): Learning rate.
    num_iters (int): Number of iterations.
    cost_function (function): Function to compute the cost.
    gradient_function (function): Function to compute the gradient.

    Returns:
    w (numpy.ndarray): Learned weight vector.
    b (float): Learned bias value.
    cost_history (list): History of cost values during optimization.
    """

    w = copy.deepcopy(w_in)
    b = b_in
    w = w_in
    J_history = []
    p_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        temp_w = w - alpha * dj_dw
        temp_b = b - alpha * dj_db

        w = temp_w
        b = temp_b

        # w -= alpha * dj_dw
        # b -= alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/100) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  "w: " + np.array2string(w, formatter={'float_kind':'{0:.3e}'.format}), "b: {0:.5e}".format(b))
            
    return w, b, J_history, p_history


