import numpy as np

def sigma(x):
    return 1 / (1 + np.exp(-x))

def initialize(hidden_size, visible_size):
    sizes = np.hstack([[visible_size], hidden_size, [visible_size]])
    W = []
    b = []
    for i in range(sizes.shape[0] - 1):
        W.append(np.random.randn(sizes[i] * sizes[i + 1]))
        b.append(np.zeros(sizes[i + 1]))
    res = np.concatenate(W + b)
    return res

def autoencoder_loss(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    sizes = np.hstack([[visible_size], hidden_size, [visible_size]])
    W = []
    b = []
    N = data.shape[0]
    pos = 0
    for i in range(sizes.shape[0] - 1):
        W.append(theta[pos : pos + sizes[i] * sizes[i + 1]].reshape(sizes[i], sizes[i + 1]))
        pos += sizes[i + 1] * sizes[i]
    for i in range(len(W)):
        b.append(theta[pos : pos + sizes[i + 1]][np.newaxis, :])
        pos += sizes[i + 1]
    
    a = [data]
    for i in range(len(W)):
        a.append(sigma(a[i].dot(W[i]) + b[i]))
    ro = []
    for i in range(len(a)):
        ro.append(a[i].mean(axis=0))
    
    loss = ((a[-1] - data) ** 2).sum() / (2 * N)
    for i in range(len(W)):
        loss += lambda_ / 2 * (W[i] ** 2).sum() 
    for i in range(1, len(ro) - 1):
        loss += beta * (sparsity_param * np.log(sparsity_param / ro[i]) + 
                        (1 - sparsity_param) * np.log((1 - sparsity_param) / (1 - ro[i]))).sum()
    
    d = [0] * len(sizes)
    d[-1] = -(data - a[-1]) * a[-1] * (1 - a[-1]) / N
    for i in reversed(range(len(W))):
        d[i] = d[i + 1].dot(W[i].T)
        if i > 0:
            d[i] += beta / N * (- sparsity_param / ro[i] + (1 - sparsity_param) / (1 - ro[i]))[np.newaxis, :]
        d[i] *= a[i] * (1 - a[i])
    dW = []
    db = []
    for i in range(len(W)):
        dW.append((a[i].T.dot(d[i + 1]) + lambda_ * W[i]).ravel())
        db.append(d[i + 1].sum(axis=0))
    gradient = np.concatenate(dW + db)
    return (loss, gradient)    

def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data):
    sizes = np.hstack([[visible_size], hidden_size, [visible_size]])
    W = []
    b = []
    pos = 0
    for i in range(sizes.shape[0] - 1):
        W.append(theta[pos : pos + sizes[i] * sizes[i + 1]].reshape(sizes[i], sizes[i + 1]))
        pos += sizes[i + 1] * sizes[i]
    for i in range(len(W)):
        b.append(theta[pos : pos + sizes[i + 1]][np.newaxis, :])
        pos += sizes[i + 1]
    
    X = data
    for i in range(layer_number):
        X = sigma(X.dot(W[i]) + b[i])
    return X   