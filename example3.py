import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

a = 1
b = 1

m0 = 1
m1 = 10
m2 = 10
m3 = 1

w1 = np.random.rand(m1, m0 + 1)
w2 = np.random.rand(m2, m1 + 1)
w3 = np.random.rand(m3, m2 + 1)

w4 = np.random.rand(m1, m0 + 1)
w5 = np.random.rand(m2, m1 + 1)
w6 = np.random.rand(m3, m2 + 1)

eta = 0.054
#n = np.arange(10001)
n = np.arange(0, 10001, 1)
yp = np.array([0])
yphat = np.array([0])

for k in n:
    u = np.sin(2 * np.pi * k / 25) + np.sin(2 * np.pi * k / 10)
    u1 = u ** 3
    p = np.array([1, yp[-1]])

    # FORWARD PASS
    v1 = np.dot(w1, p)       ###Compute the input signal u based on the values in the array n.
                                ####Perform a forward pass through the first part of the network, consisting of layers w1, w2, and w3.
    phi_v1 = a * np.tanh(b * v1) ####Apply activation functions (tanh) to the weighted inputs (v1, v2, and v3).
    y1_k = np.concatenate(([1], phi_v1)) ##Concatenate bias terms ([1]) with the activation outputs.
    v2 = np.dot(w2, y1_k)
    phi_v2 = a * np.tanh(b * v2)
    y2_k = np.concatenate(([1], phi_v2))
    v3 = np.dot(w3, y2_k)
    y3 = v3   ###Calculate the output y3.

    # BACKWARD PASS
    p2 = np.array([1, u])

    # FORWARD PASS
    v4 = np.dot(w4, p2)   ##Compute the input signal u again.
                            ##Perform a forward pass through the second part of the network, consisting of layers w4, w5, and w6.
    phi_v4 = a * np.tanh(b * v4) ##Apply activation functions to the weighted inputs (v4, v5, and v6)
    y4_k = np.concatenate(([1], phi_v4)) ##Concatenate bias terms with the activation outputs.
    v5 = np.dot(w5, y4_k)
    phi_v5 = a * np.tanh(b * v5)
    y5_k = np.concatenate(([1], phi_v5))
    v6 = np.dot(w6, y5_k)
    y6 = v6         ##Calculate the output y6

    #print(yp.shape)

    # Error calculation
    yp1 = (yp[-1] / (1 + yp[-1] ** 2)) + u1
    # print(yp.size())
    # print(yp1.size())
    yp1 = np.array([yp1], dtype=np.float32)
    yp = np.concatenate((yp, yp1))

    yphat1 = y3 + y6
    yphat = np.concatenate((yphat, yphat1))

    E = yp1 - yphat1 ##Calculate the error E between the predicted output and the target output (yp1)

    phi_v3_diff = 1
    phi_v2_diff = (b / a) * (a ** 2 - phi_v2 ** 2)  ##Compute gradients of the error with respect to the weights in the first part of the network.
    phi_v1_diff = (b / a) * (a ** 2 - phi_v1 ** 2)
    delta3 = E * phi_v3_diff
    delta_w3 = eta * np.outer(delta3, y2_k)
    delta2 = np.matmul(w3[0, 1:].reshape(1,-1).T, delta3) * phi_v2_diff
    delta_w2 = eta * np.outer(delta2, y1_k)
    delta1 = np.dot(w2[:, 1:].T, delta2) * phi_v1_diff
    delta_w1 = eta * np.outer(delta1, p)

    # Weight updation
    w1 += delta_w1  ####Update weights (w1, w2, and w3) using the calculated gradients
    w2 += delta_w2
    w3 += delta_w3

    phi_v6_diff = 1
    phi_v5_diff = (a / b) * (a ** 2 - phi_v5 ** 2)  ##Compute gradients of the error with respect to the weights in the second part of the network.
    phi_v4_diff = (a / b) * (a ** 2 - phi_v4 ** 2)

    # BACKWARD PASS
    delta6 = E * phi_v6_diff
    delta_w6 = eta * np.outer(delta6, y5_k)
    delta5 = np.matmul(w6[0,1:].reshape(1,-1).T, delta6) * phi_v5_diff
    delta_w5 = eta * np.outer(delta5, y4_k)
    delta4 = np.dot(w5[:, 1:].T, delta5) * phi_v4_diff
    delta_w4 = eta * np.outer(delta4, p2)

    # Weight updation
    w4 += delta_w4  ##Update weights (w4, w5, and w6) using the calculated gradients.
    w5 += delta_w5
    w6 += delta_w6



# Plotting Graphs


K = np.arange(10002)
plt.plot(K, yp, '-r', label='Neural Network Output')
plt.plot(K, yphat, '--g', label='Desired Output')
plt.legend()
plt.title('Example 3c Function Approximation')
plt.axis([0, 100, -10, 10])
plt.show()

# Performance Metrics
g = np.var(np.array(yp[1:]) - np.array(yphat[1:]))
h = np.var(np.array(yp[1:]))
perf = (1 - (g / h)) * 100
print(f"Performance: {perf}%")

mse = np.sum((np.array(yp[1:]) - np.array(yphat[1:])) ** 2) / len(yp[1:])
print(f"MSE: {mse}")
