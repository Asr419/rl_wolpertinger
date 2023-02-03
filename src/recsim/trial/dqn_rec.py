from rl.agents.dqn import DQNAgent

if __name__ == "__main__":
    dqn = DQNAgent.load_weights(filepath="./model/dqn_0.h5")
    a = [[94, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    t = 0
    q_value = dqn.compute_q_values(a)
    for i in range(1, 10):
        z = dqn.compute_q_values(a)
        max_v = max(z)
        t += max_v
        print(max_v)
        z = z.tolist()
        index = z.index(max_v)
        a[0][i] = index
    print(a[0])
    print(t)
