from darknet_OLD import *


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1

def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr



def predict_tactic(net, s):
    prob = 0
    d = c_array(c_float, [0.0]*256)
    tac = ''
    if not len(s):
        s = '\n'
    for c in s[:-1]:
        d[ord(c)] = 1
        pred = predict(net, d)
        d[ord(c)] = 0
    c = s[-1]
    while 1:
        d[ord(c)] = 1
        pred = predict(net, d)
        d[ord(c)] = 0
        pred = [pred[i] for i in range(256)]
        ind = sample(pred)
        c = chr(ind)
        prob += math.log(pred[ind])
        if len(tac) and tac[-1] == '.':
            break
        tac = tac + c
    return (tac, prob)

def predict_tactics(net, s, n):
    tacs = []
    for i in range(n):
        reset_rnn(net)
        tacs.append(predict_tactic(net, s))
    tacs = sorted(tacs, key=lambda x: -x[1])
    return tacs

net = load_net("../cfg/coq.test.cfg", "/home/pjreddie/backup/coq.backup", 0)
t = predict_tactics(net, "+++++\n", 10)
print t
